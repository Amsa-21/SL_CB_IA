from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from app.db.session import get_session
from typing import List, Dict, Any
import pandas as pd
import logging
import re
import json
from datetime import datetime, timedelta
import unicodedata
from thefuzz import fuzz
import markdown as _md
from app.core import config
import inspect

logger = logging.getLogger(__name__)

def log_function_call(func):
    async def async_wrapper(*args, **kwargs):
        logger.debug(f"Entering function: {func.__name__}")
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in function {func.__name__}: {e}", exc_info=True)
            raise
    
    def sync_wrapper(*args, **kwargs):
        logger.debug(f"Entering function: {func.__name__}")
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in function {func.__name__}: {e}", exc_info=True)
            raise

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

@log_function_call
def add_br_outside_blocks(text: str) -> str:
    """Ajoute <br/> uniquement s'il n'est pas déjà dans un bloc HTML (table/ul/ol/pre/code)."""
    lines = text.split("\n")
    new_lines = []
    inside_block = False
    for line in lines:
        if re.match(r"<(table|ul|ol|pre|code)[ >]", line):
            inside_block = True
        if re.match(r"</(table|ul|ol|pre|code)>", line):
            inside_block = False
            new_lines.append(line)
            continue
        if not inside_block and line.strip() != "":
            new_lines.append(line + "<br/>")
        else:
            new_lines.append(line)
    joined = "\n".join(new_lines)
    joined = re.sub(r"<br/>\n(<(table|ul|ol|pre|code)[ >])", r"\n\1", joined)
    joined = re.sub(r"(</(table|ul|ol|pre|code)>)<br/>", r"\1", joined)
    joined = re.sub(r"(<br/>[\s]*){3,}", r"<br/><br/>", joined)
    return joined

@log_function_call
def convert_markdown_to_html(md_text: str) -> str:
    """
    Convertit un texte Markdown (incluant un tableau) en HTML.
    Corrige les espaces et sauts de ligne pour permettre la conversion des tableaux.
    """
    md_text = md_text.replace("\\n", "\n")
    md_text = md_text.replace(" ", " ")
    md_text = re.sub(r"\n{3,}", "\n\n", md_text)

    extensions = [
        "tables",
        "fenced_code",
        "nl2br",
        "sane_lists",
        "toc",
        "attr_list"
    ]

    html = _md.markdown(md_text, extensions=extensions, output_format="html5")
    
    return html

@log_function_call
def format_time(delta: timedelta) -> str:
    """Formate un timedelta en texte HTML minutes/secondes."""
    total_seconds = int(delta.total_seconds())
    minutes = total_seconds // 60
    secs = total_seconds % 60
    if minutes:
        text_time = f"{minutes}m {secs:.0f}s"
    else:
        text_time = f"{secs:.0f}s"
    return f'<span style="font-size:10px;font-style:italic;text-align:right;display:block;">{text_time}</span>'

@log_function_call
async def get_rules(cat: str):
    """Récupère la liste des règles pour une catégorie, retourne les labels non vides."""
    try:
        categorie_fk = await from_cat_to_fk(cat)
        if categorie_fk is None:
            logger.warning(f"Aucune catégorie trouvée pour le label : {cat}")
            return []
        lines = await execute_sp(
            "ia.sp_categorieRegle_get",
            {"user_fk": config.USER_FK, "categorie_fk": categorie_fk}
        )
        return [line.get("regleL") for line in lines if line.get("regleL")]
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des règles : {e}")
        return []

@log_function_call
async def from_cat_to_fk(cat: str) -> int | None:
    """Retourne categorie_fk correspondant à un label de catégorie donné, ou None."""
    try:
        categories = await execute_sp("ia.sp_categorie_get", {"user_fk": config.USER_FK})
        cat_normalized = unicodedata.normalize('NFKD', cat).casefold()
        for category in categories:
            label = category.get("theLabel")
            if not label:
                continue
            label_normalized = unicodedata.normalize('NFKD', label).casefold()
            if label_normalized == cat_normalized:
                return category.get("categorie_fk")
        logger.warning(f"Aucune catégorie trouvée pour le label : {cat}")
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la catégorie '{cat}': {e}")
    return None

@log_function_call
def generate_hierarchy_codes(data: list, parent_code: str = "") -> list:
    """
    Génère les codes hiérarchiques dans l'arbre (modification In-Place).
    """
    if not isinstance(data, list):
        return []

    for index, node in enumerate(data):
        current_level_num = index + 1
        new_code = f"{parent_code}{current_level_num}."
        node['code'] = new_code
        if 'children' in node and node['children']:
            generate_hierarchy_codes(node['children'], new_code)
    return data

@log_function_call
def extract_flat_hierarchy_list(data: list[dict]) -> list:
    """
    Extrait tous les labels et leurs codes dans une liste plate.
    """
    flat_list = []
    if not isinstance(data, list):
        return flat_list
    for node in data:
        flat_list.append({
            'label': node.get('label'),
            'code': node.get('code')
        })
        if 'children' in node and node['children']:
            flat_list.extend(extract_flat_hierarchy_list(node['children']))
    return flat_list

@log_function_call
def get_descendant_labels(node: dict) -> list:
    """Fonction récursive pour extraire tous les labels descendants d'un nœud."""
    labels = []
    for child in node.get('children', []):
        labels.append(child['label'])
        labels.extend(get_descendant_labels(child))
        
    return labels

@log_function_call
def extract_all_descendants_for_list(data_list: list) -> list[list]:
    """
    Pour chaque élément de la liste de niveau supérieur, extrait la liste de tous ses descendants.
    
    :param data_list: La liste principale des dictionnaires hiérarchiques.
    :return: Une liste de listes, où chaque sous-liste contient les labels des descendants.
    """
    result = []
    for i, item in enumerate(data_list):
        descendant_labels = get_descendant_labels(item)
        descendant_labels.append(data_list[i]['label'])
        result.append(descendant_labels)
    return result

@log_function_call
def preprocessing_data(df: pd.DataFrame, simple_dict: list[dict]) -> pd.DataFrame:
    """Prétraite et catégorise le DataFrame selon les groupes définis."""
    try:
        required_columns = ['Lignes', 'Contexte', 'Nature de l\'écriture', 'Année', 'Mois', 'Montant']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"La colonne requise '{col}' est absente du DataFrame.")
        mask = (
            (df.iloc[:, 0] == df.iloc[0, 0]) &
            (df.iloc[:, 2] == "Compte d'exploitation") &
            (df.iloc[:, 8] != "Colonne variation")
        )
        df_filtered: pd.DataFrame
        df_filtered = df[mask].copy()
        
        if df_filtered.empty:
            df_filtered['Groupe'] = pd.Series(dtype='object')
            return df_filtered
            
        mask_pct_recettes = df_filtered['Lignes'] == "% DES RECETTES TOTALES"
        df_pct_recettes = df_filtered[mask_pct_recettes].copy()
        df_autres = df_filtered[~mask_pct_recettes].copy()
        if not df_pct_recettes.empty:
            df_pct_recettes = df_pct_recettes.sort_values("Montant", ascending=False).reset_index(drop=True)
            code_map = {0: 4, 1: 7, 2: 10}
            df_pct_recettes['Code Hiérarchique'] = df_pct_recettes.index.map(lambda idx: code_map.get(idx, 10))
        coded_tree = generate_hierarchy_codes(simple_dict)
        hierarchy_list = extract_flat_hierarchy_list(coded_tree)
        mapping_dict = {item['label']: item['code'] for item in hierarchy_list}
        if not df_autres.empty:
            df_autres['Code Hiérarchique'] = df_autres['Lignes'].map(mapping_dict)
        df_filtered = pd.concat([df_autres, df_pct_recettes]).sort_index(kind="stable")
        cols = list(df.columns)
        if "Code Hiérarchique" in cols and "Lignes" in cols:
            cols.remove("Code Hiérarchique")
            insert_idx = cols.index("Lignes")
            cols = cols[:insert_idx] + ["Code Hiérarchique"] + cols[insert_idx:]
            df_filtered = df_filtered[cols]
        result_list = extract_all_descendants_for_list(simple_dict)
        CHIFFRE_AFFAIRES = result_list[0] + result_list[3] + result_list[6] + result_list[9] + result_list[10]
        CHARGES = result_list[1] + result_list[4] + result_list[7] + result_list[12]
        MARGES = result_list[2] + result_list[5] + result_list[8] + result_list[11] + result_list[13]
        groupes_dict = {
            "Chiffre d'affaire": CHIFFRE_AFFAIRES,
            "Charge": CHARGES,
            "Marge": MARGES
        }
        groupes_mapping = {poste: groupe for groupe, postes in groupes_dict.items() for poste in postes}
        df_filtered['Groupe'] = df_filtered['Lignes'].map(groupes_mapping)
        df_filtered = df_filtered.drop_duplicates().reset_index(drop=True)
        return df_filtered
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement des données : {e}")
        raise

@log_function_call
async def parse_user_query(
    query: str, 
    synonymes_groupes: dict[str, list[str]], 
    simple_dict: list[dict]) -> dict[str, list[str]]:
    """Transforme une question utilisateur en paramètres de fonction."""
    def _strip_accents(text: str) -> str:
        """
        Supprime les accents et normalise la casse/ponctuation dans une chaîne pour faciliter les comparaisons.
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        s = (
            text.replace("’", "'")
                .replace("`", "'")
                .replace("‘", "'")
                .replace("–", "-")
                .replace("—", "-")
        )
        s = unicodedata.normalize("NFD", s)
        s = ''.join(
            c for c in s
            if unicodedata.category(c) != 'Mn'
        )
        s = re.sub(r"\s+", " ", s.lower()).strip()
        return s

    async def _check_detail(question: str) -> bool:
        question_norm = _strip_accents(question.lower())
        res = await get_mapping()
        keywords_details = res["Détail"]
        for kw in keywords_details:
            if fuzz.partial_ratio(question_norm, kw) > 80:
                return True
        return False

    doc = config.nlp(query)
    query_lower = query.lower()
    params = {
        'groupes': [],
        'types_valeur': [],
        'annees': [],
        'nature_ecriture': [],
        'lignes': [],
        'mois': []
    }

    # GROUPE
    def _select_groupe(question: str, threshold: float = 70):
        query_norm = _strip_accents(question)

        best_matches: dict[str, float] = {}
        max_overall_score = 0.0

        for g, mots in list(synonymes_groupes.items())[:3]:
            max_group_score = 0.0
            for m in mots:
                m_norm = _strip_accents(m)
                score = fuzz.token_set_ratio(query_norm, m_norm)
                if score > max_group_score:
                    max_group_score = score
                    
            best_matches[g] = max_group_score
            
            if max_group_score > max_overall_score:
                max_overall_score = max_group_score
        res = []
        if max_overall_score >= threshold:
            score_tolerance = 5
            for g, score in best_matches.items():
                if score >= threshold and score >= (max_overall_score - score_tolerance):
                    res.append(g)
        return res
    params['groupes'] = _select_groupe(query_lower)

    # TYPE DE VALEUR
    def _define_type(question: str):
        types_valeur = set()
        query_lower_noacc = _strip_accents(question)
        if re.search(r"reel|realise|actuel", query_lower_noacc):
            types_valeur.add('R')
        if re.search(r"budget", query_lower_noacc):
            types_valeur.add('B')
        if re.search(r"prevision|prevu|projection|project|prev", query_lower_noacc):
            types_valeur.add('P')
        for ent in doc.ents:
            ent_text_noacc = _strip_accents(ent.text.lower())
            if ent.label_ == "MISC":
                if re.search(r"reel|realise|actuel", ent_text_noacc):
                    types_valeur.add('R')
                if re.search(r"budget", ent_text_noacc):
                    types_valeur.add('B')
                if re.search(r"prevision|prevu|projection|project|prev", ent_text_noacc):
                    types_valeur.add('P')
        return sorted(types_valeur)
    params['types_valeur'] = _define_type(query_lower)

    # ANNEE
    def _define_year(question: str):
        annees = set(map(int, re.findall(r"\b20\d{2}\b", question)))
        
        pattern_2digit = r"\b(?:Budget|Prev|Reel)\s*'?(\d{2})\b"
        matches_2digit = re.findall(pattern_2digit, question, re.IGNORECASE)
        for match in matches_2digit:
            yy = int(match)
            if yy < 50:
                year = 2000 + yy
            else:
                year = 1900 + yy
            annees.add(year)

        for ent in doc.ents:
            if ent.label_ == "DATE":
                if ent.text.isdigit() and len(ent.text) == 4 and ent.text.startswith("20"):
                    annees.add(int(ent.text))
                elif re.match(r"20\d{2}[-/ ]20\d{2}", ent.text):
                    annees.update(map(int, re.findall(r"20\d{2}", ent.text)))
                else:
                    annees.update(map(int, re.findall(r"20\d{2}", ent.text)))

                ent_matches = re.findall(pattern_2digit, ent.text, re.IGNORECASE)
                for em in ent_matches:
                    yy = int(em)
                    if yy < 50:
                        year = 2000 + yy
                    else:
                        year = 1900 + yy
                    annees.add(year)

        interval_patterns_annee = [
            r"(?:entre)\s+20(\d{2})\s+(?:et)\s+20(\d{2})",
            r"(?:de)\s+20(\d{2})\s+(?:à|a)\s+20(\d{2})"
        ]
        for pat in interval_patterns_annee:
            m = re.search(pat, question)
            if m:
                y1, y2 = int("20" + m.group(1)), int("20" + m.group(2))
                if y1 <= y2:
                    annees.update(range(y1, y2 + 1))
                else:
                    annees.update(range(y2, y1 + 1))

        if annees:
            return sorted(annees)
        else:
            if re.search(r"cette (année|annee)", question):
                return [datetime.now().year]
            elif re.search(r"(année|annee) dernière|an dernier|(année|annee) (passée|passee)|(année|annee) (précédente|precedente)", question):
                return [datetime.now().year - 1]
            elif re.search(r"(l'année|l'annee) prochaine|l'an prochain", question):
                return [datetime.now().year + 1]
            else:
                return [datetime.now().year]
    params['annees'] = _define_year(query_lower)

    # MOIS
    def _define_month(question: str):
        mois_map = {
            1: [r"janv(?:ier)?", r"jan"],
            2: [r"f[ée]vr(?:ier)?", r"fev"],
            3: [r"mars?", r"mar"],
            4: [r"avr(?:il)?", r"avr"],
            5: [r"mai"],
            6: [r"juin"],
            7: [r"juil(?:let)?", r"jul"],
            8: [r"ao[uû]t?", r"aou"],
            9: [r"sept(?:embre)?", r"sep"],
            10: [r"oct(?:obre)?", r"oct"],
            11: [r"nov(?:embre)?", r"nov"],
            12: [r"d[ée]c(?:embre)?", r"dec"],
        }

        query_norm = _strip_accents(question)
        mois = set()
        interval_patterns = [
            r"(?:entre)\s+([a-zéûî\.]+)\s+(?:et)\s+([a-zéûî\.]+)",
            r"(?:de)\s+([a-zéûî\.]+)\s+(?:à|a)\s+([a-zéûî\.]+)",
        ]
        interval_match = None
        for pat in interval_patterns:
            m = re.search(pat, query_norm)
            if m:
                interval_match = m
                break

        if interval_match:
            mois1_txt, mois2_txt = interval_match.groups()
            mois1_txt = mois1_txt.replace('.', '')
            mois2_txt = mois2_txt.replace('.', '')
            mois1_num, mois2_num = None, None
            for num, patterns in mois_map.items():
                for pat in patterns:
                    if re.fullmatch(pat, mois1_txt):
                        mois1_num = num
                    if re.fullmatch(pat, mois2_txt):
                        mois2_num = num
            if mois1_num and mois2_num:
                if mois1_num <= mois2_num:
                    mois = set(range(mois1_num, mois2_num + 1))
                else:
                    mois = set(list(range(mois1_num, 13 + 1)) + list(range(1, mois2_num + 1)))
        else:
            for num, patterns in mois_map.items():
                for pat in patterns:
                    if re.search(rf"\b{pat}\b", query_norm):
                        mois.add(num)

            trimestre_regex = [
                (r"\b(1(er)?|premier|i+)[s\-]*(trimestre|trim)\b", [1, 2, 3]),
                (r"\b(2(e|ème|eme)?|deuxi[eè]me|ii+)[s\-]*(trimestre|trim)\b", [4, 5, 6]),
                (r"\b(3(e|ème|eme)?|troisi[eè]me|iii+)[s\-]*(trimestre|trim)\b", [7, 8, 9]),
                (r"\b(4(e|ème|eme)?|quatri[eè]me|iv+)[s\-]*(trimestre|trim)\b", [10, 11, 12]),
            ]
            for pat, mois_list in trimestre_regex:
                if re.search(pat, query_norm):
                    mois = set(mois_list)
                    break

            semestre_regex = [
                (r"\b(1(er)?|premier|i+)[s\-]*(semestre|sem)\b", [1, 2, 3, 4, 5, 6]),
                (r"\b(2(e|ème|eme)?|deuxi[eè]me|ii+)[s\-]*(semestre|sem)\b", [7, 8, 9, 10, 11, 12]),
            ]
            for pat, mois_list in semestre_regex:
                if re.search(pat, query_norm):
                    mois = set(mois_list)
                    break

            if re.search(r"\btous les mois\b", query_norm):
                mois = set(range(1, 13))
            elif re.search(r"\bmois courant\b", query_norm):
                mois = {datetime.now().month}
            elif re.search(r"\bmois dernier\b", query_norm):
                mois = {datetime.now().month - 1 if datetime.now().month > 1 else 12}
            elif re.search(r"\bmois prochain\b", query_norm):
                mois = {datetime.now().month + 1 if datetime.now().month < 12 else 1}
        return sorted(mois)
    params['mois'] = _define_month(query_lower)

    # NATURE ECRITURE
    def _select_nature_ecriture(question: str):
        nature_ecritures = set()
        if re.search(r"\b(mensuel(le)?|mois|trimestre|semestre|janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\b", question):
            nature_ecritures.add('Mensuelle')
        if re.search(r"\b(annuel(le)?|total|cette année|cette annee|année|annee)\b", question):
            nature_ecritures.add('Annuelle')
        if not nature_ecritures:
            if params.get('mois') and 0 < len(params['mois']) < 12:
                nature_ecritures.add('Mensuelle')
            if params.get('mois') and len(params['mois']) == 12:
                nature_ecritures.add('Annuelle')
        if not nature_ecritures:
            nature_ecritures.add('Annuelle')
        return [v for v in ['Mensuelle', 'Annuelle'] if v in nature_ecritures]
    params['nature_ecriture'] = _select_nature_ecriture(query_lower)

    # LIGNES
    result_list = extract_all_descendants_for_list(simple_dict)
    LISTE_LIGNES = [label for sublist in result_list for label in sublist]
    async def _match_lignes(question: str, threshold: int = 75, return_scores: bool = False):
        q = _strip_accents(question)
        q_tokens = set(re.findall(r"\w+", q))
        results = {}
        for ligne in LISTE_LIGNES:
            ln = _strip_accents(ligne)
            ln_tokens = set(re.findall(r"\w+", ln))
            if ln_tokens and ln_tokens.issubset({t.rstrip('s') for t in q_tokens} | q_tokens):
                results[ligne] = 100
                continue
            score = fuzz.token_set_ratio(q, ln)
            if score >= threshold:
                results[ligne] = max(results.get(ligne, 0), score)

        ordered = sorted(results.items(), key=lambda x: -x[1])
        detail = await _check_detail(question)
        if detail and ordered:
            enfants = []
            for l in ordered:
                enfants += get_children_by_label(simple_dict, l[0])
            return set(enfants + [lbl for lbl, _ in ordered])
        return ordered if return_scores else [lbl for lbl, _ in ordered]
    params['lignes'] = await _match_lignes(query_lower)

    return params

@log_function_call
async def get_data_for_llm(
    df: pd.DataFrame,
    simple_dict: list[dict],
    sa_fk: int,
    form_fk: int,
    types_valeur: list[str],
    nature_ecriture: list[str],
    groupes: list[str] = None,
    mois: list = None,
    annees: list = None,
    lignes: list[str] = None) -> str:
    """
    Args:
        df (pd.DataFrame): DataFrame des écritures catégorisées à filtrer.
        simple_dict (list[dict]): Arbre hiérarchique des lignes/postes.
        types_valeur (list[str]): Types de valeur à inclure (ex: ['R', 'B']).
        nature_ecriture (list[str]): Types d'écriture à inclure (ex: ['Annuelle', 'Mensuelle']).
        groupes (list[str], optional): Groupes analytiques à filtrer. None = tous.
        mois (list[int], optional): Mois à inclure (ex: [1,2,3]). None = tous.
        annees (list[int], optional): Années à inclure (ex: [2024,2025]). None = toutes.
        lignes (list[str], optional): Postes/écritures à inclure. None = tous.

    Returns:
        is_not_empty (boolean): 
        df_markdown (str): Tableau Markdown filtré pour le prompt, ou texte vide si aucun résultat.
    """
    df_temp: pd.DataFrame = df.copy()
    for col in ['Montant', 'Année', 'Mois']:
        if col in df_temp.columns:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
    df_temp['Groupe'] = df_temp['Groupe'].fillna('')

    filtres = (
        df_temp['Contexte'].str.lower().isin([t.lower() for t in types_valeur]) &
        df_temp["Nature de l'écriture"].str.lower().isin([n.lower() for n in nature_ecriture])
    )

    if annees:
        filtres &= df_temp['Année'].isin([int(a) for a in annees])

    if mois and len(mois) > 0:
        mois = [int(m) for m in mois]
        filtres = (
            filtres &
            (
                ((df_temp["Nature de l'écriture"].str.lower() == "mensuelle") & df_temp["Mois"].isin(mois)) |
                (df_temp["Nature de l'écriture"].str.lower() == "annuelle")
            )
        )
    
    filtres_hierarchie = df_temp['Code Hiérarchique'].apply(
        lambda x: isinstance(x, str) and x.count('.') <= 2
    )

    result_list = extract_all_descendants_for_list(simple_dict)
    CHIFFRE_AFFAIRES = result_list[0] + result_list[3] + result_list[6] + result_list[9] + result_list[10]
    CHARGES = result_list[1] + result_list[4] + result_list[7] + result_list[12]
    MARGES = result_list[2] + result_list[5] + result_list[8] + result_list[11] + result_list[13]

    if lignes and len(lignes) > 0 and groupes and len(groupes) > 0:
        lignes_valides_du_groupe: list[str] = []
        for groupe in groupes:
            if groupe == "Chiffre d'affaire":
                lignes_valides_du_groupe += CHIFFRE_AFFAIRES
            elif groupe == "Charge":
                lignes_valides_du_groupe += CHARGES
            elif groupe == "Marge":
                lignes_valides_du_groupe += MARGES
        
        toutes_lignes_coherentes = all(l in lignes_valides_du_groupe for l in lignes)
        
        if toutes_lignes_coherentes:
            filtres_lignes = df_temp["Lignes"].str.lower().isin([l.lower() for l in lignes])
            filtres &= filtres_lignes
        else:
            filtres_groupes = df_temp['Groupe'].str.lower().isin([g.lower() for g in groupes])
            filtres &= filtres_groupes
    elif lignes and len(lignes) > 0:
        filtres_lignes = df_temp["Lignes"].str.lower().isin([l.lower() for l in lignes])
        filtres &= filtres_lignes
    elif groupes and len(groupes) > 0:
        filtres_groupes = df_temp['Groupe'].str.lower().isin([g.lower() for g in groupes])
        filtres &= filtres_groupes
        filtres &= filtres_hierarchie
    else:
        filtres &= filtres_hierarchie

    df_filtre: pd.DataFrame = df_temp.loc[filtres].copy()

    all_annu = all(n.lower() == "annuelle" for n in nature_ecriture)
    all_mensu = all(n.lower() == "mensuelle" for n in nature_ecriture)

    def _code_to_tuple(code):
        if pd.isna(code):
            return (999999,)
        s = str(code).strip()
        if s.endswith('.'):
            s = s.rstrip('.')
        nums = re.findall(r'\d+', s)
        if not nums:
            return (999999,)
        return tuple(int(n) for n in nums)

    def _filter_constant_columns(cols):
        if df_filtre is not None and len(df_filtre) >= 2:
            constant_columns = cols
            for col_name in constant_columns:
                if col_name in df_filtre.columns and df_filtre[col_name].nunique() == 1:
                    cols = [c for c in cols if c != col_name]
        return cols

    def _flatten_labels(simple_dict: list[dict]) -> pd.DataFrame:
        def __flatten_labels(simple_dict: list[dict]) -> list[dict]:
            flat_list = []
            for item in simple_dict:
                label = item.get("label", "")
                code = item.get("code", "")
                flat_list.append({"code": code, "label": label})
                children = item.get("children", [])
                if children:
                    flat_list.extend(__flatten_labels(children))
            return flat_list
        return pd.DataFrame(__flatten_labels(simple_dict), columns=["code", "label"])

    def _code_parts_raw(code: str) -> list:
        if not isinstance(code, str):
            return []
        parts = code.split('.')
        parts = [p for p in parts if p != ""]
        return parts

    def _find_siblings_with_neighbors(
        label: str, 
        df_labels_codes: pd.DataFrame, 
        n_before: int = 0, 
        n_after: int = 0
    ) -> str:
        row_mask = df_labels_codes["label"] == label
        if not row_mask.any():
            return []

        idx = df_labels_codes.index[row_mask][0]
        code_central = df_labels_codes.at[idx, "code"]
        central_parts = _code_parts_raw(code_central)

        if len(central_parts) <= 1:
            def __is_root(c):
                return len(_code_parts_raw(c)) == 1
            df_roots = df_labels_codes[df_labels_codes["code"].apply(__is_root)].copy()
            label_idx_in_roots = df_roots.index[df_roots["label"] == label]
            if label_idx_in_roots.empty:
                return df_roots["label"].to_list()
            i = label_idx_in_roots[0]
            start = max(0, i - n_before)
            end = i + n_after + 1
            sel = df_roots.iloc[start:end]
            return sel["label"].to_list()

        parent_parts = central_parts[:-1]
        target_depth = len(central_parts)

        def __is_same_parent(c):
            parts = _code_parts_raw(c)
            if len(parts) != target_depth:
                return False
            cand_parent = parts[:-1]
            return cand_parent == parent_parts

        df_siblings = df_labels_codes[df_labels_codes["code"].apply(__is_same_parent)].copy()
        label_idx_in_siblings = df_siblings.index[df_siblings["label"] == label]
        if label_idx_in_siblings.empty:
            return df_siblings.reset_index(drop=True)["label"].to_list()
        i = label_idx_in_siblings[0]
        i_relative = list(df_siblings.index).index(i)
        start = max(0, i_relative - n_before)
        end = i_relative + n_after + 1
        sel: pd.DataFrame = df_siblings.iloc[start:end]
        return sel["label"].to_list()

    async def _detail_cell(
        sa_fk: int,
        theYear: int,
        context: str,
        theMonth: int,
        label: str,
        res: list[dict]
    ) -> pd.DataFrame:
        def __filter_columns_by_indices(df: pd.DataFrame, indices: list[int]):
            existing_indices = [i for i in indices if i < df.shape[1]]
            return df.iloc[:, existing_indices]

        def __get_line_fk(label: str, res: list[dict]=res):
            df = pd.DataFrame(res).iloc[:, [0, 4]]
            ser = df[df["label"] == label]["line_id"]
            return int(ser.iloc[0])

        lst = await execute_sp(
            "dbo.sp_simBudValueOne",
            {
                "user_fk": 8,
                "listSA": 0,
                "line_fk": __get_line_fk(label),
                "sa_fk": sa_fk,
                "listCR": 0,
                "cr_fk": 0,
                "theYear": theYear,
                "RB": context,
                "theMonth": theMonth
            }
        )
        if not lst or not isinstance(lst, list):
            return pd.DataFrame()
        
        indices_to_keep = [3, 10, 12, 14, 15]
        df = pd.DataFrame(lst)
        if df.shape[1] >= max(indices_to_keep) + 1:
            return __filter_columns_by_indices(df, indices_to_keep)
        else:
            existing = [i for i in indices_to_keep if i < df.shape[1]]
            if existing:
                return __filter_columns_by_indices(df, existing)
        return pd.DataFrame(lst)["Lignes"].to_list()
    
    if all_annu:
        colonnes_finales = ['Code Hiérarchique', 'Lignes', 'Année', 'Contexte', 'Montant']
        colonnes_tri = ['Code Hiérarchique', 'Année', 'Lignes']
    elif all_mensu:
        colonnes_finales = ['Code Hiérarchique', 'Lignes', 'Mois', 'Année', 'Contexte', 'Montant']
        colonnes_tri = ['Code Hiérarchique', 'Année', 'Mois', 'Lignes']
    else:
        colonnes_finales = ['Code Hiérarchique', 'Lignes', 'Mois', 'Année', 'Nature de l\'écriture', 'Contexte', 'Montant']
        colonnes_tri = ['Code Hiérarchique', 'Année']
        if 'Mois' in df_filtre.columns:
            colonnes_tri.append('Mois')
        colonnes_tri.extend(['Nature de l\'écriture', 'Lignes'])

    colonnes_finales = _filter_constant_columns(colonnes_finales)
    colonnes_tri = _filter_constant_columns(colonnes_tri)
    tri_effectif = [col for col in colonnes_tri if col in df_filtre.columns]

    if 'Code Hiérarchique' in tri_effectif and 'Code Hiérarchique' in df_filtre.columns:
        df_filtre['_code_sort'] = df_filtre['Code Hiérarchique'].apply(_code_to_tuple)
        tri_effectif = ['_code_sort' if c == 'Code Hiérarchique' else c for c in tri_effectif]

    colonnes_presentes = [col for col in colonnes_finales if col in df_filtre.columns]

    need_code_sort = '_code_sort' in tri_effectif and '_code_sort' not in colonnes_presentes
    if need_code_sort:
        cols_for_sort = ['_code_sort'] + colonnes_presentes
    else:
        cols_for_sort = colonnes_presentes

    df_final = df_filtre[cols_for_sort].sort_values(by=tri_effectif).reset_index(drop=True)
    if need_code_sort:
        df_final = df_final.drop(columns=['_code_sort'])

    # * Case: Line empty * #
    if lignes and lignes[0] not in df_final["Lignes"].values:
        logger.debug(f"La ligne '{lignes[0]}' ne contient aucune valeur")
        res = await execute_sp(
            "dbo.sp_simBudLines",
            {
                "user_fk": config.USER_FK,
                "form_fk": form_fk,
                "line_fk": 0,
                "choix": 0,
                "isVisible": 1
            }
        )
        df_labels_codes = _flatten_labels(simple_dict)
        n: int = config.N_NEIGHBORS
        lines = _find_siblings_with_neighbors(label=lignes[0], df_labels_codes=df_labels_codes, n_before=n, n_after=n)
        rows = []
        for l in lines:
            if nature_ecriture[0] == "Annuelle":
                r = await _detail_cell(
                    sa_fk=sa_fk,
                    theYear=annees[0],
                    context=types_valeur[0],
                    theMonth=0,
                    label=l,
                    res=res
                )
            else:
                r = await _detail_cell(
                    sa_fk=sa_fk,
                    theYear=annees[0],
                    context=types_valeur[0],
                    theMonth=mois[0],
                    label=l,
                    res=res
                )
            if "Source" in r.columns:
                sources = ", ".join(r['Source'].unique())
                rows.append({"Ligne": l, "Source": sources})
            else:
                rows.append({"Ligne": l, "Source": "Non disponible"})
        return False, str(
            f"La ligne '{lignes[0]}' ne contient aucune valeur. "
            "Analyse le tableau ci-dessous regroupant les lignes voisines et leurs sources pour expliquer précisément "
            "(mais brièvement) pourquoi aucune donnée n'est disponible pour cette ligne. "
            "Comparez avec les voisins pour mieux argumenter l'absence d'information.\n"
            f"{pd.DataFrame(rows, columns=['Ligne', 'Source']).to_markdown(index=False)}"
        )
    
    if df_final.empty:
        return False, None
    else:
        df_final["Montant"] = df_final["Montant"].round(0).astype(int)
        return True, df_final.to_markdown(index=False)

@log_function_call
def count_tokens(text: str) -> int:
    """Compte de tokens naïf basé sur découpage mots."""
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return len(tokens)

@log_function_call
async def execute_sp(sp_name: str, params: dict) -> List[Dict[str, Any]]:
    """
    Exécute une procédure stockée SQL et retourne la liste de dicts résultat (ou vide).
    Gère aussi bien les SP de type select que add (insert/update).
    """
    results = []
    try:
        async with get_session() as session:
            param_keys = ", ".join([f":{key}" for key in params.keys()])
            sql_query = text(f"EXEC {sp_name} {param_keys}" if param_keys else f"EXEC {sp_name}")
            result_proxy = await session.execute(sql_query, params)
            if sp_name.startswith("ia.sp_") and sp_name.endswith("_add"):
                await session.commit()
            try:
                for row in result_proxy:
                    results.append(dict(row._mapping))
            except Exception:
                pass
            return results
    except ProgrammingError as e:
        logger.error(f"SQLAlchemy ProgrammingError: {e}")
        raise
    except Exception as e:
        logger.error(f"Database connection or execution failed: {e}")
        raise

@log_function_call
async def get_mapping() -> dict:
    """Récupère le mapping des catégories et synonymes depuis la base."""
    try:
        categories = await execute_sp("ia.sp_categorie_get", {"user_fk": config.USER_FK})
        keywords = await execute_sp("ia.sp_motCle_get", {"user_fk": config.USER_FK})
        synonymes_groupes = {}
        keywords_by_cat = {}
        for keyword in keywords:
            cat_fk = keyword.get("categorie_fk")
            if cat_fk is not None:
                keywords_by_cat.setdefault(cat_fk, []).append(keyword)
        for category in categories:
            label = category.get("theLabel")
            cat_fk = category.get("categorie_fk")
            if not label or cat_fk is None:
                continue
            synonymes = set()
            synonymes.add(label)
            for keyword in keywords_by_cat.get(cat_fk, []):
                cle_label = keyword.get("cleLabel")
                if cle_label:
                    synonymes.add(cle_label)
            synonymes_groupes[label] = sorted(synonymes, key=lambda x: x.casefold())
        return synonymes_groupes
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du mapping catégories/synonymes : {e}")
        return {}

@log_function_call
def detect_keywords(question: str) -> list[str]:
    """
    Détecte les mots-clés présents dans la question, en ignorant les nombres.
    Cette version ne se base sur aucune liste de mots-clés prédéfinie.

    Args:
        question (str): La question à analyser.

    Returns:
        list[str]: Liste des mots-clés détectés (ici, tous les mots significatifs uniques, sans les nombres).
    """
    def _normalize(text):
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        return text.lower()

    stopwords = config.french_stopwords

    question_norm = _normalize(question)
    mots = re.findall(r"\b\w+\b", question_norm)
    mots_significatifs = sorted(
        set([
            mot for mot in mots
            if mot not in stopwords and len(mot) > 2 and not mot.isdigit()
        ])
    )
    return mots_significatifs

@log_function_call
def create_simplified_hierarchy(flat_data: (list[dict])):
    """
    Generates a hierarchy from a flat list, keeping only the 'label' and 'children' fields for each node.

    Args:
        flat_data (list[dict]): List of dictionaries describing flat nodes.
            Each dict must have at least 'line_id', 'parent_fk', 'label'.

    Returns:
        list[dict]: Hierarchical tree of simplified nodes with only 'label' and 'children'.
    """
    index_by_id = {}
    children_by_parent = {}
    roots = []

    for item in flat_data:
        node = item.copy()
        current_id = node['line_id']
        parent_id = node['parent_fk'] if node['parent_fk'] != 'NULL' else None

        index_by_id[current_id] = node

        if parent_id is None:
            roots.append(current_id)
        else:
            children_by_parent.setdefault(parent_id, []).append(current_id)

    def _build_simplified_node(node_id):
        """Recursively constructs a simplified node from its ID."""
        node_label = index_by_id[node_id]['label']
        children_ids = children_by_parent.get(node_id, [])
        children_list = [_build_simplified_node(child_id) for child_id in children_ids]
        return {'label': node_label, 'children': children_list}

    return [_build_simplified_node(root_id) for root_id in roots]

@log_function_call
def get_children_by_label(data: list[dict], target_label: str):
    """
    Recherche un label donné dans une structure arborescente (liste de dictionnaires
    imbriqués avec 'label' et 'children') et retourne la liste des labels de ses enfants.

    :param data: La liste des noeuds racine de la structure (votre liste principale).
    :param target_label: Le label (chaîne de caractères) dont on cherche les enfants.
    :return: Une liste de chaînes de caractères (les labels des enfants) ou None si le label n'est pas trouvé.
    """

    for node in data:
        if node.get('label') == target_label:
            return [child.get('label') for child in node.get('children', [])]

        if node.get('children'):
            result = get_children_by_label(node['children'], target_label)
            
            if result is not None:
                return result
    
    return None

@log_function_call
def find_res(question: str, df_residences: pd.DataFrame, sa_label: str, threshold: int = 50):
    def _strip_accents(text: str) -> str:
        """
        Supprime les accents et normalise la casse/ponctuation dans une chaîne pour faciliter les comparaisons.
        """
        if not isinstance(text, str) or not text.strip():
            return ""
        s = (
            text.replace("’", "'")
                .replace("`", "'")
                .replace("‘", "'")
                .replace("–", "-")
                .replace("—", "-")
        )
        s = unicodedata.normalize("NFD", s)
        s = ''.join(
            c for c in s
            if unicodedata.category(c) != 'Mn'
        )
        s = re.sub(r"\s+", " ", s.lower()).strip()
        return s

    LISTE_RES = df_residences["sa"].to_list()
    q = _strip_accents(question)
    q_tokens = set(re.findall(r"\w+", q))
    results = {}
    for ligne in LISTE_RES:
        ln = _strip_accents(ligne)
        ln_tokens = set(re.findall(r"\w+", ln))
        # 1) Strong rule: all tokens of label present in question (with plural tolerance)
        if ln_tokens and ln_tokens.issubset({t.rstrip('s') for t in q_tokens} | q_tokens):
            results[ligne] = 100
            continue
        # 2) Fuzzy fallback
        score = fuzz.token_set_ratio(q, ln)
        if score >= threshold:
            results[ligne] = max(results.get(ligne, 0), score)
    # trier par score décroissant et retourner
    ordered = sorted(results.items(), key=lambda x: -x[1])
    if ordered:
        lf = str(ordered[0][0])
        if lf != sa_label:
            return int(df_residences[df_residences["sa"]==lf]["sa_fk"].iloc[0]), lf
    return None, None

@log_function_call
async def get_ext_data_for_llm(
    question: str, 
    context_data: pd.DataFrame, 
    sa_fk: int, 
    ext_sa_fk: int, 
    form_fk: int, 
    residence: str, 
    synonymes_groupes: dict) -> str:
    """
    Récupère et prépare les données de contexte et la hiérarchie simplifiée pour un LLM à partir d'une résidence sélectionnée.
    Cette fonction exécute les procédures stockées nécessaires et prépare le prompt pour le LLM.
    Args:
        question (str): Question utilisateur à traiter
        context_data (pd.DataFrame): Données de contexte de la résidence courante
        sa_fk (int): Identifiant unique de la section analytique de la résidence courante
        form_fk (int): Identifiant unique du formulaire à analyser
        residence (str): Nom ou code de la résidence externe sélectionnée
        synonymes_groupes (dict): Dictionnaire de groupes de synonymes pour la recherche sémantique
    Returns:
        str
    """

    try:
        data = await execute_sp(
            "ia.sp_simBudFormSA_one", 
            {
                "user_fk": config.USER_FK, 
                "sa_fk": ext_sa_fk, 
                "form_fk": form_fk
            }
        )
        json_string = data[0].get('EcrituresDetails')
        
        res = await execute_sp(
            "dbo.sp_simBudLines",
            {
                "user_fk": config.USER_FK,
                "form_fk": form_fk,
                "line_fk": 0,
                "choix": 0,
                "isVisible": 1
            }
        )
        simple_dict = create_simplified_hierarchy(res)
    
        param = await parse_user_query(question, synonymes_groupes, simple_dict)
        if not json_string:
            logger.warning(f"Aucune donnée d'écriture trouvée pour la résidence demandée ({residence}).")
            success, prompt = await get_data_for_llm(context_data, simple_dict, sa_fk, form_fk, **param)
            if success:
                return param, prompt
        data_records = json.loads(json_string)
        ext_context_data = pd.DataFrame(data_records)
        logger.info(f"Données d'écriture extraites pour la résidence {residence}, nombre de lignes: {len(ext_context_data)}")

        ext_context_data = preprocessing_data(ext_context_data, simple_dict).copy()

        if str(context_data.iloc[1,0]) == str(residence):
            success, prompt = await get_data_for_llm(context_data, simple_dict, sa_fk, form_fk, **param)
            if success:
                return param, prompt
        else:
            success_1, prompt_1 = await get_data_for_llm(context_data, simple_dict, sa_fk, form_fk, **param)
            success_2, prompt_2 = await get_data_for_llm(ext_context_data, simple_dict, ext_sa_fk, form_fk, **param)
            if success_1 and success_2:
                logger.info(f"Les prompts pour les deux résidences ont été générés avec succès pour {residence}.")
                return param, (
                    f"Code analytique: {str(context_data.iloc[1,0]).split(' - ')[0]}, Nom de la résidence: {str(context_data.iloc[1,0]).split(' - ')[1]}\n"
                    f"{prompt_1}\n\n"
                    f"Code analytique: {str(residence).split(' - ')[0]}, Nom de la résidence: {str(residence).split(' - ')[1]}\n"
                    f"{prompt_2}"
                )
            else:
                logger.warning(f"Impossible de préparer les prompts nécessaires pour la résidence {residence}.")
                return param, None

    except Exception as exc:
        logger.error(f"Erreur lors de la préparation des données pour le LLM (résidence {residence}): {exc}")
        raise RuntimeError(
            f"Échec de la préparation des données pour le LLM concernant la résidence '{residence}'. Voir les logs pour plus de détails."
        ) from exc
