from datetime import datetime, timedelta
import inspect
import json
import logging
import re
from typing import Any, Dict, List
import unicodedata

import markdown as _md
import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from tabulate import tabulate
from thefuzz import fuzz

from app.core import config
from app.db.session import get_session

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
            new_lines.append(line)
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
            "dbo.sp_categorieRegle_get",
            {
                "user_fk": config.USER_FK, 
                "categorie_fk": categorie_fk
            },
            config.DATABASE_URL_IA
        )
        return [line.get("regleL") for line in lines if line.get("regleL")]
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des règles : {e}")
        return []

@log_function_call
async def from_cat_to_fk(cat: str) -> int | None:
    """Retourne categorie_fk correspondant à un label de catégorie donné, ou None."""
    try:
        categories = await execute_sp(
            "dbo.sp_categorie_get", 
            {
                "user_fk": config.USER_FK
            },
            config.DATABASE_URL_IA
        )
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
def preprocessing_data(df: pd.DataFrame, simple_dict: list[dict], colonne_type: str|None = "Année contexte") -> pd.DataFrame:
    """Prétraite et catégorise le DataFrame selon les groupes définis."""
    try:
        required_columns = ['Lignes', 'Contexte', 'Nature de l\'écriture', 'Année', 'Mois', 'Montant']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"La colonne requise '{col}' est absente du DataFrame.")
        mask = (
            (df.iloc[:, 0] == df.iloc[0, 0]) &
            (df.iloc[:, 2] == "Compte d'exploitation")
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
            sort_keys = ['Année', 'Mois', 'Contexte', "Nature de l'écriture", 'Montant']
            existing_sort_keys = [col for col in sort_keys if col in df_pct_recettes.columns]
            df_pct_recettes = (
                df_pct_recettes
                .sort_values(by=existing_sort_keys[:-1] + ['Montant'], ascending=[True]*len(existing_sort_keys[:-1])+[False])
                .reset_index(drop=True)
            )
            code_map = {0: "4.", 1: "7.", 2: "10."}
            df_pct_recettes['Code Hiérarchique'] = (
                df_pct_recettes
                .groupby(existing_sort_keys[:-1], sort=False)
                .cumcount()
                .map(lambda idx: code_map.get(idx, None))
            )

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

        mask_pct = (df_filtered['Lignes'] == "% DES RECETTES TOTALES") | (df_filtered['Colonnes'].str.contains('%', na=False))
        mask_non_pct = ~mask_pct

        # Correction : bien gérer le typage et l'arrondi sans convertir en int les NaN et préserver les floats si NaN
        df_filtered.loc[mask_non_pct, "Montant"] = df_filtered.loc[mask_non_pct, "Montant"].round(0)
        mask_non_pct_non_nan = mask_non_pct & df_filtered['Montant'].notna()
        df_filtered.loc[mask_non_pct_non_nan, "Montant"] = df_filtered.loc[mask_non_pct_non_nan, "Montant"].astype(int)

        if mask_pct.any():
            df_filtered.loc[mask_pct, "Montant"] = df_filtered.loc[mask_pct, "Montant"].round(2)
        
        if colonne_type:
            df_filtered = df_filtered[df_filtered["Type de colonnes"] == colonne_type].copy()

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

    async def _check_detail(question: str, threshold: int = 80) -> bool:
        question_norm = _strip_accents(question.lower())
        res = await get_mapping()
        keywords_details: list[str] = res["Détail"]
        for kw in keywords_details:
            if fuzz.partial_ratio(question_norm, _strip_accents(kw.lower())) > threshold:
                return True
        return False

    async def _get_col_info(question: str, threshold: int = 90) -> dict|None:
        lst = await execute_sp(
            "dbo.sp_simBudCol",
            {
                "user_fk": config.USER_FK,
                "codeMetier": 'EXP',
                "form_fk": 167,
                "codeFormType": None,
                "type_fk": 0,
                "colYear_fk": 0
            }
        )
        df_col = pd.DataFrame(lst)
        df_col = df_col[df_col["labelType"]=="Année contexte"][["label", "RB", "Mois", "theYear"]].copy()
        
        q = _strip_accents(question.lower())
        q_tokens = set(re.findall(r"\w+", q))
        LISTE_LIGNES: list[str] = df_col["label"].to_list()
        results = {}
        for ligne in LISTE_LIGNES:
            ln = _strip_accents(ligne.lower())
            ln_tokens = set(re.findall(r"\w+", ln))
            if ln_tokens and ln_tokens.issubset({str(t).rstrip('s') for t in q_tokens} | q_tokens):
                results[ligne] = 100
                continue
            score = fuzz.token_set_ratio(q, ln)
            if score >= threshold:
                results[ligne] = max(results.get(ligne, 0), score)
        if results:
            line = df_col[df_col["label"] == sorted(results.items(), key=lambda x: -x[1])[0][0]]
            return {
                "label": line["label"].iloc[0],
                "annee": int(line["theYear"].iloc[0]),
                "mois": int(line["Mois"].iloc[0]),
                "contexte": line["RB"].iloc[0]
            }    
        else:
            None

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
    col_infos = await _get_col_info(query_lower)

    # GROUPE
    def _select_groupe(question: str, threshold: float = 70) -> list:
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
    def _define_contexte(question: str) -> list:
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
        return sorted(types_valeur)#! if len(types_valeur) > 0 else ['R']
    params['types_valeur'] = _define_contexte(query_lower)

    # ANNEE
    def _define_year(question: str) -> list:
        annees = set(map(int, re.findall(r"\b20\d{2}\b", question)))
        
        pattern_2digit = r"\b(?:Budget|Réel|Reel)\s*'?(\d{2})\b"
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
    params['annees'] = _define_year(query_lower)

    # MOIS
    def _define_month(question: str) -> list:
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
                    mois.update(mois_list)
                    break

            semestre_regex = [
                (r"\b(1(er)?|premier|i+)[s\-]*(semestre|sem)\b", [1, 2, 3, 4, 5, 6]),
                (r"\b(2(e|ème|eme)?|deuxi[eè]me|ii+)[s\-]*(semestre|sem)\b", [7, 8, 9, 10, 11, 12]),
            ]
            for pat, mois_list in semestre_regex:
                if re.search(pat, query_norm):
                    mois.update(mois_list)
                    break

            if re.search(r"\btous les mois\b", query_norm):
                mois.update(range(1, 13))
            elif re.search(r"\bmois courant\b", query_norm):
                mois.update([datetime.now().month])
            elif re.search(r"\bmois dernier\b", query_norm):
                mois.update([datetime.now().month - 1 if datetime.now().month > 1 else 12])
            elif re.search(r"\bmois prochain\b", query_norm):
                mois.update([datetime.now().month + 1 if datetime.now().month < 12 else 1])
        return sorted(mois)
    params['mois'] = _define_month(query_lower)

    # NATURE ECRITURE
    def _select_nature_ecriture(question: str) -> list:
        nature_ecritures = set()
        if re.search(r"\b(mensuel(le)?|mois|trimestre|semestre|janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\b", question):
            nature_ecritures.add('Mensuelle')
        if params['mois']:
            nature_ecritures.add('Mensuelle')
        if re.search(r"\b(annuel(le)?|total|cette année|cette annee|année|annee)\b", question):
            nature_ecritures.add('Annuelle')
        if not nature_ecritures:
            if params.get('mois') and 0 < len(params['mois']) < 12:
                nature_ecritures.add('Mensuelle')
            if params.get('mois') and len(params['mois']) == 12:
                nature_ecritures.add('Annuelle')
        return [v for v in ['Mensuelle', 'Annuelle'] if v in nature_ecritures]
    params['nature_ecriture'] = _select_nature_ecriture(query_lower)

    # LIGNES
    async def _match_lignes(question: str, threshold: int = 75, return_scores: bool = False):
        result_list = extract_all_descendants_for_list(simple_dict)
        LISTE_LIGNES: list[str] = [label for sublist in result_list for label in sublist]
        q = _strip_accents(question.lower())
        q_tokens = set(re.findall(r"\w+", q))
        results = {}
        # Plus grande tolérance au singulier/pluriel : on compare formes singulier et pluriel, pour chaque token de la ligne et de la question
        def _sing_plur_forms(token):
            if token.endswith('s'):
                return {token, token[:-1]}
            else:
                return {token, token + 's'}
        
        for ligne in LISTE_LIGNES:
            ln = _strip_accents(ligne.lower())
            ln_tokens_raw = set(re.findall(r"\w+", ln))
            q_tokens_raw = q_tokens

            # Génère toutes formes singulier/pluriel pour ln_tokens et q_tokens
            ln_tokens_all = set()
            for t in ln_tokens_raw:
                ln_tokens_all.update(_sing_plur_forms(t))
            q_tokens_all = set()
            for t in q_tokens_raw:
                q_tokens_all.update(_sing_plur_forms(t))
            
            # Test : Tous les tokens 'ligne' (sing/plur) présents dans q (sing/plur)
            if ln_tokens_all and ln_tokens_all.issubset(q_tokens_all):
                results[ligne] = 100
                continue

            # Fallback fuzzy
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

    # Correct for NoneType issues and robustify list usage
    if col_infos:
        annee = col_infos.get("annee")
        # Ensure params['annees'] is always a list
        if not isinstance(params.get('annees'), list) or params['annees'] is None:
            params['annees'] = []
        if annee is not None and annee not in params['annees']:
            params['annees'].append(annee)
        mois = col_infos.get("mois", 0)
        if not isinstance(params.get('nature_ecriture'), list) or params['nature_ecriture'] is None:
            params['nature_ecriture'] = []
        if mois == 0:
            if "Annuelle" not in params['nature_ecriture']:
                params['nature_ecriture'].append("Annuelle")
        else:
            if "Mensuelle" not in params['nature_ecriture']:
                params['nature_ecriture'].append("Mensuelle")
            if not isinstance(params.get('mois'), list) or params['mois'] is None:
                params['mois'] = []
            if mois not in params['mois']:
                params['mois'].append(mois)

        contexte = col_infos.get("contexte")
        if not isinstance(params.get('types_valeur'), list) or params['types_valeur'] is None:
            params['types_valeur'] = []
        if contexte and contexte not in params['types_valeur']:
            params['types_valeur'].append(contexte)

    # Normalize all params lists (ensure always list, de-duplicate, and sort if non-empty)
    for k in ['groupes', 'types_valeur', 'annees', 'nature_ecriture', 'lignes', 'mois']:
        param_val = params.get(k)
        if not isinstance(param_val, list) or param_val is None:
            params[k] = []
        else:
            params[k] = sorted(set(param_val)) if param_val else []

    if not params['nature_ecriture']:
        params['nature_ecriture'] = ['Annuelle']

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

    def _filter_constant_columns(cols: list):
        EXCLUDE = {"Lignes", "Montant"}
        if df_filtre is not None and len(df_filtre) > 0:
            constant_columns = cols.copy()
            for col_name in constant_columns:
                if col_name in EXCLUDE:
                    continue
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
        n_before: int = config.N_NEIGHBORS, 
        n_after: int = config.N_NEIGHBORS) -> list[str]:

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

    def _find_siblings_with_col(
        annee: int, 
        mois: int,
        contexte: str,
        df_col: pd.DataFrame, 
        n_before: int = config.N_NEIGHBORS, 
        n_after: int = config.N_NEIGHBORS) -> str:

        if df_col is None or df_col.empty:
            return ""

        mask = (df_col["theYear"] == annee) & (df_col["RB"].str.lower() == contexte.lower())
        if mois == 0:
            mask = mask & (df_col["Mois"] == 0)
        else:
            mask = mask & (df_col["Mois"] == mois)
        idx_list = df_col[mask].index.tolist()
        if not idx_list:
            return ""
        idx = idx_list[0]

        if mois == 0:
            mois0_indices = df_col[df_col["Mois"] == 0].index.tolist()
            try:
                pos_in_mois0 = mois0_indices.index(idx)
            except ValueError:
                return ""
            start = max(0, pos_in_mois0 - n_before)
            end = pos_in_mois0 + n_after + 1
            indices = mois0_indices[start:end]
        else:
            year_contexte_indices = df_col[(df_col["theYear"] == annee) & (df_col["RB"].str.lower() == contexte.lower()) & (df_col["Mois"] != 0)].index.tolist()
            try:
                pos_in_year_contexte = year_contexte_indices.index(idx)
            except ValueError:
                return ""
            start = max(0, pos_in_year_contexte - n_before)
            end = pos_in_year_contexte + n_after + 1
            indices = year_contexte_indices[start:end]

        if not indices:
            return ""

        try:
            rows = df_col.loc[indices, ["theYear", "Mois", "RB"]]
        except KeyError:
            return ""
        cols_labels = ",".join([f"{int(row.theYear)},{int(row.Mois)},{row.RB}" for _, row in rows.iterrows()])
        return cols_labels

    def _get_line_fk(label: str, res: list[dict]):
        df = pd.DataFrame(res).iloc[:, [0, 4]]
        ser: pd.DataFrame = df[df["label"] == label]["line_id"]
        return int(ser.iloc[0])

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
    if lignes and list(lignes)[0] not in df_final["Lignes"].to_list():
        if not types_valeur:
            return False, None

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

        colonne_db = await execute_sp(
            "dbo.sp_simBudCol",
            {
                "user_fk": config.USER_FK,
                "codeMetier": 'EXP',
                "form_fk": form_fk,
                "codeFormType": None,
                "type_fk": 0,
                "colYear_fk": 0
            }
        )
        df_col = pd.DataFrame(colonne_db)
        df_col = df_col[df_col["labelType"]=="Année contexte"][["label", "RB", "Mois", "theYear"]].copy()
        lines = _find_siblings_with_neighbors(
            label=lignes[0], 
            df_labels_codes=df_labels_codes
        )
        mois_val = mois[0] if mois and len(mois) > 0 else 0
        cols = _find_siblings_with_col(
            annee=annees[0],
            mois=mois_val,
            contexte=types_valeur[0],
            df_col=df_col
        )

        line_fks = [str(_get_line_fk(l, res)) for l in lines]
        line_fks_str = ",".join(line_fks)
        the_line_fk = _get_line_fk(lignes[0], res)
        
        lst = await execute_sp(
            "ia.sp_simBudValueDetails",
            {
                "user_fk": config.USER_FK,
                "listSA": 0,
                "line_fk": line_fks_str,
                "sa_fk": sa_fk,
                "yearRB": cols,
            }
        )
        df_result = pd.DataFrame(lst)
        
        if df_result.empty:
            return False, None
        
        line_info = get_line_info(
            df_result, 
            the_line_fk, 
            annees[0],
            mois_val,
            types_valeur[0],
            lines,
            lambda label: _get_line_fk(label=label, res=res)
        )

        col_info = get_col_info(
            df_result, 
            the_line_fk,
            cols
        )
        return False, (
            f"La ligne '{lignes[0]}' n'a aucune valeur. "
            "Analyse les deux tableaux ci-dessous :\n"
            " - le premier montre les lignes voisines et leurs sources\n"
            " - le second montre les colonnes voisines et leurs sources\n"
            "Explique très brièvement pourquoi cette ligne n'a aucune donnée. "
            "Compare uniquement avec les voisins (lignes et colonnes) pour justifier l'absence. "
            "Si les colonnes voisines pour cette même ligne contiennent des sources, spécifie-les (année, mois, contexte, source). "
            "Réponse 1-3 phrases, sans paragraphes, directe.\n"
            f"{line_info}\n\n{col_info}"
        )
    
    if df_final.empty:
        return False, None
    else:
        df_to_display = df_final.copy()
        table_str = tabulate(df_to_display, headers="keys", tablefmt="simple", showindex=False, numalign="right", stralign="left")
        return True, table_str

""""""

@log_function_call
def data_to_pivot(df: pd.DataFrame) -> pd.DataFrame:
    if df["Section  analytique"].unique().tolist() in [[''], [], None]:
        df["Section  analytique"] = df["Liste de sélection"]
        
    # Renommage et nettoyage
    df = df.rename(
        columns={
            'Code Hiérarchique': 'Code_H', 
            'Montant': 'Montant',
            'Lignes': 'Ligne_Analytique',
            'Contexte': 'Contexte',
            'Année': 'Annee',
            'Groupe': 'Groupe',
            'Section  analytique': 'Residence'
        }
    )

    df['Annee'] = df['Annee'].astype(int)
    df['Mois'] = df['Mois'].astype(int)

    df_agg = df.groupby(
        [
            'Residence', 'Colonnes', 'Annee', 'Mois', "Nature de l'écriture", 'Contexte', 'Code_H', 'Ligne_Analytique', 'Groupe'
        ]
    )['Montant'].sum().reset_index()

    contexte_order = ['R', 'P', 'B']

    def _mois_sort_key(mois):
        try:
            return int(mois)
        except:
            return 99

    df_pivot = df_agg.pivot_table(
        index=['Groupe', 'Code_H', 'Ligne_Analytique'],
        columns=['Annee', 'Contexte', 'Mois', "Nature de l'écriture"],
        values='Montant',
        fill_value="",
        aggfunc='sum'
    )
    # Explicitly infer objects to avoid FutureWarning from fill_value on object dtype
    df_pivot = df_pivot.infer_objects(copy=False)

    if df_pivot.columns.nlevels == 4:
        nature_unique = df_pivot.columns.get_level_values(3).unique().tolist()
        if "Annuelle" in nature_unique:
            nature_unique = [n for n in nature_unique if n != "Annuelle"]
            nature_order_desc = sorted(nature_unique, reverse=True) + ["Annuelle"]
        else:
            nature_order_desc = sorted(nature_unique, reverse=True)
        nature_order_dict = {name: i for i, name in enumerate(nature_order_desc)}
        
        def _col_sort_key(x):
            return (
                int(x[0]) if str(x[0]).isdigit() else 0,
                contexte_order.index(x[1]) if x[1] in contexte_order else 99,
                _mois_sort_key(x[2]),
                nature_order_dict.get(x[3], 999)
            )
        df_pivot = df_pivot[sorted(df_pivot.columns, key=_col_sort_key)]

    df_pivot = df_pivot.reset_index()

    def _code_hierarchical_sort_key(code):
        parts = [int(part) if part.isdigit() else part for part in re.split(r'\D+', str(code).strip('.')) if part]
        return parts

    df_pivot_sorted = df_pivot.copy()
    df_pivot_sorted['__sort_key'] = df_pivot_sorted['Code_H'].apply(_code_hierarchical_sort_key)
    df_pivot_sorted = df_pivot_sorted.sort_values('__sort_key').drop(columns='__sort_key', level=0).reset_index(drop=True)

    def _format_value(val):
        try:
            if isinstance(val, (float, np.floating, int, np.integer)):
                if float(val) == int(val):
                    return int(val)
                else:
                    return "{:.2f}".format(float(val))

            if isinstance(val, str):
                num = float(val.replace(",", ".").strip())
                if num == int(num):
                    return int(num)
                else:
                    return "{:.2f}".format(num)
            return val
        except:
            return val

    for col in df_pivot_sorted.columns[3:]:
        df_pivot_sorted[col] = df_pivot_sorted[col].apply(_format_value)

    """ df_pivot_sorted = df_pivot_sorted[
        df_pivot_sorted["Code_H"].apply(
            lambda x: str(x).strip('.').split('.')[0] in ['1', '2', '3', '4']
        )
    ].reset_index(drop=True) """

    return df_pivot_sorted

@log_function_call
def get_ret_dataframe(df_pivot: pd.DataFrame, param: dict[str, list]) -> pd.DataFrame:
    if not param["groupes"] and not param["lignes"]:
        return None
    if param["groupes"]:
        df_group = df_pivot[df_pivot[df_pivot.columns.levels[0][-2]].isin(param["groupes"])]
    else:
        df_group = df_pivot

    if param['types_valeur']:
        mask_typ = df_group.columns.get_level_values(1).isin(param['types_valeur'])
        cols = df_group.columns[mask_typ].tolist()
        meta_cols = [c for c in df_group.columns if c[0] in ("Groupe", "Code_H", "Ligne_Analytique")]
        selected_cols = meta_cols + cols
        df_val = df_group.loc[:, selected_cols]
    else:
        df_val = df_group

    if param['annees']:
        mask_yrs = df_val.columns.get_level_values(0).isin(param['annees'])
        cols = df_val.columns[mask_yrs].tolist()
        meta_cols = [c for c in df_val.columns if c[0] in ("Groupe", "Code_H", "Ligne_Analytique")]
        selected_cols = meta_cols + cols
        df_annee = df_val.loc[:, selected_cols]
    else:
        df_annee = df_val

    if param['nature_ecriture']:
        mask_nat = df_annee.columns.get_level_values(3).isin(param['nature_ecriture'])
        cols = df_annee.columns[mask_nat].tolist()
        meta_cols = [c for c in df_annee.columns if c[0] in ("Groupe", "Code_H", "Ligne_Analytique")]
        selected_cols = meta_cols + cols
        df_nature = df_annee.loc[:, selected_cols]
    else:
        df_nature = df_annee

    if param["lignes"]:
        df_lignes = df_nature[df_nature[df_nature.columns.levels[0][-4]].isin(param["lignes"])]
    else:
        df_lignes = df_nature

    if param["mois"]:
        if "Annuelle" in param['nature_ecriture']:
            if 12 not in param['mois']:
                param['mois'].append(12)
        mask_mois = df_lignes.columns.get_level_values(2).isin(param['mois'])
        cols = df_lignes.columns[mask_mois].tolist()
        meta_cols = [c for c in df_lignes.columns if c[0] in ("Groupe", "Code_H", "Ligne_Analytique")]
        selected_cols = meta_cols + cols
        df_mois = df_lignes.loc[:, selected_cols]
    else:
        df_mois = df_lignes

    return df_mois

@log_function_call
def transform_for_llm(df_pivot: pd.DataFrame|None) -> tuple:
    """
        - Génère un texte métrique optimisé pour l'entrée LLM.
        - Détecte la présence de colonnes contextuelles (ex : 'Groupe', 'Code_H', 'Ligne_Analytique')
          et construit un label 'Indicateur' consolidé à partir de ces informations.
        - Requiert que les fonctions utilitaires _normalize_col et _format_euro_fr existent déjà.
    Returns :
        - metric_text : str, texte à fournir au LLM
    """
    if df_pivot is None:
        return False, None

    logger.info("transform_for_llm is used")
    def _format_euro_fr(x: float, line: str) -> str:
        """Format number in French style with 2 decimals and a non-breaking space thousands separator."""
        if pd.isna(x):
                return "N/A"
        elif str(line).startswith("%"):
            s = f"{x:,.2f}"
            s = s.replace(",", " ")
            return f"{s} %"
        else:
            s = f"{x:,.0f}"
            s = s.replace(",", " ")
            return f"{s} €"

    def _normalize_col(col):
        if isinstance(col, tuple):
            # year is always at position 0
            if len(col) >= 1:
                year = str(col[0])
            else:
                year = "unknown"
            # find Réel/Prévision/Budget if present in tuple (also accent-insensitive, lowercase!)
            typ = next(
                (str(x) for x in col if isinstance(x, str) and str(x).lower() in ("réel", "budget", "prévision", "prevision")),
                None
            )
            if typ is None:
                # fallback: any string in col
                typ = next((str(x) for x in col if isinstance(x, str)), "Réel")

            # try to recover nature and mois heuristically (commonly last two positions in tuple)
            col_strs = [str(x) for x in col if isinstance(x, str)]

            # We will look at positions from the end (to be robust to existing pivot structure)
            # assume 'nature' (ex: Mensuelle, Annuelle) is very likely last, 'mois' just before, if present
            if len(col) >= 3:
                nature = str(col[-1]) if col[-1] is not None else None
                mois = str(col[-2]) if col[-2] is not None else None
                # If col[-1] (nature) is not a valid value, set to None
                if not nature or nature.lower() in ("", "none", "nan"):
                    nature = None
                if not mois or mois.lower() in ("", "none", "nan"):
                    mois = None
            else:
                nature = None
                mois = None

            return {"year": year, "type": typ, "nature": nature, "mois": mois}
        else:
            # fallback: not a tuple
            year = str(col)
            return {"year": year, "type": "Réel", "nature": None, "mois": None}

    new_cols = list(df_pivot.columns)
    for i, col in enumerate(df_pivot.columns):
        if i >= 3:
            col_as_list = list(col)
            if col_as_list[1] == 'R':
                col_as_list[1] = 'Réel'
            elif col_as_list[1] == 'P':
                col_as_list[1] = 'Prévision'
            elif col_as_list[1] == 'B':
                col_as_list[1] = 'Budget'
            new_cols[i] = tuple(col_as_list)
    df_pivot.columns = pd.MultiIndex.from_tuples(new_cols)
    
    df = df_pivot.copy()

    # If index contains labels, reset to columns
    if df.index.name is None or df.index.name == "":
        df = df.reset_index()

    # If there are contextual columns, build a consolidated 'Indicateur' column
    context_cols = [c for c in ['Code_H', 'Ligne_Analytique', 'Indicateur'] if c in df.columns]

    if len(context_cols) > 1:
        # create a single descriptive indicator by joining available context columns (in order)
        df['Indicateur_consolide'] = df[context_cols].astype(str).apply(
            lambda row: " | ".join([str(x).strip() for x in row.values if str(x).strip() not in ['nan', 'None']]),
            axis=1
        )
        # prefer the consolidated name
        indicator_col = 'Indicateur_consolide'
    else:
        # detect a single indicator column if present, otherwise use first column
        indicator_col = None
        for possible in ['Ligne_Analytique', 'Indicateur', 'index', 0]:
            if possible in df.columns:
                indicator_col = possible
                break
        if indicator_col is None:
            indicator_col = df.columns[0]
        # if chosen indicator_col isn't already a string label, coerce to str
        if indicator_col != 'Indicateur':
            df[indicator_col] = df[indicator_col].astype(str)

    # Ensure the DataFrame has a column named exactly 'Indicateur' used downstream
    if indicator_col != 'Indicateur':
        df = df.rename(columns={indicator_col: 'Indicateur'})
    else:
        # if it already is 'Indicateur', ensure string type
        df['Indicateur'] = df['Indicateur'].astype(str)

    value_cols = [c for c in df.columns if c != 'Indicateur']

    rows = []
    for _, row in df.iterrows():
        # Avoid pandas row pretty-print for the indicator label
        if isinstance(row['Indicateur'], pd.Series):
            indicator_label = " | ".join(str(x).strip() for x in row['Indicateur'].values if str(x).strip() not in ['nan', 'None'])
        else:
            indicator_label = str(row['Indicateur']).strip()
        for col in value_cols:
            meta = _normalize_col(col)
            year = meta.get('year', 'unknown')
            typ = meta.get('type', 'Réel')
            nature = meta.get('nature', 'unknown')
            mois = meta.get('mois', 'unknown')
            try:
                val = row[col]
            except Exception:
                val = row.get(col, None)

            # Try to keep numeric
            numeric = None
            if pd.api.types.is_numeric_dtype(type(val)):
                try:
                    numeric = float(val) if not pd.isna(val) else None
                except Exception:
                    numeric = None
            else:
                try:
                    numeric = float(str(val).replace("€", "").replace("%", "").replace(" ", "").replace(",", "."))
                except Exception:
                    numeric = None

            lbl = indicator_label.split(" | ")[-1]
            txt = _format_euro_fr(numeric, lbl) if numeric is not None else "N/A"

            rows.append({
                'Indicateur': indicator_label,
                'Année': year,
                'Type': typ,
                'Nature': nature,
                'Mois': mois,
                'Valeur_num': numeric,
                'Valeur_txt': txt
            })

    df_long = pd.DataFrame(rows)

    def _context_rank(typ):
        t = str(typ).lower()
        if "réel" in t:
            return 0
        if "prevision" in t or "prévision" in t:
            return 1
        if "budget" in t:
            return 2
        return 99

    def _block_for_indicator(ind):
        label = str(ind).strip()
        lines = [f"[{label}]"]
        sub = df_long[df_long['Indicateur'].astype(str).values == str(ind)].copy()

        # Filter out technical garbage in 'Année'
        sub = sub[~sub['Année'].astype(str).str.lower().isin(['groupe', 'indicateur', 'index'])]

        # Attempt numeric year conversion for sorting; fallback keeps original order
        sub_sorted = sub.copy()
        try:
            sub_sorted["Année_num"] = pd.to_numeric(sub_sorted["Année"], errors='coerce')
        except Exception:
            sub_sorted["Année_num"] = sub_sorted["Année"]

        # First sort by Type context order, then by year
        if 'Type' in sub_sorted.columns:
            sub_sorted = sub_sorted.sort_values(
                by=["Type", "Année_num"],
                key=lambda col: col.map(_context_rank) if col.name == "Type" else col,
                ascending=[True, True]
            )
        else:
            sub_sorted = sub_sorted.sort_values(by=["Année_num"])

        # Ensure ordering Réel -> Prévision -> Budget within each year
        entries = []
        for ctx in ["Réel", "Prévision", "Budget"]:
            sub_ctx = sub_sorted[sub_sorted["Type"].astype(str).str.lower().str.contains(ctx.lower(), na=False)]
            entries.append(sub_ctx)
        if entries:
            merged = pd.concat(entries)
            merged = merged.drop_duplicates(subset=["Année", "Type", "Nature", "Mois"])
        else:
            merged = sub_sorted
        
        mois_str = [
            'Janvier',
            'Février',
            'Mars',
            'Avril',
            'Mai',
            'Juin',
            'Juillet',
            'Août',
            'Septembre',
            'Octobre',
            'Novembre',
            'Décembre'
        ]
        # Produce lines
        for _, rr in merged.iterrows():
            year = rr['Année']
            typ = rr['Type']
            nature = rr['Nature']
            mois = rr['Mois']
            txt = rr['Valeur_txt']
            # Skip rows with completely empty or N/A values for non-year labels
            if (pd.isna(year) or str(year).strip().lower() in ['nan', 'none', '']) and txt in ["0", "0 €", "N/A"]:
                continue
            if nature == "Annuelle":
                lines.append(f"- {typ} {nature} {year}: {txt}")
            else:
                lines.append(f"- {typ} {mois_str[int(mois)-1]} {year}: {txt}")
        return "\n".join(lines)

    indicators = df_long['Indicateur'].drop_duplicates().tolist()
    blocks = [_block_for_indicator(ind) for ind in indicators]

    metric_text = "\n\n".join(blocks)

    return True, metric_text

""""""

@log_function_call
def count_tokens(text: str) -> int:
    """Compte de tokens naïf basé sur découpage mots."""
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return len(tokens)

@log_function_call
async def execute_sp(sp_name: str, params: dict, database_url: str = config.DATABASE_URL) -> List[Dict[str, Any]]:
    """
    Exécute une procédure stockée SQL et retourne la liste de dicts résultat (ou vide).
    Gère aussi bien les SP de type select que add (insert/update).
    """
    results = []
    try:
        async with get_session(database_url) as session:
            param_keys = ", ".join([f":{key}" for key in params.keys()])
            sql_query = text(f"EXEC {sp_name} {param_keys}" if param_keys else f"EXEC {sp_name}")
            result_proxy = await session.execute(sql_query, params)
            if (sp_name.startswith("ia.sp_") and sp_name.endswith("_add")) or (sp_name.startswith("dbo.sp_") and sp_name.endswith("_add")):
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
        categories = await execute_sp(
            "dbo.sp_categorie_get", 
            {
                "user_fk": config.USER_FK
            },
            config.DATABASE_URL_IA
        )
        keywords = await execute_sp(
            "dbo.sp_motCle_get", 
            {
                "user_fk": config.USER_FK
            },
            config.DATABASE_URL_IA
        )
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
async def find_res(question: str, df_residences: pd.DataFrame, threshold: int = 50):
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

    async def _check_all_res(question: str, threshold: int = 80) -> tuple:
        question_norm = _strip_accents(question.lower())
        res = await get_mapping()
        keywords_autres_res: list[str] = res["Autre résidence"]
        for kw in keywords_autres_res:
            if fuzz.partial_ratio(question_norm, _strip_accents(kw.lower())) > threshold:
                exception = "Quel est le nom de la résidence ?"
                if fuzz.partial_ratio(question_norm, _strip_accents(exception.lower())) > 90:
                    return True, False
                return True, True
        return False, True

    LISTE_RES = df_residences["sa"].to_list()

    is_all_res, if_res = await _check_all_res(question)
    
    if not if_res:
        return False, None

    if is_all_res:
        res = []
        for resid in LISTE_RES:
            sa_fk_value = int(df_residences[df_residences["sa"] == resid]["sa_fk"].iloc[0])
            res.append((sa_fk_value, resid))
        return True, res
    else:
        results = {}
        q = _strip_accents(question)
        q_tokens = set(re.findall(r"\w+", q))
        for ligne in LISTE_RES:
            ln = _strip_accents(ligne)
            ln_tokens = set(re.findall(r"\w+", ln))
            # 1) Strong rule: all tokens of label present in question (with plural tolerance)
            if ln_tokens and ln_tokens.issubset({str(t).rstrip('s') for t in q_tokens} | q_tokens):
                results[ligne] = 100
                continue
            # 2) Fuzzy fallback
            score = fuzz.token_set_ratio(q, ln)
            if score >= threshold:
                results[ligne] = max(results.get(ligne, 0), score)
        # trier par score décroissant et retourner
        ordered = sorted(results.items(), key=lambda x: -x[1])
        if ordered:
            for i, ln in enumerate(ordered):
                lf = ordered[i][0]
                ordered[i]=int(df_residences[df_residences["sa"]==lf]["sa_fk"].iloc[0]), ordered[i][0]
            return True, ordered
        return False, None

@log_function_call
async def get_ext_data_for_llm(
    question: str, 
    context_data: pd.DataFrame, 
    sa_fk: int, 
    form_fk: int, 
    residences: str, 
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

    def _afficher_infos_residence(label, prompt):
        _, nom_res = str(label).split(' - ', 1)
        return (
            f"Résidence : {nom_res}\n"
            f"{prompt}\n"
        )

    try:
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
        """ if not param["types_valeur"]:
            logger.warning(f"Impossible de préparer les prompts nécessaires: 'types_valeur' est vide.")
            return param, None """
            
        datas = []
        success, prompt = await get_data_for_llm(context_data, simple_dict, sa_fk, form_fk, **param)
        if success:
            datas.append((context_data.iloc[1,0], prompt))

        for line in residences:
            ext_sa_fk, resid = line
            data = await execute_sp(
                "ia.sp_simBudFormSA_one", 
                {
                    "user_fk": config.USER_FK, 
                    "sa_fk": ext_sa_fk, 
                    "form_fk": form_fk
                }
            )
            json_string = data[0].get('EcrituresDetails')
            if not json_string:
                continue
            data_records = json.loads(json_string)
            ext_context_data = pd.DataFrame(data_records)
            ext_context_data = preprocessing_data(ext_context_data, simple_dict).copy()
            if str(context_data.iloc[1,0]) == str(ext_context_data.iloc[1,0]):
                continue
            success, prompt = await get_data_for_llm(ext_context_data, simple_dict, ext_sa_fk, form_fk, **param)
            if success:
                prompt = "\n".join(prompt.splitlines()[2:])
                datas.append((resid, prompt))

        if datas:
            mds = ""
            for resid, prompt in datas:
                mds += str(_afficher_infos_residence(resid, prompt))
            return param, mds
        else:
            logger.warning(f"Impossible de préparer les prompts nécessaires.")
            return param, None

    except Exception as exc:
        logger.error(f"Erreur lors de la préparation des données pour le LLM: {exc}")
        raise RuntimeError(
            f"Échec de la préparation des données pour le LLM. Voir les logs pour plus de détails."
        ) from exc

@log_function_call
def get_col_info(
    df_result: pd.DataFrame, 
    the_line_fk: int,
    cols: str) -> str:

    df_res = df_result[df_result["line_fk"] == the_line_fk]
    triplets = [tuple(cols.split(",")[i:i+3]) for i in range(0, len(cols.split(",")), 3)]
    rows = []

    for an, mo, co in triplets:
        df_filtered = df_res[
            (df_res["Contexte"].str.startswith(co)) &
            (df_res["dateNotFormat"].str[:4] == str(an))
        ]

        if int(mo) == 0:
            sources = (
                ", ".join(df_filtered["Source"].unique())
                if not df_filtered.empty
                else "Non disponible"
            )
            rows.append({"Année": an, "Contexte": co, "Source": sources})

        else:
            df_filtered_month = df_filtered[
                df_filtered["dateNotFormat"].str[5:7] == str(mo).zfill(2)
            ]
            sources = (
                ", ".join(df_filtered_month["Source"].unique())
                if not df_filtered_month.empty
                else "Non disponible"
            )
            rows.append({"Mois": mo, "Source": sources})

    return pd.DataFrame(rows).to_markdown(index=False)

@log_function_call
def get_line_info(
    df_result: pd.DataFrame, 
    the_line_fk: int, 
    annee: str,
    mois: str,
    contexte: str,
    lines: list[str],
    get_line_fk: callable) -> str:

    rows = []
    df_result_filtered = df_result[df_result["line_fk"] != the_line_fk]
    df_result_filtered = df_result_filtered[df_result_filtered["Contexte"].str.startswith(contexte)]
    df_result_filtered = df_result_filtered[df_result_filtered["dateNotFormat"].str[:4] == str(annee)]
    if mois != 0:
        df_result_filtered = df_result_filtered[df_result_filtered["dateNotFormat"].str[5:7] == str(mois).zfill(2)]
    for l in lines:
        r = df_result_filtered[df_result_filtered["line_fk"] == get_line_fk(label=l)]
        if "Source" in r.columns and not r["Source"].empty:
            sources = ", ".join(r['Source'].unique())
            rows.append({"Ligne": l, "Source": sources})
        else:
            rows.append({"Ligne": l, "Source": "Non disponible"})

    return pd.DataFrame(rows).to_markdown(index=False)
