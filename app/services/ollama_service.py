import asyncio
import logging

import httpx
import pandas as pd

from app.core.config import GPT, OLLAMA_URL, SOELOG
from app.services.functions import (
    add_br_outside_blocks,
    convert_markdown_to_html,
    count_tokens,
    datetime,
    find_res,
    get_data_for_llm,
    get_ext_data_for_llm,
    get_mapping,
    get_rules,
    parse_user_query,
    re,
)

logger = logging.getLogger(__name__)

async def ask_ollama(
    context_data: pd.DataFrame, 
    question:str, 
    history: list[dict], 
    category: str, 
    context_lenght: int, 
    lexiques: dict[str, list[str]], 
    simple_dict: list[dict],
    sa_fk: int,
    form_fk: int,
    residences: pd.DataFrame) -> str:
    """
    Interroge le modèle Ollama avec les données de contexte et un prompt, puis retourne la réponse.

    Args:
        context_data (pd.DataFrame): Données de contexte.
        question (str): Question utilisateur.
        history (list[dict]): Historique des échanges.
        category (str): Catégorie / modèle.
        context_lenght (int): Nombre de tours d'historique à garder.
        lexiques (dict[str, list[str]]): Lexiques métiers.
        simple_dict (list[dict]): Dictionnaire simplifié.
        sa_fk (int): Identifiant section analytique.
        form_fk (int): Identifiant formulaire.
        residences (pd.DataFrame): Résidences associées.
    Returns:
        str: La réponse du modèle.
    Raises:
        httpx.RequestError: En cas d'erreur de connexion à l'API Ollama.
        ValueError: Si la réponse de l'API est invalide.
    """
    st_prompt = datetime.now()
    full_query = ""
    parametres = None
    is_not_empty = False
    prompt_data = None
    try:
        lexiques = await get_mapping()
        ext_sa, resids = await find_res(question, residences)

        if ext_sa:
            is_not_empty=True
            parametres, prompt_data = await get_ext_data_for_llm(
                question,
                context_data,
                sa_fk,
                form_fk,
                resids,
                lexiques
            )
        else:
            parametres = await parse_user_query(question, lexiques, simple_dict)
            if parametres is None:
                parametres = {}
            is_not_empty, prompt_data = await get_data_for_llm(context_data, simple_dict, sa_fk, form_fk, **parametres)

            # On tente avec historique si pas de données pertinentes récupérées
            if (not prompt_data or prompt_data is None) and history and context_lenght > 0:
                user_messages = [h.get("content", "") for h in history if h.get("role") == "user"]
                last_user_msgs = user_messages[-context_lenght:] if context_lenght and len(user_messages) >= context_lenght else user_messages
                full_query = " ".join(last_user_msgs + [question])
                parametres = await parse_user_query(full_query, lexiques, simple_dict)
                if parametres is None:
                    parametres = {}
                is_not_empty, prompt_data = await get_data_for_llm(context_data, simple_dict, sa_fk, form_fk, **parametres)
                if prompt_data and count_tokens(str(prompt_data)) > 4096:
                    parametres = await parse_user_query(question, lexiques, simple_dict)
                    if parametres is None:
                        parametres = {}
                    is_not_empty, prompt_data = await get_data_for_llm(context_data, simple_dict, sa_fk, form_fk, **parametres)

        # Ne jamais écrire 'None' dans prompt_data.txt pour la data affichée
        pd_str = "" if prompt_data is None else str(prompt_data)

        if parametres:
            params_str = str(parametres)
        else:
            params_str = ""

        if full_query == "":
            to_write = category
            if params_str:
                to_write += f"\n{params_str}"
            to_write += "\n" + question + "\n\n" + pd_str
        else:
            to_write = category
            if params_str:
                to_write += f"\n{params_str}"
            to_write += "\n" + full_query + "\n\n" + pd_str

        with open("prompt_data.txt", "w", encoding="utf-8") as f:
            f.write(to_write)

        rules = await get_rules(category)
        prompt = ""
        prompt_system = (
            "Tu es un assistant français pour contrôleurs de gestion. "
            "Réponds de manière claire et concise. "
            "Utilise uniquement les données fournies, n'invente rien, et reste concis. "
            "ATTENTION : lors des calculs, respecte la hiérarchie des rubriques. Ne mentionne pas les codes hiérarchiques dans la réponse."
        )

        if (is_not_empty or ext_sa) and prompt_data:
            if count_tokens(str(prompt_data)) > 400:
                rules.append("Réponds en un seul paragraphe concise sans entrer dans les détails ligne à ligne.")
            prompt = (
                f"CONTEXTE: La section analytique ou résidence '{str(context_data.iloc[1,0]).split(' - ')[1]}' "
                f"et le formulaire '{context_data.iloc[1,2]}' sont analysés.\n"
                "Règles de réponse:\n"
                + "\n".join(rules) 
                + "\nRéponds en deux ou trois phrases au maximum. \n"
                f"QUESTION: {question}\n"
                f"{pd_str if pd_str != "" else "Aucune donnée trouvée."}"
            )
        else:
            prompt = (
                f"CONTEXTE: La section analytique ou résidence '{str(context_data.iloc[1,0]).split(' - ')[1]}' "
                f"et le formulaire '{context_data.iloc[1,2]}' sont analysés. "
                f"QUESTION: {question}\n"
                f"{pd_str}"
                "\nRéponds en deux phrases au maximum. "
                "Si une information manque, invite à contacter le contrôleur de gestion avec les informations sur le contexte (Réel, Budget ou Projeté) et l'année. "
                "Si c'est une salutation, réponds poliment et demande le besoin d'analyse du jour."
            )

        if not history:
            messages = [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = history + [{"role": "user", "content": prompt}]

        if category != "Formule et définition":
            model = GPT
        else:
            model = SOELOG

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": -1,
            "options": {
                "temperature": 0.1,   # réponse très déterministe
            }
        }
        et_prompt = datetime.now()
        logger.info(f"Interrogation du modèle Ollama : {model}")
        if count_tokens(prompt) > 4096:
            logger.error("Le prompt est trop long.")
            return "Le prompt soumis est trop volumineux pour être traité en une seule fois. Pour obtenir une réponse pertinente, veuillez préciser votre question ou cibler une période, catégorie ou rubrique spécifique.", st_prompt, et_prompt, None, None, prompt, model
        try:
            async with httpx.AsyncClient() as client:
                try:
                    try:
                        response = await client.post(
                            OLLAMA_URL,
                            json=payload,
                            timeout=httpx.Timeout(595)
                        )
                    except asyncio.CancelledError:
                        await client.aclose()
                        logger.warning("Requête Ollama annulée : tâche asyncio annulée (ex : client déconnecté)")
                        raise
                except httpx.TimeoutException:
                    logger.error(f"Erreur de timeout lors de la connexion à l'API Ollama")
                    return "Je suis désolé 😥, le délai de réponse du service d'analyse a été dépassé. Veuillez reformuler votre question ou réessayer dans quelques instants.", st_prompt, et_prompt, None, None, prompt, model
                response.raise_for_status()
                json_data = response.json()
                if "message" in json_data and "content" in json_data["message"]:
                    st_format = datetime.now()
                    content: str = json_data["message"]["content"]

                    parts = content.split("</think>", 1)
                    if len(parts) > 1:
                        content = parts[1]
                    else:
                        content = content
                    # TODO
                    # if not is_not_empty and prompt_data:
                    #     content += "\n"
                    #     content += prompt_data.split(".")[3]
                    content = format_response(content)
                    et_format = datetime.now()
                    return content, st_prompt, et_prompt, st_format, et_format, prompt, model
                else:
                    raise ValueError("Réponse invalide de l'API Ollama")

        except httpx.RequestError as e:
            logger.error(f"Erreur de connexion à l'API Ollama : {e}")
            raise
    except Exception as e:
        logger.error(f"Une erreur inattendue est survenue : {e}")
        raise

def format_response(content):
    content = convert_markdown_to_html(content)
    content = content.lstrip('\n')
    content = content.replace("\r\n", "\n")
    content = content.replace("\r", "\n")
    content = add_br_outside_blocks(content)
    content = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", content)
    content = re.sub(r"__(.+?)__", r"<b>\1</b>", content)
    content = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", content)
    content = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<i>\1</i>", content)
    content = re.sub(r"^<br/>\s*", "", content)
    content = re.sub(r"<br/>\s*$", "", content)
    return content


async def old_ask_ollama(
    context_data: pd.DataFrame, 
    question:str, 
    history: list[dict], 
    category: str, 
    context_lenght: int, 
    lexiques: dict[str, list[str]], 
    simple_dict: list[dict]) -> str:
    """
    Interroge le modèle Ollama avec les données de contexte et un prompt, puis retourne la réponse.

    Args:
        context_data (pd.DataFrame): Les données à utiliser comme contexte.
        question (str): The user's question.
        history (list): L'historique des questions et réponses.
    Returns:
        str: La réponse du modèle.
    Raises:
        httpx.RequestError: En cas d'erreur de connexion à l'API Ollama.
        ValueError: Si la réponse de l'API est invalide.
    """
    st_prompt = datetime.now()
    full_query = ""
    try:
        lexiques = await get_mapping()
        parametres = await parse_user_query(question, lexiques, simple_dict)
        prompt_data = get_data_for_llm(context_data, simple_dict, **parametres)

        if not prompt_data and history and context_lenght > 0:
            user_messages = [h.get("content", "") for h in history if h.get("role") == "user"]
            last_user_msgs = user_messages[-context_lenght:] if context_lenght and len(user_messages) >= context_lenght else user_messages
            full_query = " ".join(last_user_msgs + [question])
            parametres = await parse_user_query(full_query, lexiques, simple_dict)
            prompt_data = get_data_for_llm(context_data, simple_dict, **parametres)
            if prompt_data and count_tokens(str(prompt_data)) > 4096:
                parametres = await parse_user_query(question, lexiques, simple_dict)
                prompt_data = get_data_for_llm(context_data, simple_dict, **parametres)

        # TODO: Remove prompt_data.txt
        if full_query == "":
            with open("prompt_data.txt", "w", encoding="utf-8") as f:
                f.write(category + f"\n{str(parametres)}\n" + question + "\n\n" + str(prompt_data))
        else:
            with open("prompt_data.txt", "w", encoding="utf-8") as f:
                f.write(category + f"\n{str(parametres)}\n" + full_query + "\n\n" + str(prompt_data))

        rules = await get_rules(category)
        prompt = ""
        prompt_system = (
            "Tu es un assistant utilisé par les contrôleurs de gestion.\n"
            "Sois simple, clair et bref.\n"
            "Base-toi uniquement sur les données ci-dessous, n'invente rien.\n"
            "ATTENTION : Lors de toute addition ou soustraction de valeurs portant sur des rubriques (codes hiérarchiques), prends en compte la hiérarchie.\n"
            "En particulier, n'additionne pas à la fois une rubrique parent et ses enfants, "
            "et ne fais pas de calculs qui entraîneraient un double comptage ou une erreur d'agrégation liée à la structure hiérarchique des données.\n"
            "Ne mentionne pas les codes hiérarchiques dans ta réponse et ne les utilise que pour comprendre la structure des données, jamais dans le texte retourné à l'utilisateur."
        )
        if prompt_data:
            prompt = (
                f"Contexte: La résidence (ou section analytique, ou SA) analysée se nomme {str(context_data.iloc[1, 0]).split(' - ')[1]} "
                f"et le formulaire (ou onglet) concerné est {context_data.iloc[1, 2]}.\n"
                "Règles de réponse:\n"
                + "\n".join(rules)
                + "\n" + 
                f"Réponds à la question suivante par une réponse formatée en HTML: {question}\n"
                f"Données:\n{prompt_data}\n"
            )
        else:
            prompt = (
                f"Contexte: La résidence (ou section analytique, ou SA) analysée se nomme {str(context_data.iloc[1, 0]).split(' - ')[1]} "
                f"et le formulaire (ou onglet) concerné est {context_data.iloc[1, 2]}.\n"
                f"Réponds à la question suivante par une réponse formatée en HTML: {question}\n"
                "Si la question concerne des informations qui ne figurent pas dans ce prompt, "
                "adresse-toi directement au contrôleur de gestion pour obtenir plus de précisions, "
                "par exemple sur l'année ou le contexte (réel, budget, prévisionnel).\n"
                "Si la question est un simple message de salutation, réponds poliment, souhaite la bienvenue, "
                "et demande quelle question ou quel besoin d'analyse possède l'utilisateur aujourd'hui.\n"
            )
        if not history:
            messages = [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = history + [{"role": "user", "content": prompt}]

        if category != "Formule et définition":
            model = GPT
        else:
            model = SOELOG

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": -1
        }
        et_prompt = datetime.now()
        logger.info(f"Interrogation du modèle Ollama : {model}")
        if count_tokens(prompt) > 4096:
            logger.error("Le prompt est trop long.")
            return "Le prompt soumis est trop volumineux pour être traité en une seule fois. Pour obtenir une réponse pertinente, veuillez préciser votre question ou cibler une période, catégorie ou rubrique spécifique.", st_prompt, et_prompt, None, None, prompt, model
        try:
            async with httpx.AsyncClient() as client:
                try:
                    try:
                        response = await client.post(
                            OLLAMA_URL,
                            json=payload,
                            timeout=httpx.Timeout(595)
                        )
                    except asyncio.CancelledError:
                        await client.aclose()
                        logger.warning("Requête Ollama annulée : tâche asyncio annulée (ex : client déconnecté)")
                        raise
                except httpx.TimeoutException:
                    logger.error(f"Erreur de timeout lors de la connexion à l'API Ollama")
                    return "Je suis désolé 😥, le délai de réponse du service d'analyse a été dépassé. Veuillez reformuler votre question ou réessayer dans quelques instants.", st_prompt, et_prompt, None, None, prompt, model
                response.raise_for_status()
                json_data = response.json()
                if "message" in json_data and "content" in json_data["message"]:
                    st_format = datetime.now()
                    content: str = json_data["message"]["content"]

                    parts = content.split("</think>", 1)
                    if len(parts) > 1:
                        content = parts[1]
                    else:
                        content = content
                    content = convert_markdown_to_html(content)
                    content = content.lstrip('\n')
                    content = content.replace("\r\n", "\n")
                    content = content.replace("\r", "\n")
                    content = add_br_outside_blocks(content)
                    content = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", content)
                    content = re.sub(r"__(.+?)__", r"<b>\1</b>", content)
                    content = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", content)
                    content = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<i>\1</i>", content)
                    content = re.sub(r"^<br/>\s*", "", content)
                    content = re.sub(r"<br/>\s*$", "", content)
                    et_format = datetime.now()
                    return content, st_prompt, et_prompt, st_format, et_format, prompt, model
                else:
                    raise ValueError("Réponse invalide de l'API Ollama")

        except httpx.RequestError as e:
            logger.error(f"Erreur de connexion à l'API Ollama : {e}")
            raise
    except Exception as e:
        logger.error(f"Une erreur inattendue est survenue : {e}")
        raise


async def old_old_ask_ollama(
    context_data: pd.DataFrame, 
    question:str, 
    history: list[dict], 
    category: str, 
    context_lenght: int, 
    lexiques: dict[str, list[str]], 
    simple_dict: list[dict]) -> str:
    """
    Interroge le modèle Ollama avec les données de contexte et un prompt, puis retourne la réponse.

    Args:
        context_data (pd.DataFrame): Les données à utiliser comme contexte.
        question (str): The user's question.
        history (list): L'historique des questions et réponses.
    Returns:
        str: La réponse du modèle.
    Raises:
        httpx.RequestError: En cas d'erreur de connexion à l'API Ollama.
        ValueError: Si la réponse de l'API est invalide.
    """
    st_prompt = datetime.now()
    full_query = ""
    try:
        lexiques = await get_mapping()
        parametres = await parse_user_query(question, lexiques, simple_dict)
        prompt_data = get_data_for_llm(context_data, simple_dict, **parametres)

        if not prompt_data and history and context_lenght > 0:
            user_messages = [h.get("content", "") for h in history if h.get("role") == "user"]
            last_user_msgs = user_messages[-context_lenght:] if context_lenght and len(user_messages) >= context_lenght else user_messages
            full_query = " ".join(last_user_msgs + [question])
            parametres = await parse_user_query(full_query, lexiques, simple_dict)
            prompt_data = get_data_for_llm(context_data, simple_dict, **parametres)
            if prompt_data and count_tokens(str(prompt_data)) > 4096:
                parametres = await parse_user_query(question, lexiques, simple_dict)
                prompt_data = get_data_for_llm(context_data, simple_dict, **parametres)

        # TODO: Remove prompt_data.txt
        if full_query == "":
            with open("prompt_data.txt", "w", encoding="utf-8") as f:
                f.write(category + f"\n{str(parametres)}\n" + question + "\n\n" + str(prompt_data))
        else:
            with open("prompt_data.txt", "w", encoding="utf-8") as f:
                f.write(category + f"\n{str(parametres)}\n" + full_query + "\n\n" + str(prompt_data))

        rules = await get_rules(category)
        prompt = ""
        prompt_system = (
            "Tu es un assistant utilisé par les contrôleurs de gestion.\n"
            "Sois simple, clair et bref.\n"
            "Base-toi uniquement sur les données ci-dessous, n'invente rien.\n"
            "ATTENTION : Lors de toute addition ou soustraction de valeurs portant sur des rubriques (codes hiérarchiques), prends en compte la hiérarchie.\n"
            "En particulier, n'additionne pas à la fois une rubrique parent et ses enfants, "
            "et ne fais pas de calculs qui entraîneraient un double comptage ou une erreur d'agrégation liée à la structure hiérarchique des données.\n"
            "Ne mentionne pas les codes hiérarchiques dans ta réponse et ne les utilise que pour comprendre la structure des données, jamais dans le texte retourné à l'utilisateur."
        )
        if prompt_data:
            prompt = (
                f"Contexte: La résidence (ou section analytique, ou SA) analysée se nomme {str(context_data.iloc[1, 0]).split(' - ')[1]} "
                f"et le formulaire (ou onglet) concerné est {context_data.iloc[1, 2]}.\n"
                "Règles de réponse:\n"
                + "\n".join(rules)
                + "\n" + 
                f"Question: {question}\n"
                f"Données:\n{prompt_data}\n"
            )
        else:
            prompt = (
                f"Contexte: La résidence (ou section analytique, ou SA) analysée se nomme {str(context_data.iloc[1, 0]).split(' - ')[1]} "
                f"et le formulaire (ou onglet) concerné est {context_data.iloc[1, 2]}.\n"
                f"Question: {question}\n"
                "Si la question concerne des informations qui ne figurent pas dans ce prompt, "
                "adresse-toi directement au contrôleur de gestion pour obtenir plus de précisions, "
                "par exemple sur l'année ou le contexte (réel, budget, prévisionnel).\n"
                "Si la question est un simple message de salutation, réponds poliment, souhaite la bienvenue, "
                "et demande quelle question ou quel besoin d'analyse possède l'utilisateur aujourd'hui.\n"
            )
        if not history:
            messages = [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = history + [{"role": "user", "content": prompt}]

        if category != "Formule et définition":
            model = GPT
        else:
            model = SOELOG

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "keep_alive": -1
        }
        et_prompt = datetime.now()
        logger.info(f"Interrogation du modèle Ollama : {model}")
        if count_tokens(prompt) > 4096:
            logger.error("Le prompt est trop long.")
            return "Le prompt soumis est trop volumineux pour être traité en une seule fois. Pour obtenir une réponse pertinente, veuillez préciser votre question ou cibler une période, catégorie ou rubrique spécifique.", st_prompt, et_prompt, None, None, prompt, model
        try:
            async with httpx.AsyncClient() as client:
                try:
                    try:
                        response = await client.post(
                            OLLAMA_URL,
                            json=payload,
                            timeout=httpx.Timeout(595)
                        )
                    except asyncio.CancelledError:
                        await client.aclose()
                        logger.warning("Requête Ollama annulée : tâche asyncio annulée (ex : client déconnecté)")
                        raise
                except httpx.TimeoutException:
                    logger.error(f"Erreur de timeout lors de la connexion à l'API Ollama")
                    return "Je suis désolé 😥, le délai de réponse du service d'analyse a été dépassé. Veuillez reformuler votre question ou réessayer dans quelques instants.", st_prompt, et_prompt, None, None, prompt, model
                response.raise_for_status()
                json_data = response.json()
                if "message" in json_data and "content" in json_data["message"]:
                    st_format = datetime.now()
                    content: str = json_data["message"]["content"]

                    parts = content.split("</think>", 1)
                    if len(parts) > 1:
                        content = parts[1]
                    else:
                        content = content
                    content = convert_markdown_to_html(content)
                    content = content.lstrip('\n')
                    content = content.replace("\r\n", "\n")
                    content = content.replace("\r", "\n")
                    content = add_br_outside_blocks(content)
                    content = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", content)
                    content = re.sub(r"__(.+?)__", r"<b>\1</b>", content)
                    content = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", content)
                    content = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<i>\1</i>", content)
                    content = re.sub(r"^<br/>\s*", "", content)
                    content = re.sub(r"<br/>\s*$", "", content)
                    et_format = datetime.now()
                    return content, st_prompt, et_prompt, st_format, et_format, prompt, model
                else:
                    raise ValueError("Réponse invalide de l'API Ollama")

        except httpx.RequestError as e:
            logger.error(f"Erreur de connexion à l'API Ollama : {e}")
            raise
    except Exception as e:
        logger.error(f"Une erreur inattendue est survenue : {e}")
        raise
