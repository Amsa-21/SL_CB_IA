import asyncio
import logging

import httpx
import pandas as pd

from app.core.config import GPT, OLLAMA_URL, SOELOG
from app.utils.functions import (
    add_br_outside_blocks,
    convert_markdown_to_html,
    count_tokens,
    data_to_pivot,
    datetime,
    find_res,
    get_data_for_llm,
    get_ext_data_for_llm,
    get_ret_dataframe,
    get_rules,
    parse_user_query,
    re,
    transform_for_llm,
)

logger = logging.getLogger(__name__)

class LLMAPIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def post_llm(self, payload: dict):
        try:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        self.base_url,
                        json=payload,
                        timeout=httpx.Timeout(595)
                    )
                except asyncio.CancelledError:
                    await client.aclose()
                    logger.warning("Requête Ollama annulée : tâche asyncio annulée (ex : client déconnecté)")
                    raise
        except httpx.TimeoutException:
            logger.error("Erreur de timeout lors de la connexion à l'API Ollama")
            return None, "timeout"

        try:
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'], None
            else:
                logger.error(f"Réponse invalide de l'API Ollama: {response.text}")
                return None, "invalid_response"
        except Exception as e:
            logger.error(f"Erreur http lors de l'appel à Ollama: {e}")
            return None, "http_error"

class Ollama:
    def __init__(self, base_url: str = OLLAMA_URL):
        self.base_url = base_url

    async def chat(self, payload: dict):
        """
        Envoie une requête chat à l'API Ollama via httpx.
        Args:
            payload (dict): Données à envoyer à l'API Ollama.
        Returns:
            response_content (str|None): La réponse textuelle du modèle, ou None si erreur.
            error (str|None): Le code d'erreur si applicable, sinon None.
        """
        try:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        self.base_url,
                        json=payload,
                        timeout=httpx.Timeout(595)
                    )
                except asyncio.CancelledError:
                    await client.aclose()
                    logger.warning("Requête Ollama annulée : tâche asyncio annulée (ex : client déconnecté)")
                    raise
        except httpx.TimeoutException:
            logger.error("Erreur de timeout lors de la connexion à l'API Ollama")
            return None, "timeout"

        try:
            json_data = response.json()
            if "message" in json_data and "content" in json_data["message"]:
                content: str = json_data["message"]["content"]

                parts = content.split("</think>", 1)
                if len(parts) > 1:
                    content = parts[1]
                else:
                    content = content
                return content
            else:
                logger.error(f"Réponse invalide de l'API Ollama: {response.text}")
                return None, "invalid_response"
        except Exception as e:
            logger.error(f"Erreur http lors de l'appel à Ollama: {e}")
            return None, "http_error"


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
    residences: pd.DataFrame
) -> str:
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
        ext_sa, resids = await find_res(question, residences, threshold = 60)

        if ext_sa:
            is_not_empty = True
            parametres, prompt_data = await get_ext_data_for_llm(
                question,
                context_data,
                sa_fk,
                form_fk,
                resids,
                lexiques
            )
        else:
            # 1. Analyse sur la question seule
            parametres = await parse_user_query(question, lexiques, simple_dict)
            if parametres is None:
                parametres = {
                    'groupes': [],
                    'types_valeur': [],
                    'annees': [],
                    'nature_ecriture': [],
                    'lignes': [],
                    'mois': []
                }
            is_not_empty, prompt_data = await get_data_for_llm(context_data, simple_dict, sa_fk, form_fk, **parametres)

            # 2. Si aucune donnée pertinente, tentative fallback avec transform_for_llm
            if not is_not_empty:
                context_data_pivot = data_to_pivot(context_data)
                ret_context_data = get_ret_dataframe(context_data_pivot, parametres)
                is_not_empty, prompt_data = transform_for_llm(ret_context_data)

            # 3. Si toujours aucune donnée, concatène l'historique utilisateur et retente
            if (not is_not_empty or not prompt_data):
                if history and context_lenght > 0:
                    user_messages = [h.get("content", "") for h in history if h.get("role") == "user"]
                    last_user_msgs = user_messages[-context_lenght:] if context_lenght and len(user_messages) >= context_lenght else user_messages
                    full_query = " ".join(last_user_msgs + [question])
                    parametres = await parse_user_query(full_query, lexiques, simple_dict)
                    if parametres is None:
                        parametres = {
                            'groupes': [],
                            'types_valeur': [],
                            'annees': [],
                            'nature_ecriture': [],
                            'lignes': [],
                            'mois': []
                        }
                    is_not_empty, prompt_data = await get_data_for_llm(context_data, simple_dict, sa_fk, form_fk, **parametres)

                    # 4. Si concaténation historique toujours vide, utilise à nouveau le fallback transform_for_llm
                    if not is_not_empty:
                        context_data_pivot = data_to_pivot(context_data)
                        ret_context_data = get_ret_dataframe(context_data_pivot, parametres)
                        is_not_empty, prompt_data = transform_for_llm(ret_context_data)

                    # 5. Si résultat trop volumineux (tokens), retente avec la question simple, ET si ensuite toujours rien, fallback transform_for_llm
                    if prompt_data and count_tokens(str(prompt_data)) > 4096:
                        parametres = await parse_user_query(question, lexiques, simple_dict)
                        if parametres is None:
                            parametres = {
                                'groupes': [],
                                'types_valeur': [],
                                'annees': [],
                                'nature_ecriture': [],
                                'lignes': [],
                                'mois': []
                            }
                        is_not_empty, prompt_data = await get_data_for_llm(context_data, simple_dict, sa_fk, form_fk, **parametres)
                        # Fallback si toujours rien après retry prompt simple
                        if not is_not_empty:
                            context_data_pivot = data_to_pivot(context_data)
                            ret_context_data = get_ret_dataframe(context_data_pivot, parametres)
                            is_not_empty, prompt_data = transform_for_llm(ret_context_data)

        # Ne jamais écrire 'None' dans prompt_data.txt pour la data affichée
        pd_str = prompt_data if prompt_data is not None else "Vide"
        params_str = str(parametres) if parametres else "" 

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

        if (is_not_empty or ext_sa) and prompt_data:
            """if count_tokens(str(prompt_data)) > 400:
                rules.append("Réponds en un seul paragraphe concise sans entrer dans les détails ligne à ligne.")
            """
            prompt = (
                f"CONTEXTE: La section analytique ou résidence {str(context_data.iloc[1,0]).split(' - ')[1]} est analysées.\n"
                "Règles de réponse:\n"
                + "\n".join(rules) +
                f"\nQUESTION: {question}\n"
                f"{pd_str}"
            )
        else:
            prompt = (
                f"CONTEXTE: La section analytique ou résidence {str(context_data.iloc[1,0]).split(' - ')[1]} est analysées."
                f"QUESTION: {question}\n"
                f"{pd_str}\n"
                "Réponds en deux phrases au maximum. "
                #"Si une information manque, invite à contacter le contrôleur de gestion avec les informations sur le contexte (Réel, Budget ou Projeté) et l'année. "
                #"Si c'est une salutation, réponds poliment et demande le besoin d'analyse du jour."
            )

        messages = history + [{"role": "user", "content": prompt}]

        if category != "Formule et définition":
            model = GPT
        else:
            model = SOELOG

        model = "llama3:8b"
        payload = {
            # TODO
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

        # Utilisation de la classe ci-dessus
        # ? llm_client = Ollama()
        llm_client = LLMAPIClient("http://si-5:1234/v1/chat/completions")
        try:
            response, error_type = await llm_client.post_llm(payload)
            if error_type == "timeout":
                return "Je suis désolé 😥, le délai de réponse du service d'analyse a été dépassé. Veuillez reformuler votre question ou réessayer dans quelques instants.", st_prompt, et_prompt, None, None, prompt, model
            if error_type == "http_error":
                raise ValueError("Erreur HTTP lors de l'appel à l'API Ollama")

            st_format = datetime.now()
            content = format_response(response)
            et_format = datetime.now()
            return content, st_prompt, et_prompt, st_format, et_format, prompt, model

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
    content = add_br_outside_blocks(content)
    content = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", content)
    content = re.sub(r"__(.+?)__", r"<b>\1</b>", content)
    content = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", content)
    content = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<i>\1</i>", content)
    content = re.sub(r"^<br/>\s*", "", content)
    content = re.sub(r"<br/>\s*$", "", content)
    return content
