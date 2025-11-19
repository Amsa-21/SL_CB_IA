import base64
import io
import json
import logging
import pickle
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
import matplotlib.pyplot as plt
import pandas as pd
import redis
from wordcloud import WordCloud

from app.core import config
from app.models.classifier import predict_category
from app.schemas.schemas import (
    ChatRequest,
    ChatResponse,
    EndSessionRequest,
    EndSessionResponse,
    InitSessionRequest,
    InitSessionResponse,
    WordCloudResponse,
)
from app.services.functions import (
    count_tokens,
    create_simplified_hierarchy,
    detect_keywords,
    execute_sp,
    execute_sp,
    format_time,
    from_cat_to_fk,
    get_mapping,
    preprocessing_data,
)
from app.services.ollama_service import ask_ollama, datetime

router = APIRouter()
logger = logging.getLogger(__name__)

r = redis.Redis()

@router.get("/")
async def greeting():
    html_content = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <title>FastAPI SI-12</title>
        <style>
            body {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background: #22223b;
            }
            h1 {
                text-align: center;
                font-family: Arial, sans-serif;
                color: #f8fafc;
            }
        </style>
    </head>
    <body>
        <h1>FastAPI est en cours d'exécution (Chatbot)...</h1>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.post("/init_session", response_model=InitSessionResponse, summary="Initialiser la session")
async def init_session(request: InitSessionRequest):
    """
    Initialise une nouvelle session à partir d'un tableau JSON.

    - **sa_fk**: Identifiant unique du SA à analyser.
    - **form_fk**: Identifiant unique du formulaire à analyser.

    Retourne un identifiant de session unique si l'initialisation est réussie.
    """
    try:
        resids = await execute_sp(
            "dbo.sp_saResidences",
            {
                "user_fk": 14,
            }
        )
        df_residences = pd.DataFrame(resids)
        df_residences = df_residences[df_residences["sa_fk"].notna()][["sa_fk", "sa", "resp"]]
        df_residences["sa_fk"] = df_residences["sa_fk"].astype(int)
        
        session_id = str(uuid.uuid4())
        data = await execute_sp(
            "ia.sp_simBudFormSA_one", 
            {
                "user_fk": config.USER_FK, 
                "sa_fk": request.sa_fk, 
                "form_fk": request.form_fk
            }
        )
        json_string = data[0].get('EcrituresDetails')
        if json_string is None:
            context_data = pd.DataFrame()
            simple_dict = {}
            try:
                r.set(session_id, pickle.dumps([context_data, [], simple_dict, request.sa_fk, request.form_fk, df_residences]))
            except Exception as e:
                logger.error(f"Erreur lors de l'enregistrement de la session dans Redis (json_string vide) : {e}")
                raise HTTPException(status_code=500, detail="Erreur lors de l'initialisation de la session (Redis).")
            logger.info(f"Nouvelle session initialisée (json_string vide) : {session_id}")
            return {"session_id": session_id}
        
        data_records = json.loads(json_string)
        context_data = pd.DataFrame(data_records)
        res = await execute_sp(
            "dbo.sp_simBudLines",
            {
                "user_fk": config.USER_FK,
                "form_fk": request.form_fk,
                "line_fk": 0,
                "choix": 0,
                "isVisible": 1
            }
        )
        simple_dict = create_simplified_hierarchy(res)
        context_data = preprocessing_data(context_data, simple_dict).copy()
        try:
            r.set(session_id, pickle.dumps([context_data, [], simple_dict, request.sa_fk, request.form_fk, df_residences]))
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de la session dans Redis : {e}")
            raise HTTPException(status_code=500, detail="Erreur lors de l'initialisation de la session (Redis).")
        logger.info(f"Nouvelle session initialisée : {session_id} ¤ {str(context_data.iloc[1, 0]).split(' - ')[1]} - {context_data.iloc[1, 2]}")
        return {"session_id": session_id}
    except Exception as e:
        logger.error(f"Erreur inattendue lors de l'initialisation de la session : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")


@router.post("/chat", response_model=ChatResponse, summary="Discuter avec le chatbot")
async def chat(request: ChatRequest):
    """
    Envoyer une question au chatbot et recevoir une réponse.

    - **session_id**: L'ID de la session.
    - **question**: La question de l'utilisateur.
    - **user_fk**: L'ID de l'utilisateur connecté.

    Retourne la réponse du chatbot ainsi que le temps de réponse du LLM.
    """
    start_time = datetime.now()
    assistant_response = cat = llm = question_fk = algo = None
    st_prompt = et_prompt = st_format = et_format = prompt = end_time = None
    score = 0.0
    motCle_fks = []
    good = False
    res = r.get(request.session_id)
    
    if res is None:
        logger.error(f"Session sans données : {request.session_id}")
        end_time = datetime.now()
        execution_time = end_time - start_time
        formatted_time = format_time(execution_time)
        return {
            "message": "Veuillez actualiser l'assistant IA.",
            "response_time": formatted_time
        }
    try:
        context_data, history_list, simple_dict, sa_fk, form_fk, residences = pickle.loads(res)
        history = list(history_list)
        
        if len(history) > config.HISTORY_LENGTH*2+1:
            history = [history[0]] + history[-config.HISTORY_LENGTH*2:]
        lexiques = await get_mapping()
        cat, score, algo = predict_category(
            request.question,
            config.vectorizer,
            config.clf,
            config.nlp,
            config.french_stopwords,
            lexiques
        )
        logger.info(f"{cat} - {score:.2f} - {algo}")

        try:
            assistant_response, st_prompt, et_prompt, st_format, et_format, prompt, llm = await ask_ollama(
                context_data=context_data,
                question=request.question,
                history=history,
                category=cat,
                context_lenght=config.CONTEXT_LENGTH,
                lexiques=lexiques,
                simple_dict=simple_dict,
                sa_fk= sa_fk,
                form_fk=form_fk,
                residences=residences
            )
        except Exception as ollama_exc:
            logger.error(f"Erreur lors de l'appel à ask_ollama sur la session {request.session_id}: {ollama_exc}")
            raise HTTPException(status_code=500, detail="Erreur lors de l'interrogation du LLM.")
        
        try:
            keywords = detect_keywords(request.question)
        except Exception as keyword_exc:
            keywords = []
            logger.warning(f"Erreur lors de la détection des mots-clés: {keyword_exc}")
        
        motCle_fks = []
        # Exécution robuste, même si un mot-clé plante, on continue sur les autres
        for kw in keywords:
            try:
                result = await execute_sp(
                    "ia.sp_motCle_add",
                    {
                        "user_fk": config.USER_FK,
                        "categorie_fk": None,
                        "theLabel": kw.capitalize(),
                        "descr": ""
                    }
                )
                motCle_fk = 0
                if result[0].get("message"):
                    motCle_fk = result[0].get("motCle_fk", 0)
                    motCle_fks.append(str(motCle_fk))
            except Exception as motcle_exc:
                logger.warning(f"Erreur lors de l'ajout du mot-clé '{kw}': {motcle_exc}")
        try:
            categorie_fk = None
            result = None
            try:
                categorie_fk = await from_cat_to_fk(cat)
                result = await execute_sp(
                    "ia.sp_question_add",
                    {
                        "user_fk": config.USER_FK,
                        "categorie_fk": categorie_fk,
                        "theLabel": request.question,
                        "descr": ""
                    }
                )
            except Exception as category_exc:
                logger.warning(f"Erreur lors de la récupération/ajout de catégorie/question : {category_exc}")
            question_fk = 0
            if result and result[0].get("message"):
                question_fk = result[0].get("question_fk", 0)
                logger.info(result[0].get("message"))
        except Exception as e_reg_question:
            logger.warning(f"Échec de l'enregistrement de la question : {e_reg_question}")

        history.append({"role": "user", "content": request.question})
        history.append({"role": "assistant", "content": assistant_response})
        try:
            r.set(request.session_id, pickle.dumps([context_data, history, simple_dict, sa_fk, form_fk, residences]))
        except Exception as redis_exc:
            logger.error(f"Erreur lors de la sauvegarde en Redis: {redis_exc}")

        logger.info(f"Réponse complète envoyée pour la session {request.session_id}.")
        end_time = datetime.now()
        execution_time = end_time - start_time
        formatted_time = format_time(execution_time)
        good = True
        return {
            "message": assistant_response,
            "response_time": formatted_time
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue sur la session {request.session_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")

    finally:
        if good:
            try:
                def safe_dt(dt):
                    return dt if dt else None

                def safe_str(val):
                    val = "" if val is None else str(val)
                    return val

                safe_score = None
                try:
                    if score is not None:
                        safe_score = float(score)
                except Exception:
                    safe_score = None

                res = await execute_sp(
                    "ia.sp_chatBotReporting_add",
                    {
                        "user_fk": getattr(request, "user_fk", None),
                        "question_fk": question_fk,
                        "reponse": f"{count_tokens(assistant_response)} - {safe_str(assistant_response)}" if assistant_response is not None else "",
                        "prompt": f"{count_tokens(prompt)} - {safe_str(prompt.replace('\n', '<br>'))}" if prompt is not None else "",
                        "motCle": ",".join(motCle_fks) if motCle_fks else "",
                        "algorithme": algo,
                        "score": safe_score,
                        "llm": llm,
                        "theDateStart": safe_dt(start_time),
                        "theDateEnd": safe_dt(end_time),
                        "theDatePromptStart": safe_dt(st_prompt),
                        "theDatePromptEnd": safe_dt(et_prompt),
                        "theDateMEFStart": safe_dt(st_format),
                        "theDateMEFEnd": safe_dt(et_format)
                    }
                )
                logger.info(res[0]["message"])
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde: {e}")


@router.post("/end_session", response_model=EndSessionResponse, summary="Terminer une session")
async def end_session(request: EndSessionRequest):
    """
    Termine une session et supprime les données associées du cache.

    - **session_id**: L'ID de la session à terminer.

    Retourne la réussite de l'exécution.
    """
    res = r.get(request.session_id)
    if res:
        r.delete(request.session_id)
        logger.info(f"Session terminée : {request.session_id}")
        return {"message": "Session terminée."}
    else:
        raise HTTPException(status_code=404, detail="Session non trouvée.")


@router.get("/get_wordcloud", response_model=WordCloudResponse, summary="Générer un nuage de mots")
async def get_wordcloud():
    """
    Génère un nuage de mots basé sur les dernières conversations ou données agrégées.

    Retourne une image encodée en base64 représentant le nuage de mots.
    """
    try:
        res = await execute_sp(
            "ia.sp_chatBotReporting_get",
            {
                "user_fk": config.USER_FK,
            }
        )

        questions = [
            i.get("labelMotCle", "").strip()
            for i in res
            if i.get("labelMotCle")
        ]
        questions = list(filter(None, questions))
        all_questions = ' '.join(questions)

        if not all_questions:
            logger.warning("Aucune question trouvée pour générer le wordcloud.")
            raise HTTPException(status_code=404, detail="Aucune donnée disponible pour générer le nuage de mots.")

        wc = WordCloud(
            width=900,
            height=500,
            background_color='white',
            max_words=100,
            colormap='tab10',
            contour_color='steelblue',
            contour_width=1,
            prefer_horizontal=0.95,
            random_state=42,
            collocations=False
        ).generate(all_questions)

        img_buf = io.BytesIO()
        plt.figure(figsize=(10, 5), facecolor='white')
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(img_buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

        logger.info("Nuage de mots généré avec succès.")
        return {"image": f"data:image/png;base64,{img_base64}"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la génération du wordcloud: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur lors de la génération du wordcloud.")
