# Documentation du projet Soelog IA

## Vue d’ensemble
- Backend FastAPI (`app/main.py`) exposant des endpoints de chatbot et de prévision budgétaire pour les contrôleurs de gestion.
- Récupère les données analytiques via SQL Server (procédures stockées) et les met en forme avec `pandas` avant de les injecter dans un LLM (Ollama, modèle configurable via `OLLAMA_MODEL`).
- Redis sert de cache de session conversationnelle pour conserver le contexte entre les requêtes utilisateur.
- Des modèles NLP locaux (spaCy `fr_core_news_md`, RandomForest TF‑IDF, lexiques maison) catégorisent les questions et enrichissent les prompts envoyés au LLM.

## Architecture applicative
- **Entrée HTTP** : `FastAPI` + `CORSMiddleware` (origines illimitées). Les routes sont regroupées dans `app/api/v1/api.py`.
- **Sécurité** : `API_KEY` optionnelle (hook présent mais commenté dans `api_router`).
- **Initialisation** : le lifespan FastAPI appelle `initialize_ml_models()` (`app/core/config.py`) pour charger spaCy, les stopwords NLTK et entraîner le classifieur RandomForest à partir des données `app/models/__init__.py`.
- **Services métiers** : `app/services/functions.py` concentre le prétraitement des données (hiérarchie de lignes budgétaires, parsers de questions, accès BDD via `execute_sp`), plus de 1 200 lignes.
- **Client LLM** : `app/services/ollama_service.py` orchestre la génération de prompt, appelle l’API Ollama (`httpx.AsyncClient`) et normalise la réponse HTML.
- **Persistance & cache** : `app/db/session.py` gère SQLAlchemy Async pour les procédures, Redis (instancié dans `chatbot.py`) stocke `[context_data, history, simple_dict, sa_fk, form_fk, residences]` par `session_id`.
- **Observabilité** : logs détaillés dans `logs/` (info, error, fastapi, stats). Les appels LLM loguent temps de prompt/formatage et sauvegardent les prompts dans `prompt_data.txt`.

## Flux fonctionnel principal (chatbot)
1. `POST /api/v1/chatbot/init_session`
   - Charge les résidences (`dbo.sp_saResidences`) puis les écritures (`ia.sp_simBudFormSA_one`) pour `sa_fk`/`form_fk`.
   - Construit la hiérarchie des lignes (`dbo.sp_simBudLines` → `create_simplified_hierarchy`) et pré-traite les données (`preprocessing_data`).
   - Stocke le contexte en Redis et retourne `session_id`.
2. `POST /api/v1/chatbot/chat`
   - Récupère contexte + historique depuis Redis.
   - Prédit la catégorie de la question avec `predict_category` (TF‑IDF + fallback lexique).
   - Prépare les paramètres métier à partir de la question (`parse_user_query`, `get_ext_data_for_llm` ou `get_data_for_llm` selon détection de section analytique).
   - Compose le prompt (règles, contexte, données tabulaires) et appelle `ask_ollama`.
   - Formate la réponse HTML (`format_response`) et renvoie `{message, response_time}`.
   - Enregistre mots-clés (`ia.sp_motCle_add`), questions (`ia.sp_question_add`) et reporting (`ia.sp_chatBotReporting_add`).
3. `POST /api/v1/chatbot/end_session`
   - Supprime le contexte Redis.
4. `GET /api/v1/chatbot/get_wordcloud`
   - Récupère l’historique des mots-clés (`ia.sp_chatBotReporting_get`) et génère un nuage de mots `WordCloud`.

## Autres endpoints
- `GET /api/v1/chatbot/` et `GET /api/v1/prediction/` : pages HTML simples indiquant l’état du service.
- `GET /api/v1/prediction/` : placeholder « Prédiction de budget » (aucune logique métier encore implémentée).

## Configuration & variables d’environnement (`app/core/config.py`)
- `OLLAMA_URL` (défaut `http://si-5/api/chat`)
- `OLLAMA_MODEL` (alias `GPT` et `SOELOG`, défaut `gpt-oss:20b`)
- `DATABASE_URL` (défaut `mssql+aioodbc://amsata:amsata@Soelog`)
- `USER_FK`, `CONTEXT_LENGTH`, `NIVEAU_HIERARCHIE`, `HISTORY_LENGTH`, `N_NEIGHBORS`
- `PROJECT_NAME`, `PROJECT_VERSION`, `API_KEY`
Toutes passent par `python-decouple`. Les modèles NLP sont chargés au démarrage ; prévoir les ressources nécessaires et l’installation de `fr_core_news_md` + corpus NLTK `stopwords`.

## Pipeline LLM (`app/services/ollama_service.py`)
- **Sélection des données** : `parse_user_query` extrait groupes, périodes, contextes (Réel/Budget/Prévision), lignes, mois. Les réponses de `get_data_for_llm`/`get_ext_data_for_llm` sont tronquées si `count_tokens` > 4096.
- **Gestion de l’historique** : si la requête actuelle ne suffit pas à trouver des données pertinentes, concatène les derniers tours utilisateurs jusqu’à `CONTEXT_LENGTH`.
- **Garde-fous** : ajout de règles dynamiques (ex. limiter les réponses quand trop de données) et fallback avec message d’erreur si prompt > 4 096 tokens ou timeout.
- **Post-traitement** : conversion Markdown→HTML (`markdown` lib), sanitation des balises, ajout de `<br/>` hors blocs, stylisation italique/gras.

## Composants utilitaires clés (`app/services/functions.py`)
- **Préparation hiérarchique** : `generate_hierarchy_codes`, `extract_flat_hierarchy_list`, `extract_all_descendants_for_list` pour produire les codes et niveaux destinés au LLM.
- **Parsing métier** : `parse_user_query` (regex, spaCy NER, fuzzy matching `thefuzz`) pour détecter groupes, années, types de valeur, lignes cibles.
- **Accès base** : `execute_sp` (wrapper SQLAlchemy async) et helpers (`get_rules`, `get_mapping`, `find_res`…).
- **Divers** : formatage temps (`format_time`), transformation Markdown, comptage de tokens, détection de mots-clés, etc.

## Données & ressources supplémentaires
- Dossiers `lora_adapter/` contenant une LoRA et des jeux `jsonl` pour affiner un modèle GPT-OSS (non chargé automatiquement).
- `prompt_data.txt` mis à jour à chaque requête pour diagnostiquer les prompts envoyés.
- `tests/` contient des scripts exploratoires (`train.py`, `test_sp.py`) et un `data.csv` d’exemple, mais pas de suite de tests automatisée.

## Lancement local
1. Installer les dépendances (non fournies ici, typiquement via `pip install -r requirements.txt` si disponible).
2. Installer les ressources NLP :  
   `python -m spacy download fr_core_news_md`  
   `python -m nltk.downloader stopwords`
3. Configurer les variables (.env ou environnement système).
4. Démarrer l’API :  
   `uvicorn app.main:app --reload`
5. Vérifier les endpoints via `http://localhost:8000/api/v1/chatbot` ou la documentation FastAPI auto-générée (`/docs`).

## Observabilité et fichiers générés
- Logs tournants dans `logs/` (info/error/fastapi/stats). `ollamaserver.log` capture les échanges externes.
- `prompt_data.txt` écrase la dernière requête pour inspecter prompt/données/paramètres.
- `logs/QUERY.LOG` et `logs/STATS.LOG` disponibles pour audits supplémentaires.

## Tests & amélioration continue
- Pas de tests unitaires/intégration automatisés ; `tests/` contient des notebooks et scripts exploratoires.
- Recommandation : ajouter des tests pour `parse_user_query`, `preprocessing_data`, et un mock d’`ask_ollama` pour éviter les régressions.

## Points d’attention
- Les logs sont ouverts en mode `w` à chaque démarrage (perte d’historique).
- L’API n’impose pas encore la clé (`Depends(get_api_key)` commenté).
- `prompt_data.txt` est écrasé à chaque requête et stocke potentiellement des données sensibles ; penser à la gouvernance des accès.
- Les procédures stockées et leurs schémas ne sont pas versionnés ici : toute évolution côté SQL doit être synchronisée manuellement.


