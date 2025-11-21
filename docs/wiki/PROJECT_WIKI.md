# Wiki Soelog IA

## 1. Mission du projet
Offrir aux contrôleurs de gestion une interface conversationnelle capable d’exploiter les écritures budgétaires (réel/budget/prévision) et les hiérarchies métier afin de produire des analyses synthétiques directement dans les outils existants. L’API FastAPI sert de passerelle entre les données SQL Server, les règles métier, et les modèles de langage hébergés sur Ollama.

## 2. Architecture logicielle
| Couche | Description |
| --- | --- |
| **API** | `FastAPI` + `CORSMiddleware`, routes définies dans `app/api/v1/`. |
| **Sécurité** | Clé API (`app/security.py`), optionnelle pour l’instant (injection future via `Depends`). |
| **NLP / ML** | spaCy `fr_core_news_md`, stopwords NLTK, classifieur RandomForest TF‑IDF (`app/models/classifier.py`). |
| **Services métiers** | `app/services/functions.py` : prétraitements, hiérarchies, parsing des questions, appels SQL. |
| **LLM client** | `app/services/ollama_service.py` : génération des prompts, appels `httpx` vers Ollama, formatage HTML. |
| **Persistance** | SQL Server (procédures `ia.*`, `dbo.*`), Redis pour l’état de session, fichiers de log sous `logs/`. |

Diagramme textuel :
```
Client HTTP
  └─ FastAPI (main.py)
       ├─ Chatbot endpoints
       │    ├─ Redis (sessions)
       │    ├─ SQL Server (procédures budgétaires)
       │    └─ Ollama (LLM)
       └─ Prediction endpoints (placeholder)
```

## 3. Flux métier détaillé (chatbot)
1. **Initialisation (`/init_session`)**
   - Récupère les résidences (`dbo.sp_saResidences`) et les écritures du formulaire (`ia.sp_simBudFormSA_one`).
   - Construit la hiérarchie via `dbo.sp_simBudLines` → `create_simplified_hierarchy`.
   - `preprocessing_data` filtre et ajoute les codes hiérarchiques puis stocke le contexte en Redis.
2. **Question (`/chat`)**
   - Recharge le contexte Redis, tronque l’historique selon `HISTORY_LENGTH`.
   - Prédiction de catégorie (`predict_category`). Si score faible, fallback lexique.
   - Génère les paramètres d’extraction (`parse_user_query`, `get_ext_data_for_llm`, `get_data_for_llm`).
   - Compose le prompt (règles d’évitement de double comptage, limites de longueur) et appelle Ollama.
   - Formate la réponse HTML (`format_response`), stocke l’historique, enregistre mots-clés/questions/reporting.
3. **Fin (`/end_session`)**
   - Supprime l’entrée Redis.
4. **Wordcloud (`/get_wordcloud`)**
   - Construit un nuage de mots à partir des libellés enregistrés (matplotlib + WordCloud).

## 4. Configuration et variables
- `.env` géré via `python-decouple` (`app/core/config.py`).
- Variables majeures : `OLLAMA_URL`, `OLLAMA_MODEL`, `DATABASE_URL`, `API_KEY`, `USER_FK`, `CONTEXT_LENGTH`, `HISTORY_LENGTH`, `NIVEAU_HIERARCHIE`, `N_NEIGHBORS`.
- `initialize_ml_models()` est appelé à chaque démarrage : prévoir l’installation préalable des ressources spaCy/NLTK et la présence du dataset `questions/categories`.

## 5. Stockage et logs
- `logs/info.log`, `logs/error.log`, `logs/fastapi.log`, `logs/STATS.LOG`, `logs/QUERY.LOG`.
- `prompt_data.txt` conserve le dernier prompt + données (utile pour le debug, attention aux données sensibles).
- `ollamaserver.log` trace les interactions côté serveur LLM.

## 6. Données et assets
- `lora_adapter/` : configuration et checkpoints pour l’adaptation LoRA (GPT‑OSS).
- `tests/` : scripts `train.py`, `test_sp.py`, notebooks d’expérimentation (`draft.ipynb`).
- `v1.ipynb`, `v2.ipynb` : carnets d’exploration.

## 7. Déploiement & exécution
1. Installer les dépendances Python (requirements à générer si besoin).
2. Installer spaCy + stopwords.
3. Configurer accès SQL Server, Redis, Ollama (URL et modèle).
4. `uvicorn app.main:app --host 0.0.0.0 --port 8000`
5. Surveiller les logs et l’état Redis (ex. `redis-cli monitor`).

## 8. Roadmap / axes d’amélioration
- **Sécurité** : activer la clé API au niveau router, audit des permissions.
- **Observabilité** : rotation des logs, métriques Prometheus, traces des prompts (optionnel).
- **Robustesse** : mécanismes de retry côté Ollama, tests unitaires sur les fonctions critiques, validation des schémas SQL.
- **Prediction module** : implémenter la logique métier sous `/api/v1/prediction`.
- **Gouvernance données** : chiffrer ou restreindre l’accès à `prompt_data.txt`, cartographier les champs sensibles.

## 9. Références croisées
- README synthétique : `README.md`
- Présentation courte / business : `PROJECT_OVERVIEW.md`
- Code source : `app/…`, `tests/…`


