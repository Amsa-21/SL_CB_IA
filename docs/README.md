# Soelog IA – Assistant budgétaire FastAPI

## Aperçu rapide
- API FastAPI dédiée aux contrôleurs de gestion (chatbot analytique + placeholder prédiction).
- Connexion SQL Server via procédures stockées pour charger les écritures et hiérarchies budgétaires.
- Traitement NLP local : spaCy `fr_core_news_md`, TF‑IDF + RandomForest, lexiques métiers.
- Génération de réponses via Ollama (`OLLAMA_MODEL` configurable) avec formatage HTML prêt à afficher.
- Sessions utilisateur persistées dans Redis pour conserver l’historique de conversation.

## Architecture en un coup d’œil
```
Client → FastAPI (`app/main.py`)
          ├─ `/api/v1/chatbot/*`  (chat, init_session, end_session, wordcloud)
          └─ `/api/v1/prediction/*` (placeholder)
        ↳ Services (`app/services/`) : prétraitement données, prompts, Ollama
        ↳ SQL Server (procédures `ia.*`, `dbo.*`)
        ↳ Redis (cache de sessions)
```

## Démarrage rapide
1. Installer les dépendances (ex. `pip install -r requirements.txt` si disponible).
2. Installer les ressources NLP :  
   `python -m spacy download fr_core_news_md`  
   `python -m nltk.downloader stopwords`
3. Définir les variables d’environnement principales (`OLLAMA_URL`, `OLLAMA_MODEL`, `DATABASE_URL`, `API_KEY`, etc.). Voir `app/core/config.py`.
4. Lancer l’API : `uvicorn app.main:app --reload`
5. Tester :  
   - Swagger UI : `http://localhost:8000/docs`  
   - Ping chatbot : `GET /api/v1/chatbot/`

## Points clés à connaître
- `initialize_ml_models()` (lifespan FastAPI) charge spaCy, les stopwords NLTK et entraîne le classifieur RandomForest au démarrage.
- `app/services/functions.py` centralise le prétraitement (hiérarchies, parsing des requêtes, accès DB).
- `app/services/ollama_service.py` orchestre la construction des prompts et l’appel HTTPX à Ollama.
- `redis` conserve `[context_data, history, simple_dict, sa_fk, form_fk, residences]` par session pour alimenter `ask_ollama`.
- Les prompts envoyés sont sauvegardés dans `prompt_data.txt` pour faciliter le debug.

## Ressources complémentaires
- Documentation détaillée : `PROJECT_OVERVIEW.md`
- Version wiki approfondie : `docs/wiki/PROJECT_WIKI.md`
- Scripts/tests exploratoires : `tests/`
- Logs : `logs/` (info, error, fastapi, stats)


