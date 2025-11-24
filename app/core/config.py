from decouple import config
import spacy
import logging
from app.models.classifier import train_classifier

DATABASE_URL = config("DATABASE_URL", default="mssql+aioodbc://amsata:amsata@Soelog")
OLLAMA_URL = config("OLLAMA_URL", default="http://si-5/api/chat")

# * SOELOG = config("OLLAMA_MODEL", default="soelog-model")
# * GPT = config("OLLAMA_MODEL", default="gpt-oss:20b")
# * SOELOG = GPT = config("OLLAMA_MODEL", default="deepseek-r1:14b")
# * SOELOG = GPT = config("OLLAMA_MODEL", default="llama3:8b")
SOELOG = GPT = config("OLLAMA_MODEL", default="gpt-oss:20b")

NIVEAU_HIERARCHIE = config("NIVEAU_HIERARCHIE", cast=int, default=2)
CONTEXT_LENGTH = config("CONTEXT_LENGTH", cast=int, default=0)
HISTORY_LENGTH = config("HISTORY_LENGTH", cast=int, default=3)
N_NEIGHBORS = config("N_NEIGHBORS", cast=int, default=2)
USER_FK = config("USER_FK", cast=int, default=8)

# Project settings
API_KEY = config("API_KEY", default="SpyH5uBV7rzCpDA6iyJBfK5QukZeUBba")
PROJECT_NAME: str = "API Soelog"
PROJECT_VERSION: str = "1.0"

# ML Models
french_stopwords = []
vectorizer = None
lexiques={}
nlp = None
clf = None

logger = logging.getLogger(__name__)

def initialize_ml_models():
    global nlp, french_stopwords, vectorizer, clf, lexiques

    try:
        nlp = spacy.load("fr_core_news_md")
        logger.info("Modèle spaCy chargé avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle spaCy : {e}")
        raise

    try:
        from nltk.corpus import stopwords
        french_stopwords.extend(stopwords.words("french"))
        logger.info("Stopwords français chargés avec succès.")
    except LookupError:
        import nltk
        nltk.download("stopwords")
        from nltk.corpus import stopwords
        french_stopwords.extend(stopwords.words("french"))
        logger.info("Stopwords français téléchargés et chargés avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des stopwords français : {e}")
        raise

    try:
        from app.core import get_questions_and_categories
        questions, categories = get_questions_and_categories(USER_FK)
        vectorizer, clf = train_classifier(questions, categories)
        logger.info("Vectorizer et classifieur entraînés avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du vectorizer et du classifieur : {e}")
        raise