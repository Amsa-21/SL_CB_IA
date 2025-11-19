from decouple import config
import spacy
import logging
from app.models.classifier import train_classifier
from app.models.__init__ import questions, categories

OLLAMA_URL = config("OLLAMA_URL", default="http://si-5/api/chat")
# * GPT = config("OLLAMA_MODEL", default="gpt-oss:20b")
# * SOELOG = config("OLLAMA_MODEL", default="soelog-model")

# * SOELOG = GPT = config("OLLAMA_MODEL", default="deepseek-r1:14b")
# * SOELOG = GPT = config("OLLAMA_MODEL", default="llama3:8b")
SOELOG = GPT = config("OLLAMA_MODEL", default="gpt-oss:20b")

DATABASE_URL = config("DATABASE_URL", default="mssql+aioodbc://amsata:amsata@Soelog")
USER_FK = config("USER_FK", cast=int, default=8)
CONTEXT_LENGTH = config("CONTEXT_LENGTH", cast=int, default=0)
NIVEAU_HIERARCHIE = config("NIVEAU_HIERARCHIE", cast=int, default=2)
HISTORY_LENGTH = config("HISTORY_LENGTH", cast=int, default=3)
N_NEIGHBORS = config("N_NEIGHBORS", cast=int, default=2)

# Project settings
PROJECT_NAME: str = "API Soelog"
PROJECT_VERSION: str = "1.0"
API_KEY = config("API_KEY", default="SpyH5uBV7rzCpDA6iyJBfK5QukZeUBba")

# ML Models
nlp = None
vectorizer = None
clf = None
french_stopwords = []
lexiques={}

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
        vectorizer, clf = train_classifier(questions, categories)
        logger.info("Vectorizer et classifieur entraînés avec succès.")
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du vectorizer et du classifieur : {e}")
        raise