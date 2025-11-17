import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from .__init__ import *

logger = logging.getLogger(__name__)

def preprocess(text: str, nlp, french_stopwords: list) -> list:
    """
    Prétraite le texte en :
    - Minuscule
    - Tokenisation avec spaCy
    - Suppression des stopwords, ponctuation, espaces, et tokens trop courts
    - Lemmatisation
    - Conservation des nombres (dates, montants, etc.)
    """
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        # Ignore la ponctuation, les espaces, et les tokens trop courts
        if (
            not token.is_stop
            and not token.is_punct
            and not token.is_space
            and (token.is_alpha or token.like_num)
            and len(token.text) > 1
        ):
            # Si c'est un nombre, on garde le texte original (ex: "2025")
            if token.like_num:
                tokens.append(token.text)
            else:
                lemma = token.lemma_.strip()
                if lemma and lemma not in french_stopwords:
                    tokens.append(lemma)
    return tokens

def classify(text: str, nlp, french_stopwords: list, lexique: dict, min_score_threshold: float) -> tuple:
    """
    Classe un texte dans une catégorie à partir d'un lexique.
    - Prend en compte la lemmatisation, la présence exacte, la pondération par fréquence et la couverture.
    - Retourne la catégorie la plus pertinente et un score de prédiction (float).
    """
    tokens = preprocess(text, nlp, french_stopwords)
    counts = Counter(tokens)
    scores = {}
    coverage = {}

    lexique_lemmatise = {}
    for cat, mots in lexique.items():
        if cat == "Détail":
            continue
        else:
            mots_lemmatise = set()
            for mot in mots:
                mots_lemmatise.update(preprocess(mot, nlp, french_stopwords))
            lexique_lemmatise[cat] = mots_lemmatise

    text_lower = text.lower()

    for cat, mots in lexique_lemmatise.items():
        freq_score = sum(counts[tok] for tok in mots if tok in counts)
        exact_score = sum(1 for mot in lexique[cat] if mot.lower() in text_lower)
        if mots:
            coverage[cat] = len([tok for tok in mots if tok in counts]) / len(mots)
        else:
            coverage[cat] = 0
        scores[cat] = freq_score + exact_score + coverage[cat]

    max_score = max(scores.values())
    best_cats = [cat for cat, score in scores.items() if score == max_score]

    if len(best_cats) == 1:
        best_cat = best_cats[0]
    else:
        best_coverage = max(coverage[cat] for cat in best_cats)
        best_covered = [cat for cat in best_cats if coverage[cat] == best_coverage]
        if len(best_covered) == 1:
            best_cat = best_covered[0]
        else:
            presence_counts = {cat: sum(1 for tok in lexique_lemmatise[cat] if tok in counts) for cat in best_covered}
            best_cat = max(presence_counts, key=presence_counts.get)

    raw_score = scores[best_cat]
    max_total_score = max(scores.values()) if scores else 1
    if max_total_score == 0:
        prediction_score = 0.0
    else:
        prediction_score = raw_score / max_total_score

    if prediction_score < min_score_threshold:
        best_cat = "Autre"

    return best_cat

def train_classifier(questions: list, categories: list):
    """
    Entraîne un classifieur RandomForest avec une vectorisation TF-IDF sur les questions et catégories fournies.
    Cette fonction initialise et entraîne les objets vectorizer et clf en tant que variables globales.
    """

    if not questions or not categories:
        logger.error("Les listes de questions et de catégories ne doivent pas être vides.")
        raise ValueError("Les listes de questions et de catégories ne doivent pas être vides.")

    if len(questions) != len(categories):
        logger.error("Le nombre de questions et de catégories doit être identique.")
        raise ValueError("Le nombre de questions et de catégories doit être identique.")

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_df=0.85,
        )
        X = vectorizer.fit_transform(questions)

        clf = RandomForestClassifier(
            n_estimators=100,
            min_samples_split=4,
            max_depth=None,
            n_jobs=-1
        )
        clf.fit(X, categories)
        return vectorizer, clf
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du classifieur : {e}")
        raise

def predict_category(text: str, vectorizer: TfidfVectorizer, clf: RandomForestClassifier, nlp, french_stopwords: list, lexiques: dict, threshold: float = .7) -> tuple:
    """
    Prédit la catégorie d'un texte en utilisant un classifieur ML.
    Si la confiance dans la prédiction ML est insuffisante, un fallback est effectué sur une classification par mots-clés/lexique.
    
    Arguments :
        - text : str, le texte à classer
        - vectorizer : TfidfVectorizer entraîné
        - clf : RandomForestClassifier entraîné
        - nlp : modèle spaCy pour le traitement du texte
        - french_stopwords : liste de stopwords français
        - lexiques : dictionnaire de lexiques par catégorie
        - threshold : float, seuil de confiance minimum pour accepter la prédiction ML (défaut : 0.7)
    
    Retour :
        - cat : str, catégorie prédite
        - score : float, score de confiance de la prédiction
        - mode : str, "ml" pour machine learning ou "kw" pour fallback lexique
    """
    
    test_vect = vectorizer.transform([text])
    proba = clf.predict_proba(test_vect)[0]
    cat = clf.predict(test_vect)[0]
    score = max(proba)
    
    if score > threshold:
        return cat, score, "Machine Learning"
    else:
        cat = classify(text=text, nlp=nlp, french_stopwords=french_stopwords, lexique=lexiques, min_score_threshold=0.4)
        return cat, score, "Détection par mot clé"