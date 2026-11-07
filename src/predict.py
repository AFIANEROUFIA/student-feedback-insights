from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .absa import extract_aspects
from .lexicon import analyze_lexicon, analyze_vader
from .preprocessing import clean_text
from .themes import detect_themes

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_FILE = "sentiment_tfidf_logreg.joblib"


def load_ml_bundle(path: Path | None = None):
    path = path or (MODELS_DIR / MODEL_FILE)
    if not path.exists():
        return None
    bundle = joblib.load(path)
    model = bundle.get("model")
    if isinstance(model, LogisticRegression) and not hasattr(model, "multi_class"):
        # scikit-learn 1.9 removed this serialized attribute, but older
        # versions still read it in predict_proba.
        model.multi_class = "auto"
    return bundle


def predict_ml(texts: list[str], bundle=None) -> pd.DataFrame:
    """Prédiction TF-IDF + régression logistique. Le texte est nettoyé comme à l'entraînement."""
    bundle = bundle or load_ml_bundle()
    if bundle is None:
        raise FileNotFoundError("Modèle ML introuvable. Exécutez : python -m src.train_sentiment_ml")
    vectorizer = bundle["vectorizer"]
    model = bundle["model"]
    cleaned = [clean_text(t, lowercase=True) for t in texts]
    X = vectorizer.transform(cleaned)
    labels = model.predict(X)
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    out = pd.DataFrame({
        "review_text": list(texts),
        "ml_sentiment": labels,
        "ml_confidence": proba.max(axis=1).round(3),
    })
    for j, cls in enumerate(classes):
        out[f"proba_{cls}"] = proba[:, j].round(3)
    return out


def analyze_comments(texts: list[str], bundle=None, with_absa: bool = False) -> pd.DataFrame:
    """Analyse complète : lexique maison, VADER, classification TF-IDF, thèmes (et ABSA à la demande)."""
    bundle = load_ml_bundle() if bundle is None else bundle
    ml_df = predict_ml(texts, bundle) if bundle is not None else None
    rows = []
    for i, text in enumerate(texts):
        lex = analyze_lexicon(text)
        vader_label, vader_compound = analyze_vader(text)
        row = {
            "review_text": text,
            "themes": ", ".join(detect_themes(text)),
            "lexicon_sentiment": lex.label,
            "lexicon_score": lex.score,
            "lexicon_confidence": lex.confidence,
            "vader_sentiment": vader_label,
            "vader_compound": vader_compound,
        }
        if ml_df is not None:
            row["ml_sentiment"] = ml_df.loc[i, "ml_sentiment"]
            row["ml_confidence"] = ml_df.loc[i, "ml_confidence"]
        if with_absa:
            aspects = extract_aspects(text)
            row["n_aspects"] = len(aspects)
            row["absa"] = aspects
        rows.append(row)
    return pd.DataFrame(rows)
