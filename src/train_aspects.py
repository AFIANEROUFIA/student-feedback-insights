"""Entraînement léger des catégories d'aspects à partir d'annotations JSON EduRABSA."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from .data_io import load_edurabsa_json
from .preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "raw" / "avis_absa_sample.json"  # échantillon synthétique (léger). Option : le JSON EduRABSA anonymisé.
OUT = ROOT / "models" / "aspect_category.joblib"


def train(path: Path = SAMPLE) -> Path:
    df = load_edurabsa_json(path)
    rows = []
    for _, rec in df.iterrows():
        for quad in rec.get("quadruplets") or []:
            if not isinstance(quad, dict):
                continue
            cat = quad.get("category")
            if not cat:
                continue
            rows.append({"text": clean_text(rec["review_text"]), "category": cat})
    data = pd.DataFrame(rows)
    if data.empty:
        raise SystemExit("Aucune catégorie annotée dans le JSON.")
    encoder = LabelEncoder()
    y = encoder.fit_transform(data["category"])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = vectorizer.fit_transform(data["text"])
    clf = LogisticRegression(max_iter=300)
    clf.fit(X, y)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "model": clf, "encoder": encoder}, OUT)
    return OUT


if __name__ == "__main__":
    print("Modèle d'aspects :", train())
