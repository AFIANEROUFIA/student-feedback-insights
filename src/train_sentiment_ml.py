"""Entraînement et comparaison : lexique maison, VADER, TF-IDF + régression logistique.

Données : data/raw/combined_data_6.5k_anonymise.csv (EduRABSA, anonymisé) si présent,
sinon l'échantillon synthétique avis_etudiants_sample.csv.

Protocole :
- split stratifié 75 % entraînement / 25 % test (random_state=42) ;
- le classifieur n'est entraîné QUE sur l'entraînement ; lexique et VADER ne sont pas entraînés ;
- validation croisée 5 plans (macro-F1) sur le classifieur, pour mesurer la stabilité ;
- les métriques sont calculées sur le jeu de test uniquement.

Sorties : models/sentiment_tfidf_logreg.joblib, reports/metrics.json,
reports/predictions_test.csv (utilisé par l'analyse d'erreurs).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

from .data_io import load_reviews_table, prepare_frame
from .evaluate import LABELS, classification_metrics, save_metrics
from .lexicon import analyze_lexicon, analyze_vader

ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = [
    ROOT / "data" / "raw" / "combined_data_6.5k_anonymise.csv",
    ROOT / "data" / "raw" / "avis_etudiants_sample.csv",
]
MODEL_PATH = ROOT / "models" / "sentiment_tfidf_logreg.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
PREDICTIONS_PATH = ROOT / "reports" / "predictions_test.csv"


def pick_dataset(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    for candidate in DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Aucun jeu de données trouvé dans data/raw/.")


def build_classifier() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


def _per_language(gold, preds: dict[str, list], languages) -> dict:
    out = {}
    languages = pd.Series(languages).reset_index(drop=True)
    gold = pd.Series(gold).reset_index(drop=True)
    for lang, n in languages.value_counts().items():
        if n < 20:  # trop peu d'exemples pour une F1 fiable
            continue
        mask = (languages == lang).to_numpy()
        out[str(lang)] = {
            "n": int(n),
            **{
                name: round(float(f1_score(gold[mask], np.array(pred)[mask], average="macro", labels=LABELS, zero_division=0)), 3)
                for name, pred in preds.items()
            },
        }
    return out


def train(data_path: Path | None = None, test_size: float = 0.25, seed: int = 42) -> dict:
    path = pick_dataset(data_path)
    raw = load_reviews_table(path)
    df, redactions = prepare_frame(raw, anonymize=True)
    df = df[df["overall_sentiment"].notna()].reset_index(drop=True)
    if "language" not in df.columns:
        df["language"] = "unknown"
    df["language"] = df["language"].fillna("unknown")

    y = df["overall_sentiment"].to_numpy()
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)

    clf = build_classifier().fit(df["text_clean"].iloc[tr], y[tr])
    ml_pred = clf.predict(df["text_clean"].iloc[te])
    test_texts = df["review_text"].iloc[te].tolist()
    lex_results = [analyze_lexicon(t) for t in test_texts]
    lex_pred = [r.label for r in lex_results]
    vader_results = [analyze_vader(t) for t in test_texts]
    vader_pred = [r[0] for r in vader_results]

    ml_metrics = classification_metrics(y[te], ml_pred)
    lex_metrics = classification_metrics(y[te], lex_pred)
    vader_metrics = classification_metrics(y[te], vader_pred)

    # Stabilité du classifieur : 5 plans stratifiés sur tout le jeu.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = []
    for fold_tr, fold_te in skf.split(df, y):
        fold_clf = build_classifier().fit(df["text_clean"].iloc[fold_tr], y[fold_tr])
        folds.append(float(f1_score(
            y[fold_te], fold_clf.predict(df["text_clean"].iloc[fold_te]),
            average="macro", labels=LABELS, zero_division=0,
        )))

    bundle = {
        "vectorizer": clf.named_steps["tfidf"],
        "model": clf.named_steps["clf"],
        "labels": list(clf.named_steps["clf"].classes_),
        "train_ids": df["id"].iloc[tr].tolist(),
        "test_ids": df["id"].iloc[te].tolist(),
        "dataset": path.name,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    proba = clf.predict_proba(df["text_clean"].iloc[te])
    predictions = pd.DataFrame({
        "id": df["id"].iloc[te].to_numpy(),
        "niveau": df["niveau"].iloc[te].to_numpy(),
        "language": df["language"].iloc[te].to_numpy(),
        "review_text": test_texts,
        "gold": y[te],
        "lexicon": lex_pred,
        "lexicon_score": [r.score for r in lex_results],
        "vader": vader_pred,
        "vader_compound": [r[1] for r in vader_results],
        "ml": ml_pred,
        "ml_confidence": proba.max(axis=1).round(3),
    })
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8")

    payload = {
        "ml": ml_metrics,
        "lexicon": lex_metrics,
        "vader": vader_metrics,
        "cv_ml_macro_f1": {
            "folds": [round(v, 3) for v in folds],
            "mean": round(float(np.mean(folds)), 3),
            "std": round(float(np.std(folds)), 3),
        },
        "par_langue_macro_f1_test": _per_language(
            y[te], {"lexicon": lex_pred, "vader": vader_pred, "ml": ml_pred}, df["language"].iloc[te].to_numpy()
        ),
        "dataset": path.name,
        "n_total": int(len(df)),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "distribution_test": pd.Series(y[te]).value_counts().to_dict(),
        "langues_test": df["language"].iloc[te].value_counts().head(10).to_dict(),
        "anonymisation_remplacements": int(redactions),
        "split": f"{int((1 - test_size) * 100)} % entraînement / {int(test_size * 100)} % test, stratifié, random_state={seed}",
        "note": (
            "Scores calculés sur le jeu de test ; le classifieur n'a pas vu ces avis. "
            "Le lexique et VADER ne sont pas entraînés. Aucun score inventé."
        ),
    }
    save_metrics(payload, METRICS_PATH)
    return payload


if __name__ == "__main__":
    result = train()
    print("Jeu :", result["dataset"], "| n =", result["n_total"], "| test =", result["n_test"])
    print("Macro-F1 TF-IDF + LR :", round(result["ml"]["macro_f1"], 3))
    print("Macro-F1 lexique     :", round(result["lexicon"]["macro_f1"], 3))
    print("Macro-F1 VADER       :", round(result["vader"]["macro_f1"], 3))
    cv = result["cv_ml_macro_f1"]
    print(f"CV 5 plans (TF-IDF + LR) : {cv['mean']} ± {cv['std']}")
    print("Modèle :", MODEL_PATH)
