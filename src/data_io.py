"""Chargement des avis : CSV (EduRABSA combined_data_6.5k, échantillon) et JSON EduRABSA (quadruplets)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .anonymize import anonymize_frame
from .preprocessing import clean_text, normalize_label, normalize_text
from .themes import detect_themes

# Niveau de l'avis : enseignant / cours / université (demande du professeur, filtre du dashboard).
NIVEAU_ENSEIGNANT = "Enseignant"
NIVEAU_COURS = "Cours"
NIVEAU_UNIVERSITE = "Université"
NIVEAU_AUTRE = "Non précisé"
NIVEAUX = [NIVEAU_ENSEIGNANT, NIVEAU_COURS, NIVEAU_UNIVERSITE, NIVEAU_AUTRE]

# Valeurs de la colonne « type » d'EduRABSA et de nos échantillons.
TYPE_TO_NIVEAU = {
    "teacher review": NIVEAU_ENSEIGNANT,
    "enseignant": NIVEAU_ENSEIGNANT,
    "course review": NIVEAU_COURS,
    "cours": NIVEAU_COURS,
    "university review": NIVEAU_UNIVERSITE,
    "université": NIVEAU_UNIVERSITE,
    "universite": NIVEAU_UNIVERSITE,
}


def infer_niveau(text: str, declared_type=None) -> str:
    """Niveau = type déclaré si connu, sinon déduit des thèmes détectés dans le texte."""
    if declared_type is not None and not (isinstance(declared_type, float) and pd.isna(declared_type)):
        mapped = TYPE_TO_NIVEAU.get(str(declared_type).strip().lower())
        if mapped:
            return mapped
    themes = set(detect_themes(text))
    if "Enseignant / attitude" in themes:
        return NIVEAU_ENSEIGNANT
    if themes & {"Cours / contenu", "Examens / évaluation", "Difficulté", "Charge de travail", "Ressources pédagogiques"}:
        return NIVEAU_COURS
    if themes & {"Infrastructures", "Organisation"}:
        return NIVEAU_UNIVERSITE
    return NIVEAU_AUTRE


def _quads(record: dict) -> list:
    return record.get("Quadruplets") or record.get("quadruplets") or []


def load_edurabsa_json(path: str | Path) -> pd.DataFrame:
    """JSON EduRABSA : dict {id: avis} ou liste d'avis, avec « Quadruplets » (aspect, opinion, category, sentiment)."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        if "data" in payload or "reviews" in payload:
            records = payload.get("data") or payload.get("reviews")
        else:
            records = list(payload.values())
    else:
        records = payload
    rows = []
    for item in records:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("review_text") or ""
        quads = [q for q in _quads(item) if isinstance(q, dict)]
        sentiments = [normalize_label(q.get("sentiment")) for q in quads]
        overall = normalize_label(item.get("overall_sentiment"))
        if overall is None and sentiments:
            pos = sentiments.count("Positive")
            neg = sentiments.count("Negative")
            overall = "Negative" if neg > pos else "Positive" if pos > neg else "Neutral"
        rows.append({
            "id": str(item.get("id") or item.get("original_id") or len(rows)),
            "type": item.get("type"),
            "source": item.get("source", path.name),
            "review_text": text,
            "overall_sentiment": overall,
            "quadruplets": quads,
            "n_aspects": len(quads),
        })
    return pd.DataFrame(rows)


_RENAME = {
    "review_text": "review_text", "text": "review_text", "comment": "review_text",
    "commentaire": "review_text", "avis": "review_text",
    "overall_sentiment": "overall_sentiment", "sentiment": "overall_sentiment",
    "label": "overall_sentiment", "rating_cat": "overall_sentiment",
    "lang": "language", "language": "language", "langue": "language",
}


def load_reviews_table(path: str | Path) -> pd.DataFrame:
    """CSV ou JSON, colonnes normalisées : review_text, overall_sentiment, language, type, id."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_edurabsa_json(path)
    df = pd.read_csv(path, keep_default_na=True)
    rename = {}
    for col in df.columns:
        key = col.strip().lower()
        if key in _RENAME and _RENAME[key] not in df.columns:
            rename[col] = _RENAME[key]
    df = df.rename(columns=rename)
    if "review_text" not in df.columns:
        raise ValueError("Colonne de texte introuvable (review_text / text / commentaire).")
    if "overall_sentiment" in df.columns:
        df["overall_sentiment"] = df["overall_sentiment"].map(normalize_label)
    return df


def prepare_frame(df: pd.DataFrame, anonymize: bool = True) -> tuple[pd.DataFrame, int]:
    """Nettoie, anonymise, dédoublonne et ajoute niveau / thèmes.

    Retourne (DataFrame prêt à l'analyse, nombre de remplacements d'anonymisation).
    """
    out = df.copy()
    if "id" not in out.columns:
        out["id"] = [f"row_{i}" for i in range(len(out))]
    out["id"] = out["id"].astype(str)
    out["review_text"] = out["review_text"].map(normalize_text)
    out = out[out["review_text"].str.strip().ne("")]
    out = out.drop_duplicates(subset=["review_text"])
    redactions = 0
    if anonymize:
        out, redactions = anonymize_frame(out, "review_text")
    out["text_clean"] = out["review_text"].map(lambda t: clean_text(t, lowercase=True))
    declared = out["type"] if "type" in out.columns else pd.Series([None] * len(out), index=out.index)
    out["niveau"] = [infer_niveau(t, d) for t, d in zip(out["review_text"], declared)]
    out["themes"] = out["review_text"].map(lambda t: ", ".join(detect_themes(t)))
    if "overall_sentiment" in out.columns:
        out["overall_sentiment"] = out["overall_sentiment"].map(normalize_label)
    return out.reset_index(drop=True), redactions
