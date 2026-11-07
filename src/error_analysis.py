"""Analyse d'erreurs (exigence : 10 à 20 erreurs commentées).

Chaque erreur de prédiction sur le jeu de test reçoit des étiquettes automatiques :
- négation présente ; neutre confondu avec une polarité ; polarité manquée (neutre prédit) ;
- polarité inversée (positif <-> négatif) ; contraste (mais / however / cependant) ;
- sarcasme possible (motif lexical, heuristique) ; aucun mot du lexique ; texte non anglais.

Les étiquettes sont une aide au tri : elles ne remplacent pas la lecture humaine.
La sélection de 20 cas est équilibrée entre types d'erreurs, puis commentée à la main
dans docs/analyse_erreurs.md.

Usage : python -m src.error_analysis
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .lexicon import NEGATIONS, analyze_lexicon
from .preprocessing import tokenize

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "reports" / "predictions_test.csv"
ERRORS_CSV = ROOT / "reports" / "erreurs_detaillees.csv"
SELECTION_MD = ROOT / "reports" / "erreurs_selection.md"

METHODS = {"TF-IDF + LR": "ml", "Lexique maison": "lexicon", "VADER": "vader"}

CONTRAST = re.compile(r"\b(but|however|although|though|yet|mais|cependant|pourtant|toutefois|bien que)\b", re.I)
SARCASM = re.compile(
    r"(yeah right|oh great|great,? another|thanks a lot|as if|what a joke|sure,? (?:thing|right)"
    r"|wow,? (?:just|so)|nice,? (?:try|job)|bravo pour|génial,? encore|ah oui bien sûr)",
    re.I,
)

PRIORITY = [
    "polarité inversée",
    "neutre confondu avec une polarité",
    "polarité manquée (neutre prédit)",
    "sarcasme possible",
    "négation présente",
    "contraste (mais / however)",
    "aucun mot du lexique",
    "autre",
]


def error_tags(text: str, gold: str, pred: str, language: str = "en", lexicon_hits: int | None = None) -> list[str]:
    tags: list[str] = []
    if gold == pred:
        return tags
    if {gold, pred} == {"Positive", "Negative"}:
        tags.append("polarité inversée")
    if gold == "Neutral" and pred in {"Positive", "Negative"}:
        tags.append("neutre confondu avec une polarité")
    if gold in {"Positive", "Negative"} and pred == "Neutral":
        tags.append("polarité manquée (neutre prédit)")
    if SARCASM.search(text):
        tags.append("sarcasme possible")
    if any(tok in NEGATIONS for tok in tokenize(text)):
        tags.append("négation présente")
    if CONTRAST.search(text):
        tags.append("contraste (mais / however)")
    if lexicon_hits == 0:
        tags.append("aucun mot du lexique")
    if language not in {"en", "unknown"}:
        tags.append(f"texte non anglais ({language})")
    return tags or ["autre"]


def primary_tag(tags: list[str]) -> str:
    for label in PRIORITY:
        if label in tags:
            return label
    return tags[0] if tags else "autre"


def build_error_table(preds: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (avis, méthode) en erreur."""
    rows = []
    for _, row in preds.iterrows():
        text = str(row["review_text"])
        hits = len(analyze_lexicon(text).hits)
        for label, col in METHODS.items():
            pred = row[col]
            if row["gold"] == pred:
                continue
            tags = error_tags(
                text, row["gold"], pred, language=str(row.get("language", "en")),
                lexicon_hits=hits if col == "lexicon" else None,
            )
            rows.append({
                "id": row["id"],
                "methode": label,
                "niveau": row.get("niveau", ""),
                "langue": row.get("language", ""),
                "gold": row["gold"],
                "prediction": pred,
                "type_erreur": primary_tag(tags),
                "etiquettes": " ; ".join(tags),
                "texte": text,
            })
    return pd.DataFrame(rows)


QUOTAS = [
    ("neutre confondu avec une polarité", 4),
    ("négation présente", 4),
    ("polarité inversée", 4),
    ("sarcasme possible", 2),
    ("contraste (mais / however)", 3),
    ("polarité manquée (neutre prédit)", 2),
    ("autre", 1),
]


def select_examples(errors: pd.DataFrame, n: int = 20, method: str = "TF-IDF + LR") -> pd.DataFrame:
    """Sélection équilibrée des erreurs du classifieur retenu (textes courts pour la lisibilité)."""
    pool = errors[errors["methode"] == method].copy()
    pool["longueur"] = pool["texte"].str.len()
    pool = pool[pool["longueur"] <= 280].sort_values(["type_erreur", "id"])
    chosen = []
    for label, quota in QUOTAS:
        part = pool[pool["type_erreur"] == label].head(quota)
        chosen.append(part)
    selection = pd.concat(chosen) if chosen else pool.head(0)
    if len(selection) < n:
        remaining = pool[~pool["id"].isin(selection["id"])].head(n - len(selection))
        selection = pd.concat([selection, remaining])
    return selection.head(n).reset_index(drop=True)


def main() -> None:
    preds = pd.read_csv(PREDICTIONS, keep_default_na=False)
    errors = build_error_table(preds)
    ERRORS_CSV.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(ERRORS_CSV, index=False, encoding="utf-8")
    selection = select_examples(errors)
    lines = [
        "# Erreurs sélectionnées (TF-IDF + régression logistique, jeu de test)",
        "",
        f"Total d'erreurs (toutes méthodes) : {len(errors)}. Sélection : {len(selection)} cas.",
        "",
        "| id | gold | prédit | type d'erreur | texte |",
        "|---|---|---|---|---|",
    ]
    for _, r in selection.iterrows():
        text = str(r["texte"]).replace("|", "/")
        lines.append(f"| {r['id']} | {r['gold']} | {r['prediction']} | {r['type_erreur']} | {text} |")
    SELECTION_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Erreurs totales :", len(errors))
    print(errors.groupby(["methode", "type_erreur"]).size().to_string())
    print("Sélection :", SELECTION_MD)


if __name__ == "__main__":
    main()
