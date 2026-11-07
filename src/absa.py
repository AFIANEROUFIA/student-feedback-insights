"""ABSA léger : aspects, opinions, catégories et polarité par clause."""
from __future__ import annotations

from .lexicon import analyze_lexicon
from .preprocessing import split_clauses
from .themes import THEME_LEXICON, detect_themes


def extract_aspects(text: str) -> list[dict]:
    results = []
    clauses = split_clauses(text) or [text]
    for clause in clauses:
        themes = [t for t in detect_themes(clause) if t != "Autre / général"]
        if not themes:
            themes = detect_themes(text)
        lex = analyze_lexicon(clause)
        opinion_tokens = [h["token"] for h in lex.hits]
        for theme in themes:
            keywords = THEME_LEXICON.get(theme, [])
            aspect = next((k for k in keywords if k.lower() in clause.lower()), theme.split("/")[0].strip())
            results.append({
                "aspect": aspect,
                "category": theme,
                "opinion": ", ".join(opinion_tokens) if opinion_tokens else clause[:80],
                "sentiment": lex.label,
                "confidence": lex.confidence,
                "clause": clause,
            })
    # dédoublonner (aspect, category, sentiment)
    seen = set()
    unique = []
    for row in results:
        key = (row["aspect"], row["category"], row["sentiment"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique
