"""Analyse de sentiment par lexique (exigence du professeur : comparer un lexique à une classification).

Deux lexiques sont comparés à la classification TF-IDF :
- lexique maison bilingue FR/EN avec négations (analyze_lexicon) ;
- VADER (analyze_vader), lexique anglais de référence. VADER est conçu pour l'anglais :
  sur le français il renvoie surtout « Neutral » (voir docs/limites.md).
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from functools import lru_cache

from .preprocessing import tokenize

POSITIVE = {
    "excellent", "excellente", "super", "genial", "génial", "bien", "bonne", "bon",
    "clair", "claire", "utile", "utiles", "disponible", "disponibles", "interessant",
    "intéressant", "interessante", "intéressante", "enrichissant", "motivant",
    "pedagogique", "pédagogique", "organise", "organisé", "organisee", "organisée",
    "accessible", "passionnant", "passionnante", "bravo", "parfait", "parfaite",
    "recommandé", "recommande", "recommande", "agréable", "agreable", "efficace",
    "rigoureux", "rigoureuse", "constructif", "constructive", "aidant", "aidante",
    "helpful", "excellent", "great", "good", "clear", "useful", "available",
    "interesting", "engaging", "organized", "organised", "accessible", "amazing",
    "wonderful", "supportive", "friendly", "kind", "fair", "well", "love", "liked",
    "like", "best", "awesome", "fantastic", "brilliant", "positive", "easy",
    "understandable", "responsive", "patient", "encouraging", "structured",
    "qualité", "qualite", "merci", "top", "satisfait", "satisfaite", "satisfying",
    "recommend", "recommended", "enjoy", "enjoyed", "impressed", "clarity",
    "sympa", "sympathique", "gentil", "gentille", "dynamique", "interactif", "compétent",
    "thanks", "thank", "nice", "competent", "professionnel", "professionnelle", "knowledgeable", "enjoyable",
}

NEGATIVE = {
    "nul", "nulle", "mauvais", "mauvaise", "confus", "confuse", "incomprehensible",
    "incompréhensible", "absent", "absente", "injuste", "stressant", "stressante",
    "inutile", "inutiles", "chaotique", "ennuyeux", "ennuyeuse", "lourd", "lourde",
    "trop", "manque", "manquent", "probleme", "problème", "problemes", "problèmes",
    "retard", "retards", "desorganise", "désorganisé", "desorganisee", "désorganisée",
    "rude", "impoli", "impolie", "froid", "froide", "incompetent", "incompétent",
    "incompétente", "incompetente", "horrible", "catastrophe", "decevant", "décevant",
    "decevante", "décevante", "mediocre", "médiocre", "difficile", "difficult",
    "confusing", "unclear", "useless", "unfair", "stressful", "boring", "chaotic",
    "late", "disorganized", "disorganised", "rude", "unhelpful", "absent", "poor",
    "bad", "worst", "hate", "awful", "terrible", "messy", "overwhelming", "unclear",
    "inaccessible", "slow", "ignored", "ignore", "complaint", "complain", "fail",
    "failed", "failure", "hard", "impossible", "vague", "inconsistent", "bias",
    "biased", "unprepared", "monotone", "monotonous", "disappointing", "disappointed",
    "lack", "lacking", "problem", "problems", "issue", "issues", "complaint",
}

NEGATIONS = {
    "not", "no", "never", "neither", "nor", "without", "barely", "hardly",
    "pas", "non", "jamais", "rien", "aucun", "aucune", "ni", "sans", "guere", "guère",
    "plus", "ne",
    # élisions : « n'est pas » est déjà couvert par « pas » ; ces formes seules négent aussi
    "n'est", "n'a", "n'ont", "n'y", "n'était", "n'etait", "n'avait", "n'ai",
}

INTENSIFIERS = {
    "very", "really", "extremely", "so", "too", "highly", "super",
    "tres", "très", "vraiment", "extremement", "extrêmement", "trop", "hyper",
}

EXCESS_WORDS = {"trop"}  # « too » est ambigu en anglais (« me too ») : « too hard » est déjà couvert par « hard »

STOP_POLARITY = {"the", "a", "an", "le", "la", "les", "un", "une", "des", "de", "du"}


@dataclass
class LexiconResult:
    label: str
    score: float
    confidence: float
    hits: list[dict]


_MAL_PATTERN = re.compile(r"\bmal (organis|expliqu|pr[ée]par|structur|dirig|gér|ger)\w*", re.I)


def analyze_lexicon(text: str) -> LexiconResult:
    # « mal organisé » = désorganisé : « mal » inverse le mot qui suit (sinon « organisé » compte positif)
    text = _MAL_PATTERN.sub("désorganisé", str(text))
    tokens = tokenize(text)
    score = 0.0
    hits: list[dict] = []
    negate = False
    intensity = 1.0

    for idx, tok in enumerate(tokens):
        if tok in EXCESS_WORDS:
            nxt = tokens[idx + 1] if idx + 1 < len(tokens) else ""
            if nxt and nxt not in POSITIVE and nxt not in NEGATIVE and nxt not in STOP_POLARITY:
                # « trop long », « too fast » : l'excès est lui-même un reproche
                pol = -1.0 * (-1.0 if negate else 1.0)
                score += pol
                hits.append({"token": tok, "polarity": pol, "negated": negate})
                negate = False
                intensity = 1.0
                continue
        if tok in NEGATIONS:
            negate = True
            continue
        if tok in INTENSIFIERS:
            intensity = 1.6
            continue
        polarity = 0.0
        if tok in POSITIVE:
            polarity = 1.0
        elif tok in NEGATIVE:
            polarity = -1.0
        if polarity != 0.0:
            if negate:
                polarity *= -1.0
            polarity *= intensity
            score += polarity
            hits.append({"token": tok, "polarity": polarity, "negated": negate})
            negate = False
            intensity = 1.0
        elif tok not in STOP_POLARITY:
            negate = False
            intensity = 1.0

    if score > 0.4:
        label = "Positive"
    elif score < -0.4:
        label = "Negative"
    else:
        label = "Neutral"

    magnitude = min(abs(score) / 3.0, 1.0)
    if label == "Neutral":
        confidence = 0.45 + 0.15 * (1.0 - magnitude)
    else:
        confidence = 0.55 + 0.4 * magnitude
    return LexiconResult(label=label, score=round(score, 3), confidence=round(confidence, 3), hits=hits)


VADER_POS = 0.05
VADER_NEG = -0.05


@lru_cache(maxsize=1)
def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def analyze_vader(text: str) -> tuple[str, float]:
    """Baseline VADER : retourne (label, compound). Seuils standard de VADER : +/-0.05."""
    # VADER ne reconnaît pas « excellent… » (caractère « … » collé au mot) : on le ramène à « ... »
    compound = _vader().polarity_scores(str(text).replace("\u2026", "..."))["compound"]
    if compound >= VADER_POS:
        label = "Positive"
    elif compound <= VADER_NEG:
        label = "Negative"
    else:
        label = "Neutral"
    return label, round(float(compound), 3)
