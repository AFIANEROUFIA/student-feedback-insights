from __future__ import annotations

import math
import re
import unicodedata

# Négations reconnues par le lexique (voir lexicon.py pour le détail).
NEGATION_KEEP = {
    "not", "no", "never", "neither", "nor", "without",
    "pas", "non", "jamais", "rien", "aucun", "aucune", "ni", "sans", "plus",
}

_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL = re.compile(r"\S+@\S+")
_MULTI_PUNCT = re.compile(r"([!?.,]){2,}")
_APOS_TRANS = str.maketrans({"\u2019": "'", "\u2018": "'", "`": "'", "\u00b4": "'"})
_CONTRACTIONS = {
    "can't": "can not",
    "cannot": "can not",
    "won't": "will not",
    "shan't": "shall not",
    "ain't": "is not",
}
_NT = re.compile(r"\b(\w+)n't\b", re.I)

_LABEL_MAP = {
    "pos": "Positive",
    "positive": "Positive",
    "positif": "Positive",
    "positve": "Positive",
    "neg": "Negative",
    "negative": "Negative",
    "négatif": "Negative",
    "negatif": "Negative",
    "neu": "Neutral",
    "neutral": "Neutral",
    "neutre": "Neutral",
    "mixed": "Neutral",
    # EduRABSA : « Conf/Neu » = confus / neutre (classe difficile).
    "conf": "Neutral",
    "conf/neu": "Neutral",
    "conf/neutral": "Neutral",
}
_MISSING = {"", "[na]", "na", "nan", "none", "null"}


def normalize_label(value) -> str | None:
    """Ramène les labels hétérogènes (EduRABSA, sample, FR/EN) à Positive / Negative / Neutral.

    Retourne None pour les valeurs manquantes (ex. « [NA] » du CSV EduRABSA).
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip().lower()
    if text in _MISSING:
        return None
    return _LABEL_MAP.get(text, str(value).strip().title())


def normalize_text(text) -> str:
    """Nettoyage léger qui conserve la casse (affichage, anonymisation)."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = text.translate(_APOS_TRANS)
    # Les exports CSV/JSON échappent parfois les apostrophes : doesn\'t
    text = text.replace("\\'", "'").replace('\\"', '"')
    text = text.replace("\u00a0", " ").replace("\u2026", "...")
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _MULTI_PUNCT.sub(r"\1", text)
    text = _WS.sub(" ", text).strip()
    return text


def expand_contractions(text: str) -> str:
    """« isn't » -> « is not », « don't » -> « do not » : indispensable pour la négation."""
    for short, long in _CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(short)}\b", long, text, flags=re.I)
    return _NT.sub(r"\1 not", text)


def clean_text(text: str, lowercase: bool = True) -> str:
    if text is None:
        return ""
    out = expand_contractions(normalize_text(text))
    return out.lower() if lowercase else out


def tokenize(text: str) -> list[str]:
    cleaned = clean_text(text, lowercase=True)
    return re.findall(r"[a-zàâäéèêëïîôùûüçœæ0-9']+", cleaned, flags=re.I)


def split_clauses(text: str) -> list[str]:
    parts = re.split(
        r"[.!?;]|,(?=\s)| mais | but | cependant | toutefois | par contre | however | although ",
        text,
        flags=re.I,
    )
    return [p.strip() for p in parts if p and p.strip()]
