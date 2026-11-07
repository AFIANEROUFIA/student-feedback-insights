"""Anonymisation des avis : e-mails, téléphones, pseudos, URLs et noms propres.

Approche par règles (regex), sans dépendance lourde :
- e-mails, numéros de téléphone (>= 9 chiffres), pseudos (@pseudo, u/pseudo), URLs -> jeton ;
- noms propres précédés d'un titre (Dr, Prof, Mr, M., Mme...) -> le titre est gardé, le nom devient [NOM] ;
- mots capitalisés en milieu de phrase (hors liste de mots à garder : jours, outils, I...) -> [NOM].

Limite connue : heuristique. Elle rate des noms en minuscules ou en MAJUSCULES et peut
retirer des mots capitalisés qui ne sont pas des noms. Une option spaCy (entités PERSON)
est prévue via `use_spacy=True`, non testée dans cet environnement.
Voir docs/ethique_anonymisation.md.
"""
from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_NAME = "[NOM]"
TOKEN_EMAIL = "[EMAIL]"
TOKEN_TEL = "[TEL]"
TOKEN_PSEUDO = "[PSEUDO]"
TOKEN_URL = "[URL]"

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_PHONE = re.compile(r"(?<!\w)\+?\d[\d .()\-]{7,}\d(?!\w)")
_HANDLE = re.compile(r"(?<![\w/])@\w+|(?<![\w/])u/\w+", re.I)
_TITLE_NAME = re.compile(
    r"(?<!\w)(Dr|Prof|Professor|Professeur|Pr|Mr|Mrs|Ms|Miss|Mme|Mlle|Madame|Monsieur|Sir|M)(\.?)(\s+)([A-Z][\w'’-]*)"
)
_WORD = re.compile(r"[^\W\d_][\w'’-]*")
_SENTENCE_START = {"", ".", "!", "?", ":", ";", '"', "(", "["}

# Mots capitalisés à ne PAS considérer comme noms propres (minuscules pour la comparaison).
_COMMON_ADJ_SUBJECTS = {
    "great", "good", "best", "bad", "worst", "fun", "helpful", "love", "nice", "hard", "easy",
    "new", "high", "low", "first", "last", "excellent", "amazing", "awesome", "interesting",
    "boring", "clear", "useful", "fair", "tough", "kind", "smart", "bright", "brilliant",
    "terrible", "horrible", "awful", "stressful", "interested", "funny", "dry", "heavy",
    "law", "winter", "fall", "spring", "summer", "studies", "engineering", "spanish", "russian",
    "german", "chinese", "japanese", "calculus", "algebra", "linear", "design", "management",
    "music", "business", "software", "writing", "data", "theory", "lab", "science", "sciences",
    "history", "art", "arts", "geography", "statistics", "accounting", "finance", "marketing",
    "nursing", "medicine", "biology", "philosophy", "sociology", "literature", "computer",
    "computing", "programming", "networks", "security", "tech", "technology", "school",
    "faculty", "department", "class", "classes", "lecture", "lectures", "tutorial", "tutorials",
    "exam", "exams", "midterm", "final", "quiz", "project", "homework", "assignment", "week",
    "semester", "term", "year", "monday", "week", "online", "office", "hours",
    "teacher", "teachers", "instructor", "lecturer", "tutor", "staff", "student", "students",
    "ta", "tas", "prof", "profs", "faculty", "university", "uni", "college", "campus",
}
KEEP_CAPITALISED = set(_COMMON_ADJ_SUBJECTS) | set(ENGLISH_STOP_WORDS) | {
    "dr", "prof", "professeur", "pr", "mr", "mrs", "ms", "miss", "mme", "mlle", "madame", "monsieur", "sir", "m",
    "i'm", "i'll", "i've", "i'd",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "june", "july", "august", "september",
    "october", "november", "december",
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
    "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre",
    "octobre", "novembre", "décembre",
    "english", "french", "anglais", "français", "francais",
    "moodle", "canvas", "blackboard", "piazza", "quercus", "excel", "word", "powerpoint",
    "python", "java", "matlab", "microsoft", "google", "zoom", "teams", "office", "github",
    "professor", "prof", "dr", "university", "université", "college", "campus", "course",
    "canada", "france", "uk", "usa", "europe", "england", "waterloo", "exeter",
    "algeria", "algérie", "biskra",
    "math", "chemistry", "physics", "psychology", "economics",
}


def _is_title_case(word: str) -> bool:
    stripped = word.rstrip("'’-")
    return len(stripped) >= 2 and stripped[0].isupper() and any(c.islower() for c in stripped[1:])


def _redact_capitalised(text: str) -> tuple[str, int]:
    """Remplace par [NOM] les mots en Title Case en milieu de phrase.

    Un mot en début de phrase n'est retenu que s'il est suivi immédiatement d'un
    autre mot en Title Case (« Mason Smith is rude » -> « [NOM] is rude »).
    """
    matches = list(_WORD.finditer(text))
    long_words = [m.group(0) for m in matches if len(m.group(0)) >= 3]
    title_ratio = (sum(_is_title_case(w) for w in long_words) / len(long_words)) if long_words else 0.0
    if title_ratio > 0.6:
        # Avis écrit en « Title Case » : la majuscule ne signale plus un nom propre.
        return text, 0
    flags = []  # True si le mot doit être anonymisé
    for i, m in enumerate(matches):
        word = m.group(0)
        if not _is_title_case(word) or word.rstrip("'’-").lower() in KEEP_CAPITALISED:
            flags.append(False)
            continue
        prefix = text[: m.start()].rstrip()
        sentence_start = (prefix[-1] if prefix else "") in _SENTENCE_START
        if not sentence_start:
            flags.append(True)
            continue
        nxt = matches[i + 1] if i + 1 < len(matches) else None
        nxt_ok = (
            nxt is not None
            and _is_title_case(nxt.group(0))
            and nxt.group(0).rstrip("'’-").lower() not in KEEP_CAPITALISED
            and text[m.end(): nxt.start()].strip() == ""
        )
        flags.append(bool(nxt_ok))

    out: list[str] = []
    count = 0
    last = 0
    in_name = False
    for m, flag in zip(matches, flags):
        gap = text[last: m.start()]
        if flag:
            if in_name and gap.strip() == "":
                pass  # même nom : fusion dans le même [NOM]
            else:
                out.append(gap)
                out.append(TOKEN_NAME)
                count += 1
            in_name = True
        else:
            out.append(gap)
            out.append(m.group(0))
            in_name = False
        last = m.end()
    out.append(text[last:])
    return "".join(out), count


def anonymize_text(text: str, use_spacy: bool = False) -> tuple[str, int]:
    """Retourne (texte anonymisé, nombre de remplacements)."""
    if text is None:
        return "", 0
    original = str(text)
    out = original
    n = 0

    out, k = _EMAIL.subn(TOKEN_EMAIL, out)
    n += k
    out, k = _URL.subn(TOKEN_URL, out)
    n += k
    out, k = _HANDLE.subn(TOKEN_PSEUDO, out)
    n += k

    phones = [0]

    def _phone(m: re.Match) -> str:
        if sum(c.isdigit() for c in m.group(0)) >= 9:
            phones[0] += 1
            return TOKEN_TEL
        return m.group(0)

    out = _PHONE.sub(_phone, out)
    n += phones[0]

    def _title(m: re.Match) -> str:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}{TOKEN_NAME}"

    out, k = _TITLE_NAME.subn(_title, out)
    n += k

    out, k = _redact_capitalised(out)
    n += k

    if use_spacy:
        out, k = _redact_spacy(out)
        n += k
    return out, n


def _redact_spacy(text: str) -> tuple[str, int]:
    """Option : entités PERSON via spaCy si installé (non testé dans cet environnement)."""
    try:
        import spacy  # type: ignore
    except ImportError:
        return text, 0
    nlp = None
    for model in ("en_core_web_sm", "fr_core_news_sm"):
        try:
            nlp = spacy.load(model)
            break
        except OSError:
            continue
    if nlp is None:
        return text, 0
    doc = nlp(text)
    out = text
    count = 0
    for ent in reversed(doc.ents):
        if ent.label_ == "PERSON":
            out = out[: ent.start_char] + TOKEN_NAME + out[ent.end_char:]
            count += 1
    return out, count


def anonymize_series(values: Iterable[str], use_spacy: bool = False) -> tuple[list[str], int]:
    texts: list[str] = []
    total = 0
    for value in values:
        clean, k = anonymize_text(value, use_spacy=use_spacy)
        texts.append(clean)
        total += k
    return texts, total


def anonymize_frame(df: pd.DataFrame, col: str = "review_text", use_spacy: bool = False) -> tuple[pd.DataFrame, int]:
    """Anonymise une colonne de texte ; retourne le DataFrame et le nombre total de remplacements."""
    out = df.copy()
    texts, total = anonymize_series(out[col].fillna("").astype(str), use_spacy=use_spacy)
    out[col] = texts
    return out, total
