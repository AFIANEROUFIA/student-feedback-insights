"""Verdict d'un commentaire : Positif, Négatif, Neutre ou Mixte, avec explication.

Règles :
- Mixte : le lexique trouve des mots d'opinion positifs ET négatifs d'un poids comparable
  (le plus faible représente au moins la moitié du plus fort).
- Sinon : le label du classifieur TF-IDF + régression logistique (le meilleur des trois
  mesuré sur le jeu de test) ; à défaut, le lexique.

Points de vigilance : signaux heuristiques qui correspondent aux types d'erreurs mesurés
dans l'analyse d'erreurs (neutre confondu, polarité inversée, polarité manquée). Ils
indiquent un risque, ils ne prouvent pas une erreur.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from langdetect import DetectorFactory, detect

from .lexicon import analyze_lexicon, analyze_vader
from .predict import predict_ml
from .themes import detect_themes

DetectorFactory.seed = 0

LABEL_FR = {"Positive": "Positif", "Negative": "Négatif", "Neutral": "Neutre", "Mixed": "Mixte"}
MIXED_RATIO = 0.5
CONSENSUS_CONF = 0.6    # seuil quand le lexique et VADER donnent la même polarité
OVERRIDE_CONF = 0.5     # sous ce seuil, le lexique peut corriger une polarité hésitante du classifieur
SHORT_TEXT_WORDS = 12   # les règles neutres ne visent que les messages courts (question, énoncé, « sans plus »)

# Expressions explicitement mitigées : l'avis est « ni bon ni mauvais » -> Neutre.
_NEUTRAL_CUES = re.compile(
    r"\b(sans plus|ni bon ni mauvais|ni mauvais ni bon|nothing special|so[- ]so|"
    r"neither good nor bad|not bad,? not great|moyen(?:ne)?|passable|average|okay|ok)\b", re.I)
# Texte factuel (horaire, jour, salle, question) : aucune opinion exprimée.
_FACTUAL = re.compile(
    r"(\d\s?(h|am|pm)\b|\bat \d|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|room|salle|amphi|office hours?)\b)", re.I)
_QUESTION_START = re.compile(
    r"^\s*(where|when|what|who|why|how|which|"
    r"o[uù]|quand|quel(?:le|s|les)?|qui|pourquoi|comment|est-ce|combien)\b", re.I)


def _no_opinion(text: str, n_hits: int, vader_label: str) -> bool:
    """Vrai si le texte n'exprime aucune opinion : question ou énoncé factuel sans mot d'opinion.

    Le classifieur n'a que 3 classes et aucun exemple de texte purement factuel : il doit
    forcément choisir Positif/Négatif. Cette règle évite ce choix arbitraire.
    """
    if not re.search(r"[^\W\d_]", text) or text.strip().lower() in {"n/a", "na", "none", "aucun", "rien"}:
        return True   # texte vide de mots (« ... », « ? », « N/A ») : rien à juger
    if n_hits or vader_label != "Neutral" or len(text.split()) > SHORT_TEXT_WORDS:
        return False
    t = text.strip()
    return t.endswith("?") or bool(_QUESTION_START.match(t)) or bool(_FACTUAL.search(t))


@dataclass
class Verdict:
    label: str                       # Positif / Négatif / Neutre / Mixte
    explanation: str
    positive_words: list[str] = field(default_factory=list)
    negative_words: list[str] = field(default_factory=list)
    positive_share: float | None = None   # part du positif dans les signaux lexicaux (mixte)
    negative_share: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    lexicon_label: str = ""
    vader_label: str = ""
    ml_label: str = ""
    themes: list[str] = field(default_factory=list)
    warnings: list[tuple[str, str]] = field(default_factory=list)
    language: str = ""


def _language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"


def analyze_verdict(text: str, bundle=None) -> Verdict:
    lex = analyze_lexicon(text)
    vader_label, vader_compound = analyze_vader(text)
    pos_hits = [h for h in lex.hits if h["polarity"] > 0]
    neg_hits = [h for h in lex.hits if h["polarity"] < 0]
    pos_mass = sum(h["polarity"] for h in pos_hits)
    neg_mass = -sum(h["polarity"] for h in neg_hits)
    total = pos_mass + neg_mass

    ml_label, probabilities, ml_conf = "", {}, None
    if bundle is not None:
        ml = predict_ml([text], bundle).iloc[0]
        ml_label = str(ml["ml_sentiment"])
        ml_conf = float(ml["ml_confidence"])
        probabilities = {
            cls: float(ml[f"proba_{cls}"]) for cls in bundle["labels"] if f"proba_{cls}" in ml
        }

    all_negated = bool(lex.hits) and all(h["negated"] for h in lex.hits)
    is_mixed = (
        pos_mass > 0 and neg_mass > 0
        and min(pos_mass, neg_mass) / max(pos_mass, neg_mass) >= MIXED_RATIO
        and not all_negated   # « ni bon ni mauvais » : deux négations = avis neutre, pas mixte
    )
    if is_mixed:
        label = "Mixed"
        explanation = (
            f"Le commentaire contient des opinions positives et négatives de poids comparable : "
            f"environ {round(100 * pos_mass / total)} % de positif et {round(100 * neg_mass / total)} % de négatif "
            f"parmi les mots d'opinion reconnus."
        )
    elif _no_opinion(text, len(lex.hits), vader_label):
        label = "Neutral"
        explanation = (
            "Aucun mot d'opinion : le texte est une question ou un énoncé factuel, donc neutre "
            "(le classifieur n'a pas d'exemple de ce type et aurait choisi une polarité au hasard)."
        )
    elif len(text.split()) <= SHORT_TEXT_WORDS and _NEUTRAL_CUES.search(text) and not (pos_hits and neg_hits):
        label = "Neutral"
        explanation = "Expression mitigée (« sans plus », « ni bon ni mauvais », « average »…) : avis neutre."
    elif all_negated and pos_mass > 0 and neg_mass > 0:
        label = "Neutral"
        explanation = "Deux négations opposées (« ni bon ni mauvais », « not bad, not great ») : avis neutre."
    elif bundle is not None and ml_label != "Neutral" and lex.label in ("Positive", "Negative") and lex.label != ml_label and (
        ml_conf < OVERRIDE_CONF
        or (ml_conf < CONSENSUS_CONF and vader_label == lex.label)   # lexique ET VADER d'accord contre le modèle
    ):
        # Le classifieur hésite entre deux polarités et le lexique en voit une nette : on suit le lexique.
        # Un « Neutre » du classifieur n'est jamais écrasé (sinon la classe Neutre, déjà la plus faible, recule).
        label = lex.label
        explanation = (
            f"Le classifieur hésite (confiance {ml_conf:.2f}) et contredit les mots d'opinion du texte : "
            "le verdict suit le lexique."
        )
    elif bundle is not None:
        label = ml_label
        explanation = (
            f"Verdict du classifieur TF-IDF + régression logistique (confiance {ml_conf:.2f}). "
            "Le lexique et VADER servent de contrôle."
        )
    else:
        label = lex.label
        explanation = "Verdict du lexique maison (aucun modèle entraîné trouvé)."

    if not lex.hits and label in ("Positive", "Negative", "Positif", "Négatif"):
        explanation += " Peu ou pas de mots d'opinion reconnus par le lexique : le verdict repose surtout sur le modèle."

    verdict = Verdict(
        label=LABEL_FR.get(label, label),
        explanation=explanation,
        positive_words=[h["token"] for h in pos_hits],
        negative_words=[h["token"] for h in neg_hits],
        positive_share=(pos_mass / total) if (total and label == "Mixed") else None,
        negative_share=(neg_mass / total) if (total and label == "Mixed") else None,
        probabilities=probabilities,
        lexicon_label=LABEL_FR.get(lex.label, lex.label),
        vader_label=LABEL_FR.get(vader_label, vader_label),
        ml_label=LABEL_FR.get(ml_label, ml_label) if ml_label else "",
        themes=[t for t in detect_themes(text)],
        language=_language(text),
    )
    verdict.warnings = _warnings(text, label, lex, vader_compound, ml_conf, verdict.language)
    return verdict


def _warnings(text, label, lex, vader_compound, ml_conf, language) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    polar = label in ("Positive", "Negative")
    if polar and lex.label == "Neutral" and len(lex.hits) <= 1:
        out.append((
            "Risque de neutre confondu avec une polarité",
            "Très peu de mots d'opinion reconnus : le verdict polaire repose surtout sur le modèle.",
        ))
    if label == "Neutral" and (abs(lex.score) >= 1 or abs(vader_compound) >= 0.5):
        out.append((
            "Risque de polarité manquée (neutre prédit)",
            "Des mots d'opinion forts sont présents alors que le verdict est neutre.",
        ))
    if any(h["negated"] for h in lex.hits):
        out.append((
            "Risque de polarité inversée",
            "Une négation porte sur un mot d'opinion : vérifiez le sens (« not helpful » est négatif, « not boring » est positif).",
        ))
    if polar and vader_compound * (1 if label == "Positive" else -1) < -0.05:
        out.append((
            "Désaccord avec VADER",
            "VADER donne la polarité opposée : le commentaire est ambigu ou contient une tournure difficile.",
        ))
    if ml_conf is not None and ml_conf < 0.6:
        out.append(("Confiance faible du modèle", f"Probabilité maximale de {ml_conf:.2f}."))
    if language not in ("en", "unknown"):
        out.append((
            "Texte non anglais",
            f"Langue détectée : {language}. Le jeu d'évaluation est à 97,6 % anglais : résultat peu fiable.",
        ))
    return out
