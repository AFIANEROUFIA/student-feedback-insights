from src.predict import load_ml_bundle
from src.verdict import analyze_verdict

BUNDLE = load_ml_bundle()


def test_mixed_when_positive_and_negative_comparable():
    v = analyze_verdict("The teacher is friendly, but the exams are extremely difficult.", BUNDLE)
    assert v.label == "Mixte"
    assert v.positive_share is not None and v.negative_share is not None
    assert abs(v.positive_share + v.negative_share - 1) < 1e-9


def test_clear_positive_and_negative():
    assert analyze_verdict("This course is excellent and very clear.", BUNDLE).label == "Positif"
    assert analyze_verdict("He is not helpful at all.", BUNDLE).label == "Négatif"


def test_negation_triggers_inversion_warning():
    v = analyze_verdict("He is not helpful at all.", BUNDLE)
    assert any("polarité inversée" in name for name, _ in v.warnings)


def test_neutral_text_with_polar_prediction_gets_neutral_confusion_warning():
    v = analyze_verdict("The course contains five chapters and a final exam.", BUNDLE)
    if v.label == "Positif":
        assert any("neutre confondu" in name for name, _ in v.warnings)


def test_explanation_and_probabilities_present():
    v = analyze_verdict("Clear slides and helpful lectures.", BUNDLE)
    assert v.explanation
    assert set(v.probabilities) == {"Positive", "Negative", "Neutral"}


def test_question_or_fact_without_opinion_is_neutral():
    for text in ("where is the teacher", "Où est le prof ?", "Office hours are on Monday", "The course is at 9am in room B12"):
        assert analyze_verdict(text, BUNDLE).label == "Neutre", text


def test_mitigated_opinions_are_neutral_not_mixed():
    for text in ("Il est ni bon ni mauvais", "Not bad, not great", "Le cours est correct, sans plus", "The course is fine, nothing special"):
        assert analyze_verdict(text, BUNDLE).label == "Neutre", text


def test_french_mixed_sentences():
    assert analyze_verdict("Le prof est sympa mais ses cours sont ennuyeux", BUNDLE).label == "Mixte"
    assert analyze_verdict("Le cours est intéressant mais l'examen était trop long", BUNDLE).label == "Mixte"


def test_low_confidence_model_follows_clear_lexicon():
    assert analyze_verdict("The teacher is absent today", BUNDLE).label == "Négatif"


def test_empty_like_text_is_neutral_and_mal_organise_is_negative():
    for text in ("...", "?", "N/A"):
        assert analyze_verdict(text, BUNDLE).label == "Neutre", text
    assert analyze_verdict("Cours très ennuyeux et mal organisé.", BUNDLE).label == "Négatif"


def test_short_clear_positive_not_overruled_by_hesitant_model():
    assert analyze_verdict("This course is excellent.", BUNDLE).label == "Positif"
    assert analyze_verdict("This course is excellent…", BUNDLE).label == "Positif"
