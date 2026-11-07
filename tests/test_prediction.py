from src.absa import extract_aspects
from src.lexicon import analyze_lexicon, analyze_vader
from src.themes import detect_themes


def test_positive_lexicon():
    assert analyze_lexicon("The teacher is very helpful and clear.").label == "Positive"


def test_negation_flips():
    assert analyze_lexicon("The teacher is not helpful.").label == "Negative"
    assert analyze_lexicon("The course isn't clear at all.").label == "Negative"


def test_neutral():
    assert analyze_lexicon("The course contains five chapters.").label == "Neutral"


def test_vader_baseline_english():
    label, compound = analyze_vader("I absolutely loved this course, it was excellent!")
    assert label == "Positive" and compound > 0.05
    assert analyze_vader("This is the worst class ever, terrible.")[0] == "Negative"


def test_themes_and_absa():
    text = "The teacher is friendly, but the exams are extremely difficult."
    themes = detect_themes(text)
    assert any("Enseignant" in t or "Examens" in t for t in themes)
    assert len(extract_aspects(text)) >= 1


def test_unknown_theme_fallback():
    assert detect_themes("Nothing to add.") == ["Autre / général"]
