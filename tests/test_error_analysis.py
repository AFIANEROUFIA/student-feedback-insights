import pandas as pd

from src.error_analysis import error_tags, primary_tag, select_examples


def test_neutral_confused_with_positive():
    tags = error_tags("Too much reading.", "Neutral", "Positive")
    assert "neutre confondu avec une polarité" in tags


def test_polarity_inversion_and_negation_detected():
    tags = error_tags("The teacher is not helpful at all.", "Negative", "Positive")
    assert "polarité inversée" in tags and "négation présente" in tags
    assert primary_tag(tags) == "polarité inversée"


def test_correct_prediction_has_no_tags():
    assert error_tags("Great class.", "Positive", "Positive") == []


def test_selection_is_bounded():
    rows = []
    for i in range(40):
        rows.append({
            "id": str(i), "methode": "TF-IDF + LR", "niveau": "Cours", "langue": "en",
            "gold": "Neutral", "prediction": "Positive", "type_erreur": "neutre confondu avec une polarité",
            "etiquettes": "", "texte": f"Short review {i}.",
        })
    assert len(select_examples(pd.DataFrame(rows), n=20)) == 20
