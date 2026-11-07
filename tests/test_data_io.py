import json

import pandas as pd

from src.data_io import NIVEAU_COURS, NIVEAU_ENSEIGNANT, NIVEAU_UNIVERSITE, infer_niveau, load_edurabsa_json, prepare_frame


def test_niveau_from_declared_type_and_inference():
    assert infer_niveau("anything", "teacher review") == NIVEAU_ENSEIGNANT
    assert infer_niveau("anything", "university review") == NIVEAU_UNIVERSITE
    assert infer_niveau("The course has a hard exam", None) == NIVEAU_COURS
    assert infer_niveau("The classroom wifi is slow", "enquête") == NIVEAU_UNIVERSITE


def test_prepare_frame_cleans_dedups_and_anonymizes():
    raw = pd.DataFrame({
        "id": ["a", "b", "c", "d"],
        "type": ["teacher review", "course review", "course review", "course review"],
        "review_text": [
            "Dr. Zhong is a good teacher.",
            "Great course, the exam is fair.",
            "Great course, the exam is fair.",   # doublon
            "   ",                                 # vide
        ],
        "overall_sentiment": ["Pos", "Conf/Neu", "Conf/Neu", "[NA]"],
    })
    df, redactions = prepare_frame(raw)
    assert len(df) == 2
    assert df.loc[df["id"] == "a", "niveau"].item() == NIVEAU_ENSEIGNANT
    assert set(df["overall_sentiment"].dropna()) == {"Positive", "Neutral"}
    assert "Zhong" not in " ".join(df["review_text"])
    assert redactions >= 1


def test_load_edurabsa_json_dict_format(tmp_path):
    payload = {
        "1": {"id": "1", "text": "Good teacher.", "overall_sentiment": "Pos", "type": "teacher review",
              "Quadruplets": [{"aspect": "teacher", "opinion": "good", "category": "Staff - Teaching", "sentiment": "Positive"}]},
        "2": {"id": "2", "text": "Bad class.", "overall_sentiment": "", "type": "course review",
              "Quadruplets": [{"aspect": "class", "opinion": "bad", "category": "Course - Overall", "sentiment": "Negative"}]},
    }
    path = tmp_path / "x.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    df = load_edurabsa_json(path)
    assert len(df) == 2
    assert df.loc[df["id"] == "1", "overall_sentiment"].item() == "Positive"
    # Pas de label global : il est déduit des quadruplets (ici 1 négatif -> Negative).
    assert df.loc[df["id"] == "2", "overall_sentiment"].item() == "Negative"
