from pathlib import Path

from src.data_io import load_reviews_table, prepare_frame
from src.train_sentiment_ml import build_classifier

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "raw" / "avis_etudiants_sample.csv"


def test_classifier_trains_on_sample_and_predicts_valid_labels():
    df, _ = prepare_frame(load_reviews_table(SAMPLE))
    df = df[df["overall_sentiment"].notna()]
    clf = build_classifier().fit(df["text_clean"], df["overall_sentiment"])
    preds = set(clf.predict(df["text_clean"]))
    assert preds <= {"Positive", "Negative", "Neutral"}
    assert (clf.predict(df["text_clean"]) == df["overall_sentiment"]).mean() > 0.9  # sur les données d'apprentissage
