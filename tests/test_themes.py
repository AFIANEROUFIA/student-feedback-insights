import pandas as pd

from src.themes import category_profiles, lda_topics, themes_with_examples


def _quads():
    rows = []
    for i in range(12):
        rows.append({"text": f"The exam was fair and the grading clear, comment {i}.", "category": "Course - Assessment", "opinion": "fair"})
        rows.append({"text": f"Professor {i} is very friendly and helpful with students.", "category": "Staff - Attitude", "opinion": "friendly"})
    return pd.DataFrame(rows)


def test_category_profiles_has_terms_opinions_and_examples():
    prof = category_profiles(_quads())
    assert set(prof["categorie"]) == {"Course - Assessment", "Staff - Attitude"}
    row = prof[prof["categorie"] == "Course - Assessment"].iloc[0]
    assert "fair" in row["opinions_frequentes"]
    assert 1 <= len(row["exemples"].split(" | ")) <= 3
    assert row["mots_cles"]


def test_lda_returns_requested_topics():
    texts = [f"the exam was hard but fair {i}" for i in range(30)] + [f"friendly teacher helpful class {i}" for i in range(30)]
    table = lda_topics(texts, n_topics=3)
    assert len(table) == 3
    assert table["part_avis_%"].sum() > 99


def test_lda_small_corpus_is_empty():
    assert lda_topics(["only one"], n_topics=6).empty


def test_theme_examples_limited():
    df = pd.DataFrame({"review_text": ["The exam is hard.", "The exam was fair.", "The exam again.", "The exam ok."]})
    out = themes_with_examples(df, n_examples=2)
    assert out.loc[out["theme"] == "Examens / évaluation", "exemples"].item().count(" | ") <= 1
