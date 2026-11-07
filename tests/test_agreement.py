import numpy as np
import pandas as pd
import pytest

from src.theme_eval import agreement_report, fleiss_kappa, fleiss_kappa_from_counts, interpret


def test_fleiss_matches_published_example():
    # Fleiss (1971) : 10 sujets, 14 évaluateurs, 5 catégories -> kappa = 0,210
    counts = np.array([
        [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
    ])
    assert round(fleiss_kappa_from_counts(counts), 3) == 0.210


def test_perfect_and_chance_agreement():
    perfect = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [1, 1, 1]])
    assert fleiss_kappa(perfect) == pytest.approx(1.0)


def test_interpretation_bands():
    assert interpret(0.1) == "faible"
    assert interpret(0.5) == "modéré"
    assert interpret(0.9) == "presque parfait"


def test_agreement_report_on_filled_grid(tmp_path):
    grid = pd.DataFrame({
        "theme_id": [f"T{i}" for i in range(1, 7)],
        "membre_1": [3, 3, 2, 1, 3, 2],
        "membre_2": [3, 2, 2, 1, 3, 3],
        "membre_3": [3, 3, 2, 1, 2, 2],
    })
    path = tmp_path / "grille.csv"
    grid.to_csv(path, index=False)
    report = agreement_report(path, out_json=tmp_path / "accord.json")
    assert report["themes_notes_par_les_3"] == 6
    assert -1 <= report["kappa_fleiss"] <= 1
    assert "membre_1 vs membre_2" in report["kappa_pondere_par_paire"]


def test_invalid_scale_rejected(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"theme_id": ["T1", "T2"], "membre_1": [5, 1], "membre_2": [1, 1], "membre_3": [1, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        agreement_report(path, out_json=tmp_path / "accord.json")
