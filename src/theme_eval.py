"""Évaluation de la pertinence des thèmes (exigence du professeur).

Étape 1 : `python -m src.theme_eval build`
  génère docs/evaluation_themes/grille_pertinence_30_themes.csv : 30 thèmes candidats
  (24 catégories EduRABSA + 6 sujets LDA), avec mots-clés et exemples, et trois colonnes
  vides membre_1, membre_2, membre_3 (échelle 1 à 3).

Étape 2 : chaque membre note les 30 thèmes (1 = non pertinent, 2 = partiel, 3 = pertinent),
  puis l'équipe enregistre la grille remplie (ex. grille_remplie.csv).

Étape 3 : `python -m src.theme_eval agree docs/evaluation_themes/grille_remplie.csv`
  calcule le kappa de Fleiss (3 évaluateurs, catégories 1/2/3), la moyenne des kappas
  pondérés quadratiques deux à deux, l'accord exact, et liste les thèmes peu pertinents.

Aucune note n'est fabriquée par ce module : sans grille remplie, le calcul n'est pas lancé.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .data_io import load_edurabsa_json
from .themes import category_profiles, lda_topics

ROOT = Path(__file__).resolve().parents[1]
GRID_DIR = ROOT / "docs" / "evaluation_themes"
GRID_CSV = GRID_DIR / "grille_pertinence_30_themes.csv"
REPORT_JSON = ROOT / "reports" / "accord_themes.json"
RATERS = ["membre_1", "membre_2", "membre_3"]
SCALE = (1, 2, 3)
N_TOTAL = 30
JSON_FILES = [
    ROOT / "data" / "raw" / "Train_validation_Merged_ASQE_N4000_anonymise.json",
    ROOT / "data" / "raw" / "DEV_TEST_Merged_ASQE_N2500_anonymise.json",
]
CSV_SAMPLE = ROOT / "data" / "raw" / "combined_data_6.5k_anonymise.csv"


def fleiss_kappa_from_counts(counts: np.ndarray) -> float:
    """Kappa de Fleiss. `counts[i, j]` = nombre d'évaluateurs ayant donné la catégorie j au sujet i."""
    counts = np.asarray(counts, dtype=float)
    n_subjects, _ = counts.shape
    n_raters = counts[0].sum()
    if n_raters < 2:
        raise ValueError("Il faut au moins 2 évaluateurs.")
    p_i = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / (n_subjects * n_raters)
    p_e = np.sum(p_j ** 2)
    if np.isclose(1 - p_e, 0.0):
        return 1.0 if np.isclose(p_bar, 1.0) else 0.0
    return float((p_bar - p_e) / (1 - p_e))


def fleiss_kappa(ratings: np.ndarray, scale=SCALE) -> float:
    """`ratings` : matrice (sujets x évaluateurs) de valeurs de l'échelle."""
    counts = np.array([[np.sum(row == c) for c in scale] for row in ratings])
    return fleiss_kappa_from_counts(counts)


def interpret(kappa: float) -> str:
    """Grille de Landis & Koch (1977)."""
    if kappa < 0:
        return "accord inférieur au hasard"
    for limit, label in [(0.2, "faible"), (0.4, "passable"), (0.6, "modéré"), (0.8, "substantiel")]:
        if kappa <= limit:
            return label
    return "presque parfait"


def build_grid(out: Path = GRID_CSV, n_total: int = N_TOTAL) -> pd.DataFrame:
    quads = []
    for path in JSON_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Fichier manquant : {path}. Lancez scripts/prepare_edurabsa.py.")
        for _, rec in load_edurabsa_json(path).iterrows():
            for quad in rec["quadruplets"]:
                quads.append({"text": rec["review_text"], "category": quad.get("category"), "opinion": quad.get("opinion")})
    profiles = category_profiles(pd.DataFrame(quads))
    n_lda = max(0, n_total - len(profiles))
    corpus = pd.read_csv(CSV_SAMPLE, keep_default_na=False)["review_text"].astype(str).tolist()
    topics = lda_topics(corpus, n_topics=n_lda) if n_lda else pd.DataFrame()

    rows = []
    for _, r in profiles.iterrows():
        rows.append({
            "type": "Catégorie EduRABSA",
            "libelle": r["categorie"],
            "mots_cles": r["mots_cles"],
            "exemples": " | ".join(e[:200] for e in str(r["exemples"]).split(" | ")[:2]),
        })
    for _, r in topics.iterrows():
        rows.append({
            "type": "Sujet LDA",
            "libelle": r["theme_id"],
            "mots_cles": r["mots_cles"],
            "exemples": " | ".join(e[:200] for e in str(r["exemples"]).split(" | ")[:2]),
        })
    grid = pd.DataFrame(rows)
    grid.insert(0, "theme_id", [f"T{i + 1:02d}" for i in range(len(grid))])
    for rater in RATERS:
        grid[rater] = np.nan
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out, index=False, encoding="utf-8-sig")
    return grid


def load_ratings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in RATERS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes d'évaluateurs manquantes : {missing}")
    for col in RATERS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    invalid = df[RATERS].apply(lambda s: s.dropna().isin(SCALE).all()).all()
    if not invalid:
        raise ValueError("Les notes doivent être 1, 2 ou 3.")
    return df


def agreement_report(path: Path, out_json: Path = REPORT_JSON) -> dict:
    df = load_ratings(path)
    complete = df.dropna(subset=RATERS).copy()
    if len(complete) < 2:
        raise ValueError("Pas assez de thèmes notés par les 3 évaluateurs.")
    matrix = complete[RATERS].to_numpy(dtype=int)
    kappa = fleiss_kappa(matrix)
    pairwise = {}
    for a, b in combinations(RATERS, 2):
        pairwise[f"{a} vs {b}"] = float(cohen_kappa_score(complete[a].astype(int), complete[b].astype(int), weights="quadratic"))
    exact = float(np.mean([len(set(row)) == 1 for row in matrix]))
    mean_by_theme = complete[RATERS].mean(axis=1)
    id_cols = [c for c in ["theme_id", "libelle"] if c in complete.columns]
    low = complete.loc[mean_by_theme < 2, id_cols].assign(moyenne=mean_by_theme[mean_by_theme < 2].round(2))
    report = {
        "fichier": path.name,
        "themes_notes_par_les_3": int(len(complete)),
        "themes_incomplets_ignores": int(len(df) - len(complete)),
        "kappa_fleiss": round(kappa, 3),
        "interpretation_fleiss": interpret(kappa),
        "kappa_pondere_quadratique_moyen": round(float(np.mean(list(pairwise.values()))), 3),
        "kappa_pondere_par_paire": {k: round(v, 3) for k, v in pairwise.items()},
        "accord_exact_%": round(100 * exact, 1),
        "pertinence_moyenne_globale": round(float(mean_by_theme.mean()), 2),
        "themes_peu_pertinents_moyenne_<2": low.to_dict(orient="records"),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Grille de pertinence des thèmes et accord inter-évaluateurs.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="Génère la grille à 30 thèmes")
    agree = sub.add_parser("agree", help="Calcule l'accord à partir d'une grille remplie")
    agree.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        grid = build_grid()
        print(f"Grille écrite : {GRID_CSV} ({len(grid)} thèmes)")
    else:
        print(json.dumps(agreement_report(args.path), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
