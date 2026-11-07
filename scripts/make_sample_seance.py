"""Échantillon de 50 avis anonymisés, à apporter en séance (demande : 20 à 50 lignes).

Quotas : 20 avis « teacher review », 20 « course review », 10 « university review ».
Usage : python scripts/make_sample_seance.py   (après scripts/prepare_edurabsa.py)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "combined_data_6.5k_anonymise.csv"
OUT = ROOT / "data" / "raw" / "echantillon_seance_50.csv"
QUOTAS = {"teacher review": 20, "course review": 20, "university review": 10}


def main(seed: int = 42) -> None:
    df = pd.read_csv(SRC, keep_default_na=False)
    df = df[df["rating_cat"].isin(["Pos", "Neg", "Conf/Neu"])]
    parts = []
    for avis_type, quota in QUOTAS.items():
        pool = df[df["type"] == avis_type]
        parts.append(pool.sample(n=min(quota, len(pool)), random_state=seed))
    sample = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    sample.to_csv(OUT, index=False, encoding="utf-8")
    print("Écrit :", OUT, "|", len(sample), "avis")
    print(sample["type"].value_counts().to_dict())
    print(sample["rating_cat"].value_counts().to_dict())


if __name__ == "__main__":
    main()
