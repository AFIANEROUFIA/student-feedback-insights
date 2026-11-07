"""Prépare les données EduRABSA anonymisées pour le projet.

Entrée : le dossier extrait « EduRABSA_Dataset/ » (ZIP fourni avec le cours).
Sorties (dans data/raw/) :
- combined_data_6.5k_anonymise.csv : 6 500 avis, texte anonymisé, langue détectée ;
- Train_validation_Merged_ASQE_N4000_anonymise.json et DEV_TEST_Merged_ASQE_N2500_anonymise.json :
  quadruplets (aspect / opinion / catégorie / polarité) avec textes anonymisés ;
- anonymisation_rapport.json : nombre de remplacements effectués ;
- data/LICENSE-CC-BY.txt : licence du jeu de données (à citer dans le rapport).

Usage :
    python scripts/prepare_edurabsa.py --edurabsa "chemin/vers/EduRABSA_Dataset"
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anonymize import anonymize_frame, anonymize_series  # noqa: E402
from src.preprocessing import normalize_text  # noqa: E402

CSV_REL = "1_dataset_processing_sampling/data/2_sampled/combined_data_6.5k.csv"
JSON_RELS = [
    "2_annotated_dataset_files/Annotation_raw_output/Train_validation_Merged_ASQE_N4000.json",
    "2_annotated_dataset_files/Annotation_raw_output/DEV_TEST_Merged_ASQE_N2500.json",
]
LICENSE_REL = "LICENSE-CC-BY.txt"


def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:  # textes trop courts ou vides
        return "unknown"


def prepare_csv(edu: Path, out_dir: Path) -> dict:
    df = pd.read_csv(edu / CSV_REL)
    n_raw = len(df)
    df["review_text"] = df["review_text"].map(normalize_text)
    df = df[df["review_text"].str.strip().ne("")].drop_duplicates(subset=["review_text"])
    df, redactions = anonymize_frame(df, "review_text")
    df["language"] = df["review_text"].map(detect_language)
    df["id"] = df["id"].astype(str)
    keep = ["id", "type", "source", "rating", "rating_cat", "language", "review_text"]
    out = out_dir / "combined_data_6.5k_anonymise.csv"
    df[keep].to_csv(out, index=False, encoding="utf-8")
    return {
        "fichier": out.name,
        "lignes_brutes": n_raw,
        "lignes_apres_nettoyage": int(len(df)),
        "remplacements_anonymisation": int(redactions),
        "langues": df["language"].value_counts().head(10).to_dict(),
    }


def prepare_json(edu: Path, out_dir: Path) -> list[dict]:
    reports = []
    for rel in JSON_RELS:
        src = edu / rel
        data = json.loads(src.read_text(encoding="utf-8"))
        total = 0
        for record in data.values():
            text, k = anonymize_series([record.get("text", "")])
            record["text"] = text[0]
            total += k
            for quad in record.get("Quadruplets", []):
                for field in ("aspect", "opinion"):
                    if isinstance(quad.get(field), str):
                        vals, k2 = anonymize_series([quad[field]])
                        quad[field] = vals[0]
                        total += k2
        out = out_dir / src.name.replace(".json", "_anonymise.json")
        out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        reports.append({"fichier": out.name, "avis": len(data), "remplacements_anonymisation": int(total)})
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--edurabsa", required=True, help="Dossier EduRABSA_Dataset extrait")
    args = parser.parse_args()
    DetectorFactory.seed = 0

    edu = Path(args.edurabsa)
    out_dir = ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "csv": prepare_csv(edu, out_dir),
        "json": prepare_json(edu, out_dir),
        "note": "Remplacements = e-mails, URLs, pseudos, téléphones, noms propres (heuristique regex).",
    }
    (out_dir / "anonymisation_rapport.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lic_dir = ROOT / "data"
    shutil.copy(edu / LICENSE_REL, lic_dir / LICENSE_REL)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
