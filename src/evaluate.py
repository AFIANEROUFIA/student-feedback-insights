from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support


LABELS = ["Negative", "Neutral", "Positive"]


def classification_metrics(y_true, y_pred) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0, output_dict=True)
    return {
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(LABELS)
        },
        "report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "labels": LABELS,
    }


def save_metrics(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def metrics_table(metrics: dict, method: str) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "méthode": method,
            "accuracy": round(metrics["accuracy"], 3),
            "precision_macro": round(metrics["precision_macro"], 3),
            "recall_macro": round(metrics["recall_macro"], 3),
            "macro_f1": round(metrics["macro_f1"], 3),
        }
    ])
