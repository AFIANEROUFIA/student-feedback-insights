from __future__ import annotations

import pandas as pd

from .preprocessing import normalize_label


def sentiment_distribution(df: pd.DataFrame, col: str = "overall_sentiment") -> pd.Series:
    return df[col].map(normalize_label).value_counts(dropna=False)


def language_distribution(df: pd.DataFrame, col: str = "language") -> pd.Series | None:
    if col not in df.columns:
        return None
    return df[col].value_counts(dropna=False)


def missing_and_duplicates(df: pd.DataFrame, text_col: str = "review_text") -> dict:
    return {
        "n": int(len(df)),
        "missing_text": int(df[text_col].isna().sum()) if text_col in df.columns else None,
        "empty_text": int(df[text_col].fillna("").astype(str).str.strip().eq("").sum()) if text_col in df.columns else None,
        "duplicate_text": int(df[text_col].duplicated().sum()) if text_col in df.columns else None,
    }
