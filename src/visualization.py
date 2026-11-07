from __future__ import annotations

import pandas as pd
import streamlit as st


def sentiment_pie(series: pd.Series, title: str) -> None:
    counts = series.fillna("Inconnu").value_counts()
    st.subheader(title)
    st.bar_chart(counts)


def theme_bars(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Aucun thème à afficher.")
        return
    st.bar_chart(df.set_index("theme")["frequence"])
