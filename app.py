"""StudentVoice AI — interface simple.

Onglet 1 : analyser un commentaire (positif, négatif, neutre ou mixte, avec explication
et points de vigilance). Onglet 2 : tableau de bord du projet (résultats sur le jeu de test).

Lancer : streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.evaluate import LABELS, metrics_table
from src.predict import load_ml_bundle
from src.verdict import analyze_verdict

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "reports" / "metrics.json"
ERRORS = ROOT / "reports" / "erreurs_detaillees.csv"
EXAMPLE = "The teacher is friendly, but the exams are extremely difficult."

st.set_page_config(page_title="StudentVoice AI", page_icon="🎓", layout="centered")


@st.cache_resource
def bundle():
    return load_ml_bundle()


@st.cache_data
def metrics():
    return json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else None


def show_verdict(text: str) -> None:
    v = analyze_verdict(text, bundle())
    box = {"Positif": st.success, "Négatif": st.error, "Neutre": st.info, "Mixte": st.warning}[v.label]
    box(f"**Verdict : {v.label}**")
    st.write(v.explanation)

    if v.label == "Mixte":
        c1, c2 = st.columns(2)
        c1.metric("Part positive", f"{round(100 * v.positive_share)} %")
        c2.metric("Part négative", f"{round(100 * v.negative_share)} %")

    c1, c2 = st.columns(2)
    c1.markdown("**Mots positifs reconnus**")
    c1.write(", ".join(v.positive_words) or "—")
    c2.markdown("**Mots négatifs reconnus**")
    c2.write(", ".join(v.negative_words) or "—")

    st.markdown("**Thèmes détectés**")
    st.write(", ".join(v.themes))

    if v.probabilities:
        st.markdown("**Probabilités du classifieur**")
        st.bar_chart(pd.Series(v.probabilities, name="probabilité"))

    st.markdown("**Comparaison des méthodes**")
    st.table(pd.DataFrame([
        {"méthode": "Classifieur TF-IDF + LR (retenu)", "verdict": v.ml_label or "—"},
        {"méthode": "Lexique maison", "verdict": v.lexicon_label},
        {"méthode": "VADER", "verdict": v.vader_label},
    ]))

    st.markdown("**Points de vigilance (risques d'erreur)**")
    if v.warnings:
        for name, detail in v.warnings:
            st.warning(f"**{name}** — {detail}")
    else:
        st.write("Aucun point de vigilance.")
    st.caption(
        "Les points de vigilance sont des signaux heuristiques. Ils indiquent un risque d'erreur, "
        "ils ne prouvent pas que le verdict est faux."
    )


def tab_analyse() -> None:
    st.title("Analyse d'un commentaire étudiant")
    st.caption("Écrivez un commentaire en anglais pour un résultat fiable.")
    text = st.text_area("Commentaire", EXAMPLE, height=120)
    if st.button("Analyser", type="primary"):
        if not text.strip():
            st.warning("Écrivez un commentaire.")
        else:
            show_verdict(text)


def tab_dashboard() -> None:
    st.title("Tableau de bord du projet")
    m = metrics()
    if m is None:
        st.error("reports/metrics.json introuvable : lancez python -m src.train_sentiment_ml")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Avis annotés", m["n_total"])
    c2.metric("Avis de test", m["n_test"])
    c3.metric("Macro-F1 (meilleur)", f"{m['ml']['macro_f1']:.3f}")
    st.caption(f"Jeu : {m['dataset']} (EduRABSA, anonymisé, licence CC-BY). "
               f"Test : 25 %, stratifié, les avis d'entraînement sont exclus.")

    st.subheader("Comparaison des méthodes (jeu de test)")
    table = pd.concat([
        metrics_table(m["lexicon"], "Lexique maison"),
        metrics_table(m["vader"], "VADER"),
        metrics_table(m["ml"], "TF-IDF + Logistic Regression"),
    ], ignore_index=True)
    st.dataframe(table, hide_index=True)
    cv = m.get("cv_ml_macro_f1", {})
    st.write(f"Validation croisée 5 plans (classifieur) : {cv.get('mean')} ± {cv.get('std')}.")

    st.subheader("F1 par classe")
    per_class = pd.DataFrame({
        method: {label: round(m[key]["per_class"][label]["f1"], 3) for label in LABELS}
        for method, key in [("Lexique", "lexicon"), ("VADER", "vader"), ("Classifieur", "ml")]
    })
    st.dataframe(per_class)
    st.bar_chart(per_class)

    left, right = st.columns(2)
    with left:
        st.subheader("Répartition du test")
        st.bar_chart(pd.Series(m["distribution_test"], name="avis"))
    with right:
        st.subheader("Matrice de confusion (classifieur)")
        st.caption("Lignes : vérité. Colonnes : prédiction.")
        st.dataframe(pd.DataFrame(m["ml"]["confusion_matrix"], index=LABELS, columns=LABELS))

    st.subheader("Erreurs par type")
    if ERRORS.exists():
        errors = pd.read_csv(ERRORS, keep_default_na=False)
        counts = errors.groupby(["type_erreur", "methode"]).size().unstack(fill_value=0)
        st.dataframe(counts)
        st.caption(
            "Types : neutre confondu avec une polarité, polarité inversée, polarité manquée (neutre prédit). "
            "Les étiquettes sont automatiques."
        )
    else:
        st.info("reports/erreurs_detaillees.csv absent : lancez python -m src.error_analysis")

    with st.expander("Limites"):
        st.markdown(
            "- Corpus à 97,6 % anglais : aucun résultat fiable pour le français.\n"
            "- La classe neutre est la plus difficile (F1 0,42 pour le meilleur modèle).\n"
            "- Les étiquettes contiennent du bruit d'annotation.\n"
            "- La négation est gérée par le lexique, pas par le classifieur.\n"
            "- Détail complet : docs/limites.md."
        )


def main() -> None:
    tab1, tab2 = st.tabs(["Analyser un commentaire", "Tableau de bord"])
    with tab1:
        tab_analyse()
    with tab2:
        tab_dashboard()


if __name__ == "__main__":
    main()
