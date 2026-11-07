from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_io import NIVEAUX, load_edurabsa_json, load_reviews_table, prepare_frame
from src.evaluate import LABELS, classification_metrics, metrics_table
from src.lexicon import analyze_lexicon, analyze_vader
from src.predict import analyze_comments, load_ml_bundle, predict_ml
from src.absa import extract_aspects
from src.themes import category_profiles, lda_topics, themes_with_examples

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
EDU_CSV = RAW / "combined_data_6.5k_anonymise.csv"
SAMPLE = RAW / "avis_etudiants_sample.csv"
EDU_JSON = [
    RAW / "Train_validation_Merged_ASQE_N4000_anonymise.json",
    RAW / "DEV_TEST_Merged_ASQE_N2500_anonymise.json",
]
METRICS = ROOT / "reports" / "metrics.json"
ERRORS = ROOT / "reports" / "erreurs_detaillees.csv"
SELECTION = ROOT / "reports" / "erreurs_selection.md"
ACCORD = ROOT / "reports" / "accord_themes.json"
GRILLE = ROOT / "docs" / "evaluation_themes" / "grille_pertinence_30_themes.csv"
LIMITES = ROOT / "docs" / "limites.md"
ETHIQUE = ROOT / "docs" / "ethique_anonymisation.md"
ERREURS_COMMENTEES = ROOT / "docs" / "analyse_erreurs.md"

st.set_page_config(page_title="StudentVoice AI", page_icon="🎓", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #0f2744; color: #e8eef6; }
    h1, h2, h3 { color: #ffffff !important; }
    .sv-card { background:#16385f; padding:1.1rem 1.3rem; border-radius:16px; margin-bottom:1rem; }
    .sv-muted { color:#b7c6d9; }
    </style>
    """,
    unsafe_allow_html=True,
)

PAGES = [
    "Accueil",
    "Import",
    "Thèmes + exemples",
    "Analyse",
    "Comparaison",
    "Dashboard",
    "Évaluation",
    "Erreurs",
    "Limites & éthique",
]


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else f"*Fichier absent : {path.name}*"


# ---------------------------------------------------------------- données

@st.cache_data(show_spinner="Préparation des données (anonymisation, thèmes)…")
def load_prepared(kind: str, name: str, payload: bytes | None):
    if kind == "upload":
        PROCESSED.mkdir(parents=True, exist_ok=True)
        tmp = PROCESSED / f"_upload{Path(name).suffix.lower()}"
        tmp.write_bytes(payload or b"")
        raw = load_reviews_table(tmp)
    else:
        path = EDU_CSV if kind == "edurabsa" else SAMPLE
        raw = load_reviews_table(path)
    return prepare_frame(raw, anonymize=True)


@st.cache_resource
def cached_bundle():
    return load_ml_bundle()


@st.cache_data(show_spinner="Analyse des avis…")
def cached_analysis(texts: tuple[str, ...], with_absa: bool = False) -> pd.DataFrame:
    return analyze_comments(list(texts), load_ml_bundle(), with_absa=with_absa)


@st.cache_data(show_spinner="Thèmes LDA…")
def cached_lda(texts: tuple[str, ...], n_topics: int) -> pd.DataFrame:
    return lda_topics(list(texts), n_topics=n_topics)


@st.cache_data(show_spinner=False)
def cached_category_profiles() -> pd.DataFrame:
    rows = []
    for path in EDU_JSON:
        if not path.exists():
            continue
        for _, rec in load_edurabsa_json(path).iterrows():
            for quad in rec["quadruplets"]:
                rows.append({"text": rec["review_text"], "category": quad.get("category"), "opinion": quad.get("opinion")})
    if not rows:
        return pd.DataFrame()
    return category_profiles(pd.DataFrame(rows))


def get_dataset() -> tuple[pd.DataFrame, int, str]:
    """Retourne (données filtrées par type d'avis, remplacements d'anonymisation, libellé de la source)."""
    st.sidebar.header("Données")
    options = []
    if EDU_CSV.exists():
        options.append("EduRABSA anonymisé (6,5k)")
    if SAMPLE.exists():
        options.append("Échantillon synthétique (90)")
    options.append("Fichier importé")
    source = st.sidebar.radio("Source", options)
    uploaded = None
    if source == "Fichier importé":
        uploaded = st.sidebar.file_uploader("CSV ou JSON", type=["csv", "json"])
        if uploaded is None:
            st.warning("Importez un fichier CSV ou JSON, ou choisissez une autre source.")
            return pd.DataFrame(), 0, source
        df, redactions = load_prepared("upload", uploaded.name, uploaded.getvalue())
    elif source.startswith("EduRABSA"):
        df, redactions = load_prepared("edurabsa", "", None)
    else:
        df, redactions = load_prepared("sample", "", None)
    st.sidebar.caption("Les textes sont anonymisés à l'import (e-mails, pseudos, noms propres).")

    available = [n for n in NIVEAUX if n in set(df.get("niveau", []))]
    chosen = st.sidebar.multiselect("Type d'avis", available, default=available)
    if "niveau" in df.columns:
        df = df[df["niveau"].isin(chosen)].reset_index(drop=True)
    return df, redactions, source


# ---------------------------------------------------------------- pages

def page_accueil() -> None:
    st.title("StudentVoice AI")
    st.caption("Analyse des avis étudiants : thèmes récurrents, sentiments et exemples")
    st.markdown(
        """
        <div class="sv-card">
        <b>Projet 17</b> — identifier les <b>thèmes</b> et les <b>sentiments</b> des avis étudiants,
        les présenter <b>avec des exemples</b>, comparer un <b>lexique</b> à une <b>méthode de classification</b>,
        et évaluer avec le <b>F1 macro</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    metrics = _read_json(METRICS)
    if metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Macro-F1 TF-IDF + LR", f"{metrics['ml']['macro_f1']:.3f}", help="Jeu de test, non vu à l'entraînement")
        c2.metric("Macro-F1 lexique maison", f"{metrics['lexicon']['macro_f1']:.3f}")
        c3.metric("Macro-F1 VADER", f"{metrics['vader']['macro_f1']:.3f}")
        cv = metrics.get("cv_ml_macro_f1", {})
        c4.metric("CV 5 plans (TF-IDF + LR)", f"{cv.get('mean', 0):.3f} ± {cv.get('std', 0):.3f}")
        st.caption(
            f"Jeu : {metrics.get('dataset')} — {metrics.get('n_total')} avis annotés, "
            f"{metrics.get('n_test')} en test. Aucun score n'est inventé."
        )
    st.subheader("Ce que demande le professeur, et où le trouver")
    st.markdown(
        """
        | Exigence | Où dans l'application |
        |---|---|
        | Données : avis publics ou anonymisés | Import (anonymisation) et page *Limites & éthique* |
        | Thèmes récurrents avec exemples | *Thèmes + exemples* (lexique, 24 catégories, LDA) |
        | Lexique contre une classification | *Comparaison* (lexique maison, VADER, TF-IDF + LR) |
        | F1 macro et pertinence des thèmes | *Évaluation* |
        | Analyse d'erreurs (10 à 20 cas) | *Erreurs* |
        | Filtres par type d'avis | Barre latérale : *Type d'avis* |
        """
    )
    st.info(
        "Extension (hors cœur du cahier des charges) : l'analyse par aspect (ABSA) est disponible "
        "sur la page *Analyse*. Le cœur évalué est : sentiment, thèmes, exemples, lexique contre classifieur."
    )


def page_import(df: pd.DataFrame, redactions: int, source: str) -> None:
    st.header("Importation et aperçu")
    if df.empty:
        return
    st.caption(f"Source : {source}")
    cols = st.columns(4)
    cols[0].metric("Avis", len(df))
    if "overall_sentiment" in df.columns:
        cols[1].metric("Annotés", int(df["overall_sentiment"].notna().sum()))
    if "language" in df.columns:
        share_en = 100 * (df["language"] == "en").mean()
        cols[2].metric("Part anglais", f"{share_en:.1f} %")
    cols[3].metric("Remplacements d'anonymisation", redactions)
    st.subheader("Répartition par type d'avis")
    st.bar_chart(df["niveau"].value_counts())
    st.dataframe(df[["id", "niveau", "review_text"]].head(30))
    st.download_button("Exporter les avis anonymisés (CSV)", _csv_bytes(df), "avis_anonymises.csv")


def page_themes(df: pd.DataFrame) -> None:
    st.header("Thèmes récurrents et exemples")
    if df.empty:
        return
    st.caption(
        "Trois méthodes complémentaires. La 1re est transparente, la 2e est supervisée par les "
        "catégories annotées EduRABSA, la 3e est non supervisée (LDA)."
    )
    tab1, tab2, tab3 = st.tabs(["Lexique pédagogique (8 thèmes)", "24 catégories EduRABSA", "Sujets LDA"])
    with tab1:
        theme_df = themes_with_examples(df, n_examples=3)
        st.dataframe(theme_df)
        if not theme_df.empty:
            st.bar_chart(theme_df.set_index("theme")["part_avis_%"])
        st.subheader("Thème × type d'avis (nombre d'avis)")
        exploded = df.assign(theme=df["themes"].str.split(", ")).explode("theme").reset_index(drop=True)
        st.dataframe(pd.crosstab(exploded["theme"], exploded["niveau"]))
    with tab2:
        prof = cached_category_profiles()
        if prof.empty:
            st.warning("Données EduRABSA annotées absentes : lancez scripts/prepare_edurabsa.py.")
        else:
            st.caption(
                "Mots-clés = TF-IDF calculé par catégorie. Opinions = expressions les plus fréquentes. "
                "Exemples = avis qui concentrent le mieux ces mots-clés (3 par catégorie)."
            )
            st.dataframe(prof)
    with tab3:
        n_topics = st.slider("Nombre de sujets LDA", 4, 10, 6)
        topics = cached_lda(tuple(df["review_text"].tolist()), n_topics)
        st.dataframe(topics)
        st.caption("« part_avis_% » = part des avis dont le sujet dominant est ce thème. Ce n'est pas une mesure de qualité.")


def page_analyse(df: pd.DataFrame) -> None:
    st.header("Analyser un commentaire")
    text = st.text_area(
        "Commentaire",
        "The teacher is friendly, but the exams are extremely difficult.",
        height=90,
    )
    if st.button("Lancer l'analyse"):
        lex = analyze_lexicon(text)
        vader_label, vader_compound = analyze_vader(text)
        bundle = cached_bundle()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Lexique maison", lex.label, f"score {lex.score}")
        c2.metric("VADER", vader_label, f"compound {vader_compound}")
        if bundle is not None:
            ml = predict_ml([text], bundle).iloc[0]
            c3.metric("TF-IDF + LR", str(ml["ml_sentiment"]), f"confiance {ml['ml_confidence']:.2f}")
        from src.themes import detect_themes
        c4.metric("Thèmes", ", ".join(detect_themes(text)))
        st.caption("La confiance est une probabilité du modèle, pas une garantie de vérité.")
        st.subheader("Extension ABSA : aspects et opinions")
        aspects = extract_aspects(text)
        st.dataframe(pd.DataFrame(aspects)) if aspects else st.write("Aucun aspect détecté.")

    st.subheader("Thèmes récurrents (aperçu)")
    st.dataframe(themes_with_examples(df, n_examples=2) if not df.empty else pd.DataFrame())


def page_comparaison(df: pd.DataFrame) -> None:
    st.header("Comparaison : lexique contre classification")
    st.markdown(
        "Exigence du professeur : **un lexique** contre **une méthode de classification**. "
        "Lexique maison (négations, intensifieurs), VADER (lexique anglais de référence), "
        "TF-IDF + régression logistique (classifieur entraîné)."
    )
    metrics = _read_json(METRICS)
    if metrics:
        st.subheader("Résultats sur le jeu de test (officiels)")
        table = pd.concat([
            metrics_table(metrics["lexicon"], "Lexique maison"),
            metrics_table(metrics["vader"], "VADER"),
            metrics_table(metrics["ml"], "TF-IDF + Logistic Regression"),
        ], ignore_index=True)
        st.dataframe(table)
        per_lang = metrics.get("par_langue_macro_f1_test", {})
        if per_lang:
            st.caption("Macro-F1 par langue (groupes de test d'au moins 20 avis) :")
            st.json(per_lang)
    if df.empty:
        return
    analyzed = cached_analysis(tuple(df["review_text"].tolist()))
    merged = df.reset_index(drop=True).copy()
    for col in analyzed.columns:
        if col not in {"review_text"}:
            merged[col] = analyzed[col]

    c1, c2 = st.columns(2)
    sent_filter = c1.multiselect("Sentiment lexique", LABELS, default=LABELS)
    theme_filter = c2.text_input("Thème (contient)")
    view = merged[merged["lexicon_sentiment"].isin(sent_filter)]
    if theme_filter.strip():
        view = view[view["themes"].str.contains(theme_filter, case=False, na=False)]
    show_cols = [c for c in [
        "id", "niveau", "language", "review_text", "gold" if "overall_sentiment" in view.columns else None,
        "lexicon_sentiment", "vader_sentiment", "ml_sentiment", "themes", "ml_confidence",
    ] if c and c in view.columns]
    st.dataframe(view[show_cols])
    left, right = st.columns(2)
    with left:
        st.subheader("Distribution : lexique maison")
        st.bar_chart(view["lexicon_sentiment"].value_counts())
    with right:
        st.subheader("Distribution : TF-IDF + LR")
        st.bar_chart(view["ml_sentiment"].value_counts())
    st.download_button("Exporter les analyses (CSV)", _csv_bytes(view[show_cols]), "analyses_studentvoice.csv")


def page_dashboard(df: pd.DataFrame) -> None:
    st.header("Dashboard")
    if df.empty:
        return
    analyzed = cached_analysis(tuple(df["review_text"].tolist()))
    c1, c2, c3 = st.columns(3)
    c1.metric("Avis analysés", len(analyzed))
    c2.metric("Positifs (ML)", int((analyzed["ml_sentiment"] == "Positive").sum()) if "ml_sentiment" in analyzed else 0)
    c3.metric("Négatifs (ML)", int((analyzed["ml_sentiment"] == "Negative").sum()) if "ml_sentiment" in analyzed else 0)
    if "ml_sentiment" in analyzed:
        st.subheader("Sentiment par type d'avis (classifieur)")
        by_level = pd.crosstab(df["niveau"].values, analyzed["ml_sentiment"].values)
        st.bar_chart(by_level)
    st.subheader("Longueur des commentaires (nombre de mots)")
    lengths = df["review_text"].str.split().str.len().fillna(0).astype(int)
    st.bar_chart(lengths.value_counts().sort_index())
    st.subheader("Thèmes")
    st.bar_chart(themes_with_examples(df, n_examples=1).set_index("theme")["part_avis_%"])


def page_evaluation(df: pd.DataFrame) -> None:
    st.header("Évaluation")
    st.caption("F1 macro. Aucun score n'est inventé : tout est calculé sur des avis que le classifieur n'a pas vus.")
    metrics = _read_json(METRICS)
    st.subheader("1. Jeu de test officiel")
    if metrics:
        st.markdown(f"Jeu : **{metrics['dataset']}** — {metrics['split']}.")
        st.dataframe(pd.concat([
            metrics_table(metrics["lexicon"], "Lexique maison"),
            metrics_table(metrics["vader"], "VADER"),
            metrics_table(metrics["ml"], "TF-IDF + Logistic Regression"),
        ], ignore_index=True))
        cv = metrics.get("cv_ml_macro_f1", {})
        st.write(f"Validation croisée 5 plans, TF-IDF + LR : macro-F1 = {cv.get('mean')} ± {cv.get('std')} (plans : {cv.get('folds')}).")
        st.write(f"Distribution du test : {metrics.get('distribution_test')}")
    else:
        st.warning("reports/metrics.json absent : lancez python -m src.train_sentiment_ml")

    st.subheader("2. Évaluation sur les avis de la source courante (hors entraînement)")
    bundle = cached_bundle()
    if df.empty or "overall_sentiment" not in df.columns or bundle is None:
        st.info("Source sans annotations ou modèle absent.")
    else:
        seen = set(bundle.get("train_ids", []))
        gold = df[df["overall_sentiment"].notna() & ~df["id"].isin(seen)]
        st.write(f"Avis annotés hors entraînement : **{len(gold)}** (les avis d'entraînement sont exclus).")
        if len(gold) >= 10:
            lex_m = classification_metrics(gold["overall_sentiment"], [analyze_lexicon(t).label for t in gold["review_text"]])
            vad_m = classification_metrics(gold["overall_sentiment"], [analyze_vader(t)[0] for t in gold["review_text"]])
            ml_pred = predict_ml(gold["review_text"].tolist(), bundle)["ml_sentiment"]
            ml_m = classification_metrics(gold["overall_sentiment"], ml_pred)
            st.dataframe(pd.concat([
                metrics_table(lex_m, "Lexique maison"),
                metrics_table(vad_m, "VADER"),
                metrics_table(ml_m, "TF-IDF + LR"),
            ], ignore_index=True))
            st.caption("Matrice de confusion (TF-IDF + LR) : lignes = vérité, colonnes = prédiction.")
            st.dataframe(pd.DataFrame(ml_m["confusion_matrix"], index=LABELS, columns=LABELS))

    st.subheader("3. Pertinence des thèmes (évaluation humaine)")
    accord = _read_json(ACCORD)
    if accord:
        st.json(accord)
    else:
        st.info(
            "Pas encore de résultat. Chaque membre note les 30 thèmes dans "
            f"`{GRILLE.relative_to(ROOT)}` (échelle 1 à 3), puis lancez : "
            "`python -m src.theme_eval agree <grille_remplie.csv>`."
        )
    if GRILLE.exists():
        st.download_button("Télécharger la grille à noter", GRILLE.read_bytes(), GRILLE.name)


def page_erreurs() -> None:
    st.header("Analyse d'erreurs")
    st.caption("Erreurs du jeu de test. Étiquettes automatiques (négation, neutre confondu, inversion…) puis lecture humaine.")
    if ERRORS.exists():
        errors = pd.read_csv(ERRORS, keep_default_na=False)
        st.write(f"Total d'erreurs (toutes méthodes) : **{len(errors)}**")
        st.dataframe(errors.groupby(["methode", "type_erreur"]).size().rename("n").reset_index())
        methods = sorted(errors["methode"].unique())
        choice = st.selectbox("Méthode", methods, index=methods.index("TF-IDF + LR") if "TF-IDF + LR" in methods else 0)
        types = sorted(errors["type_erreur"].unique())
        chosen_types = st.multiselect("Type d'erreur", types, default=types)
        view = errors[(errors["methode"] == choice) & (errors["type_erreur"].isin(chosen_types))]
        st.dataframe(view[["id", "niveau", "gold", "prediction", "type_erreur", "texte"]])
        st.download_button("Exporter les erreurs (CSV)", _csv_bytes(view), "erreurs_filtrees.csv")
    else:
        st.warning("reports/erreurs_detaillees.csv absent : lancez python -m src.error_analysis")
    if ERREURS_COMMENTEES.exists():
        st.subheader("Erreurs commentées")
        st.markdown(_read_text(ERREURS_COMMENTEES))


def page_limites() -> None:
    st.header("Limites et éthique")
    st.markdown(_read_text(LIMITES))
    st.divider()
    st.markdown(_read_text(ETHIQUE))


# ---------------------------------------------------------------- navigation

def main() -> None:
    st.sidebar.title("StudentVoice AI")
    page = st.sidebar.radio("Navigation", PAGES)
    if page == "Accueil":
        page_accueil()
        return
    if page == "Limites & éthique":
        page_limites()
        return
    if page == "Erreurs":
        page_erreurs()
        return
    df, redactions, source = get_dataset()
    if page == "Import":
        page_import(df, redactions, source)
    elif page == "Thèmes + exemples":
        page_themes(df)
    elif page == "Analyse":
        page_analyse(df)
    elif page == "Comparaison":
        page_comparaison(df)
    elif page == "Dashboard":
        page_dashboard(df)
    elif page == "Évaluation":
        page_evaluation(df)


if __name__ == "__main__":
    main()
