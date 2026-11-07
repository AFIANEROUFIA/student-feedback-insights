"""Thèmes récurrents avec exemples (exigence du professeur).

Trois méthodes complémentaires :
1. Lexique pédagogique de 8 thèmes (règles transparentes, applicable à tout texte).
2. Profils par catégorie EduRABSA (24 catégories annotées) : TF-IDF par catégorie,
   opinions les plus fréquentes et 2 ou 3 exemples par catégorie.
3. LDA (modèle de sujets non supervisé) sur le corpus.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .preprocessing import clean_text

THEME_LEXICON = {
    "Enseignant / attitude": [
        "teacher", "professor", "prof", "enseignant", "enseignante", "personnel",
        "attitude", "rude", "friendly", "helpful", "disponible", "impoli", "gentil",
        "professeur", "intervenant",
    ],
    "Cours / contenu": [
        "course", "cours", "content", "contenu", "chapitre", "chapter", "lecture", "class",
        "matiere", "matière", "programme", "syllabus", "notion",
    ],
    "Examens / évaluation": [
        "exam", "exams", "examen", "examens", "test", "évaluation", "evaluation", "assessment",
        "notation", "grading", "grade", "note", "qcm", "contrôle", "controle",
    ],
    "Difficulté": [
        "difficult", "difficulty", "hard", "easy", "difficile", "facile", "niveau",
        "charge", "workload", "lourd", "complexe",
    ],
    "Organisation": [
        "organisation", "organization", "planning", "schedule", "emploi", "retard",
        "late", "chaotic", "chaotique", "communication", "annonce",
    ],
    "Ressources pédagogiques": [
        "slides", "support", "supports", "moodle", "resource", "ressources",
        "document", "polycopie", "polycopié", "video", "vidéo", "handout",
    ],
    "Infrastructures": [
        "salle", "classroom", "wifi", "equipment", "équipement", "amphi", "library",
        "bibliotheque", "bibliothèque", "ordinateur", "lab", "laboratoire",
    ],
    "Charge de travail": [
        "homework", "devoir", "projet", "project", "assignment", "deadline",
        "travail", "heures", "hours", "workload",
    ],
}

# Mots vides FR/EN pour TF-IDF et LDA.
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were", "be",
    "this", "that", "it", "for", "on", "with", "at", "by", "from", "as", "not", "very",
    "too", "but", "if", "so", "we", "i", "you", "they", "he", "she", "his", "her", "its",
    "my", "your", "our", "their", "do", "does", "did", "have", "has", "had", "there",
    "than", "then", "which", "who", "what", "when", "where", "how", "all", "any", "can",
    "will", "would", "could", "should", "just", "also", "about", "into", "over", "more",
    "some", "such", "only", "own", "same", "other", "each", "few", "most", "much", "no",
    "nor", "up", "down", "out", "off", "again", "once", "here", "why", "am", "been", "being",
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "est", "sont", "ce",
    "cet", "cette", "ces", "il", "elle", "ils", "elles", "nous", "vous", "pas", "plus",
    "dans", "pour", "avec", "sur", "par", "en", "au", "aux", "je", "tu", "qui", "que",
    "quoi", "ne", "se", "sa", "son", "ses", "leur", "leurs", "mais", "donc", "car", "very",
    "etre", "être", "avoir", "fait", "faire", "tres", "très", "bien", "aussi", "tout", "tous",
    # jetons produits par l'anonymisation (src/anonymize.py) : ce ne sont pas des thèmes
    "nom", "email", "tel", "pseudo", "url",
}

_TOKEN = r"(?u)\b[^\W\d_][^\W\d_'’-]+\b"


def detect_themes(text: str) -> list[str]:
    lowered = clean_text(text)
    found = []
    for theme, keywords in THEME_LEXICON.items():
        for raw in keywords:
            key = raw.strip().lower()
            if key and re.search(rf"(?<!\w){re.escape(key)}(?!\w)", lowered):
                found.append(theme)
                break
    return found or ["Autre / général"]


def themes_with_examples(df: pd.DataFrame, text_col: str = "review_text", n_examples: int = 3) -> pd.DataFrame:
    """Thèmes du lexique pédagogique : fréquence, part des avis et exemples."""
    buckets: dict[str, list[str]] = defaultdict(list)
    counts: Counter = Counter()
    n = 0
    for text in df[text_col].fillna("").astype(str):
        n += 1
        for theme in detect_themes(text):
            counts[theme] += 1
            if len(buckets[theme]) < n_examples:
                buckets[theme].append(text.strip().replace("\n", " ")[:240])
    rows = []
    for theme, count in counts.most_common():
        rows.append({
            "theme": theme,
            "frequence": count,
            "part_avis_%": round(100.0 * count / max(n, 1), 1),
            "exemples": " | ".join(buckets[theme]),
        })
    return pd.DataFrame(rows, columns=["theme", "frequence", "part_avis_%", "exemples"])


def _example_score(text_tokens: list[str], weights: dict[str, float]) -> float:
    return sum(weights.get(t, 0.0) for t in text_tokens)


def category_profiles(
    quads: pd.DataFrame,
    n_examples: int = 3,
    n_terms: int = 8,
    n_opinions: int = 5,
) -> pd.DataFrame:
    """Un profil par catégorie EduRABSA.

    `quads` a une ligne par quadruplet, avec les colonnes : text, category, opinion.
    - mots_cles : termes les plus caractéristiques de la catégorie (TF-IDF calculé
      sur un document par catégorie, donc « TF-IDF par catégorie ») ;
    - opinions_frequentes : expressions d'opinion les plus fréquentes ;
    - exemples : n_examples phrases d'avis qui concentrent le mieux ces mots-clés.
    """
    if quads.empty:
        return pd.DataFrame(columns=["categorie", "n_quadruplets", "mots_cles", "opinions_frequentes", "exemples"])
    quads = quads.dropna(subset=["category"]).copy()
    quads["opinion_norm"] = quads["opinion"].fillna("").astype(str).str.lower().str.strip(" .!?,;")
    category_docs = quads.groupby("category")["text"].apply(lambda s: " ".join(s.astype(str).unique()))
    vectorizer = TfidfVectorizer(
        stop_words=sorted(STOPWORDS),
        token_pattern=_TOKEN,
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(category_docs.values)
    vocab = vectorizer.get_feature_names_out()
    vocab_index = {term: j for j, term in enumerate(vocab)}
    analyzer = vectorizer.build_analyzer()

    rows = []
    for i, category in enumerate(category_docs.index):
        weights_vec = matrix[i].toarray().ravel()
        order = weights_vec.argsort()[::-1]
        top_terms = [vocab[j] for j in order[:n_terms] if weights_vec[j] > 0]
        weights = {vocab[j]: float(weights_vec[j]) for j in order[:200] if weights_vec[j] > 0}

        cat_quads = quads[quads["category"] == category]
        opinions = (
            cat_quads.loc[cat_quads["opinion_norm"].ne(""), "opinion_norm"]
            .value_counts()
            .head(n_opinions)
        )
        cat_texts = cat_quads["text"].astype(str).drop_duplicates()
        scored = []
        for text in cat_texts:
            tokens = analyzer(text)
            score = _example_score(tokens, weights)
            length = len(text)
            if 40 <= length <= 300:  # on privilégie des phrases lisibles
                scored.append((score, -length, text))
        scored.sort(reverse=True)
        examples = [t for _, _, t in scored[:n_examples]]
        rows.append({
            "categorie": category,
            "n_quadruplets": int(len(cat_quads)),
            "mots_cles": ", ".join(top_terms),
            "opinions_frequentes": " | ".join(f"{op} ({cnt})" for op, cnt in opinions.items()),
            "exemples": " | ".join(examples),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("n_quadruplets", ascending=False).reset_index(drop=True)


def lda_topics(
    texts: list[str],
    n_topics: int = 6,
    n_words: int = 8,
    n_examples: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """LDA non supervisé : mots-clés, part des avis dont le sujet dominant est ce thème, exemples."""
    corpus = [clean_text(t) for t in texts if str(t).strip()]
    raw = [t for t in texts if str(t).strip()]
    if len(corpus) < max(n_topics * 3, 20):
        return pd.DataFrame(columns=["theme_id", "mots_cles", "part_avis_%", "exemples"])
    vectorizer = CountVectorizer(
        stop_words=sorted(STOPWORDS), token_pattern=_TOKEN, min_df=3, max_df=0.6, max_features=4000,
    )
    X = vectorizer.fit_transform(corpus)
    lda = LatentDirichletAllocation(
        n_components=n_topics, random_state=random_state, learning_method="batch", max_iter=25,
    )
    doc_topic = lda.fit_transform(X)
    vocab = vectorizer.get_feature_names_out()
    dominant = doc_topic.argmax(axis=1)
    rows = []
    for k, component in enumerate(lda.components_):
        top = component.argsort()[::-1][:n_words]
        members = np.where(dominant == k)[0]
        share = 100.0 * len(members) / len(corpus)
        best = members[np.argsort(-doc_topic[members, k])][:n_examples] if len(members) else []
        rows.append({
            "theme_id": f"LDA {k + 1}",
            "mots_cles": ", ".join(vocab[j] for j in top),
            "part_avis_%": round(share, 1),
            "exemples": " | ".join(raw[int(i)][:240] for i in best),
        })
    return pd.DataFrame(rows).sort_values("part_avis_%", ascending=False).reset_index(drop=True)
