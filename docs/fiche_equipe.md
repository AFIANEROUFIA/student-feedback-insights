# Avant la prochaine séance : une page par équipe

**Projet 17 — StudentVoice AI**
**3 étudiants** — Analyse des avis étudiants (thèmes et sentiments)

## Membres et trois choix classés
À compléter par l'équipe :

| Rang | Projet | Pourquoi |
|---|---|---|
| 1 | **Projet 17 — Analyse des avis étudiants** | Données textuelles réelles (EduRABSA), NLP, comparaison lexique et classifieur, prototype visualisable |
| 2 | *(2e choix de l'équipe)* | |
| 3 | *(3e choix de l'équipe)* | |

Membres : 1. ________   2. ________   3. ________

## Problème précis
Comment extraire automatiquement, à partir d'avis d'étudiants anonymisés, **les thèmes récurrents** (enseignant, cours, examens, organisation, ressources…) et **le sentiment** (positif, négatif, neutre), puis les présenter avec des **exemples** exploitables par une équipe pédagogique ?

## Premier prototype
Application Streamlit **StudentVoice AI** (`streamlit run app.py`) :
1. Import CSV ou JSON, anonymisation à l'import, filtre par type d'avis (enseignant, cours, université).
2. Thèmes récurrents avec exemples : lexique pédagogique, 24 catégories EduRABSA, sujets LDA.
3. Sentiment par lexique maison, VADER, et TF-IDF + régression logistique.
4. Comparaison, F1 macro, matrice de confusion, analyse d'erreurs.
5. Extension : analyse par aspect (ABSA) et export CSV.

## Source de données et référence
- **Source** : EduRABSA (6 500 avis, licence CC-BY), anonymisé avant analyse. Les 90 textes de l'échantillon synthétique sont fictifs et ne sont pas utilisés pour les résultats.
- **Référence** : approches classiques de sentiment (Pang & Lee) ; EduRABSA pour le jeu annoté ; VADER (Hutto & Gilbert, 2014) pour le lexique de référence.

## Critères de réussite
- Le prototype affiche thèmes, sentiments et exemples.
- Comparaison chiffrée lexique contre classifieur (accuracy, precision, recall, macro-F1).
- Pertinence des thèmes : grille notée par les 3 membres, accord calculé (kappa de Fleiss).
- Aucun score inventé ; split train et test documenté ; limites écrites.
- Interface utilisable en démonstration : import, filtres, export.

## Répartition des tâches
| Membre | Rôle |
|---|---|
| 1 | Données, anonymisation, analyse exploratoire, langues |
| 2 | Lexique (maison et VADER), TF-IDF + régression logistique, métriques, analyse d'erreurs |
| 3 | Streamlit, dashboard, exemples par thème, grille de pertinence, export, démonstration |

## À apporter
- `data/raw/echantillon_seance_50.csv` (50 avis anonymisés).
- Grille `docs/evaluation_themes/grille_pertinence_30_themes.csv`, notée individuellement par chaque membre.
