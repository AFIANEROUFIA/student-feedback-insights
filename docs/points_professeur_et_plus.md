# Exigences du professeur : état du rendu

| Exigence du professeur | État dans le rendu | Où |
|---|---|---|
| **Données** : textes publics ou avis anonymisés | Fait. EduRABSA (CC-BY) anonymisé à l'import ; éthique documentée. | `scripts/prepare_edurabsa.py`, `src/anonymize.py`, `docs/ethique_anonymisation.md` |
| **Prototype** : thèmes récurrents et sentiments avec exemples | Fait. Thèmes par lexique pédagogique, par 24 catégories EduRABSA (TF-IDF par catégorie + 3 exemples), et LDA. | `src/themes.py`, page *Thèmes + exemples* |
| **Comparaison** : un lexique contre une méthode de classification | Fait. Lexique maison et VADER comparés à TF-IDF + régression logistique. | `src/lexicon.py`, `src/train_sentiment_ml.py`, page *Comparaison* |
| **Évaluation** : F1 macro et pertinence des thèmes | F1 macro fait (test + validation croisée). Pertinence : grille prête, **notes humaines à saisir par l'équipe**. | `reports/metrics.json`, `src/theme_eval.py`, `docs/evaluation_themes/` |
| **Avant la séance** : une page par équipe | Fait, à compléter : membres et choix classés. | `docs/fiche_equipe.md` |
| **Apporter un échantillon** | Fait : 50 avis anonymisés. | `data/raw/echantillon_seance_50.csv` |

## Points demandés en complément

| Point | État | Remarque |
|---|---|---|
| 1. Baseline lexique : VADER ou FEEL | VADER fait. FEEL non fait. | VADER est anglais ; FEEL serait la piste pour le français (voir limites). |
| 2. Thèmes : BERTopic ou LDA + TF-IDF par catégorie | LDA et TF-IDF par catégorie faits. BERTopic non fait. | Choix LDA : plus léger, sans GPU. |
| 3. Évaluation des thèmes par 3 membres, accord | Outil prêt. Notes non saisies. | Aucun accord n'est annoncé tant que la grille n'est pas remplie. |
| 4. Anonymisation, CC-BY | Fait (règles regex). spaCy non testé. | Heuristique : voir les limites. |
| 5. Analyse d'erreurs : 10 à 20 cas commentés | Fait : 20 cas commentés. | `docs/analyse_erreurs.md`. Pas de cas de sarcasme détecté. |
| 6. Dashboard : onglet thèmes + exemples, filtres par type d'avis | Fait. Filtre *Type d'avis* : Enseignant, Cours, Université. | Type dérivé du champ `type` d'EduRABSA. |
| 7. Section limites | Fait. | `docs/limites.md` |
| 8. ABSA présentée comme extension | Fait. ABSA sur la page *Analyse*, non évaluée. | Le cœur est : sentiment, thèmes, exemples, lexique contre classifieur. |

## Résultats à citer, et ceux à ne pas citer

À citer (jeu de test EduRABSA, 1 612 avis) : TF-IDF + LR macro-F1 **0,626** (CV 0,624 ± 0,014) ; lexique maison **0,479** ; VADER **0,485**. Classe neutre : F1 0,42 pour le classifieur.

À ne pas citer : le score de 0,87 obtenu sur l'échantillon synthétique dans une version antérieure. Le lexique a été écrit sur ces phrases, le chiffre est optimiste.
