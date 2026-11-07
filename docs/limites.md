# Limites du prototype

Cette page fait partie du rendu. Elle décrit ce que le système ne fait pas, et ce qu'il ne faut pas en conclure.

## 1. Biais anglophone
Le jeu EduRABSA est à **97,6 % en anglais** (6 336 avis sur 6 490 après nettoyage). Sur le jeu de test, il n'y a que **7 avis français** : aucune mesure fiable n'existe pour le français. Le lexique maison contient des mots français, mais ils n'ont pas été évalués de façon statistiquement utile. VADER est conçu pour l'anglais : sur le français, il renvoie surtout « Neutral ».
Conséquence : **aucune garantie n'est donnée sur les avis en français**. La piste propre est FEEL (lexique français) ou un jeu annoté en français.

## 2. La classe « Neutral » est difficile
Dans le jeu EduRABSA, la classe neutre correspond à « Conf/Neu » (confus / neutre), soit 823 avis avant nettoyage. Sur le jeu de test (206 avis neutres) :

| Méthode | F1 Negative | F1 Neutral | F1 Positive | Macro-F1 |
|---|---|---|---|---|
| TF-IDF + régression logistique | 0,694 | 0,415 | 0,769 | 0,626 |
| Lexique maison | 0,517 | 0,171 | 0,749 | 0,479 |
| VADER | 0,591 | 0,070 | 0,795 | 0,485 |

Les avis mélangés (« le cours est bon mais l'examen est injuste ») sont souvent annotés neutres, alors qu'un lecteur peut les lire comme positifs ou négatifs. Le macro-F1 pénalise fortement cette classe.

## 3. Bruit d'annotation
Certaines étiquettes sont discutables : « Kermit the Frog! » est annoté négatif, « Good class... Definitely recommend » est annoté neutre. L'analyse d'erreurs (docs/analyse_erreurs.md) en cite plusieurs. Le score mesure l'accord avec ces étiquettes, pas la vérité absolue.

## 4. Négation et portée
Les négations sont gérées par le lexique maison (« not », « pas », « n'est pas », contractions). Le classifieur TF-IDF est un **sac de mots** : il ne voit pas la portée de la négation. Dans l'analyse d'erreurs, 261 erreurs du classifieur contiennent une négation.

## 5. Sarcasme et ironie
Aucune erreur du jeu de test n'a été étiquetée « sarcasme possible » par l'heuristique (motifs de type « yeah right », « thanks a lot »). Cela signifie que **l'heuristique est trop faible pour mesurer le sarcasme**, pas qu'il est absent. Le sarcasme n'est pas traité.

## 6. Vocabulaire hors lexique
Le lexique maison ne couvre que quelques centaines de mots. Des formulations pourtant claires sont manquées : « doesn't help » est neutre pour le lexique, « sick », « excuses » ne sont pas reconnus. Le lexique est transparent mais limité ; c'est un argument pour la comparaison, pas un produit fini.

## 7. Échantillon synthétique
Les 90 textes de `data/raw/avis_etudiants_sample.csv` ont été **rédigés pour le prototype**. Ce ne sont pas des avis réels. Le lexique maison a été écrit en regardant ces phrases : ses scores sur cet échantillon (macro-F1 de l'ordre de 0,87 dans une version antérieure) sont **optimistes et ne doivent pas être cités**. Les résultats de référence sont ceux de EduRABSA (docs/limites.md, tableau ci-dessus).

## 8. Anonymisation imparfaite
L'anonymisation est une heuristique par règles (voir docs/ethique_anonymisation.md). Des noms en minuscules ou en MAJUSCULES échappent au filtre, et des mots capitalisés qui ne sont pas des noms peuvent être masqués. Un avis entièrement en casse de titre n'est pas traité par la règle des noms en milieu de phrase.

## 9. Thèmes
- Les **24 catégories EduRABSA** sont des catégories d'annotation, pas des thèmes validés par l'équipe pédagogique.
- La **LDA** donne des sujets dont l'interprétation reste humaine ; le pourcentage d'avis par sujet n'est pas une mesure de qualité.
- La **pertinence** des thèmes doit être jugée par les trois membres de l'équipe (grille docs/evaluation_themes/). Tant que cette grille n'est pas remplie, aucun accord n'est annoncé.

## 10. Périmètre
- L'analyse par aspect (ABSA) est une **extension**, non évaluée quantitativement dans ce rendu.
- XLM-RoBERTa (`src/train_sentiment_dl.py`) n'a pas été entraîné ni évalué. Il ne doit pas être présenté comme un résultat.
- Le système ne doit pas servir à évaluer un enseignant individuellement : c'est une aide à la lecture de retours collectifs.
