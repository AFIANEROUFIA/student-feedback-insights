# Évaluation de la pertinence des thèmes

## Fichier
`grille_pertinence_30_themes.csv` : 30 thèmes candidats.
- T01 à T24 : les 24 catégories annotées dans EduRABSA (ex. « Course - Assessment », « Staff - Attitude »), avec leurs mots-clés (TF-IDF par catégorie) et deux exemples d'avis.
- T25 à T30 : six sujets LDA, avec leurs mots-clés et deux exemples.

## Consigne pour chaque membre
Pour chaque thème, notez de **1 à 3** sa pertinence pour un retour pédagogique :
- **1** : non pertinent (le thème ne correspond pas aux avis, ou il est mal défini) ;
- **2** : partiellement pertinent (le thème existe mais les exemples sont mélangés à d'autres sujets) ;
- **3** : pertinent (le thème est clair et les exemples l'illustrent).

Notez **indépendamment**, sans concertation, en vous appuyant sur les mots-clés et les exemples. Remplissez votre colonne (`membre_1`, `membre_2` ou `membre_3`) et n'altérez pas les autres colonnes.

## Calcul de l'accord
Enregistrez la grille remplie (par exemple `grille_remplie.csv`, même dossier), puis lancez :

    python -m src.theme_eval agree docs/evaluation_themes/grille_remplie.csv

Le rapport (`reports/accord_themes.json`) donne :
- le **kappa de Fleiss** (3 évaluateurs, catégories 1, 2, 3) ;
- la moyenne des **kappas pondérés quadratiques** deux à deux, adaptée à une échelle ordonnée ;
- l'accord exact (en %) ;
- la liste des thèmes dont la moyenne est inférieure à 2 : candidats à la fusion ou au retrait.

Interprétation (Landis & Koch) : moins de 0,2 faible ; 0,2 à 0,4 passable ; 0,4 à 0,6 modéré ; 0,6 à 0,8 substantiel ; au-delà, presque parfait.

## Règle de rendu
Les notes doivent être saisies à la main. Le module ne génère aucune note : sans grille remplie, il ne calcule rien. Rapportez le kappa tel quel, même s'il est faible.
