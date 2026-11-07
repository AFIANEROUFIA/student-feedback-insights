# Éthique, anonymisation et licence

## Données utilisées
- **EduRABSA** (dépôt fourni avec le cours) : 6 500 avis (`combined_data_6.5k.csv`) et deux fichiers d'annotations ASQE (4 000 et 2 500 avis). Les avis proviennent de sites d'évaluation de cours et d'enseignants, et les identifiants d'avis ne sont pas des noms de personnes.
- **Licence** : CC-BY (fichier `data/LICENSE-CC-BY.txt`). La citation du jeu de données doit figurer dans le rapport, avec la licence.
- **Échantillon synthétique** (`data/raw/avis_etudiants_sample.csv`) : 90 textes rédigés pour le prototype. Ils ne proviennent d'aucune personne réelle.
- **Échantillon de séance** (`data/raw/echantillon_seance_50.csv`) : 50 avis anonymisés, tirés du jeu EduRABSA anonymisé, avec quotas par type d'avis.

## Ce qui est fait avant toute analyse
`scripts/prepare_edurabsa.py` produit des copies anonymisées : les fichiers bruts (avec noms et pseudos) ne sont pas copiés dans le dépôt. `src/anonymize.py` applique, dans l'ordre :

1. e-mails → `[EMAIL]` ; URLs → `[URL]` ; pseudos (`@pseudo`, `u/pseudo`) → `[PSEUDO]` ;
2. numéros de téléphone (9 chiffres ou plus) → `[TEL]` ; les années comme « 2023-2024 » sont conservées ;
3. noms précédés d'un titre (Dr, Prof, Mr, Mme, M.…) → le titre est gardé, le nom devient `[NOM]` ;
4. mots en casse de titre en milieu de phrase, hors liste de mots à garder (jours, matières, outils, rôles comme « teacher ») → `[NOM]`. Un texte entièrement en casse de titre n'est pas traité par cette règle, pour ne pas effacer les mots de sentiment.

Volumes sur le jeu EduRABSA : 2 122 remplacements dans les avis (6 490 avis), 2 347 et 1 582 dans les deux fichiers ASQE. Les chiffres exacts sont dans `data/raw/anonymisation_rapport.json`.

## Limites de l'anonymisation
- C'est une **heuristique**. Un nom en minuscules, ou un nom mal capitalisé, passe. Un lieu ou un sigle peut être masqué à tort.
- Pendant la mise au point, une douzaine d'exemples ont été relus à la main pour ajuster les règles. Ce n'est pas un audit systématique : une relecture d'un échantillon plus large est nécessaire avant toute diffusion.
- Une option spaCy (entités PERSON) est prévue (`use_spacy=True`). Elle n'a **pas été testée** dans cet environnement.
- Les textes anonymisés restent des avis personnels. Ne pas tenter de réidentifier une personne à partir d'un avis.

## Usage pédagogique
- Le système analyse des **retours collectifs** pour aider une équipe pédagogique à repérer des thèmes. Il ne sert pas à noter un enseignant ou à prendre une décision individuelle.
- Les exemples affichés dans l'app sont des extraits d'avis anonymisés. Ils ne doivent pas être publiés hors de l'équipe sans vérification.
- Les avis ont été écrits par des tiers, sans consentement spécifique au projet. Le cadre est celui de la licence CC-BY et de la publication d'origine.

## Ce que l'application ne fait pas
- Elle ne conserve aucun identifiant d'auteur.
- Les fichiers importés sont écrits dans `data/processed/` (ignoré par git) et anonymisés avant tout affichage.
