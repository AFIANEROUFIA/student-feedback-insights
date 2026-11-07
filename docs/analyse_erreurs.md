# Analyse d'erreurs (20 cas commentés)

Source : `reports/erreurs_selection.md`, généré par `python -m src.error_analysis` à partir du jeu de test (TF-IDF + régression logistique, 1 612 avis). Les types d'erreurs sont des étiquettes automatiques ; les commentaires ci-dessous sont une lecture humaine de chaque cas.

Répartition des erreurs du classifieur sur le test : 261 contiennent une négation ; 107 sont des neutres pris pour une polarité ; 213 sont des polarités inversées ; 172 sont des polarités manquées (neutre prédit).

**Sélection** : 14 neutres confondus avec une polarité, 4 polarités inversées, 2 polarités manquées. Aucun cas de sarcasme n'a été détecté par l'heuristique, donc il n'y en a pas dans cette sélection (voir docs/limites.md, §5).

## A. Neutre confondu avec une polarité (14 cas)

Dans la plupart de ces cas, l'erreur vient de l'étiquette autant que du modèle : le texte exprime un avis clair, ou au contraire un avis mélangé que la classe neutre ne représente pas bien.

1. **1_108_753_4214** — « [NOM] is a really good teacher … I highly recommend [NOM]! » (gold Neutral, prédit Positive). Avis nettement positif. Erreur d'annotation probable.
2. **1_11_1855_6496** — « this guy is horrible, would advise not taking his classes, biggest jerk in the world » (gold Neutral, prédit Negative). Avis très négatif. Erreur d'annotation nette : le modèle a raison.
3. **1_130_1457_11714** — « good instructor … Book is horrible … class isn't too hard, but attendance is mandatory » (gold Neutral, prédit Negative). Avis mélangé : deux points positifs, deux négatifs. Le mot « horrible » tire la prédiction vers le négatif. Ce cas illustre la difficulté de la classe neutre.
4. **1_134_758_7425** — « Too much reading. » (gold Neutral, prédit Negative). Phrase courte à polarité négative implicite. Le modèle lit correctement le sens ; l'étiquette neutre est discutable.
5. **1_137_1774_4257** — « Super calm teacher … study guides, they really help in preparation for the tests » (gold Neutral, prédit Positive). Avis favorable. Le modèle est plausible ; l'étiquette est discutable.
6. **1_141_921_14309** — « Hard homeworks and easy tests. Do not buy the textbook as it is useless. You only need his notes. » (gold Neutral, prédit Negative). Avis mélangé avec une négation (« Do not buy ») : le modèle retient « useless » sans en voir la portée sur le livre.
7. **1_149_245_7847** — « Horvit is annoying … boring … useless amount of reading. » (gold Neutral, prédit Negative). Avis nettement négatif. Erreur d'annotation probable.
8. **1_153_1130_1810** — « If you do the work it's not too bad, but … your grade drops significantly » (gold Neutral, prédit Positive). Tournure atténuée : « not too bad » est une évaluation modérée et le modèle la lit comme positive. Un lexique à négations isolerait mal ce cas.
9. **1_153_549_1421** — « Cares that you understand the material … Wish she taught more classes. » (gold Neutral, prédit Positive). Avis positif, avec une pointe de regret. Étiquette neutre discutable.
10. **1_156_158_6066** — « Amazing prof! … Can be arrogant and sarcastic, but is still overall entertaining » (gold Neutral, prédit Positive). Avis globalement positif malgré un défaut. Le modèle est raisonnable ; l'étiquette neutre est difficile à justifier.
11. **1_164_52_15657** — « Brought real world experience … Too much time spent on Excel stuff. » (gold Neutral, prédit Negative). Avis mélangé. Le modèle retient la critique.
12. **1_165_2684_14840** — « not as bad as people say … has some hard concepts … accent is extremely hard to understand » (gold Neutral, prédit Negative). Le mot « hard » est répété et sert ici à décrire la difficulté des cours, pas un jugement sur l'enseignant. Le modèle le lit comme négatif. Limite du sac de mots, et de la négation « not as bad » qu'il ne prend pas en compte.
13. **1_165_2684_14842** — « Good class … Definitely recommend. » (gold Neutral, prédit Positive). Avis clairement positif. Erreur d'annotation nette.
14. **1_172_1852_16632** — « He is easy going. However, he did not give any outline in class. » (gold Neutral, prédit Negative). Contraste (« However ») et négation (« did not give ») : le premier point est positif, le second négatif. Le modèle ne sait pas les pondérer.

## B. Polarité inversée (4 cas)

15. **1_101_1146_3644** — « Kermit the Frog! » (gold Negative, prédit Positive). Aucun contenu évaluatif. Le label ne peut pas être justifié à partir du texte. Bruit d'annotation.
16. **1_149_568_12328** — « This is by far my favorite class … I learned alot and didn't mind doing all the work … » (gold Positive, prédit Negative). Le modèle associe « work » à une charge négative. La négation « didn't mind » inverse le sens de « work ». Erreur de portée de négation, typique d'un sac de mots.
17. **1_149_983_8173** — « Unless you love journalism, steer clear. While Professor [NOM] has somewhat improved … » (gold Negative, prédit Positive). Le conseil est négatif, mais la phrase contient « love » et « improved ». Le modèle se laisse guider par des mots positifs. Un avis conditionnel (« Unless ») n'est pas compris.
18. **1_172_2226_8649** — « New professor … extremely helpful, very clear lessons. Doesn't fill lessons with useless theorems that confuse you … » (gold Positive, prédit Negative). La négation « Doesn't fill … with useless theorems » porte sur « useless » : l'avis est positif. Le modèle lit « useless » et « confuse » comme négatifs. Cas emblématique de la négation.

## C. Polarité manquée : neutre prédit (2 cas)

19. **1_104_1628_5574** — « Always letting class out early, or sick AGAIN! I paid for an education, not excuses. » (gold Negative, prédit Neutral). Critique forte, mais exprimée sans mot de sentiment explicite (« sick », « excuses » sont hors lexique). Le lexique et le modèle ne captent pas l'ironie du ton.
20. **1_104_305_14004** — « I absolutely loved [NOM]. I would take more classes with him … he's just adorable. » (gold Positive, prédit Neutral). Avis très positif. Le modèle, qui pondère les mots fréquents, hésite. Les phrases courtes et explicites sont plus souvent bien classées : la longueur dilue le signal.

## Ce qu'on retient

- **La négation** est la cause la plus fréquente d'erreurs sur les avis polarisés (cas 8, 14, 16, 18). Un lexique à portée de négation est plus adapté ici, mais il manque de couverture (cas 19).
- **La classe neutre** concentre les désaccords : une partie des erreurs vient de l'étiquetage (cas 2, 7, 13), une autre de vrais avis mélangés (cas 3, 6, 11).
- **Le sarcasme et l'ironie** n'apparaissent pas dans cette sélection. Une mesure fiable demanderait une annotation dédiée.
- **Le bruit d'annotation** limite le score maximal atteignable. Il faut le dire dans le rapport plutôt que d'optimiser le modèle sur des étiquettes discutables.
