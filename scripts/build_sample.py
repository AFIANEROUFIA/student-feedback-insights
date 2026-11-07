"""Génère l'échantillon SYNTHÉTIQUE du prototype (data/raw/avis_etudiants_sample.csv et avis_absa_sample.json).

ATTENTION : ces 90 textes ont été rédigés pour tester l'application. Ce ne sont PAS des avis réels.
Pour une évaluation réelle, utiliser le jeu EduRABSA anonymisé (scripts/prepare_edurabsa.py).
Le lexique maison a été écrit en regardant ces phrases : ses scores sur cet échantillon sont optimistes.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

ROWS = [
    # English Positive
    ("r001", "en", "course_eval", "synthétique (rédigé pour le prototype)", "The teacher is very helpful and explains every concept clearly.", "Positive", "Enseignant / attitude, Cours / contenu", 5),
    ("r002", "en", "course_eval", "synthétique (rédigé pour le prototype)", "Great course content and well organized lectures.", "Positive", "Cours / contenu, Organisation", 5),
    ("r003", "en", "forum", "synthétique (rédigé pour le prototype)", "The professor is friendly, available and really supportive.", "Positive", "Enseignant / attitude", 5),
    ("r004", "en", "course_eval", "synthétique (rédigé pour le prototype)", "I liked the slides and the extra resources on Moodle.", "Positive", "Ressources pédagogiques", 4),
    ("r005", "en", "course_eval", "synthétique (rédigé pour le prototype)", "Fair exams and clear grading criteria.", "Positive", "Examens / évaluation", 4),
    ("r006", "en", "survey", "synthétique (rédigé pour le prototype)", "The classroom is comfortable and the wifi works well.", "Positive", "Infrastructures", 4),
    ("r007", "en", "course_eval", "synthétique (rédigé pour le prototype)", "Assignments are useful and the project is interesting.", "Positive", "Charge de travail, Cours / contenu", 4),
    ("r008", "en", "forum", "synthétique (rédigé pour le prototype)", "Best teacher I had this semester. Very patient.", "Positive", "Enseignant / attitude", 5),
    ("r009", "en", "course_eval", "synthétique (rédigé pour le prototype)", "The course is structured and easy to follow.", "Positive", "Cours / contenu, Organisation", 4),
    ("r010", "en", "survey", "synthétique (rédigé pour le prototype)", "Recommended. Clear lectures and helpful feedback.", "Positive", "Cours / contenu, Enseignant / attitude", 5),
    # English Negative
    ("r011", "en", "course_eval", "synthétique (rédigé pour le prototype)", "He is not helpful at all. Very rude.", "Negative", "Enseignant / attitude", 1),
    ("r012", "en", "course_eval", "synthétique (rédigé pour le prototype)", "The exams are extremely difficult and unfair.", "Negative", "Examens / évaluation, Difficulté", 1),
    ("r013", "en", "forum", "synthétique (rédigé pour le prototype)", "The course is confusing and the slides are useless.", "Negative", "Cours / contenu, Ressources pédagogiques", 2),
    ("r014", "en", "survey", "synthétique (rédigé pour le prototype)", "The teacher is often absent and never answers emails.", "Negative", "Enseignant / attitude, Organisation", 1),
    ("r015", "en", "course_eval", "synthétique (rédigé pour le prototype)", "Too much homework and impossible deadlines.", "Negative", "Charge de travail", 2),
    ("r016", "en", "forum", "synthétique (rédigé pour le prototype)", "The classroom is noisy and the equipment is terrible.", "Negative", "Infrastructures", 2),
    ("r017", "en", "course_eval", "synthétique (rédigé pour le prototype)", "Grading is inconsistent and the feedback is poor.", "Negative", "Examens / évaluation", 2),
    ("r018", "en", "survey", "synthétique (rédigé pour le prototype)", "The organization is chaotic. Announcements arrive too late.", "Negative", "Organisation", 1),
    ("r019", "en", "course_eval", "synthétique (rédigé pour le prototype)", "The content is outdated and boring.", "Negative", "Cours / contenu", 2),
    ("r020", "en", "forum", "synthétique (rédigé pour le prototype)", "Worst lecture ever. Completely unprepared professor.", "Negative", "Enseignant / attitude, Cours / contenu", 1),
    # English Neutral / mixed
    ("r021", "en", "course_eval", "synthétique (rédigé pour le prototype)", "The course contains five chapters and a final exam.", "Neutral", "Cours / contenu, Examens / évaluation", 3),
    ("r022", "en", "survey", "synthétique (rédigé pour le prototype)", "Lectures are on Monday morning in room B12.", "Neutral", "Organisation, Infrastructures", 3),
    ("r023", "en", "forum", "synthétique (rédigé pour le prototype)", "The teacher is friendly, but the exams are extremely difficult.", "Negative", "Enseignant / attitude, Examens / évaluation, Difficulté", 3),
    ("r024", "en", "course_eval", "synthétique (rédigé pour le prototype)", "The material is available, nothing special.", "Neutral", "Ressources pédagogiques", 3),
    ("r025", "en", "survey", "synthétique (rédigé pour le prototype)", "Average class. Some parts are clear, some are not.", "Neutral", "Cours / contenu", 3),
    ("r026", "en", "course_eval", "synthétique (rédigé pour le prototype)", "The professor is okay but the workload is heavy.", "Negative", "Enseignant / attitude, Charge de travail", 3),
    ("r027", "en", "forum", "synthétique (rédigé pour le prototype)", "I attended all sessions. The course follows the syllabus.", "Neutral", "Cours / contenu, Organisation", 3),
    ("r028", "en", "survey", "synthétique (rédigé pour le prototype)", "Library resources are mentioned in the first lecture.", "Neutral", "Ressources pédagogiques, Infrastructures", 3),
    # French Positive
    ("r029", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Le professeur est très disponible et explique clairement.", "Positive", "Enseignant / attitude", 5),
    ("r030", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Cours intéressant, bien organisé et enrichissant.", "Positive", "Cours / contenu, Organisation", 5),
    ("r031", "fr", "forum", "synthétique (rédigé pour le prototype)", "Les supports Moodle sont utiles et bien faits.", "Positive", "Ressources pédagogiques", 4),
    ("r032", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Les examens sont justes et le barème est clair.", "Positive", "Examens / évaluation", 4),
    ("r033", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Salle confortable, wifi efficace, bon équipement.", "Positive", "Infrastructures", 4),
    ("r034", "fr", "forum", "synthétique (rédigé pour le prototype)", "Projet intéressant et charge de travail raisonnable.", "Positive", "Charge de travail, Cours / contenu", 4),
    ("r035", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Enseignante pédagogue, patiente et très claire.", "Positive", "Enseignant / attitude", 5),
    ("r036", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Je recommande ce cours, vraiment top.", "Positive", "Cours / contenu", 5),
    ("r037", "fr", "forum", "synthétique (rédigé pour le prototype)", "Organisation parfaite et communication efficace.", "Positive", "Organisation", 5),
    ("r038", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Les chapitres sont accessibles et bien structurés.", "Positive", "Cours / contenu", 4),
    # French Negative
    ("r039", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Le prof n'est pas du tout disponible. Très impoli.", "Negative", "Enseignant / attitude", 1),
    ("r040", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Les examens sont trop difficiles et injustes.", "Negative", "Examens / évaluation, Difficulté", 1),
    ("r041", "fr", "forum", "synthétique (rédigé pour le prototype)", "Cours confus, supports inutiles et slides pauvres.", "Negative", "Cours / contenu, Ressources pédagogiques", 2),
    ("r042", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Toujours en retard, organisation chaotique.", "Negative", "Organisation", 1),
    ("r043", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Trop de devoirs et des délais impossibles.", "Negative", "Charge de travail", 2),
    ("r044", "fr", "forum", "synthétique (rédigé pour le prototype)", "La salle est bruyante et le matériel est mauvais.", "Negative", "Infrastructures", 2),
    ("r045", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Notation incohérente, aucun retour utile.", "Negative", "Examens / évaluation", 2),
    ("r046", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Contenu décevant, cours ennuyeux et trop vague.", "Negative", "Cours / contenu", 2),
    ("r047", "fr", "forum", "synthétique (rédigé pour le prototype)", "Manque de pédagogie. Explications incompréhensibles.", "Negative", "Enseignant / attitude, Cours / contenu", 1),
    ("r048", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Le wifi ne marche pas et l'amphi est trop petit.", "Negative", "Infrastructures", 2),
    # French Neutral / mixed
    ("r049", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Le cours contient cinq chapitres et un examen final.", "Neutral", "Cours / contenu, Examens / évaluation", 3),
    ("r050", "fr", "forum", "synthétique (rédigé pour le prototype)", "Les séances ont lieu le lundi matin en salle B12.", "Neutral", "Organisation, Infrastructures", 3),
    ("r051", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Le professeur est gentil, mais les examens sont extrêmement difficiles.", "Negative", "Enseignant / attitude, Examens / évaluation, Difficulté", 3),
    ("r052", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Les documents sont en ligne, rien de particulier.", "Neutral", "Ressources pédagogiques", 3),
    ("r053", "fr", "forum", "synthétique (rédigé pour le prototype)", "Cours moyen. Certaines parties sont claires, d'autres non.", "Neutral", "Cours / contenu", 3),
    ("r054", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Le prof est correct mais la charge de travail est trop lourde.", "Negative", "Enseignant / attitude, Charge de travail", 3),
    ("r055", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Le programme suit le syllabus annoncé.", "Neutral", "Cours / contenu, Organisation", 3),
    ("r056", "fr", "forum", "synthétique (rédigé pour le prototype)", "La bibliothèque est citée dans le premier cours.", "Neutral", "Infrastructures, Ressources pédagogiques", 3),
    # Extra diversity
    ("r057", "en", "course_eval", "synthétique (rédigé pour le prototype)", "Not a bad course, but the teacher is not very clear.", "Negative", "Cours / contenu, Enseignant / attitude", 2),
    ("r058", "en", "course_eval", "synthétique (rédigé pour le prototype)", "The exam was fair even if the course is difficult.", "Positive", "Examens / évaluation, Difficulté", 4),
    ("r059", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Ce n'est pas un mauvais cours, mais le prof n'est pas clair.", "Negative", "Cours / contenu, Enseignant / attitude", 2),
    ("r060", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Examen juste même si le cours est difficile.", "Positive", "Examens / évaluation, Difficulté", 4),
    ("r061", "en", "forum", "synthétique (rédigé pour le prototype)", "Moodle resources are excellent and always updated.", "Positive", "Ressources pédagogiques", 5),
    ("r062", "fr", "forum", "synthétique (rédigé pour le prototype)", "Les ressources Moodle sont excellentes et à jour.", "Positive", "Ressources pédagogiques", 5),
    ("r063", "en", "survey", "synthétique (rédigé pour le prototype)", "The staff attitude is cold and unhelpful.", "Negative", "Enseignant / attitude", 1),
    ("r064", "fr", "enquête", "synthétique (rédigé pour le prototype)", "L'attitude du personnel est froide et peu aidante.", "Negative", "Enseignant / attitude", 1),
    ("r065", "en", "course_eval", "synthétique (rédigé pour le prototype)", "Assessment methods are clear and well explained.", "Positive", "Examens / évaluation", 4),
    ("r066", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Les méthodes d'évaluation sont claires et bien expliquées.", "Positive", "Examens / évaluation", 4),
    ("r067", "en", "course_eval", "synthétique (rédigé pour le prototype)", "The project deadline is realistic and the instructions are good.", "Positive", "Charge de travail", 4),
    ("r068", "fr", "enquête", "synthétique (rédigé pour le prototype)", "La deadline du projet est réaliste et les consignes sont bonnes.", "Positive", "Charge de travail", 4),
    ("r069", "en", "forum", "synthétique (rédigé pour le prototype)", "I do not like the vague feedback after the exam.", "Negative", "Examens / évaluation", 2),
    ("r070", "fr", "forum", "synthétique (rédigé pour le prototype)", "Je n'aime pas les retours vagues après l'examen.", "Negative", "Examens / évaluation", 2),
    ("r071", "en", "survey", "synthétique (rédigé pour le prototype)", "There are 12 lectures and 2 labs this term.", "Neutral", "Cours / contenu, Infrastructures", 3),
    ("r072", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Il y a 12 cours et 2 séances de laboratoire ce semestre.", "Neutral", "Cours / contenu, Infrastructures", 3),
    ("r073", "en", "course_eval", "synthétique (rédigé pour le prototype)", "Amazing professor, but the classroom chairs are awful.", "Positive", "Enseignant / attitude, Infrastructures", 4),
    ("r074", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Professeur génial, mais les chaises de la salle sont horribles.", "Positive", "Enseignant / attitude, Infrastructures", 4),
    ("r075", "en", "forum", "synthétique (rédigé pour le prototype)", "The course is not organized at all.", "Negative", "Organisation, Cours / contenu", 1),
    ("r076", "fr", "forum", "synthétique (rédigé pour le prototype)", "Le cours n'est pas du tout organisé.", "Negative", "Organisation, Cours / contenu", 1),
    ("r077", "en", "course_eval", "synthétique (rédigé pour le prototype)", "Very interesting content and highly recommended.", "Positive", "Cours / contenu", 5),
    ("r078", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Contenu très intéressant, je recommande fortement.", "Positive", "Cours / contenu", 5),
    ("r079", "en", "survey", "synthétique (rédigé pour le prototype)", "The homework is too heavy compared to the credits.", "Negative", "Charge de travail", 2),
    ("r080", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Les devoirs sont trop lourds par rapport aux crédits.", "Negative", "Charge de travail", 2),
    ("r081", "en", "course_eval", "synthétique (rédigé pour le prototype)", "He explains well. The exam is fair.", "Positive", "Enseignant / attitude, Examens / évaluation", 4),
    ("r082", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Il explique bien. L'examen est juste.", "Positive", "Enseignant / attitude, Examens / évaluation", 4),
    ("r083", "en", "forum", "synthétique (rédigé pour le prototype)", "Nothing to add. Standard university course.", "Neutral", "Cours / contenu", 3),
    ("r084", "fr", "forum", "synthétique (rédigé pour le prototype)", "Rien à ajouter. Cours universitaire standard.", "Neutral", "Cours / contenu", 3),
    ("r085", "en", "survey", "synthétique (rédigé pour le prototype)", "The teacher is not available during office hours.", "Negative", "Enseignant / attitude", 2),
    ("r086", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Le professeur n'est pas disponible pendant les permanences.", "Negative", "Enseignant / attitude", 2),
    ("r087", "en", "course_eval", "synthétique (rédigé pour le prototype)", "Clear slides, engaging lectures, great support.", "Positive", "Ressources pédagogiques, Cours / contenu", 5),
    ("r088", "fr", "enquête", "synthétique (rédigé pour le prototype)", "Slides clairs, cours motivant, super support.", "Positive", "Ressources pédagogiques, Cours / contenu", 5),
    ("r089", "en", "forum", "synthétique (rédigé pour le prototype)", "The final is impossible and the grading is biased.", "Negative", "Examens / évaluation, Difficulté", 1),
    ("r090", "fr", "forum", "synthétique (rédigé pour le prototype)", "L'examen final est impossible et la notation est injuste.", "Negative", "Examens / évaluation, Difficulté", 1),
]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    csv_path = RAW / "avis_etudiants_sample.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id", "language", "type", "source", "review_text",
                "overall_sentiment", "themes_gold", "rating",
            ],
        )
        writer.writeheader()
        for row in ROWS:
            writer.writerow({
                "id": row[0],
                "language": row[1],
                "type": row[2],
                "source": row[3],
                "review_text": row[4],
                "overall_sentiment": row[5],
                "themes_gold": row[6],
                "rating": row[7],
            })

    json_rows = []
    for row in ROWS[:12]:
        text = row[4]
        sentiment = {"Positive": "Pos", "Negative": "Neg", "Neutral": "Neu"}[row[5]]
        json_rows.append({
            "id": row[0],
            "original_id": row[0],
            "text": text,
            "overall_sentiment": sentiment,
            "type": row[2],
            "Quadruplets": [
                {
                    "aspect": row[6].split(",")[0].strip(),
                    "opinion": text[:60],
                    "category": row[6].split(",")[0].strip(),
                    "sentiment": sentiment,
                }
            ],
        })
    json_path = RAW / "avis_absa_sample.json"
    json_path.write_text(json.dumps(json_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote", csv_path, "n=", len(ROWS))
    print("Wrote", json_path)


if __name__ == "__main__":
    main()
