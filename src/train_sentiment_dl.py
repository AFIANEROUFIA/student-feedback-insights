"""Fine-tuning optionnel XLM-RoBERTa (plus-value, pas exigé pour le prototype)."""
from __future__ import annotations

import argparse
from pathlib import Path

from .data_io import load_reviews_table, prepare_frame

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning XLM-RoBERTa (optionnel).")
    parser.add_argument("--data", default=str(ROOT / "data" / "raw" / "avis_etudiants_sample.csv"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Installez torch, transformers et datasets pour lancer le Deep Learning.\n"
            "Le prototype du professeur (lexique vs classification TF-IDF) fonctionne déjà sans cela."
        ) from exc

    df, _ = prepare_frame(load_reviews_table(args.data))
    df = df.dropna(subset=["overall_sentiment"])
    label2id = {lab: i for i, lab in enumerate(sorted(df["overall_sentiment"].unique()))}
    df["label"] = df["overall_sentiment"].map(label2id)
    dataset = Dataset.from_pandas(df[["text_clean", "label"]])
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def tokenize(batch):
        return tokenizer(batch["text_clean"], truncation=True, padding="max_length", max_length=args.max_length)

    tokenized = dataset.map(tokenize, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base", num_labels=len(label2id)
    )
    out_dir = ROOT / "models" / "sentiment_xlmr"
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=2e-5,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print("Modèle sauvegardé dans", out_dir)


if __name__ == "__main__":
    main()
