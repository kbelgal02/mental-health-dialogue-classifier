"""Train the mental health dialogue classifier.

Usage:
    python -m src.train --data data/sample_data.csv
    python -m src.train --data data/sample_data.csv --output_dir models/my_model --epochs 5
"""
import argparse
import random
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from .config import LABEL2ID, ID2LABEL, TrainingConfig
from .dataset import MentalHealthDataset
from .evaluate import compute_metrics, full_report


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: TrainingConfig, data_path: str) -> None:
    set_seed(config.seed)

    print(f"Loading data from: {data_path}")
    ds_loader = MentalHealthDataset(config)
    dataset = ds_loader.load_from_csv(data_path)

    print(f"Train: {len(dataset['train'])}  Val: {len(dataset['validation'])}  Test: {len(dataset['test'])}")

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        fp16=config.fp16 and torch.cuda.is_available(),
        seed=config.seed,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Starting training...")
    trainer.train()

    print("\nEvaluating on test set...")
    test_output = trainer.predict(dataset["test"])
    preds = test_output.predictions.argmax(axis=-1).tolist()
    labels = test_output.label_ids.tolist()
    full_report(preds, labels, output_path=f"{config.output_dir}/test_report.txt")

    print(f"\nSaving model to: {config.output_dir}")
    trainer.save_model(config.output_dir)
    ds_loader.tokenizer.save_pretrained(config.output_dir)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train mental health dialogue classifier")
    parser.add_argument("--data", required=True, help="Path to CSV training data")
    parser.add_argument("--model_name", default="distilbert-base-uncased")
    parser.add_argument("--output_dir", default="./models/mental_health_classifier")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    config = TrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_length,
        fp16=args.fp16,
    )
    train(config, args.data)


if __name__ == "__main__":
    main()
