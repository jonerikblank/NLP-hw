import argparse
import json
import os

import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from helpers import compute_accuracy  # reuse existing accuracy fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="google/electra-small-discriminator",
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the model and metrics.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=3.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Train batch size per device.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max sequence length for hypothesis-only inputs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional: limit number of training samples for quick debugging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load SNLI
    print("Loading SNLI...")
    raw = datasets.load_dataset("snli")

    # Remove examples with no label (label = -1), same as run.py
    print("Filtering unlabeled examples...")
    raw = raw.filter(lambda ex: ex["label"] != -1)

    train_dataset = raw["train"]
    eval_dataset = raw["validation"]

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(args.max_train_samples))

    # 2. Tokenizer and model
    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3
    )

    # ELECTRA contiguity fix (mirrors run.py pattern)
    if hasattr(model, "electra"):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

    # 3. Hypothesis-only preprocessing
    def preprocess_hypothesis_only(examples):
        # Only use the hypothesis text as input
        tokenized = tokenizer(
            examples["hypothesis"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        tokenized["label"] = examples["label"]
        return tokenized

    print("Tokenizing (hypothesis-only)...")
    train_enc = train_dataset.map(
        preprocess_hypothesis_only,
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    eval_enc = eval_dataset.map(
        preprocess_hypothesis_only,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    # 4. Training config — *no* evaluation_strategy here
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        logging_steps=100,
        save_steps=10000,  # large so we rarely save mid-training
        save_total_limit=1,
    )

    def compute_metrics_fn(eval_preds):
        return compute_accuracy(eval_preds)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_enc,
        eval_dataset=eval_enc,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
    )

    # 5. Train & evaluate
    print("***** Training hypothesis-only model *****")
    trainer.train()

    print("***** Evaluating hypothesis-only model *****")
    eval_results = trainer.evaluate()
    print("Eval results:", eval_results)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved eval metrics to {metrics_path}")

    # 6. Save predictions (so we can reuse the artifact analyzer)
    print("Generating predictions on validation set...")
    pred_output = trainer.predict(eval_enc)
    preds = pred_output.predictions.argmax(axis=1).tolist()

    # Reload original eval_dataset to keep premise & hypothesis text in the JSONL
    print("Writing eval_predictions.jsonl (hypothesis-only)...")
    preds_path = os.path.join(args.output_dir, "eval_predictions.jsonl")
    with open(preds_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(eval_dataset):
            row = {
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
                "label": int(ex["label"]),
                "predicted_scores": pred_output.predictions[i].tolist(),
                "predicted_label": int(preds[i]),
            }
            f.write(json.dumps(row))
            f.write("\n")
    print(f"Saved predictions to {preds_path}")


if __name__ == "__main__":
    main()
