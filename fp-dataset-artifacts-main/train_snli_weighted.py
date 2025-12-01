import argparse
import json
import os

import datasets
import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from helpers import compute_accuracy  # reuse your existing metric fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="google/electra-small-discriminator",
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--weights_path",
        type=str,
        required=True,
        help="Path to snli_hyp_conf_train.json produced by compute_snli_weights.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the debiased model and metrics.",
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
        help="Max sequence length for NLI inputs.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional: subsample train set for quick debugging.",
    )
    return parser.parse_args()


class WeightedTrainer(Trainer):
    """
    Trainer that expects a 'weight' field in each batch and uses it
    to reweight per-example cross-entropy loss.
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract weights and labels
        weights = inputs.pop("weight", None)
        labels = inputs.get("labels", None)
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits

        if labels is None:
            loss = outputs.loss if hasattr(outputs, "loss") else None
        else:
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            # logits: [B, num_labels], labels: [B]
            per_example_loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            if weights is not None:
                # weights: [B]
                per_example_loss = per_example_loss * weights.view(-1)
            loss = per_example_loss.mean()

        return (loss, outputs) if return_outputs else loss


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load SNLI and filter unlabeled examples
    print("Loading SNLI...")
    raw = datasets.load_dataset("snli")
    print("Filtering unlabeled examples...")
    raw = raw.filter(lambda ex: ex["label"] != -1)

    train_dataset = raw["train"]
    eval_dataset = raw["validation"]

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(args.max_train_samples))

    print(f"Train examples (filtered): {len(train_dataset)}")
    print(f"Validation examples (filtered): {len(eval_dataset)}")

    # 2. Load weights
    print(f"Loading weights from {args.weights_path} ...")
    with open(args.weights_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    weights = data["weights"]

    if len(weights) != len(train_dataset):
        raise ValueError(
            f"Number of weights ({len(weights)}) != number of train examples ({len(train_dataset)})"
        )

    # Attach weights as a column
    train_dataset = train_dataset.add_column("weight", weights)

    # 3. Tokenizer and model
    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=3,
    )

    # ELECTRA contiguity fix
    if hasattr(model, "electra"):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

    # 4. Preprocess: full NLI (premise + hypothesis)
    def preprocess_nli(examples):
        tokenized = tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        tokenized["labels"] = examples["label"]
        # keep weight if present
        if "weight" in examples:
            tokenized["weight"] = examples["weight"]
        return tokenized

    print("Tokenizing train set...")
    train_enc = train_dataset.map(
        preprocess_nli,
        batched=True,
        remove_columns=train_dataset.column_names,  # old columns removed; we keep only tokens + labels + weight
    )

    print("Tokenizing validation set...")
    eval_enc = eval_dataset.map(
        preprocess_nli,
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    # Set format for torch
    train_enc.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "weight"],
    )
    eval_enc.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # 5. TrainingArguments (minimal, compatible)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        logging_steps=100,
        save_steps=10000,
        save_total_limit=1,
    )

    def compute_metrics_fn(eval_preds):
        return compute_accuracy(eval_preds)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_enc,
        eval_dataset=eval_enc,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_fn,
    )

    # 6. Train & evaluate
    print("***** Training weighted (debiased) NLI model *****")
    trainer.train()

    print("***** Evaluating weighted (debiased) NLI model *****")
    eval_results = trainer.evaluate()
    print("Eval results:", eval_results)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved eval metrics to {metrics_path}")

    # 7. Save predictions for artifact analysis
    print("Generating predictions on validation set...")
    pred_output = trainer.predict(eval_enc)
    preds = pred_output.predictions.argmax(axis=1).tolist()

    preds_path = os.path.join(args.output_dir, "eval_predictions.jsonl")
    print(f"Writing eval_predictions.jsonl to {preds_path} ...")
    with open(preds_path, "w", encoding="utf-8") as f:
        for i, ex in enumerate(eval_dataset):
            row = {
                "premise": ex["premise"],
                "hypothesis": ex["hypothesis"],
                "label": int(ex["label"]),
                "predicted_scores": pred_output.predictions[i].tolist(),
                "predicted_label": int(preds[i]),
            }
            f.write(json.dumps(row) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
