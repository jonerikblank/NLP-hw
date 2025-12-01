import argparse
import json
import os

import datasets
import torch
from torch.utils.data import DataLoader
from torch import nn

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


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
        help="Where to save the hyp-only model (optional) and logs.",
    )
    parser.add_argument(
        "--weights_output",
        type=str,
        default="snli_hyp_conf_train.json",
        help="Path to save the train-set weights JSON.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=1.0,
        help="Number of training epochs for hyp-only model.",
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
        help="Optional: limit number of training samples for faster debugging.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load SNLI train split and filter label = -1
    print("Loading SNLI train split...")
    raw = datasets.load_dataset("snli")
    print("Filtering unlabeled examples...")
    train_dataset = raw["train"].filter(lambda ex: ex["label"] != -1)
    print(f"Train examples after filtering: {len(train_dataset)}")

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(args.max_train_samples))
        print(f"Subsampled train examples: {len(train_dataset)}")

    # 2. Tokenizer and model
    print(f"Loading tokenizer and model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3
    )

    # ELECTRA contiguity fix (like in your previous scripts)
    if hasattr(model, "electra"):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

    # 3. Hypothesis-only preprocessing
    def preprocess_hyp_only(examples):
        tok = tokenizer(
            examples["hypothesis"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        tok["labels"] = examples["label"]
        return tok

    print("Tokenizing (hypothesis-only)...")
    train_enc = train_dataset.map(
        preprocess_hyp_only,
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    # Set format for training
    train_enc.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # 4. TrainingArguments (no evaluation needed here)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        logging_steps=100,
        save_steps=10000,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_enc,
    )

    # 5. Train hyp-only model
    print("***** Training hypothesis-only model (for weights) *****")
    trainer.train()

    # 6. Compute confidences on the train set
    print("***** Computing confidences on train set *****")
    model.to(device)
    model.eval()

    loader = DataLoader(train_enc, batch_size=args.per_device_train_batch_size)
    softmax = nn.Softmax(dim=-1)
    all_confidences = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = softmax(outputs.logits)  # [B, num_labels]
            conf, _ = probs.max(dim=-1)      # [B]

        all_confidences.extend(conf.cpu().tolist())

    assert len(all_confidences) == len(train_dataset), "Length mismatch between confidences and train set!"

    # 7. Convert confidences -> weights
    # weight = 1 - conf, but keep a floor at 0.1 so easy examples are not completely ignored
    print("Converting confidences to weights...")
    weights = []
    for c in all_confidences:
        w = 1.0 - c
        w = 0.1 + 0.9 * w  # map to [0.1, 1.0]
        weights.append(w)

    out = {
        "num_examples": len(train_dataset),
        "weights": weights,
    }

    with open(args.weights_output, "w", encoding="utf-8") as f:
        json.dump(out, f)

    print(f"Saved weights to {args.weights_output} with {len(weights)} entries.")


if __name__ == "__main__":
    main()

