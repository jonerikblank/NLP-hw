import argparse
import json
import os

import datasets
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hyp_model_dir",
        type=str,
        required=True,
        help="Path to the trained hypothesis-only model directory (e.g., outputs/snli-hyp-only).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="snli_hyp_conf_train.json",
        help="Where to store the list of weights as JSON.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max sequence length for hypothesis-only tokenization.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load SNLI train split and filter unlabeled examples (label = -1)
    print("Loading SNLI train split...")
    raw = datasets.load_dataset("snli")
    print("Filtering unlabeled examples...")
    train_dataset = raw["train"].filter(lambda ex: ex["label"] != -1)

    print(f"Train examples after filtering: {len(train_dataset)}")

    # 2. Load hyp-only model + tokenizer
    print(f"Loading hypothesis-only model from {args.hyp_model_dir} ...")
    # Tokenizer comes from the *base* model family, not the fine-tuned directory.
    tokenizer = AutoTokenizer.from_pretrained("google/electra-small-discriminator", use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.hyp_model_dir)
    model.to(device)
    model.eval()

    # 3. Preprocess: hypothesis-only tokenization
    def preprocess_hyp_only(examples):
        tok = tokenizer(
            examples["hypothesis"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        tok["labels"] = examples["label"]
        return tok

    print("Tokenizing hypotheses...")
    enc_train = train_dataset.map(
        preprocess_hyp_only,
        batched=True,
    )

    # Keep only needed columns and set torch format
    enc_train.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    loader = DataLoader(enc_train, batch_size=args.batch_size)

    all_confidences = []

    print("Running hyp-only model on train set to compute confidences...")
    softmax = torch.nn.Softmax(dim=-1)

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = softmax(outputs.logits)  # [B, num_labels]
            conf, _ = probs.max(dim=-1)      # [B]

        all_confidences.extend(conf.cpu().tolist())

    assert len(all_confidences) == len(train_dataset), "Length mismatch!"

    # Convert confidence -> weight (higher weight = harder example)
    # weight = 1 - conf, but keep a small floor so easy examples are not fully ignored.
    weights = []
    for c in all_confidences:
        w = 1.0 - c
        # optional floor/ceiling
        w = 0.1 + 0.9 * w  # so weights live in [0.1, 1.0]
        weights.append(w)

    out = {
        "num_examples": len(train_dataset),
        "weights": weights,
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out, f)

    print(f"Saved weights to {args.output_path} with {len(weights)} entries.")


if __name__ == "__main__":
    main()
