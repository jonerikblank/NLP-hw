import argparse
import json
import os
import re
import string
from collections import Counter, defaultdict
from typing import List, Dict, Any


NEGATION_RE = re.compile(r"\b(no|not|never|none|nothing|nobody|nowhere|neither|nor|cannot|can't|dont|don't|doesnt|doesn't|isnt|isn't|wasnt|wasn't|won't|wouldn't|shouldn't|couldn't)\b", re.I)


def load_predictions(path: str) -> List[Dict[str, Any]]:
    """Load eval_predictions.jsonl into a list of dicts."""
    preds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            preds.append(json.loads(line))
    if not preds:
        raise ValueError(f"No predictions found in {path}")
    return preds


def overall_accuracy(preds: List[Dict[str, Any]]) -> float:
    correct = sum(1 for ex in preds if ex["predicted_label"] == ex["label"])
    return correct / len(preds)


def confusion_matrix(preds: List[Dict[str, Any]]):
    """Return confusion matrix as dict[(gold, pred)] -> count and label sets."""
    labels = sorted({ex["label"] for ex in preds} | {ex["predicted_label"] for ex in preds})
    cm = {(g, p): 0 for g in labels for p in labels}
    for ex in preds:
        cm[(ex["label"], ex["predicted_label"])] += 1
    return labels, cm


def tokenize(text: str):
    text = text.lower()
    for p in string.punctuation:
        text = text.replace(p, " ")
    return [t for t in text.split() if t]


def hypothesis_length(ex: Dict[str, Any]) -> int:
    return len(tokenize(ex["hypothesis"]))


def overlap_ratio(ex: Dict[str, Any]) -> float:
    """|premise ∩ hypothesis| / |hypothesis|."""
    p_tokens = set(tokenize(ex["premise"]))
    h_tokens = set(tokenize(ex["hypothesis"]))
    if not h_tokens:
        return 0.0
    return len(p_tokens & h_tokens) / len(h_tokens)


def split_by_negation(preds: List[Dict[str, Any]]):
    neg, non_neg = [], []
    for ex in preds:
        if NEGATION_RE.search(ex["hypothesis"]):
            neg.append(ex)
        else:
            non_neg.append(ex)
    return neg, non_neg


def accuracy_for_subset(subset: List[Dict[str, Any]]) -> float:
    if not subset:
        return float("nan")
    return sum(1 for ex in subset if ex["predicted_label"] == ex["label"]) / len(subset)


def bucket_by_length(preds: List[Dict[str, Any]], boundaries=(5, 10, 20)):
    """
    Buckets:
      <= b1, (b1, b2], (b2, b3], > b3
    """
    buckets = defaultdict(list)
    for ex in preds:
        n = hypothesis_length(ex)
        if n <= boundaries[0]:
            key = f"<= {boundaries[0]}"
        elif n <= boundaries[1]:
            key = f"{boundaries[0]+1}-{boundaries[1]}"
        elif n <= boundaries[2]:
            key = f"{boundaries[1]+1}-{boundaries[2]}"
        else:
            key = f"> {boundaries[2]}"
        buckets[key].append(ex)
    return buckets


def bucket_by_overlap(preds: List[Dict[str, Any]], threshold=0.5):
    high, low = [], []
    for ex in preds:
        r = overlap_ratio(ex)
        if r > threshold:
            high.append(ex)
        else:
            low.append(ex)
    return high, low


def format_confusion_matrix(labels, cm):
    # nice text table
    header = "gold\\pred" + "".join(f"\t{p}" for p in labels)
    lines = [header]
    for g in labels:
        row = [str(g)]
        for p in labels:
            row.append(str(cm[(g, p)]))
        lines.append("\t".join(row))
    return "\n".join(lines)


def label_distribution(preds: List[Dict[str, Any]]):
    gold_counts = Counter(ex["label"] for ex in preds)
    pred_counts = Counter(ex["predicted_label"] for ex in preds)
    return gold_counts, pred_counts


def generate_report(preds: List[Dict[str, Any]]) -> str:
    lines = []
    n = len(preds)
    acc = overall_accuracy(preds)
    lines.append("# SNLI Artifact Analysis Report\n")
    lines.append(f"Total eval examples: {n}")
    lines.append(f"Overall accuracy: {acc:.4f}\n")

    # Label distributions
    gold_counts, pred_counts = label_distribution(preds)
    lines.append("## Label distributions\n")
    lines.append("Gold label counts:")
    for lbl, c in sorted(gold_counts.items()):
        lines.append(f"  - {lbl}: {c} ({c / n:.3%})")
    lines.append("Predicted label counts:")
    for lbl, c in sorted(pred_counts.items()):
        lines.append(f"  - {lbl}: {c} ({c / n:.3%})")
    lines.append("")

    # Confusion matrix
    labels, cm = confusion_matrix(preds)
    lines.append("## Confusion matrix (rows = gold, cols = predicted)\n")
    lines.append("Labels are integer IDs as used in the model/dataset.\n")
    lines.append("```")
    lines.append(format_confusion_matrix(labels, cm))
    lines.append("```")
    lines.append("")

    # Negation artifact
    neg, non_neg = split_by_negation(preds)
    acc_neg = accuracy_for_subset(neg)
    acc_non = accuracy_for_subset(non_neg)

    lines.append("## Negation heuristic\n")
    lines.append(
        "Here we check an artifact where hypotheses containing negation words "
        "('no', 'not', 'never', etc.) are often predicted as contradiction."
    )
    lines.append(f"- Negation subset size: {len(neg)}")
    lines.append(f"- Non-negation subset size: {len(non_neg)}")
    if len(neg) > 0:
        lines.append(f"- Accuracy on negation subset: {acc_neg:.4f}")
    if len(non_neg) > 0:
        lines.append(f"- Accuracy on non-negation subset: {acc_non:.4f}")
    lines.append("")

    # Overlap artifact
    high, low = bucket_by_overlap(preds, threshold=0.5)
    acc_high = accuracy_for_subset(high)
    acc_low = accuracy_for_subset(low)

    lines.append("## Lexical overlap heuristic\n")
    lines.append(
        "We compute lexical overlap as |premise ∩ hypothesis| / |hypothesis|. "
        "High overlap (> 0.5) often correlates with entailment predictions."
    )
    lines.append(f"- High-overlap subset size (> 0.5): {len(high)}")
    lines.append(f"- Low-overlap subset size (<= 0.5): {len(low)}")
    if len(high) > 0:
        lines.append(f"- Accuracy on high-overlap subset: {acc_high:.4f}")
    if len(low) > 0:
        lines.append(f"- Accuracy on low-overlap subset: {acc_low:.4f}")
    lines.append("")

    # Length buckets
    buckets = bucket_by_length(preds)
    lines.append("## Hypothesis length buckets\n")
    lines.append("We bucket hypotheses by tokenized length:")
    for name, subset in buckets.items():
        a = accuracy_for_subset(subset)
        lines.append(f"- {name}: {len(subset)} examples, accuracy = {a:.4f}")
    lines.append("")

    lines.append(
        "You can now use these slice accuracies to argue about which artifacts "
        "your model appears to rely on (e.g., big gaps between subsets)."
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to eval_predictions.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis",
        help="Directory to write artifact_report.txt",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    preds = load_predictions(args.predictions)
    report = generate_report(preds)

    # Save report
    out_path = os.path.join(args.output_dir, "artifact_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Also print a short summary to stdout
    print(report)


if __name__ == "__main__":
    main()

