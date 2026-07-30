"""Evaluate a trained pair classifier on a pair JSONL file.

Reports pointwise classification metrics (accuracy, P/R/F1, AUC, confusion
matrix). If the input contains `_meta.query_id` (NQ-style), also computes
per-query ranking metrics (Recall@1, Recall@5, MRR) by grouping pairs by
query_id and ranking by predicted P(label=1).

Usage:
    python scripts/eval/eval_pair_classifier.py \
        --checkpoint checkpoints/nq_pair_clf \
        --eval-file data/nq_pair_eval.jsonl \
        --output-file results/nq_pair_clf_eval.json
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from corpus_reasoning.lib.io import load_jsonl


@torch.no_grad()
def run_inference(model, tokenizer, rows, max_length, batch_size, device):
    ds = Dataset.from_list([
        {"text_a": r["text_a"], "text_b": r["text_b"], "label": int(r["label"])}
        for r in rows
    ])

    def tok(batch):
        return tokenizer(
            batch["text_a"], batch["text_b"],
            truncation="longest_first", max_length=max_length,
        )

    ds = ds.map(tok, batched=True, remove_columns=["text_a", "text_b"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        labels = batch.pop("labels", batch.pop("label", None))
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy() if torch.is_tensor(labels) else np.array(labels))
    return np.concatenate(all_probs), np.concatenate(all_labels)


def ranking_metrics(rows, probs):
    """Compute Recall@K and MRR by grouping rows on _meta.query_id."""
    groups: dict = defaultdict(list)
    for r, p in zip(rows, probs):
        meta = r.get("_meta") or {}
        qid = meta.get("query_id")
        if qid is None:
            return None
        groups[qid].append((p, int(r["label"])))

    r1, r5, rr = [], [], []
    for qid, pairs in groups.items():
        pairs_sorted = sorted(pairs, key=lambda x: -x[0])
        labels_ranked = [lab for _, lab in pairs_sorted]
        if 1 not in labels_ranked:
            continue
        rank_of_first_gold = labels_ranked.index(1) + 1
        r1.append(1.0 if rank_of_first_gold == 1 else 0.0)
        r5.append(1.0 if rank_of_first_gold <= 5 else 0.0)
        rr.append(1.0 / rank_of_first_gold)
    if not r1:
        return None
    return {
        "n_queries": len(r1),
        "recall@1": float(np.mean(r1)),
        "recall@5": float(np.mean(r5)),
        "mrr": float(np.mean(rr)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to trained model dir")
    p.add_argument("--eval-file", required=True, help="Pair JSONL eval file")
    p.add_argument("--output-file", required=True)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    rows = load_jsonl(args.eval_file)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint).to(args.device)

    probs, labels = run_inference(
        model, tokenizer, rows, args.max_length, args.batch_size, args.device,
    )
    preds = (probs >= args.threshold).astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    out = {
        "checkpoint": args.checkpoint,
        "eval_file": args.eval_file,
        "n_examples": int(len(rows)),
        "threshold": args.threshold,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix": {"tn_fp_fn_tp": [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]},
    }
    try:
        out["auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        out["auc"] = None

    ranking = ranking_metrics(rows, probs)
    if ranking is not None:
        out["ranking"] = ranking

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
