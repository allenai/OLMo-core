"""Build a combined multi-task *unified* JSONL for training.

Concatenates examples from multiple task sources **without** converting to
alpaca. Each example is written in the unified schema used elsewhere in the
codebase (`documents`, `queries`, etc.) with two extra tag fields added:

    _task      -> string task key consumed by `build_prompt_parts`
                  ({outlier, contradiction, reorder, grouping_labeled,
                    cot_retrieval, retrieval})
    _cot_mode  -> string cot_mode argument (task-dependent)

`scripts/train/train.py`'s alpaca converter dispatches on these tags
per-example, so prompt building happens dynamically at train time — same as
a single-task unified file, just per-row.

Usage:
    python scripts/data/build_combined_unified.py \
        --out data/combined_train.jsonl
"""
import argparse
import random
from pathlib import Path

from corpus_reasoning.lib.io import load_jsonl, save_jsonl


TASKS = [
    {
        "name": "outlier",
        "src":  "data/review_outlier_v2_n100_train_5000.jsonl",
        "task": "outlier",
        "cot_mode": "label",
        "n_train": 2000,
    },
    {
        "name": "contradiction",
        "src":  "data/contradiction_train_pubmed_both_n100_k3.jsonl",
        "task": "contradiction",
        "cot_mode": "template",
        "n_train": 2000,
    },
    {
        "name": "reorder",
        "src":  "data/reorder_gutenberg_n5_train_1995.jsonl",
        "task": "reorder",
        "cot_mode": "successor",
        "n_train": 1995,
    },
    {
        "name": "grouping",
        "src":  "data/openalex_grouping_n20_levels_train_5000.jsonl",
        "task": "grouping_labeled",
        "cot_mode": "label",
        "n_train": 2000,
    },
    {
        "name": "hotpotqa",
        "src":  "data/hotpotqa_train_k100_bridge_unified_hn98_2000_cot.jsonl",
        "task": "cot_retrieval",
        "cot_mode": "label",
        "n_train": 2000,
    },
    {
        "name": "nq",
        "src":  "data/nq_train_k100_hn10_10000.jsonl",
        "task": "retrieval",
        "cot_mode": "none",
        "n_train": 2000,
    },
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output unified JSONL path.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = []
    per_task_counts = {}

    for t in TASKS:
        src = Path(t["src"])
        if not src.exists():
            raise FileNotFoundError(f"Missing data file for {t['name']}: {src}")
        exs = list(load_jsonl(src))
        if t["n_train"] > len(exs):
            print(f"[warn] {t['name']}: requested {t['n_train']} but only {len(exs)} available — taking all.")
            n = len(exs)
        else:
            n = t["n_train"]
            rng.shuffle(exs)
            exs = exs[:n]
        for ex in exs:
            ex = dict(ex)
            ex["_task"] = t["task"]
            ex["_cot_mode"] = t["cot_mode"]
            out.append(ex)
        per_task_counts[t["name"]] = n

    rng.shuffle(out)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(str(out_path), out)

    print(f"=== combined unified JSONL written: {out_path} ===")
    print(f"total examples = {len(out)}")
    for name, n in per_task_counts.items():
        print(f"  {name:14s} {n}")


if __name__ == "__main__":
    main()
