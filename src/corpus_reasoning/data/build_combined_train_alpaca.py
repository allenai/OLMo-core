"""Build a combined multi-task alpaca JSONL for training.

Concatenates examples from 6 task settings, each prompted via
`build_prompt_parts` with its own best-known CoT mode and a shared
`--query-position`. Output is a single alpaca-format JSONL that axolotl can
consume directly.

Task manifest (best CoT per task; see results/*.md for evidence):
    outlier          cot_mode=label    task=outlier            (outlier v2 n=100)
    contradiction    cot_mode=template task=contradiction      (pubmed n=100)
    reorder          cot_mode=successor task=reorder           (gutenberg n=5)
    grouping         --                task=grouping_labeled   (openalex n=20)
    qa (hotpotqa)    --                task=cot_retrieval      (k=100)
    nq retrieval     --                task=retrieval          (k=100, no CoT)

Usage:
    python scripts/data/build_combined_train_alpaca.py \
        --query-position both \
        --out data/combined_train_qboth.jsonl
"""
import argparse
import random
from pathlib import Path

from corpus_reasoning.lib.io import load_jsonl, save_jsonl
from corpus_reasoning.lib.data_format import build_prompt_parts


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
    ap.add_argument("--query-position", choices=["both", "after", "before"], required=True)
    ap.add_argument("--out", required=True, help="Output alpaca JSONL path.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle-cross-task", action="store_true", default=True,
                    help="Shuffle combined examples across tasks (default on).")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_alpaca = []
    per_task_counts = {}

    for t in TASKS:
        src = Path(t["src"])
        if not src.exists():
            raise FileNotFoundError(f"Missing data file for {t['name']}: {src}")
        examples = list(load_jsonl(src))
        if t["n_train"] > len(examples):
            print(f"[warn] {t['name']}: requested {t['n_train']} but only {len(examples)} available — taking all.")
            n = len(examples)
        else:
            n = t["n_train"]
            rng.shuffle(examples)
            examples = examples[:n]

        for ex in examples:
            instr, inp, out = build_prompt_parts(
                ex,
                task=t["task"],
                query_position=args.query_position,
                use_titles=True,
                before_dummy=0,
                after_dummy=0,
                cot_mode=t["cot_mode"],
            )
            all_alpaca.append({
                "instruction": instr,
                "input": inp,
                "output": out,
                "_source_task": t["name"],
            })
        per_task_counts[t["name"]] = n

    if args.shuffle_cross_task:
        rng.shuffle(all_alpaca)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(str(out_path), all_alpaca)

    print(f"=== combined alpaca written: {out_path} ===")
    print(f"query_position = {args.query_position}")
    print(f"total examples = {len(all_alpaca)}")
    for name, n in per_task_counts.items():
        print(f"  {name:14s} {n}")


if __name__ == "__main__":
    main()
