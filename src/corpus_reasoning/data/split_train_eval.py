"""Carve a JSONL into disjoint train/eval subsets (a held-out split of the same
distribution). Useful for tasks that only ship one split (e.g. a benchmark's
test set) — shuffle deterministically and split, so train and eval are matched
in distribution with no overlap.

Usage:
    python scripts/data/split_train_eval.py \\
        --input data/beir_scifact_test_k20_300.jsonl --train-frac 0.8
    # -> data/beir_scifact_test_k20_300_split{train,eval}.jsonl
"""

import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default=None,
                    help="defaults to the input's directory")
    args = ap.parse_args()

    rows = [l for l in open(args.input) if l.strip()]
    random.Random(args.seed).shuffle(rows)
    n_train = int(len(rows) * args.train_frac)

    import os
    stem = args.input[:-6] if args.input.endswith(".jsonl") else args.input
    if args.output_dir:
        stem = os.path.join(args.output_dir, os.path.basename(stem))
    for label, chunk in [("train", rows[:n_train]), ("eval", rows[n_train:])]:
        path = f"{stem}_split{label}.jsonl"
        with open(path, "w") as f:
            f.writelines(chunk)
        print(f"{label}: {len(chunk)} -> {path}")


if __name__ == "__main__":
    main()
