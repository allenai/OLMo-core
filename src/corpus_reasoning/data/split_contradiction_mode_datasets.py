"""Build scale-controlled obvious-vs-subtle contradiction datasets.

Takes a simple-mode (obvious) and a subtle-mode contradiction task JSONL,
subsamples BOTH to the same number of examples (the smaller of the two), then
splits each into train/eval with an identical ratio. This guarantees the two
modes are compared at exactly matched data scale.

Emits 4 task-format JSONL files (same schema as the generator output), which
prepare_contradiction_pair_classifier.py then turns into pair JSONLs.

Usage:
    python scripts/data/split_contradiction_mode_datasets.py \
        --simple data/contradiction_train_pubmed_simple_n20_k3.jsonl \
        --subtle data/contradiction_train_pubmed_subtle_n20_k3.jsonl \
        --out-prefix data/contradiction_modecmp \
        --eval-frac 0.2
"""

import argparse
import random

from corpus_reasoning.lib.io import load_jsonl, save_jsonl


def split(rows, eval_frac):
    n_eval = int(round(len(rows) * eval_frac))
    return rows[n_eval:], rows[:n_eval]  # train, eval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--simple", required=True, help="simple-mode (obvious) task JSONL")
    p.add_argument("--subtle", required=True, help="subtle-mode task JSONL")
    p.add_argument("--out-prefix", required=True,
                   help="output path prefix; writes <prefix>_<mode>_<split>.jsonl")
    p.add_argument("--eval-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    obvious = load_jsonl(args.simple)
    subtle = load_jsonl(args.subtle)

    n = min(len(obvious), len(subtle))
    print(f"obvious={len(obvious)}  subtle={len(subtle)}  -> matched scale n={n}")

    rng.shuffle(obvious)
    rng.shuffle(subtle)
    obvious = obvious[:n]
    subtle = subtle[:n]

    for mode, rows in [("obvious", obvious), ("subtle", subtle)]:
        tr, ev = split(rows, args.eval_frac)
        for split_name, split_rows in [("train", tr), ("eval", ev)]:
            path = f"{args.out_prefix}_{mode}_{split_name}.jsonl"
            save_jsonl(path, split_rows)
            print(f"  wrote {len(split_rows):5d} -> {path}")


if __name__ == "__main__":
    main()
