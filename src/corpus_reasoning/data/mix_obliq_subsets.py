"""Mix OBLIQ per-subset JSONLs into one unified train+test pair.

Inputs are the per-subset files emitted by `generate_obliq_data.py`. Each
example already carries `source: obliq_<subset>`, so the mix is a plain
concatenation. A deterministic shuffle keeps subsets from clumping inside
the file (helps batch composition in training).

Usage:
    python scripts/data/mix_obliq_subsets.py \\
        --train data/obliq_writing_train_k30_p3_*.jsonl \\
                data/obliq_congress_train_k30_p3_*.jsonl \\
                data/obliq_twitter_train_k500_p3_*.jsonl \\
                data/obliq_wildchat_train_k50_p3_*.jsonl \\
        --test  data/obliq_writing_test_k30_*.jsonl \\
                data/obliq_congress_test_k30_*.jsonl \\
                data/obliq_twitter_test_k500_*.jsonl \\
                data/obliq_wildchat_test_k50_*.jsonl \\
        --out-tag mix4_kmixed
"""

import argparse
import glob
import json
import random
from collections import Counter
from pathlib import Path

from corpus_reasoning.lib.io import save_jsonl


def _load_many(paths):
    examples = []
    for pat in paths:
        for path in sorted(glob.glob(pat)):
            with open(path) as f:
                rows = [json.loads(l) for l in f]
            print(f"  {path}: {len(rows)} examples  ({rows[0].get('source', '?')})")
            examples.extend(rows)
    return examples


def _summarize(examples, label):
    src_counts = Counter(ex.get("source", "?") for ex in examples)
    print(f"\n{label}: {len(examples)} examples")
    for src, n in sorted(src_counts.items()):
        print(f"  {src}: {n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", required=True,
                        help="Per-subset train JSONLs (globs ok).")
    parser.add_argument("--test", nargs="+", required=True,
                        help="Per-subset test JSONLs (globs ok).")
    parser.add_argument("--out-tag", default="mix4",
                        help="Tag baked into output filenames.")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    rng = random.Random(args.seed)

    print("Loading train files:")
    train = _load_many(args.train)
    rng.shuffle(train)
    train_path = out_dir / f"obliq_{args.out_tag}_train_{len(train)}.jsonl"
    save_jsonl(str(train_path), train)
    _summarize(train, f"Train → {train_path}")

    print("\nLoading test files:")
    test = _load_many(args.test)
    # Don't shuffle test — keep subset blocks contiguous for easier inspection
    test_path = out_dir / f"obliq_{args.out_tag}_test_{len(test)}.jsonl"
    save_jsonl(str(test_path), test)
    _summarize(test, f"Test → {test_path}")


if __name__ == "__main__":
    main()
