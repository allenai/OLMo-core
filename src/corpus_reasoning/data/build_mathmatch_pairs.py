"""Build balanced (text_a, text_b, label) pair JSONLs for the mathmatch
classifier difficulty experiment.

For a given tolerance x:
  - positive: two expressions whose answers satisfy  0 < |val_a - val_b| <= x
  - negative: two expressions whose answers satisfy      |val_a - val_b| > x

(Excluding |diff|=0 from positives so the model can't trivially pattern-match
"identical answers"; at x=0 this means there are no positives, so we use
|diff|<=x with x>=1 only here. For x=0, see --include-zero-diff.)

Both positives and negatives are sampled by drawing pairs of values
independently from a fixed integer range, then expressions are constructed
deterministically. This way the only difference across x configs is the
labeling threshold — answer / expression distributions are identical.
"""

import argparse
import random

from corpus_reasoning.data.generate_mathmatch_data import build_expression
from corpus_reasoning.lib.io import save_jsonl


def sample_value_pair(rng, vmin, vmax, diff_lo, diff_hi):
    """Sample (v1, v2) with |v1-v2| uniformly in [diff_lo, diff_hi]."""
    while True:
        v1 = rng.randint(vmin, vmax)
        d = rng.randint(diff_lo, diff_hi)
        sign = rng.choice([-1, 1])
        v2 = v1 + sign * d
        if vmin <= v2 <= vmax:
            return v1, v2


def build_pairs(n, label, diff_lo, diff_hi, vmin, vmax, rng):
    out = []
    for _ in range(n):
        v1, v2 = sample_value_pair(rng, vmin, vmax, diff_lo, diff_hi)
        e1 = build_expression(v1, rng)
        e2 = build_expression(v2, rng)
        out.append({
            "text_a": e1, "text_b": e2, "label": label,
            "_meta": {"val_a": v1, "val_b": v2, "diff": abs(v1 - v2)},
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=int, required=True,
                    help="x: positives have |diff|<=x AND |diff|>=1; "
                         "negatives have |diff|>x")
    ap.add_argument("--max-diff", type=int, default=100,
                    help="upper bound on |diff| for negatives sampling")
    ap.add_argument("--ans-min", type=int, default=-50)
    ap.add_argument("--ans-max", type=int, default=50)
    ap.add_argument("--n-train-per-class", type=int, default=5000)
    ap.add_argument("--n-eval-per-class", type=int, default=1000)
    ap.add_argument("--include-zero-diff", action="store_true",
                    help="allow |diff|=0 in positives (default off)")
    ap.add_argument("--output-prefix", required=True,
                    help="writes <prefix>_train.jsonl and <prefix>_eval.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pos_lo = 0 if args.include_zero_diff else 1
    pos_hi = args.tolerance
    neg_lo = args.tolerance + 1
    neg_hi = args.max_diff

    if pos_hi < pos_lo:
        raise SystemExit(
            f"No valid positive diff range: x={args.tolerance} with "
            f"include_zero_diff={args.include_zero_diff}"
        )

    for split, n in [("train", args.n_train_per_class),
                     ("eval", args.n_eval_per_class)]:
        pos = build_pairs(n, 1, pos_lo, pos_hi, args.ans_min, args.ans_max, rng)
        neg = build_pairs(n, 0, neg_lo, neg_hi, args.ans_min, args.ans_max, rng)
        rows = pos + neg
        rng.shuffle(rows)
        path = f"{args.output_prefix}_{split}.jsonl"
        save_jsonl(path, rows)
        print(f"  {split}: {len(pos)} pos + {len(neg)} neg -> {path}")


if __name__ == "__main__":
    main()
