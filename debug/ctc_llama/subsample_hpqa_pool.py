"""Subsample the BUILD_MATRIX-§2 hotpotqa pool to the 10k discrete-uniform ladder.

The pool holds 4000 examples per document-count bucket (n in 11/24/50/100/205, bridge questions,
BM25 hard negatives at n/10) and was mined once for the Qwen build. Re-mining it needs pyserini,
whose JVM hangs on the jsteinhardt compute nodes, so the pool is reused verbatim and only
re-tokenized per model family.

Taking 2000 from each of the 5 buckets gives the 10k examples the recipe specifies, uniform over n
-- which is what makes the training length distribution match the 2k/4k/8k/16k eval ladder.

:param --seed: Fixed at 42 by the recipe so every family trains on the SAME examples; that is what
    makes the cross-family comparison a comparison of models rather than of data draws.
"""

import argparse
import glob
import os
import random


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", required=True, help="dir of hotpotqa_train_k*_bridge_hn*_4000.jsonl")
    ap.add_argument("--out", required=True, help="dir to write the per-bucket subsamples into")
    ap.add_argument("--per-bucket", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.pool, "hotpotqa_train_k*_bridge_hn*.jsonl")))
    if not paths:
        raise SystemExit(f"no pool files in {args.pool}")
    os.makedirs(args.out, exist_ok=True)
    total = 0
    for path in paths:
        with open(path) as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        if len(lines) < args.per_bucket:
            raise SystemExit(
                f"{os.path.basename(path)} has only {len(lines)} examples, need {args.per_bucket}"
            )
        # A fresh Random per bucket keyed on the same seed: the draw for one bucket must not depend
        # on how many buckets came before it, or adding/reordering a bucket would silently change
        # every other bucket's sample.
        rng = random.Random(args.seed)
        rng.shuffle(lines)
        keep = lines[: args.per_bucket]
        out = os.path.join(args.out, os.path.basename(path).replace(".jsonl", "_sub.jsonl"))
        with open(out, "w") as f:
            f.write("\n".join(keep) + "\n")
        total += len(keep)
        print(f"  {os.path.basename(path)}: {len(lines)} -> {len(keep)}", flush=True)
    print(f"subsampled total={total} across {len(paths)} buckets (seed {args.seed})")


if __name__ == "__main__":
    main()
