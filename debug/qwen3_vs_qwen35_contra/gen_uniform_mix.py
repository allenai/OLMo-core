#!/usr/bin/env python
"""Generate a uniform 8k..256k length mix of contradiction examples (10k datapoints).

Per-example: draw a target doc-count n ~ Uniform[N_MIN, N_MAX] (spanning ~8k..256k rendered
tokens at ~46.8 tok/doc), then resize a gold-pair example from the recomb pool to that n by adding
fresh PubMed fillers (expand_example -- keeps the gold pairs, no LLM). This is the CTC joint-training
protocol (n drawn over the ladder) extended to the 256k rung. Output feeds the SAME
convert_unified_to_document_landmark.py + PackingInstanceSource path as the fixed-length shards.
"""
import argparse
import json
import random
import sys

sys.path.insert(0, "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src")
from corpus_reasoning.data.generate_pubmed_contradiction_data import (  # noqa: E402
    expand_example,
    load_jsonl,
    load_pubmed_pool,
    save_jsonl,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="recomb gold pool JSONL (n=50, has documents+gold)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=10000)
    ap.add_argument("--n-min", type=int, default=175)  # ~8k tokens
    ap.add_argument("--n-max", type=int, default=5400)  # ~250k tokens
    ap.add_argument("--pool-abstracts", type=int, default=80000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = load_jsonl(args.src)
    print(f"loaded {len(src)} source gold examples from {args.src}", flush=True)
    _, filler_pool = load_pubmed_pool(args.pool_abstracts, args.seed + 1)
    print(f"filler pool: {len(filler_pool)} abstracts", flush=True)

    rng = random.Random(args.seed)
    out = []
    hist = {"8-32k": 0, "32-64k": 0, "64-128k": 0, "128-256k": 0}
    for i in range(args.num):
        base = src[i % len(src)]
        n = rng.randint(args.n_min, args.n_max)
        ex_rng = random.Random(args.seed * 1_000_003 + i)
        out.append(expand_example(base, filler_pool, n, ex_rng))
        tok = 289 + 46.8 * n
        if tok < 32000:
            hist["8-32k"] += 1
        elif tok < 64000:
            hist["32-64k"] += 1
        elif tok < 128000:
            hist["64-128k"] += 1
        else:
            hist["128-256k"] += 1
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{args.num}", flush=True)
    save_jsonl(args.out, out)
    print(f"wrote {len(out)} examples -> {args.out}", flush=True)
    print(f"approx length histogram (rendered tokens): {json.dumps(hist)}", flush=True)


if __name__ == "__main__":
    main()
