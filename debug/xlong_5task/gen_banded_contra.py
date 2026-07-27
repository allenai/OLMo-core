"""Generate the contradiction 2k->256k short-skewed pool (banded, streaming, shardable).

Generalizes ``debug/qwen3_vs_qwen35_contra/gen_uniform_mix.py`` from a flat uniform draw to the
banded short-heavy spec in ``bands.py``. Per example: pick a target rendered-token count from the
band plan, convert to a document count via the measured **Qwen3.5** calibration, then resize an
audited gold-pair example with fresh PubMed fillers (``expand_example`` -- keeps the gold pairs
intact, no LLM).

Streams output line-by-line: at n=6176 docs an example is ~1MB of JSON and the full pool is ~8.2M
document copies, so accumulating in a list before writing would need several GB.

``--num-shards/--shard-index`` split the work across array tasks; the band plan is shuffled first
so every shard gets a similar length mix (otherwise one shard holds all the 256k examples).
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src")

from corpus_reasoning.data.generate_pubmed_contradiction_data import (  # noqa: E402
    expand_example,
    load_jsonl,
    load_pubmed_pool,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bands import band_label, draw_plan, n_for_tokens  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="audited gold pool JSONL (documents + gold pairs)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pool-abstracts", type=int, default=80000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--limit-src", type=int, default=4000, help="gold examples to cycle over")
    args = ap.parse_args()

    plan = draw_plan(args.seed)
    mine = [(i, t) for i, t in enumerate(plan) if i % args.num_shards == args.shard_index]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(mine)} of {len(plan)} examples", flush=True)

    # Only the gold pairs and their source docs are reused; the rest is fresh filler, so a few
    # thousand base examples is plenty of diversity and keeps the load cheap.
    src = load_jsonl(args.src)[: args.limit_src]
    print(f"loaded {len(src)} source gold examples from {args.src}", flush=True)

    max_n = max(n_for_tokens("contradiction", t) for _, t in mine) if mine else 0
    print(f"max n_docs this shard = {max_n}", flush=True)
    _, filler_pool = load_pubmed_pool(args.pool_abstracts, args.seed + 1)
    print(f"filler pool: {len(filler_pool)} abstracts", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    hist: dict = {}
    t0 = time.time()
    with open(args.out, "w") as f:
        for j, (gi, target_tok) in enumerate(mine):
            n = n_for_tokens("contradiction", target_tok)
            base = src[gi % len(src)]
            # Global index seeds the RNG so a re-run with different --num-shards is identical.
            ex_rng = random.Random(args.seed * 1_000_003 + gi)
            ex = expand_example(base, filler_pool, n, ex_rng)
            f.write(json.dumps(ex) + "\n")
            lab = band_label(target_tok)
            hist[lab] = hist.get(lab, 0) + 1
            if (j + 1) % 250 == 0:
                el = time.time() - t0
                print(
                    f"  {j + 1}/{len(mine)}  {el:.0f}s  {el / (j + 1):.2f}s/ex  last_n={n}",
                    flush=True,
                )

    print(f"wrote {len(mine)} examples -> {args.out}", flush=True)
    print(f"band histogram: {json.dumps(hist, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
