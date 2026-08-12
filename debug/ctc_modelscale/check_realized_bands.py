#!/usr/bin/env python3
"""Verify the realized length distribution of a generated OOLONG pool (the capacity-trap guard).

WHY THIS IS NOT OPTIONAL. `generate_oolong_ladder_data.py` packs items drawn WITHOUT REPLACEMENT
from a single source dataset's item pool and stops once the token budget is met. If that pool holds
fewer tokens than the target, the loop simply runs out and emits a **short example with no error**
(the trap documented in `debug/xlong_5task/gen_oolong_ultra.sbatch`, where a 2M rung could silently
be a 700k rung). So a band's *requested* range proves nothing about what it contains.

This reads the generated JSONL pool directly, tokenizes each example's packed item stream, and
reports the realized length distribution per band plus the fraction of examples that actually
landed inside their requested [lo, hi] window. It exits non-zero if any band's in-window rate falls
below --min-in-band, so it can gate the conversion step.

Unlike `debug/xlong_5task/check_bands.py`, which reads tokenized shards, this runs on the raw pool
BEFORE conversion -- the point is to catch a bad pool before spending a tokenization pass on it.

    python debug/ctc_modelscale/check_realized_bands.py --pool-root /data/prasann/ctc_oolong_ctc/pools
"""
import argparse
import glob
import json
import os
import statistics
import sys

# Band table must match gen_oolong_ctc_bands.sbatch.
BANDS = [(2048, 3973), (4096, 7946), (8192, 15892), (16384, 31784), (32768, 63569)]
TOKENIZER = "/data/prasann/hf_models/Qwen3.5-4B-Base"


def example_text(ex):
    """The packed item stream of one oolong example (one 'document' holding all items)."""
    docs = ex.get("documents") or []
    parts = []
    for d in docs:
        parts.append(d.get("text", "") if isinstance(d, dict) else str(d))
    for q in ex.get("queries") or []:
        parts.append(q)
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool-root", required=True,
                    help="dir holding band<B>_shard<S>/ subdirs of generated JSONL")
    ap.add_argument("--sample", type=int, default=60, help="examples tokenized per band")
    ap.add_argument("--min-in-band", type=float, default=0.80,
                    help="fail if a band has fewer than this fraction inside [lo,hi]")
    ap.add_argument("--tokenizer", default=TOKENIZER)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    print(f"{'band':>6s} {'requested':>16s} {'files':>6s} {'ex':>7s} {'p10':>8s} {'p50':>8s} "
          f"{'p90':>8s} {'in-band':>8s}")
    failed = []
    for b, (lo, hi) in enumerate(BANDS):
        files = sorted(glob.glob(os.path.join(args.pool_root, f"band{b}_shard*", "*.jsonl")))
        if not files:
            print(f"{b:6d} {f'[{lo},{hi}]':>16s} {'0':>6s}  -- NO FILES")
            failed.append((b, "no files"))
            continue
        rows, total = [], 0
        for f in files:
            with open(f) as fh:
                for line in fh:
                    total += 1
                    if len(rows) < args.sample:
                        rows.append(json.loads(line))
        lens = [len(tok(example_text(r), add_special_tokens=False)["input_ids"]) for r in rows]
        if not lens:
            print(f"{b:6d} {f'[{lo},{hi}]':>16s} {len(files):6d} {'0':>7s}  -- EMPTY")
            failed.append((b, "empty"))
            continue
        lens.sort()
        p10 = lens[int(0.1 * (len(lens) - 1))]
        p50 = statistics.median(lens)
        p90 = lens[int(0.9 * (len(lens) - 1))]
        in_band = sum(1 for x in lens if lo <= x <= hi) / len(lens)
        flag = "" if in_band >= args.min_in_band else "  <-- BELOW THRESHOLD"
        print(f"{b:6d} {f'[{lo},{hi}]':>16s} {len(files):6d} {total:7d} {p10:8.0f} {p50:8.0f} "
              f"{p90:8.0f} {in_band:7.1%}{flag}")
        if in_band < args.min_in_band:
            failed.append((b, f"in-band {in_band:.1%}"))

    if failed:
        print("\nVERDICT: FAIL -- " + "; ".join(f"band {b}: {why}" for b, why in failed))
        print("A short band means the packer ran out of items (capacity trap), NOT a config typo.")
        sys.exit(1)
    print("\nVERDICT: PASS -- every band realized inside its requested window")


if __name__ == "__main__":
    main()
