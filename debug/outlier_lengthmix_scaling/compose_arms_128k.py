"""Compose ONLY the new 128k-transfer-test arms from the per-rung outlier pools.

Deliberately separate from compose_arms_lengthmix.py: that script rewrites *every* arm jsonl
(7.3 GB) on each run, which is both slow and unsafe while training/eval jobs are reading the
existing arms. This one touches only the four new arms.

Same composition rule as compose_arms_lengthmix.py so the new arms stay comparable:
nested PREFIXES of outlier_lm_n{n}_train.jsonl, concatenated, then order-shuffled with
SHUFFLE_SEED=7113.

Arms:
  p128k_500    {880: 500}    pure-128k, ~64M tokens
  p128k_1500   {880: 1500}   pure-128k, ~192M tokens
  mix_sx64M    short-heavy with a 128k tail, ~64M tokens
  mix_sx200M   same shape, ~200M tokens (scaled down proportionally if a pool caps)

The mix arms are specified by a TOKEN-SHARE vector, not by example counts: counts are derived
from the MEASURED median tokens/example of each rung so the realised token split matches the
intent. Everything is printed.
"""

import argparse
import json
import pathlib
import random

SHUFFLE_SEED = 7113

# Measured median tokens/example (Qwen/Qwen3.5-0.8B-Base, query_position=after, +eos).
# n14/28/57/111 from /data/prasann/outlier_lengthmix/tokenized/n{n}_train/metadata.json,
# n220 from arms_tokenized/p32k_2000/metadata.json, n880 measured by the pilot (--n880-median).
MEDIAN_TOKENS = {14: 2109, 28: 4120, 57: 8306, 111: 16129, 220: 32027}

# Token shares for the mix_sx family: 42/25/15/8/4/6 % at 2k/4k/8k/16k/32k/128k.
MIX_SHARES = {14: 0.42, 28: 0.25, 57: 0.15, 111: 0.08, 220: 0.04, 880: 0.06}
MIX_BUDGETS = {"mix_sx64M": 64_000_000, "mix_sx200M": 200_000_000}

PURE = {"p128k_500": {880: 500}, "p128k_1500": {880: 1500}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/data/prasann/outlier_lengthmix")
    ap.add_argument("--n880-median", type=int, required=True,
                    help="Measured median tokens/example of the n=880 pool (from the pilot).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    work = pathlib.Path(args.work)
    out_root = work / "arms"
    out_root.mkdir(exist_ok=True)
    MEDIAN_TOKENS[880] = args.n880_median

    pools = {}
    for n in (14, 28, 57, 111, 220, 880):
        pools[n] = (work / f"outlier_lm_n{n}_train.jsonl").read_text().splitlines()
        print(f"pool n{n:<3}: {len(pools[n]):>6} examples  (median {MEDIAN_TOKENS[n]:>7,} tok/ex)")

    specs: dict[str, dict[int, int]] = dict(PURE)
    for arm, budget in MIX_BUDGETS.items():
        raw = {n: budget * s / MEDIAN_TOKENS[n] for n, s in MIX_SHARES.items()}
        # Proportional scale-down if any pool caps (user protocol: never regenerate pools).
        scale = min(1.0, min(len(pools[n]) / v for n, v in raw.items() if v > 0))
        if scale < 1.0:
            capped = [n for n, v in raw.items() if len(pools[n]) / v == scale]
            print(f"!! {arm}: pool n{capped} caps the arm -> scaling ALL rungs by {scale:.4f}")
        specs[arm] = {n: int(v * scale) for n, v in raw.items()}

    manifest = {}
    for arm, spec in specs.items():
        rows, tok_by_rung = [], {}
        for n, cnt in sorted(spec.items()):
            assert len(pools[n]) >= cnt, f"{arm}: pool n{n} has {len(pools[n])} < {cnt}"
            rows += pools[n][:cnt]
            tok_by_rung[n] = cnt * MEDIAN_TOKENS[n]
        total = sum(tok_by_rung.values())
        share = {n: f"{100*t/total:.1f}%" for n, t in tok_by_rung.items()}
        print(f"\narm {arm}: {len(rows)} examples, ~{total/1e6:.1f}M tokens")
        for n in sorted(spec):
            print(f"    n{n:<3} x{spec[n]:<6} = {tok_by_rung[n]/1e6:>6.1f}M tok  ({share[n]:>5})")
        if not args.dry_run:
            rng = random.Random(SHUFFLE_SEED)
            rng.shuffle(rows)
            (out_root / f"{arm}.jsonl").write_text("\n".join(rows) + "\n")
        manifest[arm] = {
            "spec": {str(k): v for k, v in sorted(spec.items())},
            "n_examples": len(rows),
            "est_tokens": total,
            "est_tokens_by_rung": {str(k): v for k, v in sorted(tok_by_rung.items())},
            "token_share": {str(k): v for k, v in sorted(share.items())},
            "median_tokens_per_example": {str(k): MEDIAN_TOKENS[k] for k in sorted(spec)},
            "shuffle_seed": SHUFFLE_SEED,
            "composition": "nested prefixes of outlier_lm_n{n}_train.jsonl",
        }
    if not args.dry_run:
        (out_root / "MANIFEST_128k.json").write_text(json.dumps(manifest, indent=2))
        print("\nMANIFEST_128k.json written")


if __name__ == "__main__":
    main()
