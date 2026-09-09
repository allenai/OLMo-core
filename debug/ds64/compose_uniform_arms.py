"""
Compose SHORT-HEAVY 2k-64k length-mix training arms (the 2026-09-08 data-scaling campaign,
records/ds64-scaling-plan.md) as NESTED PREFIXES from per-rung pools: token shares SHARES over
2k / 4k / 8k / 16k / 32k / 56k (the top rung is 56k so every example fits the 65536 packing window
with its prompt; evals still score a 64k rung), example counts from each rung's nominal token
count, nested so a smaller budget is a strict subset of every larger one, then order-shuffled
with a fixed seed. Budgets a pool cannot cover are skipped, never silently shrunk.

    python debug/ds64/compose_uniform_arms.py --task outlier --pools-dir POOLS --out-dir ARMS --budgets 32M,64M,128M
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

RUNGS = ["2k", "4k", "8k", "16k", "32k", "56k"]
RUNG_TOKENS = {"2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "56k": 57344}
# SHORT-HEAVY (Prasann 2026-09-08: "train from 2k to 64k, could use short-heavy data"): the standing
# 45/27/16/8/4 shape over 2k-32k, re-weighted so the 32k and 56k rungs carry enough mass to learn
# from (a 4% tail at 128M is ~90 examples at 56k). Token shares:
SHARES = {"2k": 0.30, "4k": 0.20, "8k": 0.15, "16k": 0.13, "32k": 0.12, "56k": 0.10}
SHUFFLE_SEED = 7113


def parse_budget(s: str) -> int:
    s = s.strip().upper()
    mult = {"M": 1_000_000, "K": 1_000, "B": 1_000_000_000}
    return int(float(s[:-1]) * mult[s[-1]]) if s[-1] in mult else int(s)


def counts_for(budget_tokens: int, medians: dict | None = None) -> dict:
    toks = medians or RUNG_TOKENS
    return {r: int(round(budget_tokens * SHARES[r] / toks[r])) for r in RUNGS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--pools-dir", required=True, help="holds <task>_<rung>/<task>/train.jsonl per rung")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budgets", default="16M,32M,64M,128M")
    ap.add_argument("--medians", default="", help="optional JSON {rung: measured median tokens}; default = nominal")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    budgets = [parse_budget(b) for b in args.budgets.split(",")]
    medians = json.load(open(args.medians)) if args.medians else None
    pools_dir, out_dir = pathlib.Path(args.pools_dir), pathlib.Path(args.out_dir)
    pools = {}
    for r in RUNGS:
        f = pools_dir / f"{args.task}_{r}" / args.task / "train.jsonl"
        rows = [ln for ln in open(f) if ln.strip()]
        random.Random(SHUFFLE_SEED + len(r)).shuffle(rows)
        pools[r] = rows
        print(f"pool {r}: {len(rows)} rows")
    manifest = {}
    for B in budgets:
        counts = counts_for(B, medians)
        short = [r for r in RUNGS if len(pools[r]) < counts[r]]
        if short:
            print(f"[SKIP] {B/1e6:.0f}M not buildable: " + ", ".join(f"{r} pool {len(pools[r])} < {counts[r]}" for r in short))
            continue
        lines = [ln for r in RUNGS for ln in pools[r][: counts[r]]]
        random.Random(SHUFFLE_SEED).shuffle(lines)
        arm = f"{args.task}_u{int(B/1e6)}M"
        manifest[arm] = {"spec": counts, "n_examples": len(lines), "target_tokens": B, "shape": "short-heavy 2k-56k " + str(SHARES),
                         "nominal_tokens": sum(counts[r] * (medians or RUNG_TOKENS)[r] for r in RUNGS)}
        print(f"{arm}: {len(lines)} ex, {counts}, ~{manifest[arm]['nominal_tokens']/1e6:.1f}M tok")
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{arm}.jsonl").write_text("".join(lines))
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"MANIFEST_{args.task}.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
