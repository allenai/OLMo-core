"""
Compose SHORT-HEAVY length-mix training arms at several token budgets, as NESTED PREFIXES, from
per-rung pools (jsonl space). One script for every task of the FLOP-scaling study
(``records/flop-scaling-ffn-kv-plan.md`` §3).

Shape (the standing directive from the outlier length-mix campaign): 45 / 27 / 16 / 8 / 4 % of
TOKENS at the 2k / 4k / 8k / 16k / 32k rungs. Example counts use each rung's NOMINAL token count
(the ladder labels are calibrated token targets); the tokenizer's exact count is recorded by the
converter afterwards and lands in the shard metadata.

Nesting: every pool file is shuffled ONCE with a fixed seed, and arm B takes the first
``count_r(B)`` rows of each rung pool; since ``count_r`` is monotone in the budget, a smaller arm
is a strict subset of every larger one -- the data-scaling ladder varies only in HOW MUCH data.
The concatenated arm is then order-shuffled with a fixed seed (so a run does not see all 32k
examples last) and written as one jsonl, plus a manifest with the exact counts.

    python compose_shortheavy_arms.py --task outlier --pools-dir POOLS --out-dir ARMS \\
        --budgets 8M,16M,32M,64M,128M
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

RUNGS = ["2k", "4k", "8k", "16k", "32k"]
RUNG_TOKENS = {"2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768}
SHORT_HEAVY = {"2k": 0.45, "4k": 0.27, "8k": 0.16, "16k": 0.08, "32k": 0.04}
SHUFFLE_SEED = 7113  # same seed the outlier campaign used


def parse_budget(s: str) -> int:
    s = s.strip().upper()
    mult = {"M": 1_000_000, "K": 1_000, "B": 1_000_000_000}
    return int(float(s[:-1]) * mult[s[-1]]) if s[-1] in mult else int(s)


def counts_for(budget_tokens: int) -> dict:
    return {r: int(round(budget_tokens * SHORT_HEAVY[r] / RUNG_TOKENS[r])) for r in RUNGS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--pools-dir", required=True, help="holds <task>_<rung>/<task>/train.jsonl per rung")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budgets", default="8M,16M,32M,64M,128M")
    ap.add_argument("--dry-run", action="store_true", help="print the counts, write nothing")
    args = ap.parse_args()

    budgets = [parse_budget(b) for b in args.budgets.split(",")]
    pools_dir = pathlib.Path(args.pools_dir)
    out_dir = pathlib.Path(args.out_dir)

    # Load + shuffle each rung pool once (fixed seed => nested prefixes across budgets).
    pools: dict = {}
    for r in RUNGS:
        f = pools_dir / f"{args.task}_{r}" / args.task / "train.jsonl"
        if not f.exists():
            raise SystemExit(f"missing pool {f}")
        rows = f.read_text().splitlines()
        rng = random.Random(SHUFFLE_SEED + RUNGS.index(r))
        rng.shuffle(rows)
        pools[r] = rows
        print(f"[pool] {args.task} {r}: {len(rows)} examples")

    need_max = counts_for(max(budgets))
    short = {r: need_max[r] - len(pools[r]) for r in RUNGS if need_max[r] > len(pools[r])}
    if short:
        raise SystemExit(f"pools too small for the largest budget: need more by {short}")

    manifest = {"task": args.task, "shape": SHORT_HEAVY, "rung_tokens": RUNG_TOKENS, "arms": {}}
    out_dir.mkdir(parents=True, exist_ok=True)
    for b in budgets:
        counts = counts_for(b)
        label = f"{args.task}_sh{b // 1_000_000}M"
        nominal = sum(counts[r] * RUNG_TOKENS[r] for r in RUNGS)
        manifest["arms"][label] = {"budget": b, "counts": counts, "nominal_tokens": nominal}
        print(f"[arm] {label}: {counts} -> {sum(counts.values())} examples, {nominal / 1e6:.1f}M nominal tokens")
        if args.dry_run:
            continue
        rows = []
        for r in RUNGS:
            rows.extend(pools[r][: counts[r]])
        random.Random(SHUFFLE_SEED + 1000 + b % 997).shuffle(rows)
        (out_dir / f"{label}.jsonl").write_text("\n".join(rows) + "\n")
    if not args.dry_run:
        (out_dir / f"{args.task}_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
        print(f"wrote {len(budgets)} arms + manifest -> {out_dir}")


if __name__ == "__main__":
    main()
