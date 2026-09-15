"""
Slice **2k-only** nested-prefix training arms out of an existing ds64 per-rung pool
(``debug/ds64/build_ds64_data_beaker.sh`` writes ``<task>_2k/<task>/train.jsonl``).

The ds64 arms are a short-heavy MIX over 2k..56k (``compose_uniform_arms.py``). This composer keeps
only the 2k rung, so a run at any of these budgets trains on exactly the length bucket the 2k eval
rung scores. That is the whole point of the fast2k loop: compaction saves the same FRACTION of
FLOPs at any length in the linear-cost regime, so accuracy-vs-FLOPs at matched budget is well posed
at 2k, and a screening run costs minutes instead of GPU-hours. **Only length generalisation needs
the full ladder.**

Budgets are named in NOMINAL tokens (2048/example, the rung's nominal length) purely to match the
ds64 naming; what they actually fix is the EXAMPLE count, and every arm -- dense and soft alike --
trains unpacked on the same rows in the same order, so the budget is matched by construction.

    python debug/ds64_fast2k/compose_fast2k.py --task outlier --pool POOL.jsonl --out-dir ARMS \
        --budgets 2M,4M,8M
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

SHUFFLE_SEED = 7113  # same constant as compose_uniform_arms.py; nested prefixes, fixed order
RUNG_TOKENS = 2048


def parse_budget(s: str) -> int:
    s = s.strip().upper()
    mult = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    return int(float(s[:-1]) * mult[s[-1]]) if s[-1] in mult else int(s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--pool", required=True, help="the 2k rung's train.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budgets", default="2M,4M,8M")
    ap.add_argument("--tokens-per-example", type=int, default=RUNG_TOKENS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [ln for ln in open(args.pool) if ln.strip()]
    random.Random(SHUFFLE_SEED).shuffle(rows)
    print(f"pool 2k: {len(rows)} rows <- {args.pool}")
    out_dir = pathlib.Path(args.out_dir)
    manifest = {}
    for b in args.budgets.split(","):
        B = parse_budget(b)
        n = int(round(B / args.tokens_per_example))
        if n > len(rows):
            print(f"[SKIP] {b} needs {n} rows, pool has {len(rows)}")
            continue
        arm = f"{args.task}_f{b.strip()}"
        manifest[arm] = {"n_examples": n, "nominal_tokens": B, "rung": "2k",
                         "tokens_per_example_nominal": args.tokens_per_example, "pool": args.pool}
        print(f"{arm}: {n} examples (~{B/1e6:.1f}M nominal tokens)")
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{arm}.jsonl").write_text("".join(rows[:n]))
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"MANIFEST_{args.task}.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
