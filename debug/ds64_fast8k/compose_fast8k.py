"""
Slice **8k-only** training arms out of the ds64 campaign's existing 8k rung pool -- the length
sibling of ``debug/ds64_fast2k/compose_fast2k.py`` (whose ``parse_budget`` this reuses verbatim).

Why a second composer instead of the 2k one with a different ``--tokens-per-example``: fast8k needs
one thing fast2k does not, the **disjoint tail slice**. The two-phase arm trains phase 1 on the
first 85% of a budget and phase 2 on the remaining 15%; with prefix-only slicing phase 2 would
re-train on rows phase 1 already saw, and "a short real-body finetune rescues the arm" would be
indistinguishable from "a second epoch on the same 1660 rows". So every slice here is an explicit
``name:start:count`` window into the shuffled pool, and the named budgets are sugar for
``<task>_g<B>:0:round(B / tokens_per_example)``.

The shuffle seed is the ds64 constant, so slice ``[0:n]`` here is the same prefix of the same
ordering fast2k would produce -- only the rung and the tokens/example differ.

    python debug/ds64_fast8k/compose_fast8k.py --task outlier --pool POOL.jsonl --out-dir ARMS \
        --budgets 8M,16M,20M --tokens-per-example 8192 --slices P1:0:1660,P2:1660:293
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ds64_fast2k"))

from compose_fast2k import SHUFFLE_SEED, parse_budget  # noqa: E402

RUNG_TOKENS = 8192


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--pool", required=True, help="the 8k rung's train.jsonl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budgets", default="8M,16M,20M")
    ap.add_argument("--slices", default="", help="extra windows, name:start:count[,...]")
    ap.add_argument("--tokens-per-example", type=int, default=RUNG_TOKENS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = [ln for ln in open(args.pool) if ln.strip()]
    random.Random(SHUFFLE_SEED).shuffle(rows)
    print(f"pool 8k: {len(rows)} rows <- {args.pool}", flush=True)

    windows = []
    for b in args.budgets.split(","):
        b = b.strip()
        if not b:
            continue
        windows.append((b, 0, int(round(parse_budget(b) / args.tokens_per_example))))
    for s in args.slices.split(","):
        s = s.strip()
        if not s:
            continue
        name, start, count = s.split(":")
        windows.append((name, int(start), int(count)))

    out_dir = pathlib.Path(args.out_dir)
    manifest = {}
    for name, start, count in windows:
        if start + count > len(rows):
            # A short pool is a real constraint, not a rounding detail: silently truncating it
            # would break the "same rows, same order, same steps" guarantee the budget rests on.
            print(f"[SKIP] {name} needs rows [{start}:{start + count}], pool has {len(rows)}")
            continue
        arm = f"{args.task}_g{name}"
        manifest[arm] = {"n_examples": count, "row_start": start, "rung": "8k",
                         "nominal_tokens": count * args.tokens_per_example,
                         "tokens_per_example_nominal": args.tokens_per_example, "pool": args.pool}
        print(f"{arm}: rows [{start}:{start + count}] = {count} examples "
              f"(~{count * args.tokens_per_example / 1e6:.1f}M nominal tokens)")
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{arm}.jsonl").write_text("".join(rows[start:start + count]))
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"MANIFEST_{args.task}.json").write_text(json.dumps(manifest, indent=2))
    # The two-phase arms depend on P1 and P2 being DISJOINT; assert it rather than trust the caller.
    spans = {n: (s, s + c) for n, s, c in windows if not n[0].isdigit()}
    for a in spans:
        for b in spans:
            if a < b and spans[a][0] < spans[b][1] and spans[b][0] < spans[a][1]:
                raise SystemExit(f"!!! named slices {a} and {b} OVERLAP: {spans[a]} vs {spans[b]}")


if __name__ == "__main__":
    main()
