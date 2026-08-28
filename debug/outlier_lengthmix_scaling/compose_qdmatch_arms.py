"""Compose the qdmatch_nq length-mix ARMS from the per-length pools (jsonl space).

Exact analogue of compose_arms.py (outlier): each arm is a deterministic NESTED PREFIX of the
pool's train file, so smaller arms are strict subsets of larger ones and the data-scaling ladder
varies only in how much data, never in which distribution. Arm names are q-prefixed so they can
live in the same S3 arms/ prefix as the outlier arms without colliding.

Pool key = documents (== queries) per example: 9 -> ~2k context, 42 -> ~8k context (read off the
shipped qdmatch_nq eval rungs).
"""

import argparse
import json
import pathlib
import random

ARMS = {
    # --- requested set ---
    "q2k_1250": {9: 1250},
    "q2k_2500": {9: 2500},
    "q2k_5000": {9: 5000},
    "q8k_1000": {42: 1000},
    "q8k_2000": {42: 2000},
    "q8k_4000": {42: 4000},
    # --- headroom arms (same nested-prefix family; extend the ladder if wanted) ---
    "q2k_10000": {9: 10000},
    "q2k_20000": {9: 20000},
    "q8k_8000": {42: 8000},
}
SHUFFLE_SEED = 7113  # same seed as the outlier arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/data/prasann/qdmatch_lengthmix")
    ap.add_argument("--only", default="", help="comma-separated arm subset")
    args = ap.parse_args()
    work = pathlib.Path(args.work)
    out_root = work / "arms"
    out_root.mkdir(exist_ok=True)

    wanted = set(args.only.split(",")) if args.only else set(ARMS)
    pools = {}
    for n, tag in ((9, "q9"), (42, "q42")):
        path = work / f"qdmatch_nq_{tag}_train.jsonl"
        pools[n] = path.read_text().splitlines()
        print(f"pool n{n} ({tag}): {len(pools[n])} examples", flush=True)

    mpath = out_root / "MANIFEST.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    for arm, spec in ARMS.items():
        if arm not in wanted:
            continue
        lines = []
        for n, cnt in spec.items():
            assert len(pools[n]) >= cnt, f"{arm}: pool n{n} has {len(pools[n])} < {cnt}"
            lines += pools[n][:cnt]
        rng = random.Random(SHUFFLE_SEED)
        rng.shuffle(lines)
        out = out_root / f"{arm}.jsonl"
        out.write_text("\n".join(lines) + "\n")
        manifest[arm] = {"spec": {str(k): v for k, v in spec.items()},
                         "n_examples": len(lines), "shuffle_seed": SHUFFLE_SEED,
                         "task": "qdmatch", "source": "qdmatch_nq",
                         "composition": "nested prefixes of qdmatch_nq_q{9,42}_train.jsonl"}
        print(f"arm {arm}: {len(lines)} examples -> {out}", flush=True)
    mpath.write_text(json.dumps(manifest, indent=2))
    print("MANIFEST written", flush=True)


if __name__ == "__main__":
    main()
