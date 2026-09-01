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

# pool key -> pool-file tag. The 16k/32k/64k entries are registered at runtime from the
# --n16k/--n32k/--m64k+--n64k flags, because their shapes are MEASURED by
# build_qdmatch_pools.py --calibrate, never assumed. Keys are ints for the symmetric rungs
# (the key IS M == N) and the string "q64k" for the asymmetric 64k rung -- see --n64k in the
# builder for why 64k cannot be symmetric.
POOL_TAG = {9: "q9", 42: "q42"}

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
    ap.add_argument("--arms", default="",
                    help="alias of --only (the certainty-wave bigmix sbatch calls with this "
                         "spelling); also registers the big q32k_32000 arm when requested")
    ap.add_argument("--n16k", type=int, default=0,
                    help="M=N behind the 16k pool; enables the q16k_* arms")
    ap.add_argument("--n32k", type=int, default=0,
                    help="M=N behind the 32k pool; enables the q32k_* arms")
    ap.add_argument("--m64k", type=int, default=0, help="M behind the (asymmetric) 64k pool")
    ap.add_argument("--n64k", type=int, default=0,
                    help="N behind the 64k pool; with --m64k enables the q64k_* arms")
    args = ap.parse_args()
    if args.arms:
        assert not args.only, "--arms is an alias of --only; give one"
        args.only = args.arms
    # the 16x arm of the qdmatch-ceiling test (M=N=172 measured in BUILD_REPORT_32k64k.json);
    # its pool is extended by debug_build_q32k_big.py, so no --n32k needed to address it
    if "q32k_32000" in set(args.only.split(",")):
        POOL_TAG[172] = "q32k"
        ARMS["q32k_32000"] = {172: 32000}
    work = pathlib.Path(args.work)
    out_root = work / "arms"
    out_root.mkdir(exist_ok=True)

    if args.n16k:
        POOL_TAG[args.n16k] = "q16k"
        ARMS["q16k_2000"] = {args.n16k: 2000}
        ARMS["q16k_8000"] = {args.n16k: 8000}
    if args.n32k:
        POOL_TAG[args.n32k] = "q32k"
        ARMS["q32k_2000"] = {args.n32k: 2000}
        ARMS["q32k_8000"] = {args.n32k: 8000}
    if args.m64k or args.n64k:
        assert args.m64k and args.n64k, "--m64k and --n64k must be given together"
        POOL_TAG["q64k"] = "q64k"          # asymmetric: the pool key is the tag itself
        ARMS["q64k_1000"] = {"q64k": 1000}
        ARMS["q64k_4000"] = {"q64k": 4000}

    wanted = set(args.only.split(",")) if args.only else set(ARMS)
    assert wanted <= set(ARMS), f"unknown arms: {sorted(wanted - set(ARMS))}"
    # Load ONLY the pools the wanted arms actually need (a q16k-only extension must not read the
    # 550 MB q9/q42 train files).
    pools = {}
    for arm in wanted:
        for n in ARMS[arm]:
            if n in pools:
                continue
            path = work / f"qdmatch_nq_{POOL_TAG[n]}_train.jsonl"
            pools[n] = path.read_text().splitlines()
            print(f"pool n{n} ({POOL_TAG[n]}): {len(pools[n])} examples", flush=True)

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
                         "shape": {str(k): {"M": args.m64k, "N": args.n64k} if k == "q64k"
                                   else {"M": k, "N": k} for k in spec},
                         "composition": "nested prefixes of " + ", ".join(
                             f"qdmatch_nq_{POOL_TAG[k]}_train.jsonl" for k in spec)}
        print(f"arm {arm}: {len(lines)} examples -> {out}", flush=True)
    mpath.write_text(json.dumps(manifest, indent=2))
    print("MANIFEST written", flush=True)


if __name__ == "__main__":
    main()
