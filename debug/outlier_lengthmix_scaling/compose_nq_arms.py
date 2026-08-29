"""Compose the pure-length **nq** ARMS from the per-length pools (jsonl space).

Exact analogue of compose_arms.py (outlier) / compose_qdmatch_arms.py (qdmatch_nq): each arm is a
deterministic NESTED PREFIX of that pool's train file, so a smaller arm is a strict subset of a
larger one and the data-scaling ladder varies only in HOW MUCH data, never in which distribution.

Arm names are `nq`-prefixed so they can share the S3 `arms/` prefix with the outlier `p*` arms and
the qdmatch `q*` arms without colliding.

Pool key = documents per example (k): the 2k and 8k pools, measured -- not assumed -- by
build_nq_pools.py --calibrate.
"""

import argparse
import json
import pathlib
import random

ARMS = {
    "nq2k_1250": ("n2k", 1250),
    "nq2k_2500": ("n2k", 2500),
    "nq2k_5000": ("n2k", 5000),
    "nq2k_10000": ("n2k", 10000),
    "nq2k_20000": ("n2k", 20000),
    "nq8k_1000": ("n8k", 1000),
    "nq8k_2000": ("n8k", 2000),
    "nq8k_4000": ("n8k", 4000),
    "nq8k_8000": ("n8k", 8000),
}
SHUFFLE_SEED = 7113  # same seed as the outlier and qdmatch arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/data/prasann/nq_lengthmix")
    ap.add_argument("--only", default="", help="comma-separated arm subset")
    args = ap.parse_args()
    work = pathlib.Path(args.work)
    out_root = work / "arms"
    out_root.mkdir(exist_ok=True)

    wanted = set(args.only.split(",")) if args.only else set(ARMS)
    pools = {}
    for tag in ("n2k", "n8k"):
        pools[tag] = (work / f"nq_{tag}_train.jsonl").read_text().splitlines()
        print(f"pool {tag}: {len(pools[tag])} examples", flush=True)

    mpath = out_root / "MANIFEST.json"
    manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
    short = []
    for arm, (tag, cnt) in ARMS.items():
        if arm not in wanted:
            continue
        have = len(pools[tag])
        if have < cnt:
            # Report the ACTUAL count, never the target: emit the whole pool and flag it.
            print(f"!!! {arm}: pool {tag} has {have} < {cnt}; emitting {have}", flush=True)
            short.append(arm)
        lines = list(pools[tag][:cnt])
        rng = random.Random(SHUFFLE_SEED)
        rng.shuffle(lines)
        out = out_root / f"{arm}.jsonl"
        out.write_text("\n".join(lines) + "\n")
        manifest[arm] = {"pool": tag, "requested": cnt, "n_examples": len(lines),
                         "shuffle_seed": SHUFFLE_SEED, "task": "retrieval", "source": "nq",
                         "short_of_request": len(lines) < cnt,
                         "composition": f"nested prefix of nq_{tag}_train.jsonl"}
        print(f"arm {arm}: {len(lines)} examples -> {out}", flush=True)
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"MANIFEST written{'; SHORT ARMS: ' + ','.join(short) if short else ''}", flush=True)


if __name__ == "__main__":
    main()
