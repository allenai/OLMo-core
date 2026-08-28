"""Compose the length-mix experiment's training ARMS from the per-rung outlier pools (jsonl space).

Each arm = deterministic NESTED PREFIXES of the rung train files (so smaller arms are strict
subsets of larger ones, and the transfer arms' fixed target-rung examples are identical across
arms), mixed and then order-shuffled with a fixed seed. Output: one jsonl per arm + manifest.

Arms (counts are EXAMPLES from each rung pool; rung key = doc count n):
  lr2k5000        n14:5000                          (LR sweep / smoke, both variants)
  p8k_250/1k/4k   n57:{250,1000,4000}               (pure-8k anchors)
  p16k_250/1k/4k  n111:{250,1000,4000}              (pure-16k anchors)
  t8k_2k1000/4k   n57:500 + n14:{1000,4000}         (2k->8k transfer)
  t16k_4k1000/4k  n111:500 + n28:{1000,4000}        (4k->16k transfer)
  v_mix           n111:1000 + n28:2000 + n14:2000   (pre-registered validation mix)
"""

import argparse
import json
import pathlib
import random

ARMS = {
    "lr2k5000": {14: 5000},
    "p8k_250": {57: 250},
    "p8k_1000": {57: 1000},
    "p8k_4000": {57: 4000},
    "p16k_250": {111: 250},
    "p16k_1000": {111: 1000},
    "p16k_4000": {111: 4000},
    "t8k_2k1000": {57: 500, 14: 1000},
    "t8k_2k4000": {57: 500, 14: 4000},
    "t16k_4k1000": {111: 500, 28: 1000},
    "t16k_4k4000": {111: 500, 28: 4000},
    "v_mix": {111: 1000, 28: 2000, 14: 2000},
    # ---- check thread v2 (2026-08-27) ----
    "m8k_mix": {57: 4000, 14: 4000},   # check2 (b)
    "p8k_5000": {57: 5000},            # check2 FLOP-matched control for (b)
    "p8k_8000": {57: 8000},            # check2 (c)
    "p2k_1250": {14: 1250},            # check3 2k scaling (0.25x of 5000)
    "p2k_2500": {14: 2500},
    "p2k_10000": {14: 10000},
    "p2k_20000": {14: 20000},
    "p32k_2000": {220: 2000},          # check4 32k candidate
    # ---- wave 3: scaling families for BOTH possible check-2 winners ----
    "msb_1000": {57: 1000, 14: 1000},   # 0.25x of mix (b)
    "msb_2000": {57: 2000, 14: 2000},   # 0.5x
    "msb_8000": {57: 8000, 14: 8000},   # 2x
    "msb_16000": {57: 16000, 14: 16000},# 4x
    "p8k_2000": {57: 2000},             # 0.25x of (c)
    "p8k_16000": {57: 16000},           # 2x
    "p8k_32000": {57: 32000},           # 4x
}
SHUFFLE_SEED = 7113


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/data/prasann/outlier_lengthmix")
    args = ap.parse_args()
    work = pathlib.Path(args.work)
    out_root = work / "arms"
    out_root.mkdir(exist_ok=True)

    pools = {}
    for n in (14, 28, 57, 111, 220):
        path = work / f"outlier_lm_n{n}_train.jsonl"
        pools[n] = path.read_text().splitlines()
        print(f"pool n{n}: {len(pools[n])} examples")

    manifest = {}
    for arm, spec in ARMS.items():
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
                         "composition": "nested prefixes of outlier_lm_n{n}_train.jsonl"}
        print(f"arm {arm}: {len(lines)} examples -> {out}")
    (out_root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("MANIFEST written")


if __name__ == "__main__":
    main()
