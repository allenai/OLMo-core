"""Measure the realized context-length distribution of the built shards.

The band plan in ``bands.py`` targets *rendered token counts* via a fitted calibration, so the
realized distribution drifts a little from the plan (the fit has 1.2-3.3% MAPE and the converter
drops anything over ``--seq-len``). This reads the actual shards and counts instances per band, so
the 128-256k floor of 300 can be verified rather than assumed.

Instances are EOS-terminated in the flat uint32 stream (dense emit), so instance lengths are the
gaps between EOS ids.
"""

import argparse
import glob
import os

import numpy as np

EOS_QWEN3_5 = 248044
BANDS = [
    (0, 2048),
    (2048, 4096),
    (4096, 8192),
    (8192, 16384),
    (16384, 32768),
    (32768, 65536),
    (65536, 131072),
    (131072, 262145),
]
FLOOR_BAND = (131072, 262145)
FLOOR = 300


def instance_lengths(shard_dir: str, eos: int) -> np.ndarray:
    lens = []
    for f in sorted(glob.glob(os.path.join(shard_dir, "token_ids_part_*.npy"))):
        a = np.fromfile(f, dtype=np.uint32)
        idx = np.flatnonzero(a == eos)
        prev = 0
        for i in idx:
            lens.append(int(i) - prev + 1)
            prev = int(i) + 1
    return np.array(lens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/prasann/xlong5/shards")
    ap.add_argument("--eos", type=int, default=EOS_QWEN3_5)
    args = ap.parse_args()

    labels = [f"{lo // 1024}-{hi // 1024}k" for lo, hi in BANDS]
    print("task".ljust(16) + "".join(lbl.rjust(10) for lbl in labels) + "      max   FLOOR")
    ok = True
    for d in sorted(glob.glob(os.path.join(args.root, "*_train"))):
        task = os.path.basename(d).replace("_train", "")
        lens = instance_lengths(d, args.eos)
        if not len(lens):
            print(f"{task:16s}  (no instances)")
            ok = False
            continue
        row = "".join(
            str(int(((lens >= lo) & (lens < hi)).sum())).rjust(10) for lo, hi in BANDS
        )
        top = int(((lens >= FLOOR_BAND[0]) & (lens < FLOOR_BAND[1])).sum())
        verdict = "PASS" if top >= FLOOR else f"FAIL({top}<{FLOOR})"
        ok = ok and top >= FLOOR
        print(f"{task:16s}{row}{int(lens.max()):>9}   {verdict}")
    print("\nALL BANDS OK" if ok else "\nFLOOR VIOLATION -- top up the 128-256k band")


if __name__ == "__main__":
    main()
