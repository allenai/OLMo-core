#!/usr/bin/env python3
"""FIX 1: merge the dense +CPT-mix RULER (L1024/L2048) block + ruler_avg_recall back into the
dense +CPT-mix native ladder JSON (which ran --skip-ruler during OOM recovery, leaving ruler={}
and ruler_avg_recall=None).
"""
import json
import os

EVDIR = "/scratch/users/prasann/corpus-reasoning/outputs/eval_results"
SRC = os.path.join(EVDIR, "q4b-dense-cptmix_ruler_L1024L2048.json")
DST = os.path.join(EVDIR, "q4b-dense-cptmix-32k-native_ladder.json")


def main():
    if not os.path.exists(SRC):
        print(f"[fix1] SRC not ready: {SRC}")
        return
    src = json.load(open(SRC))
    ruler = src.get("ruler") or {}
    if len(ruler) < 14:
        print(f"[fix1] SRC ruler block incomplete (n={len(ruler)}); not merging yet.")
        return
    dst = json.load(open(DST))
    dst["ruler"] = ruler
    recalls = [v["recall"] for v in ruler.values() if isinstance(v, dict) and "recall" in v]
    dst["ruler_avg_recall"] = sum(recalls) / len(recalls) if recalls else None
    json.dump(dst, open(DST, "w"), indent=2)
    print(f"[fix1] merged {len(ruler)} RULER cells, ruler_avg_recall={dst['ruler_avg_recall']:.4f} -> {DST}")


if __name__ == "__main__":
    main()
