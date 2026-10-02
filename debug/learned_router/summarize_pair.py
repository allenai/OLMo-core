"""Paired-budget re-score: each router keeps, per row, as many routed tokens as gold_fl20p8_noslot keeps on
that same row (``router_<name>@pair``); paired dCE vs the bar with tails. eval_size 16/16/8 -- flagged.

    python debug/learned_router/summarize_pair.py [--res results_router_pair]
"""
import argparse
import glob
import json
import math
import os

import numpy as np

from summarize_len5 import verdict

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(os.path.dirname(HERE), "devloss_grid")
BAR = "gold_fl20p8_noslot"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_router_pair")
    a = ap.parse_args()
    for f in sorted(glob.glob(os.path.join(GRID, a.res, "*.json"))):
        d = json.load(open(f))
        pr = d["per_row"]
        full, b = np.asarray(pr["full"]["ce"]), np.asarray(pr[BAR]["ce"])
        cB, dB = float(np.mean(pr[BAR]["compaction"])), float((b - full).mean())
        print(f"\n{os.path.basename(f)}  bar ×{cB:.3f} {dB:+.3f}  (⚠ eval_size {len(full)})")
        for s in pr:
            if not s.endswith("@pair"):
                continue
            r = np.asarray(pr[s]["ce"])
            diff = r - b
            p = {"comp": float(np.mean(pr[s]["compaction"])), "dce": float((r - full).mean()), "paired": float(diff.mean()),
                 "se": float(diff.std(ddof=1) / math.sqrt(len(diff)))}
            print(f"  {s[len('router_'):-len('@pair')]:24s} ×{p['comp']:.3f} paired {p['paired']:+.3f} ± {p['se']:.3f} (med {np.median(diff):+.3f}, "
                  f"max {diff.max():+.3f}, {(diff > 0.1).sum()} rows >0.1) -> {verdict(p, cB, dB)}")


if __name__ == "__main__":
    main()
