"""Length transfer of the 2k-trained canonical-recipe routers to 8k / 32k vs gold_fl20p8_noslot at the same rung.

    python debug/learned_router/summarize_len5.py [--res results_router_v5len]

Per task x rung: (a) per-row exact T2/T = the bar's T2/T at that rung (``@c<x>``, x closest to the bar's
mean T2/T); (b) the fixed label-free rule "same routed-keep fraction as at 2k" (``@k<rho>``). Each paired vs
the bar +- SE, with the summarize_vsfl verdict of that single point (tol = max(SE, 0.005)).
eval_size 16 (8k) / 8 (32k) -- flagged.
"""
import argparse
import glob
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(os.path.dirname(HERE), "devloss_grid")
BAR = "gold_fl20p8_noslot"
TOL = 0.005


def verdict(p, cB, dB):
    tol = max(p["se"], TOL)
    if p["comp"] <= 1.02 * cB and p["dce"] <= dB + tol and (p["comp"] <= 0.95 * cB or p["paired"] < -tol):
        return "beats"
    return "matches" if p["comp"] <= 1.05 * cB and abs(p["paired"]) <= tol else "loses"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results_router_v5len")
    ap.add_argument("--rungs", default="8k,32k")
    ap.add_argument("--match", default="", help="only files whose name contains this")
    a = ap.parse_args()
    tally = {}
    out = {}
    for rung in a.rungs.split(","):
        print(f"\n### {rung} (eval_size {8 if rung == '32k' else 16} ⚠)\n| task | bar ×c ΔCE | (a) exact T2/T: paired ± SE | (b) 2k keep: ×c paired ± SE | verdict (a) / (b) |")
        print("|---|---|---|---|---|")
        for f in sorted(glob.glob(os.path.join(GRID, a.res, f"*_{rung}.json"))):
            if a.match and a.match not in os.path.basename(f):
                continue
            t = os.path.basename(f)[: -len(f"_{rung}.json")]
            d = json.load(open(f))
            pr = d["per_row"]
            full, b = np.asarray(pr["full"]["ce"]), np.asarray(pr[BAR]["ce"])
            cB, dB = float(np.mean(pr[BAR]["compaction"])), float((b - full).mean())
            pts = {}
            for s in pr:
                if s.startswith("router_"):
                    r = np.asarray(pr[s]["ce"])
                    diff = r - b
                    pts[s] = {"comp": float(np.mean(pr[s]["compaction"])), "dce": float((r - full).mean()), "paired": float(diff.mean()),
                              "se": float(diff.std(ddof=1) / math.sqrt(len(diff))), "median": float(np.median(diff)),
                              "max": float(diff.max()), "n_bad": int((diff > 0.1).sum())}
            cs = {k: v for k, v in pts.items() if "@c" in k}
            pa = cs[min(cs, key=lambda k: abs(float(k.rsplit("@c", 1)[1]) - cB))] if cs else None
            pk = next((v for k, v in pts.items() if "@k" in k), None)
            va, vb = (verdict(pa, cB, dB) if pa else "—"), (verdict(pk, cB, dB) if pk else "—")
            for k_, v_ in (("exact", va), ("keep2k", vb)):
                tally.setdefault(rung, {}).setdefault(k_, {}).setdefault(v_, 0)
                tally[rung][k_][v_] += 1
            out[f"{t}_{rung}"] = {"bar": {"comp": cB, "dce": dB}, "exact": pa, "keep2k": pk, "verdict": {"exact": va, "keep2k": vb}}
            fm = lambda p: (f"×{p['comp']:.3f} {p['paired']:+.3f} ± {p['se']:.3f} (med {p['median']:+.3f}, max {p['max']:+.3f}, "  # noqa: E731
                            f"{p['n_bad']}>0.1)")
            print(f"| {t} | ×{cB:.3f} {dB:+.3f} | {fm(pa) if pa else '—'} | {fm(pk) if pk else '—'} | {va} / {vb} |")
    print("\n" + json.dumps(tally))
    if a.res == "results_router_v5len":
        json.dump({"cells": out, "tally": tally}, open(os.path.join(HERE, "headline_v5len.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
