"""Final table on the 64-row disjoint test set (``test64``): paired budget (``@pair``) vs gold_fl20p8_noslot,
for the v6 router (``results_router_t64/<task>_v6fin_2k.json``) and, from one job each, the restart-AVERAGED
router vs that job's best single restart (``results_router_v6/<task>_v6avg_s0_2k.json``). eval_size 64
(qdmatch_hpqa 54; obliq has no fresh rows) -- below the 500 policy, flagged.

    python debug/learned_router/summarize_t64.py
"""
import glob
import json
import math
import os

import numpy as np

from summarize_len5 import verdict

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(os.path.dirname(HERE), "devloss_grid")
BAR = "gold_fl20p8_noslot"


def pair_point(f, key_end):
    if not os.path.exists(f):
        return None, None
    d = json.load(open(f))
    pr = d["per_row"]
    full, b = np.asarray(pr["full"]["ce"]), np.asarray(pr[BAR]["ce"])
    cB, dB = float(np.mean(pr[BAR]["compaction"])), float((b - full).mean())
    ks = [s for s in pr if s.endswith(key_end)]
    if not ks:
        return None, (cB, dB, len(full))
    r = np.asarray(pr[ks[0]]["ce"])
    diff = r - b
    p = {"comp": float(np.mean(pr[ks[0]]["compaction"])), "dce": float((r - full).mean()), "paired": float(diff.mean()),
         "se": float(diff.std(ddof=1) / math.sqrt(len(diff))), "median": float(np.median(diff)), "max": float(diff.max()),
         "n_bad": int((diff > 0.1).sum())}
    p["verdict"] = verdict(p, cB, dB)
    return p, (cB, dB, len(full))


def fmt(p):
    return "—" if p is None else (f"{p['paired']:+.3f} ± {p['se']:.3f} (med {p['median']:+.3f}, max {p['max']:+.2f}, {p['n_bad']}>0.1) "
                                  f"**{p['verdict']}**")


def main():
    tasks = sorted({os.path.basename(f).split("_v6fin")[0] for f in glob.glob(os.path.join(GRID, "results_router_t64", "*_v6fin_2k.json"))}
                   | {os.path.basename(f).split("_v6avg")[0] for f in glob.glob(os.path.join(GRID, "results_router_v6", "*_v6avg_s0_2k.json"))})
    tally = {"v6": {}, "avg": {}, "best": {}}
    print("| task | n | bar ×c ΔCE | v6 (single run) | v6 + restart-avg | same job: best restart |")
    print("|---|---|---|---|---|---|")
    for t in tasks:
        p6, m = pair_point(os.path.join(GRID, "results_router_t64", f"{t}_v6fin_2k.json"), "v6fin_s0_rhobar@pair")
        fa = os.path.join(GRID, "results_router_v6", f"{t}_v6avg_s0_2k.json")
        pa, m2 = pair_point(fa, "v6avg_s0_rhobar@pair")
        pb, _ = pair_point(fa, "v6avg_s0_rhobar_best@pair")
        mm = m or m2
        for k, p in (("v6", p6), ("avg", pa), ("best", pb)):
            if p:
                tally[k][p["verdict"]] = tally[k].get(p["verdict"], 0) + 1
        print(f"| {t} | {mm[2] if mm else '—'} ⚠ | ×{mm[0]:.3f} {mm[1]:+.3f} | {fmt(p6)} | {fmt(pa)} | {fmt(pb)} |" if mm else f"| {t} | pending |")
    print("\n" + json.dumps(tally))


if __name__ == "__main__":
    main()
