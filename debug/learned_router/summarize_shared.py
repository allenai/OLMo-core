"""Shared (cross-task) routers on test64 (paired budget): per router, held-in vs held-out verdicts.

    python debug/learned_router/summarize_shared.py --routers xs3,xs6,xs9,xt17f,xt17v6a

Training sets are read from ``runs/<name>/<name>_s0_rhobar.json`` (argv --xtasks). eval_size 64 -- flagged.
"""
import argparse
import json
import math
import os

import numpy as np

from summarize_len5 import verdict

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(os.path.dirname(HERE), "devloss_grid")
BAR = "gold_fl20p8_noslot"
TASKS = "nq contradiction scifact strmatch outlier fiqa msmarco oolong outlier_amzn reorder qdmatch_hpqa rerank textgroups grouping niah absence".split()


def point(f, key):
    d = json.load(open(f))
    pr = d["per_row"]
    full, b = np.asarray(pr["full"]["ce"]), np.asarray(pr[BAR]["ce"])
    cB, dB = float(np.mean(pr[BAR]["compaction"])), float((b - full).mean())
    r = np.asarray(pr[key]["ce"])
    diff = r - b
    p = {"comp": float(np.mean(pr[key]["compaction"])), "dce": float((r - full).mean()), "paired": float(diff.mean()),
         "se": float(diff.std(ddof=1) / math.sqrt(len(diff))), "n_bad": int((diff > 0.1).sum())}
    p["verdict"] = verdict(p, cB, dB)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routers", default="xs3,xs6,xs9,xt17f,xt17v6a")
    a = ap.parse_args()
    out = {}
    for nm in a.routers.split(","):
        rp = os.path.join(HERE, "runs", nm, f"{nm}_s0_rhobar.json")
        argv = json.load(open(rp))["argv"] if os.path.exists(rp) else []
        train = set(argv[argv.index("--xtasks") + 1].split(",")) if "--xtasks" in argv else set()
        rows, tally = [], {"held-in": {}, "held-out": {}}
        for t in TASKS:
            f = os.path.join(GRID, "results_router_t64", f"{t}_{nm}_2k.json")
            if not os.path.exists(f):
                continue
            p = point(f, f"router_{nm}@pair")
            split = "held-in" if t in train else "held-out"
            tally[split][p["verdict"]] = tally[split].get(p["verdict"], 0) + 1
            rows.append((t, split, p))
        out[nm] = {"train": sorted(train), "tally": tally, "rows": {t: dict(p, split=s) for t, s, p in rows}}
        print(f"\n## {nm}  (trained on {len(train)}: {','.join(sorted(train))})  held-in {tally['held-in']}  held-out {tally['held-out']}")
        for t, s, p in rows:
            print(f"  {t:13s} {s:8s} {p['paired']:+.3f} ± {p['se']:.3f} ({p['n_bad']}>0.1) {p['verdict']}")
    json.dump(out, open(os.path.join(HERE, "headline_shared.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
