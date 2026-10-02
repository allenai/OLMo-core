"""Retrained routers (relpos features + routed markers + rho matched to gold_fl20p8_noslot, fix per
diagnosis) vs the exact heuristic on train/val and vs the bar on the test rows.

    python debug/learned_router/summarize_rel.py [--prefix e2erel] [--tasks rerank,textgroups,...]

Per task and seed: train / val paired dCE vs the exact heuristic at equal T2/T (``runs/<task>/<name>.json``
``diag``), the train-val gap, gold-token keep; the seed with the lowest VAL paired is selected, and its
test-row verdict vs the bar (summarize_vsfl.py's rule) is reported. eval_size 16 -- flagged.
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


def test_points(f):
    d = json.load(open(f))
    pr = d["per_row"]
    full, b = np.asarray(pr["full"]["ce"], float), np.asarray(pr[BAR]["ce"], float)
    cB, dB = float(np.mean(pr[BAR]["compaction"])), float((b - full).mean())
    pts = []
    for s in pr:
        if s.startswith("router_") and "hand" not in s:
            r = np.asarray(pr[s]["ce"], float)
            diff = r - b
            pts.append({"scheme": s, "comp": float(np.mean(pr[s]["compaction"])), "dce": float((r - full).mean()),
                        "paired": float(diff.mean()), "se": float(diff.std(ddof=1) / math.sqrt(len(diff))),
                        "median": float(np.median(diff)), "max": float(diff.max()), "n_bad": int((diff > 0.1).sum())})
    tol = lambda p: max(p["se"], TOL)  # noqa: E731
    if any(p["comp"] <= 1.02 * cB and p["dce"] <= dB + tol(p) and (p["comp"] <= 0.95 * cB or p["paired"] < -tol(p)) for p in pts):
        v = "beats"
    elif any(p["comp"] <= 1.05 * cB and abs(p["paired"]) <= tol(p) for p in pts):
        v = "matches"
    else:
        v = "loses"
    at = min((p for p in pts if re.search(r"_c[0-9.]+$", p["scheme"])), key=lambda p: abs(p["comp"] - cB), default=None)
    return {"cB": cB, "dB": dB, "at": at, "points": pts, "verdict": v, "n": len(full)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="e2erel")
    ap.add_argument("--tasks", default="rerank,textgroups,absence,grouping,niah")
    ap.add_argument("--res", default=os.path.join(GRID, "results_router_rel"))
    a = ap.parse_args()
    out = {}
    print("| task | seed | train paired vs heur | val paired vs heur | train-val gap | gold keep (router/heur, val) | test @c_B: ×c ΔCE (paired ± SE) | test verdict |")
    print("|---|---|---|---|---|---|---|---|")
    for t in a.tasks.split(","):
        rows = []
        for f in sorted(glob.glob(os.path.join(HERE, "runs", t, f"{a.prefix}_s*_rhobar.json"))):
            r = json.load(open(f))
            dg = r.get("diag")
            if not dg:
                continue
            seed = int(re.search(r"_s(\d+)_", os.path.basename(f)).group(1))
            tf = os.path.join(a.res, f"{t}_s{seed}_2k.json")
            tp = test_points(tf) if os.path.exists(tf) else None
            rows.append({"seed": seed, "train": dg["train"]["paired"], "train_se": dg["train"]["paired_se"], "val": dg["val"]["paired"],
                         "val_se": dg["val"]["paired_se"], "gk_r": dg["val"]["gold_keep_router"], "gk_h": dg["val"]["gold_keep_heuristic"],
                         "rho": r.get("rho"), "test": tp})
        if not rows:
            continue
        best = min(rows, key=lambda x: x["val"])
        for x in rows:
            tp = x["test"]
            at = tp["at"] if tp else None
            gk = "—" if x["gk_r"] is None else f"{x['gk_r']:.2f}/{x['gk_h']:.2f}"
            print(f"| {t} | {x['seed']}{' ★' if x is best else ''} | {x['train']:+.3f} ± {x['train_se']:.3f} | {x['val']:+.3f} ± {x['val_se']:.3f} | "
                  f"{x['val'] - x['train']:+.3f} | {gk} | "
                  + (f"×{at['comp']:.3f} {at['dce']:+.3f} ({at['paired']:+.3f} ± {at['se']:.3f})" if at else "—")
                  + f" | {tp['verdict'] if tp else '—'} (n {tp['n'] if tp else '—'} ⚠) |")
        out[t] = {"seeds": rows, "selected_seed": best["seed"]}
    json.dump(out, open(os.path.join(HERE, f"headline_{a.prefix}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
