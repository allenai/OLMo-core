"""Headline for the END-TO-END router: per (task, rung) cell, is the router's Pareto curve at or
below the BEST EXISTING scheme's (compaction, dCE) point?

    python debug/learned_router/analyze_e2e.py [--out debug/learned_router/headline_e2e.json]

* Existing schemes: every non-router scheme in ``debug/devloss_grid/grid.json`` except ``full``, the
  label-using oracle ``grad20`` and the layer-skip variants (their token count is gold_fl20p8's).
* Best existing point of a cell: the most compact existing scheme at parity (mean dCE <= 0.02);
  if none reaches parity, the one with the lowest mean dCE.
* Router curve: the ``router_e2e_rho*`` points of ``results_router_e2e/<task>_<rung>.json`` plus the
  trivial keep-all point (1.0, 0.0); lower envelope over them.
* Verdict (tol = max(0.02, paired SE of the best point)):
  ``win``  -- the router reaches dCE <= d_best + tol at compaction <= 0.9 c_best, or its envelope at
              c_best is below d_best - tol;
  ``tie``  -- it reaches d_best + tol at compaction <= 1.1 c_best (and does not win);
  ``loss`` -- otherwise.
  Also reported: the tau-selected routers (``router_e2e_tau*``) against the same point.
xabsence is excluded (VOID checkpoint). Eval sets are 16/16/8 rows -- flagged inline.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(os.path.dirname(HERE), "devloss_grid", "grid.json")
RES = os.path.join(os.path.dirname(HERE), "devloss_grid", "results_router_e2e")
EXCL = {"full", "grad20", "gold_fl20p8_skipodd", "gold_fl20p8_skipeven", "gold_fl20p8_skipgdn2"}
PAR = 0.02


def env_at(pts, c):
    pts = sorted(pts)
    env = []
    for ci, di in pts:
        if not env or di < env[-1][1]:
            env.append((ci, di))
    if c < env[0][0]:
        return None
    for (c1, d1), (c2, d2) in zip(env, env[1:]):
        if c1 <= c < c2:
            return d1 + (d2 - d1) * (c - c1) / (c2 - c1)
    return env[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=GRID)
    ap.add_argument("--res", default=RES)
    ap.add_argument("--out", default=os.path.join(HERE, "headline_e2e.json"))
    a = ap.parse_args()
    recs = json.load(open(a.grid))["records"]
    cells = {}
    for r in recs:
        if r["task"] in ("xabsence", "cpt80") or r["scheme"] in EXCL or r["scheme"].startswith("router_"):
            continue
        cells.setdefault((r["task"], r["rung"]), []).append(r)
    out, cnt, cnt_tau = {}, Counter(), {t: Counter() for t in ("0.05", "0.1", "0.2")}
    print("| task | rung | n | best existing (×c, ΔCE) | router curve: ρ → ×c ΔCE | router ×c at best's ΔCE+tol | env ΔCE at best's ×c | verdict | τ0.1 router |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (task, rung), rr in sorted(cells.items()):
        par = [r for r in rr if r["dce"] <= PAR]
        best = min(par, key=lambda r: r["compaction"]) if par else min(rr, key=lambda r: r["dce"])
        f = os.path.join(a.res, f"{task}_{rung}.json")
        if not os.path.exists(f):
            out[f"{task}@{rung}"] = {"best": best["scheme"], "verdict": "not run"}
            cnt["not run"] += 1
            continue
        d = json.load(open(f))
        full = np.asarray(d["per_row"]["full"]["ce"], float)
        pts, pts_named, tau_pts = [(1.0, 0.0)], [], {}
        for s in d["per_row"]:
            if not s.startswith("router_e2e_"):
                continue
            diff = np.asarray(d["per_row"][s]["ce"], float) - full
            c = float(np.mean(d["per_row"][s]["compaction"]))
            if s.startswith("router_e2e_rho"):
                pts.append((c, float(diff.mean())))
                pts_named.append((s[len("router_e2e_rho"):], c, float(diff.mean()), float(np.median(diff))))
            elif s.startswith("router_e2e_tau"):
                tau_pts[s[len("router_e2e_tau"):]] = (c, float(diff.mean()))
        tol = max(PAR, best.get("dce_se") or 0.0)
        cb, db = best["compaction"], best["dce"]
        reach = [c for c, dd in pts if dd <= db + tol]
        c_r = min(reach) if reach else None
        e = env_at(pts, cb)
        if (c_r is not None and c_r <= 0.9 * cb) or (e is not None and e < db - tol):
            v = "win"
        elif c_r is not None and c_r <= 1.1 * cb:
            v = "tie"
        else:
            v = "loss"
        cnt[v] += 1
        tv = {}
        for t, (c, dd) in tau_pts.items():
            tv[t] = "win" if (dd <= db + tol and c <= 0.9 * cb) or (c <= cb * 1.1 and dd < db - tol) else (
                "tie" if dd <= db + tol and c <= 1.1 * cb else "loss")
            cnt_tau[t][tv[t]] += 1
        n = d["eval_size"]
        out[f"{task}@{rung}"] = {"eval_size": n, "best": {"scheme": best["scheme"], "comp": cb, "dce": db, "se": best.get("dce_se")},
                                 "router_points": pts_named, "router_comp_at_best_dce": c_r, "router_env_at_best_comp": e,
                                 "verdict": v, "tau_points": tau_pts, "tau_verdicts": tv}
        curve = " ".join(f"{r}:×{c:.2f} {dd:+.3f}" for r, c, dd, _ in sorted(pts_named, key=lambda z: float(z[0])))
        t01 = tau_pts.get("0.1")
        print(f"| {task} | {rung} | {n}{' ⚠' if n < 500 else ''} | {best['scheme']} ×{cb:.2f} {db:+.3f} | {curve} | "
              f"{'—' if c_r is None else f'×{c_r:.2f}'} | {'—' if e is None else f'{e:+.3f}'} | {v} | "
              f"{'—' if not t01 else f'×{t01[0]:.2f} {t01[1]:+.3f} ({tv.get(chr(48) + chr(46) + chr(49), chr(45))})'} |")
    json.dump({"cells": out, "counts": dict(cnt), "counts_tau": {k: dict(v) for k, v in cnt_tau.items()}}, open(a.out, "w"), indent=1)
    print(f"\nPareto-curve verdicts over {sum(cnt.values())} cells: {dict(cnt)}")
    for t, c in cnt_tau.items():
        print(f"tau {t} selected router verdicts: {dict(c)}")
    print(f"(eval sets 16/16/8 rows; tol = max({PAR}, SE)) wrote {a.out}")


if __name__ == "__main__":
    main()
