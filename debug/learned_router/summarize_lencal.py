"""Length transfer of the per-task e2e routers (trained at 2k) vs gold_fl20p8_noslot at 8k / 32k.

    python debug/learned_router/summarize_lencal.py

Three label-free thresholds per task x rung, each scored paired against the bar on the rung's test rows:
  fixed  -- the router's 2k thresholds unchanged (results_router_e2e_len; the best of its points)
  (i)    -- ``_k2k_<rung>``: keep the SAME fraction of routed tokens as on the 2k val rows, offset set on
            unlabeled rows of the rung (results_router_e2e_lencal)
  (ii)   -- ``_c<c_B>_<rung>``: offset set on unlabeled rows of the rung so mean T2/T == the bar's T2/T there
Verdict per point uses summarize_vsfl.py's rule (tol = max(paired SE, 0.005)). eval_size 16 (8k) / 8 (32k) -- flagged.
"""
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


def load(f):
    d = json.load(open(f))
    full = np.asarray(d["per_row"]["full"]["ce"], float)
    b = np.asarray(d["per_row"][BAR]["ce"], float)
    cB, dB = float(np.mean(d["per_row"][BAR]["compaction"])), float((b - full).mean())
    pts = {}
    for s in d["per_row"]:
        if s.startswith("router_"):
            r = np.asarray(d["per_row"][s]["ce"], float)
            diff = r - b
            pts[s] = {"comp": float(np.mean(d["per_row"][s]["compaction"])), "dce": float((r - full).mean()),
                      "paired": float(diff.mean()), "se": float(diff.std(ddof=1) / math.sqrt(len(diff)))}
    return pts, cB, dB, len(full)


def verdict(ps, cB, dB):
    tol = lambda p: max(p["se"], TOL)  # noqa: E731
    if any(p["comp"] <= 1.02 * cB and p["dce"] <= dB + tol(p) and (p["comp"] <= 0.95 * cB or p["paired"] < -tol(p)) for p in ps):
        return "beats"
    return "matches" if any(p["comp"] <= 1.05 * cB and abs(p["paired"]) <= tol(p) for p in ps) else "loses"


def fmt(p):
    return f"×{p['comp']:.3f} {p['dce']:+.3f} ({p['paired']:+.3f} ± {p['se']:.3f})"


def main():
    out, tally = {}, {}
    for rung in ("8k", "32k"):
        print(f"\n### {rung}\n| task | n | bar ×c, ΔCE | fixed 2k cutoff (best) | (i) same routed keep as 2k | (ii) bar's T2/T | verdict fixed / (i) / (ii) |")
        print("|---|---|---|---|---|---|---|")
        for f in sorted(glob.glob(os.path.join(GRID, "results_router_e2e_lencal", f"*_{rung}.json"))):
            t = os.path.basename(f)[: -len(f"_{rung}.json")]
            pl, cB, dB, n = load(f)
            p1 = next((v for k, v in pl.items() if k.endswith(f"_k2k_{rung}")), None)
            cs = {float(m.group(1)): v for k, v in pl.items() if (m := re.search(rf"_c([0-9.]+)_{rung}$", k))}
            p2 = cs[min(cs, key=lambda c: abs(c - cB))] if cs else None
            ff = os.path.join(GRID, "results_router_e2e_len", f"{t}_{rung}.json")
            pf = list(load(ff)[0].values()) if os.path.exists(ff) else []
            vf = verdict(pf, cB, dB) if pf else "—"
            v1 = verdict([p1], cB, dB) if p1 else "—"
            v2 = verdict(list(cs.values()), cB, dB) if cs else "—"
            bestf = min(pf, key=lambda p: p["paired"]) if pf else None
            print(f"| {t} | {n} ⚠ | ×{cB:.3f} {dB:+.3f} | {fmt(bestf) if bestf else '—'} | {fmt(p1) if p1 else '—'} | "
                  f"{fmt(p2) if p2 else '—'} | {vf} / {v1} / {v2} |")
            out[f"{t}_{rung}"] = {"eval_size": n, "bar": {"comp": cB, "dce": dB}, "fixed": pf, "keep2k": p1, "barcomp": p2,
                                  "all_barcomp_points": cs, "verdict": {"fixed": vf, "keep2k": v1, "barcomp": v2}}
            for k, v in (("fixed", vf), ("keep2k", v1), ("barcomp", v2)):
                tally.setdefault(rung, {}).setdefault(k, {}).setdefault(v, 0)
                tally[rung][k][v] += 1
    print("\n", json.dumps(tally))
    json.dump({"cells": out, "tally": tally}, open(os.path.join(HERE, "headline_lencal.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
