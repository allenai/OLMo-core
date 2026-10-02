"""Per-task table: e2e router vs ``gold_fl20p8_noslot`` at 2k (results_router_e2e_vsfl/<task>_2k.json).

    python debug/learned_router/summarize_vsfl.py [--rung 2k]

For each task, with B = gold_fl20p8_noslot (T2/T c_B, mean dCE d_B) and every router point R in the file
(label-free val-calibrated thresholds ``_c<x>`` and the router's default threshold), paired per row:
  (a) the router point calibrated to c_B: realised T2/T, dCE, paired dCE(R) - dCE(B) +- SE;
  (b) the smallest router T2/T whose dCE <= d_B + tol, tol = max(SE_paired(R, B), 0.005 nats).
Verdict (Pareto): ``beats`` = some R with T2/T <= 1.02 c_B and dCE <= d_B + tol, strictly better on one
axis (T2/T <= 0.95 c_B, or paired diff < -tol); ``matches`` = some R with T2/T <= 1.05 c_B and
|paired diff| <= tol; ``loses`` otherwise. eval_size 16 -- flagged.
"""
import argparse
import glob
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "devloss_grid", "results_router_e2e_vsfl")
BAR = "gold_fl20p8_noslot"
TOL = 0.005  # practical floor on the tolerance (nats): at near-zero CE the paired SE can be ~1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="2k")
    ap.add_argument("--res", default=RES)
    ap.add_argument("--out", default=os.path.join(HERE, "headline_vsfl.json"))
    ap.add_argument("--only", default="", help="regex: restrict router points to matching scheme names")
    a = ap.parse_args()
    out, verdicts = {}, {}
    print(f"| task | n | {BAR} ×c, ΔCE | (a) router @ its ×c: ×c, ΔCE, paired Δ ± SE | (b) smallest ×c within ΔCE_B + SE | verdict |")
    print("|---|---|---|---|---|---|")
    for f in sorted(glob.glob(os.path.join(a.res, f"*_{a.rung}.json"))):
        task = os.path.basename(f)[: -len(f"_{a.rung}.json")]
        d = json.load(open(f))
        full = np.asarray(d["per_row"]["full"]["ce"], float)
        b = np.asarray(d["per_row"][BAR]["ce"], float)
        cB, dB = float(np.mean(d["per_row"][BAR]["compaction"])), float((b - full).mean())
        pts = []
        for s in d["per_row"]:
            if not s.startswith("router_") or (a.only and not re.search(a.only, s)):
                continue
            r = np.asarray(d["per_row"][s]["ce"], float)
            diff = r - b
            m = re.search(r"_c([0-9.]+)$", s)
            pts.append({"scheme": s, "target": float(m.group(1)) if m else None, "comp": float(np.mean(d["per_row"][s]["compaction"])),
                        "dce": float((r - full).mean()), "dce_median": float(np.median(r - full)), "paired": float(diff.mean()),
                        "paired_se": float(diff.std(ddof=1) / math.sqrt(len(diff)))})
        if not pts:
            continue
        at = min((p for p in pts if p["target"] is not None), key=lambda p: abs(p["target"] - cB), default=None)
        tol = lambda p: max(p["paired_se"], TOL)  # noqa: E731
        ok = [p for p in pts if p["dce"] <= dB + tol(p)]
        small = min(ok, key=lambda p: p["comp"]) if ok else None
        beats = [p for p in pts if p["comp"] <= 1.02 * cB and p["dce"] <= dB + tol(p)
                 and (p["comp"] <= 0.95 * cB or p["paired"] < -tol(p))]
        matches = [p for p in pts if p["comp"] <= 1.05 * cB and abs(p["paired"]) <= tol(p)]
        v = "beats" if beats else ("matches" if matches else "loses")
        verdicts[v] = verdicts.get(v, 0) + 1
        out[task] = {"eval_size": d["eval_size"], "bar": {"comp": cB, "dce": dB}, "points": pts, "at_bar": at, "smallest": small, "verdict": v}
        fa = "—" if at is None else f"×{at['comp']:.3f} {at['dce']:+.3f} ({at['paired']:+.3f} ± {at['paired_se']:.3f})"
        fb = "—" if small is None else f"×{small['comp']:.3f} ({small['dce']:+.3f})"
        print(f"| {task} | {d['eval_size']} ⚠ | ×{cB:.3f} {dB:+.3f} | {fa} | {fb} | **{v}** |")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nverdicts: {verdicts}  (eval_size 16 per task at {a.rung}) wrote {a.out}")


if __name__ == "__main__":
    main()
