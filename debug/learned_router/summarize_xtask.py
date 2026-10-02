"""Cross-task router (xtask3: nq+contradiction+scifact, 16 rows each at 2k) vs gold_fl20p8_noslot and vs the per-task routers.

    python debug/learned_router/summarize_xtask.py [--name xtask_3t]

Per task x rung: the shared-threshold point (``router_<name>``, one global offset for all tasks, p>0.5),
the label-free per-task calibrated point (``router_<name>_c<c_B>``, threshold matched on unlabeled rows to
the bar's T2/T) with its paired dCE vs the bar +- SE, and (2k only) the per-task router's paired dCE at the
same calibration (results_router_e2e_vsfl), so "cost of sharing" = xtask paired - per-task paired.
Verdicts come from summarize_vsfl.py's rule. eval_size 16/16/8 -- flagged.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = os.path.join(os.path.dirname(HERE), "devloss_grid")
BAR = "gold_fl20p8_noslot"


def pts(d, pat):
    full = np.asarray(d["per_row"]["full"]["ce"], float)
    b = np.asarray(d["per_row"][BAR]["ce"], float)
    out = {}
    for s in d["per_row"]:
        if re.fullmatch(pat, s):
            r = np.asarray(d["per_row"][s]["ce"], float)
            diff = r - b
            out[s] = (float(np.mean(d["per_row"][s]["compaction"])), float((r - full).mean()), float(diff.mean()),
                      float(diff.std(ddof=1) / math.sqrt(len(diff))))
    return out, float(np.mean(d["per_row"][BAR]["compaction"])), float((b - full).mean()), len(full)


def verdicts(res, rung):
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"_xt_{rung}.json")
    subprocess.run([sys.executable, os.path.join(HERE, "summarize_vsfl.py"), "--res", res, "--rung", rung, "--out", tmp],
                   check=True, capture_output=True)
    return {k: v["verdict"] for k, v in json.load(open(tmp)).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="xtask_3t")
    ap.add_argument("--rungs", default="2k,8k,32k")
    ap.add_argument("--res-dir", default="results_router_xtask", help="results dir under debug/devloss_grid")
    ap.add_argument("--held-in", default="nq,contradiction,scifact", help="training tasks ('all' for the pooled run)")
    a = ap.parse_args()
    xres, pres = os.path.join(GRID, a.res_dir), os.path.join(GRID, "results_router_e2e_vsfl")
    held = None if a.held_in == "all" else set(a.held_in.split(","))
    for rung in a.rungs.split(","):
        vx = verdicts(xres, rung)
        vp = verdicts(pres, rung) if rung == "2k" else {}
        print(f"\n### {rung}\n| task | split | n | bar ×c, ΔCE | shared thr ×c, ΔCE | xtask @bar ×c: paired ± SE | per-task @bar paired | cost of sharing | xtask verdict | per-task verdict |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for t in sorted(vx):
            f = os.path.join(xres, f"{t}_{rung}.json")
            d = json.load(open(f))
            p, cB, dB, n = pts(d, rf"router_{a.name}(_c[0-9.]+)?")
            sh = p.get(f"router_{a.name}")
            cal = next((v for k, v in p.items() if k != f"router_{a.name}"), None)
            per, cost = "", ""
            pf = os.path.join(pres, f"{t}_{rung}.json")
            if rung == "2k" and os.path.exists(pf):
                pp, *_ = pts(json.load(open(pf)), r"router_.*_c[0-9.]+")
                if pp:
                    k = min(pp, key=lambda s: abs(pp[s][0] - cB))
                    per = f"{pp[k][2]:+.3f} ± {pp[k][3]:.3f}"
                    if cal:
                        cost = f"{cal[2] - pp[k][2]:+.3f}"
            flag = " ⚠" if n < 500 else ""
            print(f"| {t} | {'held-in' if held is None or t in held else 'held-out'} | {n}{flag} | ×{cB:.3f} {dB:+.3f} | "
                  + (f"×{sh[0]:.3f} {sh[1]:+.3f}" if sh else "—") + " | "
                  + (f"×{cal[0]:.3f} {cal[2]:+.3f} ± {cal[3]:.3f}" if cal else "—")
                  + f" | {per} | {cost} | {vx.get(t, '')} | {vp.get(t, '')} |")


if __name__ == "__main__":
    main()
