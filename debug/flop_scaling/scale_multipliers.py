"""Matched-compute multipliers for the model-scale ladder: for each (task, scale, method arm, budget)
row of results_scale.csv, interpolate the scale's dense f1 log-linearly in PFLOPs between its two
budgets, invert it to the FLOPs dense would need to reach the method's f1, and divide by the
method's FLOPs (>1 = method is compute-optimal; ~ = outside the dense range, extrapolated).
    python debug/flop_scaling/scale_multipliers.py [--md]"""
import csv, math, sys
from collections import defaultdict
ROWS = [r for r in csv.DictReader(open("results/flop_scaling/results_scale.csv")) if r["task"] in ("oolong", "contradiction") and r.get("mean_f1") not in ("", None)]
dense = defaultdict(list)
for r in ROWS:
    if r["arm"] == "dense":
        dense[(r["task"], r["scale"])].append((math.log(float(r["actual_pflops"])), float(r["mean_f1"])))
def line(task, scale):
    pts = sorted(dense[(task, scale)]); (x0, y0), (x1, y1) = pts[0], pts[-1]
    b = (y1 - y0) / (x1 - x0); a = y0 - b * x0
    return a, b, x0, x1
md = "--md" in sys.argv
out = []
for r in sorted(ROWS, key=lambda r: (r["task"], ["0.8b", "2b", "4b", "9b", "27b"].index(r["scale"]), r["arm"], int(r["tokens"]))):
    if r["arm"] == "dense" or (r["task"], r["scale"]) not in dense: continue
    a, b, x0, x1 = line(r["task"], r["scale"]); pf = float(r["actual_pflops"]); f1 = float(r["mean_f1"])
    dense_same = a + b * math.log(pf); x_need = (f1 - a) / b; mult = math.exp(x_need) / pf
    flag = "" if x0 - 1e-9 <= x_need <= x1 + 1e-9 else "~"
    out.append((r["task"], r["scale"], r["arm"], r["budget"], f1, pf, dense_same, mult, flag))
if md:
    print("| task | scale | arm | budget | f1 | PF | dense f1 at same PF | FLOPs for dense to match, ÷ method PF |"); print("|---|---|---|---|---|---|---|---|")
    for t, s, arm, bud, f1, pf, ds, m, fl in out: print(f"| {t} | {s} | {arm} | {bud} | {f1:.3f} | {pf:.0f} | {ds:.3f} | {m:.2f}{fl} |")
else:
    for t, s, arm, bud, f1, pf, ds, m, fl in out: print(f"{t:13s} {s:5s} {arm:12s} {bud:>4s} f1={f1:.3f} PF={pf:7.0f} dense@PF={ds:.3f} mult={m:.2f}{fl}")
