"""Collect the model-scale ladder (orchestrate_scale.py state files) into results_scale.csv, with
FLOPs priced per scale, plus the matching 4B rows from results35.csv; prints a per-(task, scale)
table.    python debug/flop_scaling/collect_scale.py"""
import csv, os, sys
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
import collect_results35 as c

OUT = c.OUT
rows = []
for scale, tag in [("0.8b", "s08b"), ("2b", "s2b"), ("4b", "s4b"), ("9b", "s9b"), ("27b", "s27b")]:
    sp = f"{D}/orchestrate_{tag}_state.json"
    if os.path.exists(sp):
        rows += c.main(state_path=sp, out_csv=f"{OUT}/results_scale_{tag}.csv", scale=scale, prior_dense=False, run_fit=False)
if os.path.exists(f"{OUT}/results35.csv"):
    for r in csv.DictReader(open(f"{OUT}/results35.csv")):
        if r["task"] in ("oolong", "contradiction") and r["arm"] in ("dense", "kv17", "kv33", "ffnmoe-t10", "ffnmoe-t10p") and r["budget"] in ("20M", "80M", "14M", "56M"):
            r = dict(r); r["scale"] = "4b"; rows.append(r)
keys = ["task", "scale", "arm", "budget", "tokens", "actual_pflops", "dense_equiv_pflops", "flop_ratio", "mean_f1", "partial", "f1_2k", "f1_8k", "f1_16k", "f1_32k", "run"]
with open(f"{OUT}/results_scale.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
order = {"0.8b": 0, "2b": 1, "4b": 2, "9b": 3, "27b": 4}; aorder = {"dense": 0, "kv17": 1, "kv33": 1, "ffnmoe-t10": 2, "ffnmoe-t10p": 3}
print(f"{'task':13s} {'scale':5s} {'budget':>6s} {'arm':10s} {'mean':>6s} {'32k':>5s} {'PF':>7s} {'x':>5s}")
for r in sorted(rows, key=lambda r: (r["task"], int(str(r["tokens"])), order.get(r["scale"], 9), aorder.get(r["arm"], 9))):
    pf = r.get("actual_pflops"); x = r.get("flop_ratio"); m = r.get("mean_f1"); k32 = r.get("f1_32k")
    fmt = lambda v, p: ("-" if v in (None, "", "None") else f"{float(v):{p}}")
    print(f"{r['task']:13s} {r['scale']:5s} {r['budget']:>6s} {r['arm']:10s} {fmt(m,'.3f'):>6s} {fmt(k32,'.2f'):>5s} {fmt(pf,'.0f'):>7s} {fmt(x,'.2f'):>5s}")
