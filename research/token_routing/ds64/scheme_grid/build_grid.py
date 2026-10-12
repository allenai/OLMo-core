"""Build the method x task grid (f1 @ PF, matched-FLOP delta vs dense) from results/ds64/results.csv.

Outlier uses the 4-rung mean (2k/8k/16k/32k): the 64k rung is the broken v2 xlong file (~0 for every arm).
Soft arms: the -b128f3- generation only (the wall-clock-faithful recipe); dense: all budgets.
Dense reference at a soft arm's PF: log-log linear interpolation on the dense (PF, f1) curve; flagged
'extrap' when the PF sits below the cheapest measured dense point.
"""
import json, math
import pandas as pd

df = pd.read_csv("results/ds64/results.csv")
df = df[df.eval_state == "DONE"].copy()
df["f1"] = df.mean_f1
o = df.task == "outlier"
df.loc[o, "f1"] = df.loc[o, ["f1_2k", "f1_8k", "f1_16k", "f1_32k"]].mean(axis=1)
soft = df[(df.arm != "dense") & df.run.str.contains("-b128f3-") & (df.steps > 2)]
dense = df[df.arm == "dense"]

def dense_at(task, pf):
    d = dense[dense.task == task].sort_values("flops_meter")
    xs, ys = d.flops_meter.tolist(), d.f1.tolist()
    if pf < xs[0]:
        # extrapolate on the first segment
        (x0, y0), (x1, y1) = (xs[0], ys[0]), (xs[1], ys[1])
        return y0 + (y1 - y0) * (math.log(pf) - math.log(x0)) / (math.log(x1) - math.log(x0)), True
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x0 <= pf <= x1:
            return y0 + (y1 - y0) * (math.log(pf) - math.log(x0)) / (math.log(x1) - math.log(x0)), False
    return ys[-1], True

def dominates(task, f1, pf):
    """PF ratio to the most expensive dense point this arm is >= in f1 (0 if none)."""
    d = dense[(dense.task == task) & (dense.f1 <= f1 + 1e-9)]
    if d.empty:
        return None
    best = d.sort_values("flops_meter").iloc[-1]
    return best.budget, best.flops_meter / pf

out = {}
for _, r in soft.iterrows():
    ref, ex = dense_at(r.task, r.flops_meter)
    dom = dominates(r.task, r.f1, r.flops_meter)
    out.setdefault(r.task, {}).setdefault(r.arm, {})[r.budget] = dict(
        f1=round(r.f1, 3), pf=round(r.flops_meter, 0), dense_ref=round(ref, 3), delta=round(r.f1 - ref, 3),
        extrap=ex, dom_budget=dom[0] if dom else None, dom_ratio=round(dom[1], 2) if dom else None,
        rungs=[r.f1_2k, r.f1_8k, r.f1_16k, r.f1_32k, r.f1_64k])
for _, r in dense.iterrows():
    out.setdefault(r.task, {}).setdefault("dense", {})[r.budget] = dict(
        f1=round(r.f1, 3), pf=round(r.flops_meter, 0), rungs=[r.f1_2k, r.f1_8k, r.f1_16k, r.f1_32k, r.f1_64k])
json.dump(out, open("research/token_routing/ds64/scheme_grid/grid.json", "w"), indent=1)
for task in out:
    print("==", task)
    for arm, b in out[task].items():
        print(f"  {arm:10s}", "  ".join(f"{k}: {v['f1']:.3f}@{v['pf']:.0f}" + (f" Δ{v['delta']:+.3f}{'*' if v['extrap'] else ''}" if 'delta' in v else "") + (f" [{v['dom_ratio']}x>{v['dom_budget']}]" if v.get('dom_ratio') else "") for k, v in sorted(b.items(), key=lambda kv: int(kv[0][:-1]))))
