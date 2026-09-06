"""
Performance vs (a) training wall-clock GPU-hours, (b) FFN FLOPs actually spent, (c) total training
FLOPs, for every FLOP-scaling arm with several budgets per arm.

Inputs
  results/flop_scaling/results35.csv                 Qwen3.5-4B grid (dense / ffnmoe-* / kv*)
  results/flop_scaling/results_scale_s4b{attn,flex}.csv   KV-route and joint-budget arms (4B)
  results/flop_scaling/results_q3s4b{dense2,flex2,flexs2}.csv   Qwen3-4B (dense / 2-router / 3-router)
  results/flop_scaling/results_scale_s{08b,2b,4b,9b,27b}.csv    model-scale ladder
  results/flop_scaling/harvest/runs/<run>/flops.json  meter summary (tokens, routing fractions)
  results/flop_scaling/walltime.csv                   per-run step-1 -> last-step seconds + GPUs
                                                      (debug/flop_scaling/collect_walltime.py)

Wall-clock for the prior-campaign dense anchors comes from their own Beaker training jobs
(debug/flop_scaling/prior_dense_state.json maps prior-dense-<task>-<B> to the tsl-full-* /
lmx-full-* experiments; collect_walltime.py --states adds them to walltime.csv). Any anchor without
a measured job falls back to a dense GPU-hours-per-Mtok rate fitted on the grid's own small dense
anchors and is drawn hollow.

FFN FLOPs = tokens processed x dense FFN FLOPs/token x mean routed FFN cost (1 for dense and KV
soft-token arms, whose saving is in tokens, not width).

Outputs visualizations/flop_scaling/{grid35,routed4b,qwen3,scale}_{mean,32k}.png
"""

import csv
import glob
import json
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
D = f"{REPO}/debug/flop_scaling"
OUT = f"{REPO}/results/flop_scaling"
VIZ = f"{REPO}/visualizations/flop_scaling"
sys.path.insert(0, D)
import collect_results35 as C  # noqa: E402

TASKSCALE_RATE = 0.00552 * 8  # GPU-hours per M tokens, taskscale dense on one 8xH100 node
GPUS_DEFAULT = {"0.8b": 4, "2b": 4, "4b": 4, "9b": 8, "27b": 16}
GPUS_OVERRIDE = {"27b": 16}  # two nodes: the log's DP world size (8) x Ulysses CP 2


def fs35_dense_rate(wt):
    """GPU-hours per M tokens fitted on the grid's own dense anchors (4B, 4 GPUs, same recipe as
    the routed arms): hours = a + b * Mtok, least squares over the fs35-*-dense-* runs."""
    xs, ys = [], []
    for run, (sec, g) in wt.items():
        m = re.match(r"fs35-\w+-dense-s(\d+)M$", run)
        if m:
            xs.append(int(m.group(1))); ys.append(sec * (g or 4) / 3600)
    if len(xs) < 2:
        return TASKSCALE_RATE
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    print(f"fs35 dense rate: {b:.4f} GPU-h per Mtok from {n} anchors (taskscale 8-GPU rate {TASKSCALE_RATE:.4f})")
    return b

STYLE = {
    "dense": dict(color="#222222", marker="o", label="dense"),
    "ffnmoe-t10": dict(color="#0E6B66", marker="s", label="routed FFN L12+, two-sided 0.10"),
    "ffnmoe-t10p": dict(color="#4FB8B0", marker="s", label="routed FFN L12+, frozen tail"),
    "ffnmoe-s1": dict(color="#2A9D8F", marker="^", label="routed FFN L12+, hinge 0.01 (stage 1)"),
    "ffnmoe-s2": dict(color="#6C5B7B", marker="v", label="routed FFN all layers (stage 2)"),
    "ffnmoe-a10": dict(color="#9B5DE5", marker="D", label="routed FFN all layers, 0.10"),
    "kv17": dict(color="#B9552A", marker="P", label="KV soft tokens 1/6"),
    "kv33": dict(color="#E08A5C", marker="X", label="KV soft tokens 1/3"),
    "kvb17": dict(color="#B9552A", marker="p", label="KV soft tokens 1/6 (gold-blind)"),
    "kvb33": dict(color="#E08A5C", marker="h", label="KV soft tokens 1/3 (gold-blind)"),
    "attnroute-c50": dict(color="#1F77B4", marker="<", label="KV router keep 0.50"),
    "attnroute-c25": dict(color="#4C9BE0", marker="<", label="KV router keep 0.25"),
    "attnroute-c10": dict(color="#8FC1F0", marker="<", label="KV router keep 0.10"),
    "flex-c70": dict(color="#D62728", marker="*", label="FFN+KV joint 0.70"),
    "flex-c60": dict(color="#E8636A", marker="*", label="FFN+KV joint 0.60"),
    "flex-c45": dict(color="#F29A9F", marker="*", label="FFN+KV joint 0.45"),
    "flexs-c30": dict(color="#7B2D8E", marker="H", label="3 routers joint 0.30"),
    "flexs-c15": dict(color="#A65BB8", marker="H", label="3 routers joint 0.15"),
    "flexs-c05": dict(color="#CFA0DB", marker="H", label="3 routers joint 0.05"),
}


def read_csv(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def walltime():
    wt = {}
    for r in read_csv(f"{OUT}/walltime.csv"):
        if r.get("train_seconds"):
            wt[r["run"]] = (float(r["train_seconds"]), int(r["gpus"] or 0))
    return wt


def flops_json(run):
    p = f"{OUT}/harvest/runs/{run}/flops.json"
    return json.load(open(p)) if os.path.exists(p) else None


def ffn_cost(row, fl):
    arm = row["arm"]
    if fl is None or arm == "dense" or arm.startswith("kv"):
        return 1.0
    if arm.startswith(("flex",)):
        return float(fl.get("ffn_cost_frac") or 1.0)
    if arm.startswith("attnroute"):
        return 1.0
    ffn_frac = C.ffn_per_token() / C.fpt(65536)
    return max(0.0, min(1.0, 1.0 - (1.0 - float(fl["actual_over_dense"])) / ffn_frac))


def enrich(rows, wt, family, scale, dense_rate=TASKSCALE_RATE):
    C.FAMILY, C.SCALE = family, scale
    ffn_pt = C.ffn_per_token()
    out = []
    for r in rows:
        if not r.get("mean_f1"):
            continue
        run = r["run"]
        fl = flops_json(run)
        tokens = float(fl["tokens_processed"]) if fl else float(r["tokens"])
        c = ffn_cost(r, fl)
        rec = dict(r)
        rec["mean_f1"] = float(r["mean_f1"])
        rec["f1_32k"] = float(r["f1_32k"]) if r.get("f1_32k") else None
        rec["total_pf"] = float(r["actual_pflops"]) if r.get("actual_pflops") else None
        rec["ffn_pf"] = tokens * ffn_pt * c / 1e15
        rec["ffn_cost"] = c
        if r["arm"] == "ffnmoe-s2":  # two-stage: add stage-1 FFN FLOPs
            s1 = flops_json(run.replace("ffnmoe-s2", "ffnmoe-s1"))
            if s1:
                c1 = ffn_cost({"arm": "ffnmoe-s1"}, s1)
                rec["ffn_pf"] += float(s1["tokens_processed"]) * ffn_pt * c1 / 1e15
        if run in wt:
            sec, g = wt[run]
            g = GPUS_OVERRIDE.get(scale, g or GPUS_DEFAULT.get(scale, 4))
            rec["gpu_hours"] = sec * g / 3600
            rec["gpus"] = g
            rec["wt_measured"] = True
            if r["arm"] == "ffnmoe-s2" and run.replace("ffnmoe-s2", "ffnmoe-s1") in wt:
                s, g1 = wt[run.replace("ffnmoe-s2", "ffnmoe-s1")]
                rec["gpu_hours"] += s * GPUS_OVERRIDE.get(scale, g1 or GPUS_DEFAULT.get(scale, 4)) / 3600
        elif run.startswith("prior-dense") and scale == "4b" and family == "qwen3_5":
            # no measured job for this anchor: fall back to the fitted dense rate (drawn hollow)
            rec["gpu_hours"] = float(r["tokens"]) / 1e6 * dense_rate
            rec["wt_measured"] = False
        else:
            rec["gpu_hours"] = None
            rec["wt_measured"] = False
        out.append(rec)
    return out


AXES = [("gpu_hours", "training wall-clock (H100 GPU-hours; step 1 to last step)"), ("ffn_pf", "FFN FLOPs spent (PFLOPs)"),
        ("total_pf", "total training FLOPs (PFLOPs)")]


def panel_figure(recs, tasks, title, path, ykey="mean_f1", ylabel="mean f1 over rungs"):
    fig, axes = plt.subplots(len(AXES), len(tasks), figsize=(4.2 * len(tasks), 3.4 * len(AXES)), squeeze=False)
    handles = {}
    for j, task in enumerate(tasks):
        trs = [r for r in recs if r["task"] == task and r.get(ykey) is not None]
        for i, (xk, xl) in enumerate(AXES):
            ax = axes[i][j]
            arms = sorted({r["arm"] for r in trs}, key=lambda a: (a != "dense", a))
            for arm in arms:
                st = STYLE.get(arm, dict(color="#888888", marker="o", label=arm))
                pts = sorted([r for r in trs if r["arm"] == arm and r.get(xk)], key=lambda r: r[xk])
                if not pts:
                    continue
                xs = [r[xk] for r in pts]
                ys = [r[ykey] for r in pts]
                ls = "-" if arm == "dense" else "--"
                (h,) = ax.plot(xs, ys, ls, color=st["color"], lw=1.4 if arm == "dense" else 1.0, alpha=0.9)
                for r in pts:
                    hollow = xk == "gpu_hours" and not r.get("wt_measured")
                    ax.plot(r[xk], r[ykey], st["marker"], color=st["color"], ms=7,
                            mfc="white" if hollow else st["color"], mew=1.2)
                handles.setdefault(arm, plt.Line2D([], [], color=st["color"], marker=st["marker"], ls=ls, label=st["label"]))
            ax.set_xscale("log")
            ax.grid(True, which="both", alpha=0.25)
            if i == 0:
                ax.set_title(task, fontsize=12)
            if i == len(AXES) - 1:
                ax.set_xlabel(xl)
            else:
                ax.set_xlabel(xl, fontsize=8, color="#666666")
            if j == 0:
                ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=13)
    fig.legend(handles=list(handles.values()), loc="lower center", ncol=min(4, len(handles)), fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    gp = sorted({(r.get("gpus") or 0) for r in recs if r.get("wt_measured")} - {0})
    fig.text(0.5, -0.05 - 0.02 * (len(handles) // 4), "GPU-hours = wall-clock x GPUs; runs used " + "/".join(map(str, gp)) +
             " GPUs (dense anchors from the earlier campaign ran on 8, most routed 4B arms on 4). Hollow = estimated from a fitted dense rate.",
             ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.06 + 0.02 * (len(handles) // 4), 1, 0.96))
    os.makedirs(VIZ, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def dump_table(recs, path):
    cols = ["family", "scale", "task", "arm", "budget", "mean_f1", "f1_32k", "gpu_hours", "gpus", "wt_measured", "ffn_pf", "total_pf", "ffn_cost", "run"]
    with open(path, "w") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow({k: (f"{r[k]:.4g}" if isinstance(r.get(k), float) else r.get(k)) for k in cols})
    print("wrote", path)


def main():
    wt = walltime()
    print(f"{len(wt)} runs with measured wall-clock")
    rate = fs35_dense_rate(wt)
    tasks4 = ["contradiction", "nq", "oolong", "outlier"]
    # 1. Qwen3.5-4B grid
    grid = enrich(read_csv(f"{OUT}/results35.csv"), wt, "qwen3_5", "4b", rate)
    for r in grid:
        r["family"], r["scale"] = "qwen3_5", "4b"
    # 2. routed attention arms on the same base (dense points shared with the grid)
    routed = []
    for p in ("results_scale_s4battn.csv", "results_scale_s4bflex.csv"):
        routed += [r for r in enrich(read_csv(f"{OUT}/{p}"), wt, "qwen3_5", "4b", rate) if not r["run"].startswith("prior-dense")]
    for r in routed:
        r["family"], r["scale"] = "qwen3_5", "4b"
    dense_grid = [r for r in grid if r["arm"] in ("dense", "ffnmoe-t10", "kv33", "kv17") and r["task"] in ("contradiction", "oolong")]
    # 3. Qwen3-4B
    q3 = []
    for p in ("results_q3s4bdense2.csv", "results_q3s4bflex2.csv", "results_q3s4bflexs2.csv"):
        q3 += [r for r in enrich(read_csv(f"{OUT}/{p}"), wt, "qwen3", "4b") if not r["run"].startswith("prior-dense")]
    for r in q3:
        r["family"], r["scale"] = "qwen3", "4b"
    # 4. scale ladder
    ladder = []
    for sc, p in (("0.8b", "s08b"), ("2b", "s2b"), ("4b", "s4b"), ("9b", "s9b"), ("27b", "s27b")):
        rs = [r for r in enrich(read_csv(f"{OUT}/results_scale_{p}.csv"), wt, "qwen3_5", sc) if not r["run"].startswith("prior-dense")]
        for r in rs:
            r["family"], r["scale"] = "qwen3_5", sc
        ladder += rs
    allrecs = grid + routed + q3 + ladder
    dump_table(allrecs, f"{OUT}/axes_points.csv")

    for ykey, suffix, yl in (("mean_f1", "mean", "mean f1 over rungs"), ("f1_32k", "32k", "f1 at the 32k rung")):
        panel_figure(grid, tasks4, f"Qwen3.5-4B grid: performance vs compute axes ({yl})", f"{VIZ}/grid35_{suffix}.png", ykey, yl)
        panel_figure(dense_grid + routed, ["contradiction", "oolong"],
                     f"Qwen3.5-4B routed attention / joint budget vs dense ({yl})", f"{VIZ}/routed4b_{suffix}.png", ykey, yl)
        panel_figure(q3, ["contradiction", "oolong"], f"Qwen3-4B: dense vs 2-router vs 3-router ({yl})", f"{VIZ}/qwen3_{suffix}.png", ykey, yl)
    # scale ladder: one panel per scale x task, mean f1 only, dense/ffnmoe/kv
    scales = ["0.8b", "2b", "4b", "9b", "27b"]
    for task in ("contradiction", "oolong"):
        fig, axes = plt.subplots(len(AXES), len(scales), figsize=(3.6 * len(scales), 3.2 * len(AXES)), squeeze=False)
        handles = {}
        for j, sc in enumerate(scales):
            trs = [r for r in ladder if r["task"] == task and r["scale"] == sc]
            if sc == "4b":
                trs += [r for r in grid if r["task"] == task and r["arm"] in ("dense", "ffnmoe-t10", "kv17", "kv33")]
            for i, (xk, xl) in enumerate(AXES):
                ax = axes[i][j]
                for arm in sorted({r["arm"] for r in trs}, key=lambda a: (a != "dense", a)):
                    st = STYLE.get(arm, dict(color="#888888", marker="o", label=arm))
                    pts = sorted([r for r in trs if r["arm"] == arm and r.get(xk) and r.get("mean_f1") is not None], key=lambda r: r[xk])
                    if not pts:
                        continue
                    ax.plot([r[xk] for r in pts], [r["mean_f1"] for r in pts], "-" if arm == "dense" else "--", color=st["color"], lw=1.2)
                    for r in pts:
                        hollow = xk == "gpu_hours" and not r.get("wt_measured")
                        ax.plot(r[xk], r["mean_f1"], st["marker"], color=st["color"], ms=6, mfc="white" if hollow else st["color"], mew=1.1)
                    handles.setdefault(arm, plt.Line2D([], [], color=st["color"], marker=st["marker"], ls="-" if arm == "dense" else "--", label=st["label"]))
                ax.set_xscale("log")
                ax.grid(True, which="both", alpha=0.25)
                if i == 0:
                    ax.set_title(f"{sc}", fontsize=12)
                ax.set_xlabel(xl, fontsize=8 if i < len(AXES) - 1 else 10, color="#666666" if i < len(AXES) - 1 else "black")
                if j == 0:
                    ax.set_ylabel("mean f1 over rungs")
        fig.suptitle(f"Model-scale ladder, {task}: performance vs compute axes (Qwen3.5)", fontsize=13)
        fig.legend(handles=list(handles.values()), loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02), frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 0.96))
        p = f"{VIZ}/scale_{task}.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("wrote", p)


if __name__ == "__main__":
    main()
