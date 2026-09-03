"""
Collect the Qwen3.5 track: new-arm evals (Beaker result datasets of the eval jobs tracked in
orchestrator35_state.json) + their flops.json (harvest) + the prior campaigns' DENSE points
(debug/taskscale_lengthmix/points.json, dense FLOPs computed analytically from the arm's token
budget), into results/flop_scaling/results35.csv, then fit (fit_scaling.py --in results35.csv).

    python debug/flop_scaling/collect_results35.py
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import subprocess
import sys

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
D = f"{REPO}/debug/flop_scaling"
OUT = f"{REPO}/results/flop_scaling"
HARVEST = f"{OUT}/harvest"
EVALS = f"{OUT}/evals35"
STATE = f"{D}/orchestrator35_state.json"
POINTS = f"{REPO}/debug/taskscale_lengthmix/points.json"
sys.path.insert(0, D)
from launch_grid35 import ARMS  # noqa: E402

RUNGS = {"contradiction": ["2k", "8k", "16k", "32k"], "nq": ["2k", "8k", "16k", "32k"],
         "outlier": ["8k", "16k", "32k"], "oolong": ["2k", "8k", "16k", "32k"]}
TASK_KEY = {"contradiction": "contra", "nq": "nq", "outlier": "outlier", "oolong": "oolong"}
ENV = dict(os.environ, PATH="/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:" + os.environ.get("PATH", ""))


def budget_tokens(b):
    return int(b[:-1]) * 1_000_000


_MODELS = {}
SCALE = "4b"  # FLOP pricing follows the model scale (collect_scale.py switches this per ladder rung)


def _model():
    from olmo_core.nn.transformer import TransformerConfig
    if SCALE not in _MODELS:
        fac = {"0.8b": TransformerConfig.qwen3_5_0_8B, "2b": TransformerConfig.qwen3_5_2B, "4b": TransformerConfig.qwen3_5_4B,
               "9b": TransformerConfig.qwen3_5_9B, "27b": TransformerConfig.qwen3_5_27B}[SCALE]
        _MODELS[SCALE] = fac(vocab_size=248320).build(init_device="meta")
    return _MODELS[SCALE]


def fpt(L):
    """Qwen3.5-4B training FLOPs/token for an example of length L (the meter's own formula)."""
    return int(_model().num_flops_per_token(int(L)))


def ffn_per_token():
    return int(sum(b.feed_forward.num_flops_per_token(1) for b in _model().blocks.values()))


_FPT_CACHE = {}


def arm_flops(lengths, ffn_cost=1.0):
    """Sum over examples of L * [fpt(L) - ffn_per_tok * (1 - ffn_cost)], with attention priced at
    each example's REAL length (the dense arms train packed with per-example masking; the padded
    65536 window the meter uses would inflate attention ~10x). Lengths are bucketed to 256."""
    ffn = ffn_per_token()
    total = 0.0
    for L in lengths:
        b = max(256, (int(L) + 255) // 256 * 256)
        if (SCALE, b) not in _FPT_CACHE:
            _FPT_CACHE[(SCALE, b)] = fpt(b)
        total += L * (_FPT_CACHE[(SCALE, b)] - ffn * (1.0 - ffn_cost))
    return total / 1e15


def arm_lengths(arm_dir_name):
    p = f"{HARVEST}/arms/{arm_dir_name}_lengths.json"
    return json.load(open(p))["lengths"] if os.path.exists(p) else None


def dense_flops_per_token():
    return fpt(65536)


_LOG_CACHE = {}


def f1_from_log(ex):
    """``{<task>_<rung>: f1}`` parsed from an eval experiment's ``[ladder:task@rung] f1=`` lines
    (cached on disk under evals35/logs so the collect stays fast on re-runs)."""
    if not ex:
        return {}
    p = f"{EVALS}/logs/{ex}.json"
    if os.path.exists(p):
        return json.load(open(p))
    if ex in _LOG_CACHE:
        return _LOG_CACHE[ex]
    try:
        out = subprocess.run(["beaker", "experiment", "logs", ex], env=ENV, capture_output=True, text=True, timeout=300).stdout
    except Exception:
        out = ""
    hits = re.findall(r"\[ladder:(\w+)@(\d+k)\] (?:f1|score)=([0-9.]+)", out)  # oolong prints score=
    res = {f"{t}_{rg}": float(v) for t, rg, v in hits}
    _LOG_CACHE[ex] = res
    if res and "=== DONE" in out:
        os.makedirs(f"{EVALS}/logs", exist_ok=True)
        json.dump(res, open(p, "w"))
    return res


def fetch_eval(run, ex):
    d = f"{EVALS}/{run}"
    if glob.glob(f"{d}/**/*multirung*.json", recursive=True):
        return d
    os.makedirs(d, exist_ok=True)
    subprocess.run(["beaker", "experiment", "results", ex, "-o", d], env=ENV, capture_output=True, text=True, timeout=600)
    return d if glob.glob(f"{d}/**/*multirung*.json", recursive=True) else None


def main(state_path=STATE, out_csv=None, scale="4b", prior_dense=True, run_fit=True):
    global SCALE
    SCALE = scale
    out_csv = out_csv or f"{OUT}/results35.csv"
    st = json.load(open(state_path)) if os.path.exists(state_path) else {"runs": {}, "evals": {}}
    fpt65k = dense_flops_per_token()
    rows = []
    # dense baseline from the prior campaigns (+ rungs those campaigns never scored, launched by
    # hand for this study and listed in dense_extra_evals.tsv: run, savedir, ex, task, rungs, when)
    pts = json.load(open(POINTS)) if prior_dense else {}
    extra = {}
    xp = f"{D}/dense_extra_evals.tsv"
    if os.path.exists(xp):
        for line in open(xp):
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5 or not cols[2]:
                continue
            run, ex, task = cols[0], cols[2], cols[3]
            m = re.search(r"mixs(\d+M)", run)
            d = fetch_eval(f"dense-{run}", ex)
            if m and d:
                res = json.load(open(glob.glob(f"{d}/**/*multirung*.json", recursive=True)[0]))
                for rg in RUNGS[task]:
                    v = res.get(f"{TASK_KEY[task]}_{rg}", res.get(f"{task}_{rg}"))
                    if v is not None:
                        extra[(task, m.group(1), rg)] = v
    for task in ARMS:
        dense = pts.get(task, {}).get("dense", {})
        for b in ARMS[task]:
            tok = budget_tokens(b)
            f1 = {r: dense.get(r, {}).get(str(tok)) for r in RUNGS[task]}
            for r in RUNGS[task]:
                if f1[r] is None and (task, b, r) in extra:
                    f1[r] = extra[(task, b, r)]
            vals = [v for v in f1.values() if v is not None]
            if not vals:
                continue
            lens = arm_lengths(os.path.basename(ARMS[task][b]))
            pf = arm_flops(lens) if lens else tok * fpt65k / 1e15  # fallback: padded-window formula
            rows.append({"task": task, "arm": "dense", "budget": b, "tokens": tok, "actual_pflops": pf,
                         "dense_equiv_pflops": pf, "flop_ratio": 1.0, "mean_f1": sum(vals) / len(vals),
                         "partial": int(len(vals) < len(RUNGS[task])),
                         **{f"f1_{r}": v for r, v in f1.items()}, "run": f"prior-dense-{task}-{b}"})
    # new arms
    for run, e in st.get("evals", {}).items():
        if e.get("state") != "DONE" or not e.get("ex"):
            continue
        r = st["runs"][run]
        # docchunk-evaluated (KV) runs: the per-rung f1 lines of THIS eval's Beaker log are the
        # primary source -- the weka _eval_results/<run>_<task>_multirung.json is a merge target
        # that any later eval of the same run overwrites (2026-09-02: "cancelled" garbage-id evals
        # kept running and finished after the fixed ones). Harvest file = fallback only.
        res = f1_from_log(e["ex"]) if r["arm"].startswith("kv") else {}
        hv = glob.glob(f"{HARVEST}/evals/{run}_*multirung*.json")
        if res:
            pass
        elif hv:
            res = json.load(open(hv[0]))
        else:
            d = fetch_eval(run, e["ex"])
            if not d:
                print("no result file for", run)
                continue
            res = json.load(open(glob.glob(f"{d}/**/*multirung*.json", recursive=True)[0]))
        tkey = TASK_KEY[r["task"]]
        f1 = {rg: (res.get(f"{tkey}_{rg}") if res.get(f"{tkey}_{rg}") is not None else res.get(f"{r['task']}_{rg}")) for rg in RUNGS[r["task"]]}
        vals = [v for v in f1.values() if v is not None]
        flp = f"{HARVEST}/runs/{run}/flops.json"
        fl = json.load(open(flp)) if os.path.exists(flp) else None
        data_budget = r.get("data_budget", r["budget"])  # sub-budget dense points train on a bigger arm
        lens = arm_lengths(os.path.basename(ARMS[r["task"]][data_budget]))
        frac = budget_tokens(r["budget"]) / budget_tokens(data_budget)
        dpf = arm_flops(lens) * frac if lens else (fl["dense_equivalent_pflops"] if fl else None)
        pf = None
        if r["arm"] == "dense":
            pf = dpf  # --max-tokens dense anchor: priced like the prior dense points, scaled to its budget
        elif fl:
            if r["arm"].startswith("ffnmoe"):
                # back out the mean routed FFN cost from the meter's own (padded-window) ratio,
                # then re-price with real lengths: ratio = 1 - ffn_frac65k * (1 - c)
                ffn_frac = ffn_per_token() / fpt(65536)
                c = 1.0 - (1.0 - fl["actual_over_dense"]) / ffn_frac
                pf = arm_flops(lens, ffn_cost=max(0.0, min(1.0, c))) if lens else fl["actual_pflops"]
            else:
                pf = fl["actual_pflops"]  # KV: the meter already prices the compacted rows at their real length
        if r["arm"] == "ffnmoe-s2":
            s1run = run.replace("ffnmoe-s2", "ffnmoe-s1"); f1p = f"{HARVEST}/runs/{s1run}/flops.json"
            if os.path.exists(f1p) and pf is not None:
                s1 = json.load(open(f1p)); ffn_frac = ffn_per_token() / fpt(65536)
                c1 = 1.0 - (1.0 - s1["actual_over_dense"]) / ffn_frac
                pf += arm_flops(lens, ffn_cost=max(0.0, min(1.0, c1))) if lens else s1["actual_pflops"]
                dpf = (dpf or 0) + (arm_flops(lens) if lens else s1["dense_equivalent_pflops"])
        rows.append({"task": r["task"], "arm": r["arm"], "budget": r["budget"], "tokens": budget_tokens(r["budget"]),
                     "actual_pflops": pf, "dense_equiv_pflops": dpf, "flop_ratio": (pf / dpf) if (pf and dpf) else None,
                     "mean_f1": sum(vals) / len(vals) if vals else None, "partial": int(len(vals) < len(RUNGS[r["task"]])),
                     **{f"f1_{rg}": v for rg, v in f1.items()}, "run": run})
    os.makedirs(OUT, exist_ok=True)
    if not rows:
        print("nothing to collect"); return
    for r in rows:
        r["scale"] = scale
    keys = ["task", "scale", "arm", "budget", "tokens", "actual_pflops", "dense_equiv_pflops", "flop_ratio", "mean_f1", "partial"] + sorted({k for r in rows for k in r if k.startswith("f1_")}) + ["run"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} rows)")
    for r in sorted(rows, key=lambda x: (x["task"], x["arm"], x["tokens"])):
        print(f"{r['task']:14s} {r['arm']:10s} {r['budget']:>5s} mean_f1={None if r['mean_f1'] is None else round(r['mean_f1'],3)} "
              f"PF={None if r['actual_pflops'] is None else round(r['actual_pflops'],1)} ratio={None if r['flop_ratio'] is None else round(r['flop_ratio'],3)}")
    if run_fit:
        subprocess.run([sys.executable, f"{D}/fit_scaling.py", "--in", out_csv, "--tag", "35"], env=ENV)
    return rows


if __name__ == "__main__":
    main()
