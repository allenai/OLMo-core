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


def dense_flops_per_token():
    """Qwen3.5-4B dense training FLOPs/token at the packed 65536 window (meta build, no weights)."""
    from olmo_core.nn.transformer import TransformerConfig
    cfg = TransformerConfig.qwen3_5_4B(vocab_size=248320)
    model = cfg.build(init_device="meta")
    return int(model.num_flops_per_token(65536))


def fetch_eval(run, ex):
    d = f"{EVALS}/{run}"
    if glob.glob(f"{d}/*multirung*.json"):
        return d
    os.makedirs(d, exist_ok=True)
    subprocess.run(["beaker", "experiment", "results", ex, "-o", d], env=ENV, capture_output=True, text=True, timeout=600)
    return d if glob.glob(f"{d}/*multirung*.json") else None


def main():
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}}
    fpt = dense_flops_per_token()
    rows = []
    # dense baseline from the prior campaigns
    pts = json.load(open(POINTS))
    for task in ARMS:
        dense = pts.get(task, {}).get("dense", {})
        for b in ARMS[task]:
            tok = budget_tokens(b)
            f1 = {r: dense.get(r, {}).get(str(tok)) for r in RUNGS[task]}
            vals = [v for v in f1.values() if v is not None]
            if not vals:
                continue
            pf = tok * fpt / 1e15
            rows.append({"task": task, "arm": "dense", "budget": b, "tokens": tok, "actual_pflops": pf,
                         "dense_equiv_pflops": pf, "flop_ratio": 1.0, "mean_f1": sum(vals) / len(vals),
                         **{f"f1_{r}": v for r, v in f1.items()}, "run": f"prior-dense-{task}-{b}"})
    # new arms
    for run, e in st.get("evals", {}).items():
        if e.get("state") != "DONE" or not e.get("ex"):
            continue
        r = st["runs"][run]
        d = fetch_eval(run, e["ex"])
        if not d:
            print("no result file for", run)
            continue
        res = json.load(open(glob.glob(f"{d}/*multirung*.json")[0]))
        tkey = TASK_KEY[r["task"]]
        f1 = {rg: res.get(f"{tkey}_{rg}") for rg in RUNGS[r["task"]]}
        vals = [v for v in f1.values() if v is not None]
        flp = f"{HARVEST}/runs/{run}/flops.json"
        fl = json.load(open(flp)) if os.path.exists(flp) else None
        pf = fl["actual_pflops"] if fl else None
        dpf = fl["dense_equivalent_pflops"] if fl else None
        if r["arm"] == "ffnmoe-s2":
            f1p = f"{HARVEST}/runs/{run.replace('ffnmoe-s2', 'ffnmoe-s1')}/flops.json"
            if os.path.exists(f1p) and pf is not None:
                s1 = json.load(open(f1p)); pf += s1["actual_pflops"]; dpf += s1["dense_equivalent_pflops"]
        rows.append({"task": r["task"], "arm": r["arm"], "budget": r["budget"], "tokens": budget_tokens(r["budget"]),
                     "actual_pflops": pf, "dense_equiv_pflops": dpf, "flop_ratio": (pf / dpf) if (pf and dpf) else None,
                     "mean_f1": sum(vals) / len(vals) if vals else None, **{f"f1_{rg}": v for rg, v in f1.items()}, "run": run})
    os.makedirs(OUT, exist_ok=True)
    if not rows:
        print("nothing to collect"); return
    keys = ["task", "arm", "budget", "tokens", "actual_pflops", "dense_equiv_pflops", "flop_ratio", "mean_f1"] + sorted({k for r in rows for k in r if k.startswith("f1_")}) + ["run"]
    with open(f"{OUT}/results35.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT}/results35.csv ({len(rows)} rows)")
    for r in sorted(rows, key=lambda x: (x["task"], x["arm"], x["tokens"])):
        print(f"{r['task']:14s} {r['arm']:10s} {r['budget']:>5s} mean_f1={None if r['mean_f1'] is None else round(r['mean_f1'],3)} "
              f"PF={None if r['actual_pflops'] is None else round(r['actual_pflops'],1)} ratio={None if r['flop_ratio'] is None else round(r['flop_ratio'],3)}")
    subprocess.run([sys.executable, f"{D}/fit_scaling.py", "--in", f"{OUT}/results35.csv", "--tag", "35"], env=ENV)


if __name__ == "__main__":
    main()
