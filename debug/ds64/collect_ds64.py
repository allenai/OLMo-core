"""
Collect the ds64 campaign (records/ds64-scaling-plan.md) into results/ds64/results.csv: one row per
training run with per-rung f1 (parsed from the eval job's `[ladder:task@rung] f1=` lines), mean f1,
the FLOP meter's token accounting (tokens in / out, compaction) from the harvested flops.json,
and wall-clock (train seconds, GPUs, GPU-hours) from the training job's Beaker log.

    python debug/ds64/collect_ds64.py            # 4B state file
    DS64_SCALE=27b python debug/ds64/collect_ds64.py
"""

import csv
import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
D = f"{REPO}/debug/ds64"
SCALE = os.environ.get("DS64_SCALE", "4b")
STATE = f"{D}/orchestrator_ds64{'' if SCALE == '4b' else '_' + SCALE}_state.json"
OUT = f"{REPO}/results/ds64"
HARVEST = f"{OUT}/harvest"
CACHE = f"{OUT}/logs"
ENV = dict(os.environ, PATH="/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:" + os.environ.get("PATH", ""))
RUNGS = ["2k", "8k", "16k", "32k", "64k"]
TASK_KEY = {"contradiction": "contra", "nq": "nq", "outlier": "outlier", "oolong": "oolong"}


def beaker_log(ex, kind, final=False):
    """Fetch (and cache only when ``final``: a log cached mid-run would freeze partial results)."""
    os.makedirs(CACHE, exist_ok=True)
    p = f"{CACHE}/{kind}_{ex}.log"
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return open(p).read()
    try:
        out = subprocess.run(["beaker", "experiment", "logs", ex], env=ENV, capture_output=True, text=True, timeout=600).stdout
    except Exception:  # noqa: BLE001
        return ""
    if final and out:
        open(p, "w").write(out)
    return out


def f1_from_eval(ex, final=False):
    out = beaker_log(ex, "eval", final=final)
    res = {}
    for task, rung, val, skipped in re.findall(r"\[ladder:(\w+)@(\d+k)\] (?:f1|score)=([0-9.]+) \(n=\d+, skipped_too_long=(\d+)\)", out):
        res[rung] = float(val)
        if int(skipped):
            res[f"skipped_{rung}"] = int(skipped)
    return res


def walltime(ex):
    """(gpus, steps, train_seconds) from the training log: world size + first/last step timestamps."""
    out = beaker_log(ex, "train", final=True)
    lines = [re.sub(r"^(\S+Z) ", r"\1 ", ln) for ln in out.splitlines()]
    gpus = None
    m = re.search(r"world[_ ]size[=: ]+(\d+)", out, re.I)
    if m:
        gpus = int(m.group(1))
    stamps = []
    for ln in lines:
        m2 = re.match(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z .*\[step=(\d+)/", ln)  # console_logger: [step=1/31,epoch=1]
        if m2:
            stamps.append((datetime.fromisoformat(m2.group(1)), int(m2.group(2))))
    if len(stamps) < 2:
        return gpus, len(stamps), None
    return gpus, stamps[-1][1], (stamps[-1][0] - stamps[0][0]).total_seconds()


def main():
    os.makedirs(OUT, exist_ok=True)
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}}
    rows = []
    for run, r in st["runs"].items():
        e = st["evals"].get(run, {})
        f1 = f1_from_eval(e["ex"], final=(e.get("state") == "DONE")) if e.get("ex") else {}
        vals = [f1[k] for k in RUNGS if k in f1]
        fl = {}
        p = f"{HARVEST}/runs/{run}/flops.json"
        if os.path.exists(p):
            fl = json.load(open(p))
        gpus, steps, secs = walltime(r["ex"]) if r.get("ex") else (None, None, None)
        rows.append({"run": run, "scale": SCALE, "task": r["task"], "arm": r["arm"], "budget": r["budget"], "train_state": r["state"],
                     "eval_state": e.get("state"), "mean_f1": (sum(vals) / len(vals)) if vals else None, "n_rungs": len(vals),
                     **{f"f1_{k}": f1.get(k) for k in RUNGS},
                     "skipped_too_long": ";".join(f"{k}:{f1[f'skipped_{k}']}" for k in RUNGS if f"skipped_{k}" in f1),
                     "tokens_in": fl.get("tokens_in") or fl.get("soft_token_compaction", {}).get("tokens_in"),
                     "tokens_out": fl.get("tokens_out") or fl.get("soft_token_compaction", {}).get("tokens_out"),
                     "flops_meter": fl.get("total_flops") or fl.get("flops"),
                     "gpus": gpus, "steps": steps, "train_seconds": secs,
                     "gpu_hours": (gpus * secs / 3600) if (gpus and secs) else None,
                     "train_ex": r.get("ex"), "eval_ex": e.get("ex")})
    cols = list(rows[0].keys()) if rows else ["run"]
    with open(f"{OUT}/results{'' if SCALE == '4b' else '_' + SCALE}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    done = [r for r in rows if r["mean_f1"] is not None]
    print(f"{len(rows)} runs, {len(done)} scored -> {OUT}/results.csv")
    for r in sorted(done, key=lambda x: (x["task"], x["arm"], x["budget"])):
        print(f"  {r['run']:40} mean f1 {r['mean_f1']:.3f} " + " ".join(f"{k}={r['f1_'+k]:.2f}" for k in RUNGS if r['f1_'+k] is not None)
              + (f"  {r['gpu_hours']:.1f} GPUh" if r["gpu_hours"] else ""))


if __name__ == "__main__":
    main()
