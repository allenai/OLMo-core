"""
Per-run training wall-clock for the FLOP-scaling campaigns (4B grid, model-scale ladder, KV-route
/ flex / Qwen3 arms): parse each training experiment's Beaker log for the data-parallel world size
and the timestamps of the first and last logged step, plus the job's started/exited times.

Writes results/flop_scaling/walltime/<run>.json (cached; re-runs only fetch missing runs) and a
merged results/flop_scaling/walltime.csv with columns
    run, ex, gpus, steps, train_seconds (step 1 -> last step line), job_seconds (started -> exited)

    python debug/flop_scaling/collect_walltime.py            # all state files
    python debug/flop_scaling/collect_walltime.py --workers 8
"""

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
D = f"{REPO}/debug/flop_scaling"
OUT = f"{REPO}/results/flop_scaling/walltime"
ENV = dict(os.environ, PATH="/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:" + os.environ.get("PATH", ""))
STATES = [f"{D}/orchestrator35_state.json", f"{D}/prior_dense_state.json"] + sorted(glob.glob(f"{D}/orchestrate_s*_state.json")) + sorted(
    glob.glob(f"{D}/orchestrate_q3s4b*2_state.json")
)

TS = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?)Z")
STEP = re.compile(r"\[step=(\d+)/(\d+)")
WORLD = re.compile(r"Data parallel world size = (\d+)")
CP = re.compile(r"[Cc]ontext parallel.*?(\d+)")


def _t(s):
    return datetime.fromisoformat(s.replace("Z", ""))


def parse_log(text):
    gpus = None
    first = last = None
    steps = None
    n_cp = 1
    for line in text.splitlines():
        m = TS.match(line)
        if not m:
            continue
        ts = m.group(1)
        if gpus is None:
            w = WORLD.search(line)
            if w:
                gpus = int(w.group(1))
        s = STEP.search(line)
        if s:
            if first is None:
                first = ts
            last = ts
            steps = int(s.group(1))
    return {"gpus": gpus, "steps": steps, "t_first_step": first, "t_last_step": last,
            "train_seconds": (_t(last) - _t(first)).total_seconds() if first and last else None}


def job_times(ex):
    try:
        out = subprocess.run(["beaker", "experiment", "get", ex, "--format", "json"], env=ENV,
                             capture_output=True, text=True, timeout=120).stdout
        d = json.loads(out)
        e = d[0] if isinstance(d, list) else d
        j = e["jobs"][-1]["status"]
        return {"job_started": j.get("started"), "job_exited": j.get("exited"),
                "job_seconds": (_t(j["exited"]) - _t(j["started"])).total_seconds() if j.get("started") and j.get("exited") else None,
                "n_jobs": len(e["jobs"])}
    except Exception as exc:  # noqa: BLE001
        return {"job_error": str(exc)[:200]}


def one(run, meta):
    p = f"{OUT}/{run}.json"
    if os.path.exists(p):
        return run, json.load(open(p))
    ex = meta["ex"]
    try:
        log = subprocess.run(["beaker", "experiment", "logs", ex], env=ENV, capture_output=True, text=True, timeout=600).stdout
    except Exception as exc:  # noqa: BLE001
        log = ""
        print("log fetch failed", run, exc, file=sys.stderr)
    rec = {"run": run, "ex": ex, "task": meta.get("task"), "arm": meta.get("arm"), "budget": meta.get("budget"),
           "scale": meta.get("scale", "4b"), **parse_log(log), **job_times(ex)}
    if rec.get("train_seconds") is not None:
        json.dump(rec, open(p, "w"), indent=1)
    return run, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--states", nargs="*", default=STATES)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    runs = {}
    for sp in a.states:
        st = json.load(open(sp))
        for run, r in st.get("runs", {}).items():
            if r.get("state") == "DONE" and r.get("ex"):
                runs[run] = r
    print(f"{len(runs)} finished runs across {len(a.states)} state files", flush=True)
    recs = {}
    done = 0
    with ThreadPoolExecutor(a.workers) as pool:
        futs = [pool.submit(one, run, meta) for run, meta in runs.items()]
        for f in as_completed(futs):
            run, rec = f.result()
            recs[run] = rec
            done += 1
            if done in (1, 2, 5, 10) or done % 20 == 0:
                print(f"[{done}/{len(runs)}] {run}: gpus={rec.get('gpus')} steps={rec.get('steps')} "
                      f"train={rec.get('train_seconds')} job={rec.get('job_seconds')}", flush=True)
    cols = ["run", "ex", "task", "arm", "budget", "scale", "gpus", "steps", "train_seconds", "job_seconds", "t_first_step", "t_last_step"]
    with open(f"{OUT}.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for run in sorted(recs):
            w.writerow(recs[run])
    missing = [r for r, v in recs.items() if v.get("train_seconds") is None]
    print(f"wrote {OUT}.csv ({len(recs)} rows); {len(missing)} without step timing: {missing[:10]}", flush=True)


if __name__ == "__main__":
    main()
