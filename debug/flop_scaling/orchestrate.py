"""
Unattended orchestrator for the FLOP-scaling study (records/flop-scaling-ffn-kv-plan.md).

Runs on the LOGIN node (compute nodes have no outbound access to Beaker/GitHub/S3) as a
detached process; state lives in debug/flop_scaling/orchestrator_state.json so it can be
killed and restarted at any point. Every cycle (5 min):

  A. data: once the outlier shards are on S3, run the S3->weka sync job and wait for it;
  B. grid: launch every (task, budget, arm in dense/kv17/kv33/ffnmoe-s1) not yet launched;
  C. training runs: poll Beaker; finalized+ok -> launch the multi-rung eval (and, for ffnmoe-s1,
     stage 2 which warm-starts from its export); failed -> relaunch ONCE under the same name;
  D. evals: poll; failed -> relaunch once;
  E. every ~90 min and at the end: harvest weka -> S3 -> local, collect + fit, so partial
     results are always readable. Prints ALL_DONE when nothing is left to wait for.

    setsid nohup python debug/flop_scaling/orchestrate.py > debug/flop_scaling/orchestrator.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
D = f"{REPO}/debug/flop_scaling"
STATE = f"{D}/orchestrator_state.json"
PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=f"{REPO}/src",
           PATH="/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:" + os.environ.get("PATH", ""),
           AWS_PROFILE="S3")
TASKS = ["outlier", "nq", "oolong", "contradiction"]
BUDGETS = {"contradiction": ["8M", "16M", "32M", "48M"]}
DEFAULT_BUDGETS = ["8M", "16M", "32M", "64M", "128M"]
ARMS = ["dense", "kv17", "kv33", "ffnmoe-s1"]
PILOTS = {  # already launched by hand
    "fs-contradiction-dense-sh16M": "01M1GB22BK7THD9GR345RBKD4Y",
    "fs-contradiction-kv17-sh16M": "01M1GB38PKKAFNQ2ATFDW68ZKT",
    "fs-contradiction-ffnmoe-s1-sh16M": "01M1GB4T12VY82Q4E1J59BJ5TZ",
}
CYCLE = 300
HARVEST_EVERY = 90 * 60


def log(msg: str) -> None:
    print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} {msg}", flush=True)


def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"synced": False, "sync_ex": None, "grid_launched": False, "runs": {}, "evals": {},
            "last_harvest": 0, "harvest_ex": None, "done": False}


def save_state(st):
    tmp = STATE + ".tmp"
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, STATE)


def sh(cmd, timeout=900):
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=REPO, env=ENV, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def beaker_status(ex: str):
    """-> ('S'|'R'|'F', exit_code or None); ('?', None) on error."""
    rc, out = sh(["beaker", "experiment", "get", ex, "--format", "json"], timeout=120)
    if rc != 0:
        return "?", None
    try:
        j = json.loads(out)
        j = j[0] if isinstance(j, list) else j
        s = j["jobs"][-1]["status"]
        if s.get("finalized"):
            return "F", s.get("exitCode")
        return ("R" if s.get("started") else "S"), None
    except Exception:
        return "?", None


def parse_id(out: str):
    m = (re.search(r"SUBMITTED id=(\S+)", out) or re.search(r"submitted: (\S+)", out)
         or re.search(r"beaker\.org/ex/([A-Z0-9]{26})", out) or re.search(r"rc=0: ([A-Z0-9]{26})", out))
    return m.group(1) if m else None


def all_runs():
    for t in TASKS:
        for b in BUDGETS.get(t, DEFAULT_BUDGETS):
            for a in ARMS:
                yield t, b, a, f"fs-{t}-{a}-sh{b}"


def launch_train(st, task, budget, arm):
    name = f"fs-{task}-{arm}-sh{budget}"
    rc, out = sh([PY, f"{D}/launch_grid.py", "--tasks", task, "--budgets", budget, "--arms", arm, "launch"], timeout=1200)
    ex = parse_id(out)
    prev = st["runs"].get(name, {})
    st["runs"][name] = {"ex": ex, "task": task, "budget": budget, "arm": arm, "state": "S" if ex else "LAUNCH-FAILED",
                        "rc": None, "retries": prev.get("retries", 0), "eval": prev.get("eval"), "s2": prev.get("s2")}
    log(f"launch {name} -> {ex or 'FAILED: ' + out[-200:]}")
    save_state(st)


def launch_eval(st, name):
    rc, out = sh([PY, f"{D}/launch_evals.py", name], timeout=1200)
    ex = parse_id(out)
    st["evals"][name] = {"ex": ex, "state": "S" if ex else "LAUNCH-FAILED", "rc": None, "retries": st["evals"].get(name, {}).get("retries", 0)}
    st["runs"][name]["eval"] = ex
    log(f"eval {name} -> {ex or 'FAILED: ' + out[-200:]}")
    save_state(st)


def s3_has_outlier():
    rc, out = sh("aws s3 ls s3://ai2-llm/checkpoints/prasanns/flop_scaling/shards/outlier_sh128M/", timeout=120)
    return "metadata.json" in out


def harvest(st):
    rc, out = sh(f"bash {D}/harvest_to_s3.sh", timeout=600)
    ex = parse_id(out)
    st["harvest_ex"] = ex
    st["last_harvest"] = time.time()
    log(f"harvest job -> {ex}")
    save_state(st)


def collect(st):
    rc, out = sh("aws s3 sync s3://ai2-llm/checkpoints/prasanns/flop_scaling/harvest results/flop_scaling/harvest --only-show-errors", timeout=600)
    rc2, out2 = sh([PY, f"{D}/collect_results.py"], timeout=600)
    log("collect: " + "\n".join(out2.strip().splitlines()[-12:]))
    rc3, out3 = sh([PY, f"{D}/fit_scaling.py"], timeout=600)
    log("fit: " + ("ok" if rc3 == 0 else out3[-300:]))


def main():
    st = load_state()
    for name, ex in PILOTS.items():
        if name not in st["runs"]:
            t, a, b = name.split("-")[1], "-".join(name.split("-")[2:-1]), name.split("-sh")[-1]
            st["runs"][name] = {"ex": ex, "task": t, "budget": b, "arm": a, "state": "S", "rc": None, "retries": 0, "eval": None, "s2": None}
    save_state(st)
    log("orchestrator up; runs tracked: %d" % len(st["runs"]))
    while True:
        try:
            cycle(st)
        except Exception as e:  # never die
            log(f"cycle error: {e!r}")
        if st.get("done"):
            log("ALL_DONE")
            return
        time.sleep(CYCLE)


def cycle(st):
    # A. data sync
    if not st["synced"]:
        if st["sync_ex"] is None:
            if s3_has_outlier():
                rc, out = sh(f"bash {D}/sync_s3_to_weka.sh", timeout=600)
                st["sync_ex"] = parse_id(out)
                log(f"sync job -> {st['sync_ex']}")
            else:
                log("waiting for outlier shards on S3")
            save_state(st)
            return
        s, rc = beaker_status(st["sync_ex"])
        if s == "F":
            st["synced"] = rc == 0
            log(f"sync finalized rc={rc}")
            if rc != 0:
                st["sync_ex"] = None
            save_state(st)
            if not st["synced"]:
                return  # relaunch next cycle; do NOT fall through to the grid (bug 2026-09-01)
        else:
            log(f"sync {s}")
            return
    # B. grid (only with the data on weka)
    if st["synced"] and not st["grid_launched"]:
        for t, b, a, name in all_runs():
            if name in st["runs"]:
                continue
            launch_train(st, t, b, a)
        st["grid_launched"] = True
        save_state(st)
    # C. training runs
    for name, r in list(st["runs"].items()):
        if r["state"] in ("DONE", "FAILED") or not r.get("ex"):
            if r["state"] == "LAUNCH-FAILED" and r["retries"] < 1:
                r["retries"] += 1
                launch_train(st, r["task"], r["budget"], r["arm"])
            continue
        s, rc = beaker_status(r["ex"])
        if s == "?":
            continue
        if s != r["state"] and s in ("R",):
            log(f"{name} running")
        r["state"] = s
        if s == "F":
            r["rc"] = rc
            if rc == 0:
                r["state"] = "DONE"
                log(f"{name} DONE")
                if not r.get("eval"):
                    launch_eval(st, name)
                if r["arm"] == "ffnmoe-s1" and not r.get("s2"):
                    launch_train(st, r["task"], r["budget"], "ffnmoe-s2")
                    r["s2"] = st["runs"][f"fs-{r['task']}-ffnmoe-s2-sh{r['budget']}"]["ex"]
            else:
                if r["retries"] < 1:
                    r["retries"] += 1
                    log(f"{name} failed rc={rc}; relaunching (resume)")
                    launch_train(st, r["task"], r["budget"], r["arm"])
                    st["runs"][name]["retries"] = r["retries"]
                else:
                    r["state"] = "FAILED"
                    log(f"{name} FAILED twice rc={rc}")
        save_state(st)
    # D. evals
    for name, e in list(st["evals"].items()):
        if e["state"] in ("DONE", "FAILED") or not e.get("ex"):
            if e["state"] == "LAUNCH-FAILED" and e["retries"] < 1:
                e["retries"] += 1
                launch_eval(st, name)
            continue
        s, rc = beaker_status(e["ex"])
        if s == "?":
            continue
        e["state"] = s
        if s == "F":
            e["rc"] = rc
            if rc == 0:
                e["state"] = "DONE"
                log(f"eval {name} DONE")
            elif e["retries"] < 1:
                e["retries"] += 1
                log(f"eval {name} failed rc={rc}; relaunching")
                launch_eval(st, name)
                st["evals"][name]["retries"] = e["retries"]
            else:
                e["state"] = "FAILED"
                log(f"eval {name} FAILED twice")
        save_state(st)
    # E. harvest / completion
    pending_runs = [n for n, r in st["runs"].items() if r["state"] not in ("DONE", "FAILED")]
    pending_evals = [n for n, e in st["evals"].items() if e["state"] not in ("DONE", "FAILED")]
    missing_evals = [n for n, r in st["runs"].items() if r["state"] == "DONE" and not r.get("eval")]
    log(f"status: runs pending {len(pending_runs)}, evals pending {len(pending_evals)}, done runs {sum(r['state']=='DONE' for r in st['runs'].values())}, done evals {sum(e['state']=='DONE' for e in st['evals'].values())}")
    if st.get("harvest_ex"):
        s, rc = beaker_status(st["harvest_ex"])
        if s == "F":
            st["harvest_ex"] = None
            collect(st)
            save_state(st)
            if st.get("finishing"):
                st["done"] = True
                save_state(st)
                return
    finished = st["grid_launched"] and not pending_runs and not pending_evals and not missing_evals
    if (time.time() - st["last_harvest"] > HARVEST_EVERY or finished) and not st.get("harvest_ex"):
        if finished:
            st["finishing"] = True
        harvest(st)


if __name__ == "__main__":
    main()
