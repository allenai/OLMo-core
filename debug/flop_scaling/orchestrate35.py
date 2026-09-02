"""
Unattended orchestrator for the Qwen3.5 track (records/flop-scaling-ffn-kv-plan.md §9).
Runs detached on the LOGIN node; state in orchestrator35_state.json; restartable.

Each 5-min cycle:
  A. ffnmoe-s1 on every (task, budget) dense arm (launch once);
  B. KV data: when the marker shards for a task are on S3 (mooney jobs) or on weka (Beaker jobs),
     sync S3->weka once and launch kv17/kv33 for that task;
  C. poll training runs: finalized ok -> eval via beaker_native_lengthmix_eval.py (the dense
     campaigns' launcher, per-task ladder config) and, for ffnmoe-s1, stage 2; failed -> one relaunch;
  D. poll evals; failed -> one relaunch;
  E. harvest every 90 min (weka flops.json + eval JSONs -> S3 -> local) and at the end; ALL_DONE.

    setsid nohup python debug/flop_scaling/orchestrate35.py >> debug/flop_scaling/orchestrator35.log 2>&1 &
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
STATE = f"{D}/orchestrator35_state.json"
PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=f"{REPO}/src",
           PATH="/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:" + os.environ.get("PATH", ""),
           AWS_PROFILE="S3")
sys.path.insert(0, D)
from launch_grid35 import ARMS, run_name  # noqa: E402

CYCLE = 300
HARVEST_EVERY = 90 * 60
# per-task eval config (debug/taskscale_lengthmix/fire_evals.sh + the outlier/nq campaign)
EVAL_CFG = {
    "oolong": ("2k,8k,16k,32k", "_eval_bundle_eval500_v2_clean", ""),
    "contradiction": ("2k,8k,16k,32k", "_eval_bundle_eval500_v3", "--ladder-version v3"),
    "outlier": ("8k,16k,32k", "outlier_lengthmix/eval_rungs", ""),
    "nq": ("2k,8k,16k,32k", "outlier_lengthmix/eval_rungs", ""),
}
KV_S3_TASKS = {"outlier", "nq"}  # marker shards built on mooney -> S3 -> weka
KV_WEKA_TASKS = {"contradiction", "oolong"}  # built straight onto weka by the gantry jobs
KV_WEKA_JOBS = {"contradiction": "01M1GCP893SDYYRTKP6JGSCVAB", "oolong": "01M1GCPCMH81BY9E6FX6HDVBG7"}


def log(m):
    print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} {m}", flush=True)


def load():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"runs": {}, "evals": {}, "kv_synced": {}, "kv_launched": {}, "s1_launched": False,
            "sync_ex": None, "last_harvest": 0, "harvest_ex": None, "done": False}


def save(st):
    json.dump(st, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def sh(cmd, timeout=900):
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=REPO, env=ENV, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def status(ex):
    rc, out = sh(["beaker", "experiment", "get", ex, "--format", "json"], timeout=120)
    if rc != 0:
        return "?", None
    try:
        j = json.loads(out)
        j = j[0] if isinstance(j, list) else j
        s = j["jobs"][-1]["status"]
        return ("F", s.get("exitCode")) if s.get("finalized") else (("R" if s.get("started") else "S"), None)
    except Exception:
        return "?", None


def parse_id(out):
    m = (re.search(r"SUBMITTED id=(\S+)", out) or re.search(r"submitted: (\S+)", out)
         or re.search(r"beaker\.org/ex/([A-Z0-9]{26})", out) or re.search(r"rc=0: ([A-Z0-9]{26})", out))
    return m.group(1) if m else None


def launch_train(st, task, budget, arm):
    name = run_name(task, arm, budget)
    rc, out = sh([PY, f"{D}/launch_grid35.py", "--tasks", task, "--budgets", budget, "--arms", arm, "launch"], timeout=1200)
    ex = parse_id(out)
    prev = st["runs"].get(name, {})
    st["runs"][name] = {"ex": ex, "task": task, "budget": budget, "arm": arm, "state": "S" if ex else "LAUNCH-FAILED",
                        "rc": None, "retries": prev.get("retries", 0), "eval": prev.get("eval"), "s2": prev.get("s2")}
    log(f"launch {name} -> {ex or 'FAILED: ' + out[-300:]}")
    save(st)


W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
# The KV arms train on MARKER-wrapped prompts, so they are scored with the marker-aware
# (docchunk, dense-emitter) evaluator on the SAME rung files the dense campaign used; the FFN
# arms train on the dense arms' plain prompts and use the dense campaign's evaluator as-is.
KV_RUNG_FILES = {
    "outlier": {"outlier": {"8k": f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_8192.jsonl",
                            "16k": f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_16384.jsonl",
                            "32k": f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_32768.jsonl"}},
    "nq": {"nq": {"2k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl",
                  "8k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_8192.jsonl",
                  "16k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_16384.jsonl",
                  "32k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_32768.jsonl"}},
    "contradiction": {"contradiction": {"2k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl",
                                        "8k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl",
                                        "16k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n385_k3.jsonl",
                                        "32k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n765_k3.jsonl"}},
    "oolong": {"oolong": {"2k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl",
                          "8k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl",
                          "16k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx16384_spliteval.jsonl",
                          "32k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx32768_spliteval.jsonl"}},
}
TASK_KEY = {"contradiction": "contra", "nq": "nq", "outlier": "outlier", "oolong": "oolong"}


def launch_eval(st, name):
    r = st["runs"][name]
    rungs, root, extra = EVAL_CFG[r["task"]]
    if r["arm"].startswith("kv"):
        cmd = [PY, "-u", f"{REPO}/src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py", name, "ai2/neptune",
               "--task", TASK_KEY[r["task"]], "--variant", "docchunk", "--ckpt", f"{W}/ctc_suite/ckpts/{name}",
               "--query-position", "after", "--cot-mode", "none", "--tokenizer", "Qwen/Qwen3.5-0.8B-Base",
               "--ngpu", "2", "--max-test", "600", "--priority", "urgent",
               "--dc-rung-files", json.dumps(KV_RUNG_FILES[r["task"]]),
               "--dc-rungs", ",".join(next(iter(KV_RUNG_FILES[r["task"]].values())).keys())]
    else:
        cmd = [PY, "-u", f"{REPO}/debug/outlier_lengthmix_scaling/beaker_native_lengthmix_eval.py", name, name,
               "--ladder-tasks", r["task"], "--ladder-rungs", rungs, "--cluster", "ai2/neptune", "--eval500-root", root] + (extra.split() if extra else [])
    rc, out = sh(cmd, timeout=1200)
    ex = parse_id(out)
    st["evals"][name] = {"ex": ex, "state": "S" if ex else "LAUNCH-FAILED", "rc": None, "retries": st["evals"].get(name, {}).get("retries", 0)}
    r["eval"] = ex
    log(f"eval {name} -> {ex or 'FAILED: ' + out[-300:]}")
    save(st)


def kv_data_ready(task):
    """All of the task's marker shards present at their expected place."""
    if task in KV_S3_TASKS:
        for b in ARMS[task]:
            rc, out = sh(f"aws s3 ls s3://ai2-llm/checkpoints/prasanns/flop_scaling35/shards/{task}_s{b}_mk/", timeout=120)
            if "metadata.json" not in out:
                return False
        return True
    return True  # weka-native builds: verified by the gantry job's KVDATA_DONE (checked below)


def harvest(st):
    rc, out = sh(f"bash {D}/harvest_to_s3.sh", timeout=600)
    st["harvest_ex"] = parse_id(out)
    st["last_harvest"] = time.time()
    log(f"harvest job -> {st['harvest_ex']}")
    save(st)


def collect(st):
    sh("aws s3 sync s3://ai2-llm/checkpoints/prasanns/flop_scaling/harvest results/flop_scaling/harvest --only-show-errors", timeout=600)
    rc2, out2 = sh([PY, f"{D}/collect_results35.py"], timeout=600)
    log("collect: " + "\n".join(out2.strip().splitlines()[-14:]))


def cycle(st):
    # A. stage-1 FFN on every dense arm
    if not st["s1_launched"]:
        for task in ARMS:
            for b in ARMS[task]:
                if run_name(task, "ffnmoe-s1", b) not in st["runs"]:
                    launch_train(st, task, b, "ffnmoe-s1")
        st["s1_launched"] = True
        save(st)
    # B. KV arms once data is in place AND the soft-token mechanism is validated on the hybrid
    #    (state flag kv_ok, set by hand after the fs35-smoke-softtoken run passes)
    for task in ARMS:
        if st["kv_launched"].get(task) or not st.get("kv_ok"):
            continue
        if task in KV_S3_TASKS:
            if not st["kv_synced"].get(task):
                if st.get("sync_ex"):
                    s, rc = status(st["sync_ex"])
                    if s == "F":
                        st["sync_ex"] = None
                        if rc == 0:
                            for t in KV_S3_TASKS:
                                if kv_data_ready(t):
                                    st["kv_synced"][t] = True
                        save(st)
                    continue
                if kv_data_ready(task):
                    rc, out = sh(f"PFX=flop_scaling35 bash {D}/sync_s3_to_weka.sh", timeout=600)
                    st["sync_ex"] = parse_id(out)
                    log(f"kv sync ({task} ready) -> {st['sync_ex']}")
                    save(st)
                continue
        else:
            if not st["kv_synced"].get(task):
                s_, rc_ = status(KV_WEKA_JOBS[task])
                if s_ == "F" and rc_ == 0:
                    st["kv_synced"][task] = True
                    log(f"kv data for {task} built on weka")
                    save(st)
                else:
                    continue
        for b in ARMS[task]:
            for arm in ("kv17", "kv33"):
                if run_name(task, arm, b) not in st["runs"]:
                    launch_train(st, task, b, arm)
        st["kv_launched"][task] = True
        save(st)
    # C. training runs
    for name, r in list(st["runs"].items()):
        if r["state"] in ("DONE", "FAILED") or not r.get("ex"):
            if r["state"] == "LAUNCH-FAILED" and r["retries"] < 1:
                r["retries"] += 1
                launch_train(st, r["task"], r["budget"], r["arm"])
            continue
        s, rc = status(r["ex"])
        if s == "?":
            continue
        if s == "R" and r["state"] != "R":
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
                    r["s2"] = st["runs"][run_name(r["task"], "ffnmoe-s2", r["budget"])]["ex"]
            elif r["retries"] < 1:
                r["retries"] += 1
                log(f"{name} failed rc={rc}; relaunching")
                launch_train(st, r["task"], r["budget"], r["arm"])
                st["runs"][name]["retries"] = r["retries"]
            else:
                r["state"] = "FAILED"
                log(f"{name} FAILED twice rc={rc}")
        save(st)
    # D. evals
    for name, e in list(st["evals"].items()):
        if e["state"] in ("DONE", "FAILED") or not e.get("ex"):
            if e["state"] == "LAUNCH-FAILED" and e["retries"] < 1:
                e["retries"] += 1
                launch_eval(st, name)
            continue
        s, rc = status(e["ex"])
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
        save(st)
    # E. harvest / completion
    pending_runs = [n for n, r in st["runs"].items() if r["state"] not in ("DONE", "FAILED")]
    pending_evals = [n for n, e in st["evals"].items() if e["state"] not in ("DONE", "FAILED")]
    missing_evals = [n for n, r in st["runs"].items() if r["state"] == "DONE" and not r.get("eval")]
    kv_all = all(st["kv_launched"].get(t) for t in ARMS)
    log(f"status: runs {sum(r['state']=='DONE' for r in st['runs'].values())}/{len(st['runs'])} done, "
        f"{len(pending_runs)} pending; evals {sum(e['state']=='DONE' for e in st['evals'].values())}/{len(st['evals'])} done, "
        f"{len(pending_evals)} pending; kv launched {sorted(k for k,v in st['kv_launched'].items() if v)}")
    if st.get("harvest_ex"):
        s, rc = status(st["harvest_ex"])
        if s == "F":
            st["harvest_ex"] = None
            collect(st)
            save(st)
            if st.get("finishing"):
                st["done"] = True
                save(st)
                return
    finished = kv_all and not pending_runs and not pending_evals and not missing_evals
    if (time.time() - st["last_harvest"] > HARVEST_EVERY or finished) and not st.get("harvest_ex"):
        if finished:
            st["finishing"] = True
        harvest(st)


def main():
    st = load()
    log("orchestrator35 up; runs tracked: %d" % len(st["runs"]))
    while True:
        try:
            cycle(st)
        except Exception as e:
            log(f"cycle error: {e!r}")
        if st.get("done"):
            log("ALL_DONE")
            return
        time.sleep(CYCLE)


if __name__ == "__main__":
    main()
