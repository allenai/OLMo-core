"""
Unattended orchestrator for the short-heavy 2k-64k data-scaling campaign (records/ds64-scaling-plan.md).
Runs detached on the LOGIN node; state in orchestrator_ds64[_27b]_state.json; restartable.

Each 5-min cycle:
  A. per task: when its Beaker data build (debug/ds64/data_build_jobs.tsv) has finalized rc=0,
     launch every (arm, budget) of that task (launch_ds64.py);
  B. poll training runs: finalized ok -> eval on the 16k/32k/64k rungs with the marker-aware
     docchunk evaluator (2k/8k/16k/32k/64k); failed -> one relaunch;
  C. poll evals; failed -> up to 3 relaunches;
  D. harvest flops.json + walltime every 90 min and at the end; ALL_DONE.

    setsid nohup python debug/ds64/orchestrate_ds64.py >> debug/ds64/orchestrator_ds64.log 2>&1 &
    DS64_SCALE=27b setsid nohup python debug/ds64/orchestrate_ds64.py >> debug/ds64/orchestrator_ds64_27b.log 2>&1 &
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
D = f"{REPO}/debug/ds64"
SCALE = os.environ.get("DS64_SCALE", "4b")
STATE = f"{D}/orchestrator_ds64{'' if SCALE == '4b' else '_' + SCALE}_state.json"
PY = sys.executable
ENV = dict(os.environ, PYTHONPATH=f"{REPO}/src",
           PATH="/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:" + os.environ.get("PATH", ""),
           AWS_PROFILE="S3", DS64_SCALE=SCALE)
sys.path.insert(0, D)
from launch_ds64 import BUDGETS, TASK_ARMS, TASKS, run_name  # noqa: E402

CYCLE = 300
HARVEST_EVERY = 90 * 60
W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
EVAL_CLUSTER = os.environ.get("DS64_EVAL_CLUSTER", {"4b": "ai2/jupiter-cirrascale-2", "27b": "ai2/titan-cirrascale"}[SCALE])
EVAL_NGPU = {"4b": "2", "27b": "4"}[SCALE]
TOKENIZER = f"{W}/hf_tokenizers/Qwen3.5-0.8B-Base"
RUNG_FILES = {
    "contradiction": {"contradiction": {"2k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n100_k3.jsonl",
                                        "8k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n190_k3.jsonl",
                                        "16k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n385_k3.jsonl",
                                        "32k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n765_k3.jsonl",
                                        "64k": f"{W}/_eval_bundle_eval500_v3/contra/contradiction_eval_pubmed_realistic_n1525_k3_xlong_64k.jsonl"}},
    "nq": {"nq": {"2k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_2048.jsonl",
                  "8k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_8192.jsonl",
                  "16k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_16384.jsonl",
                  "32k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_32768.jsonl",
                  "64k": f"{W}/outlier_lengthmix/eval_rungs/nq/rung_65536.jsonl"}},
    "outlier": {"outlier": {"2k": f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_2048.jsonl",
                            "8k": f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_8192.jsonl",
                            "16k": f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_16384.jsonl",
                            "32k": f"{W}/outlier_lengthmix/eval_rungs/outlier/rung_32768.jsonl",
                            "64k": f"{W}/_eval_bundle_eval500_v2/outlier/outlier_wiki100w_n448_k3_eval_xlong_64k.jsonl"}},
    "oolong": {"oolong": {"2k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx2048_spliteval.jsonl",
                          "8k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx8192_spliteval.jsonl",
                          "16k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx16384_spliteval.jsonl",
                          "32k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx32768_spliteval.jsonl",
                          "64k": f"{W}/_eval_bundle_eval500_v2_clean/oolong/oolong_test_synth_ctx65536_spliteval.jsonl"}},
}
TASK_KEY = {"contradiction": "contra", "nq": "nq", "outlier": "outlier", "oolong": "oolong"}


def log(m):
    print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} {m}", flush=True)


def load():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"runs": {}, "evals": {}, "data_ok": {}, "launched": {}, "last_harvest": 0, "harvest_ex": None, "done": False}


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


def data_jobs():
    out = {}
    for line in open(f"{D}/data_build_jobs.tsv"):
        if line.strip():
            task, ex, *_ = line.rstrip("\n").split("\t")
            out[task] = ex  # the LAST entry for a task wins (relaunches append)
    return out


def launch_train(st, task, budget, arm):
    name = run_name(task, arm, budget)
    rc, out = sh([PY, f"{D}/launch_ds64.py", "--tasks", task, "--budgets", budget, "--arms", arm, "launch"], timeout=1200)
    ex = parse_id(out)
    prev = st["runs"].get(name, {})
    st["runs"][name] = {"ex": ex, "task": task, "budget": budget, "arm": arm, "state": "S" if ex else "LAUNCH-FAILED",
                        "rc": None, "retries": prev.get("retries", 0), "eval": prev.get("eval")}
    log(f"launch {name} -> {ex or 'FAILED: ' + out[-300:]}")
    save(st)


def launch_eval(st, name):
    r = st["runs"][name]
    files = RUNG_FILES[r["task"]]
    cmd = [PY, "-u", f"{REPO}/src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py", name, EVAL_CLUSTER,
           "--task", TASK_KEY[r["task"]], "--variant", "docchunk", "--ckpt", f"{W}/ctc_suite/ckpts/{name}",
           "--query-position", "after", "--cot-mode", "none", "--tokenizer", TOKENIZER,
           "--ngpu", EVAL_NGPU, "--max-test", "500", "--max-length", "70000", "--priority", "urgent",
           "--dc-rung-files", json.dumps(files), "--dc-rungs", ",".join(next(iter(files.values())).keys())]
    rc, out = sh(cmd, timeout=1200)
    ex = parse_id(out)
    st["evals"][name] = {"ex": ex, "state": "S" if ex else "LAUNCH-FAILED", "rc": None, "retries": st["evals"].get(name, {}).get("retries", 0)}
    r["eval"] = ex
    log(f"eval {name} -> {ex or 'FAILED: ' + out[-300:]}")
    save(st)


def harvest(st):
    rc, out = sh(f"bash {D}/harvest_ds64.sh", timeout=600)
    st["harvest_ex"] = parse_id(out)
    st["last_harvest"] = time.time()
    log(f"harvest job -> {st['harvest_ex']}")
    save(st)


def collect(st):
    sh("aws s3 sync s3://ai2-llm/checkpoints/prasanns/ds64/harvest results/ds64/harvest --only-show-errors", timeout=600)
    rc2, out2 = sh([PY, f"{D}/collect_ds64.py"], timeout=900)
    log("collect: " + "\n".join(out2.strip().splitlines()[-12:]))


def cycle(st):
    # A. launch each task's grid once its data build is done
    for task, ex in data_jobs().items():
        if st["launched"].get(task):
            continue
        if not st["data_ok"].get(task):
            s, rc = status(ex)
            if s == "F" and rc == 0:
                st["data_ok"][task] = True
                log(f"data for {task} built ({ex})")
                save(st)
            elif s == "F":
                log(f"data build for {task} FAILED rc={rc} ({ex}) -- relaunch by hand: TASK={task} bash debug/ds64/build_ds64_data_beaker.sh")
                continue
            else:
                continue
        for b in BUDGETS:
            for arm in TASK_ARMS[task]:
                if run_name(task, arm, b) not in st["runs"]:
                    launch_train(st, task, b, arm)
        st["launched"][task] = True
        save(st)
    # B. training runs
    for name, r in list(st["runs"].items()):
        if r["state"] == "DONE" and not r.get("eval"):
            launch_eval(st, name)
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
            elif r["retries"] < 1:
                r["retries"] += 1
                log(f"{name} failed rc={rc}; relaunching")
                launch_train(st, r["task"], r["budget"], r["arm"])
                st["runs"][name]["retries"] = r["retries"]
            else:
                r["state"] = "FAILED"
                log(f"{name} FAILED twice rc={rc}")
        save(st)
    # C. evals
    for name, e in list(st["evals"].items()):
        if e["state"] in ("DONE", "FAILED") or not e.get("ex"):
            if e["state"] == "LAUNCH-FAILED" and e["retries"] < 3:
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
            elif e["retries"] < 3:
                e["retries"] += 1
                log(f"eval {name} failed rc={rc}; relaunching")
                launch_eval(st, name)
                st["evals"][name]["retries"] = e["retries"]
            else:
                e["state"] = "FAILED"
                log(f"eval {name} FAILED 4x")
        save(st)
    # D. harvest / completion
    pending_runs = [n for n, r in st["runs"].items() if r["state"] not in ("DONE", "FAILED")]
    pending_evals = [n for n, e in st["evals"].items() if e["state"] not in ("DONE", "FAILED")]
    missing_evals = [n for n, r in st["runs"].items() if r["state"] == "DONE" and not r.get("eval")]
    all_launched = all(st["launched"].get(t) for t in TASKS)
    log(f"status: data_ok {sorted(st['data_ok'])} | runs {sum(r['state']=='DONE' for r in st['runs'].values())}/{len(st['runs'])} done, "
        f"{len(pending_runs)} pending | evals {sum(e['state']=='DONE' for e in st['evals'].values())}/{len(st['evals'])} done, {len(pending_evals)} pending")
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
    finished = all_launched and st["runs"] and not pending_runs and not pending_evals and not missing_evals
    if (time.time() - st["last_harvest"] > HARVEST_EVERY or finished) and not st.get("harvest_ex") and st["runs"]:
        if finished:
            st["finishing"] = True
        harvest(st)


def main():
    st = load()
    log(f"orchestrator start (scale {SCALE}, state {STATE})")
    while not st.get("done"):
        try:
            cycle(st)
        except Exception as e:  # noqa: BLE001
            log(f"cycle error: {e!r}")
        time.sleep(CYCLE)
    log("ALL_DONE")


if __name__ == "__main__":
    main()
