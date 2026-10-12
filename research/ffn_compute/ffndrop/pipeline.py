"""
Login-node pipeline for the drop-CPT study (records/ffndrop-cpt-plan.md): wait for the drop-CPT run,
then (a) its drop-robustness dev-loss eval and (b) routed-FFN SFT from its export on the chosen fs35
settings, then (c) score each SFT run with the same native ladder evaluator the base-model fs35 points
used. Idempotent state file; restart it and it resumes.

    setsid nohup python research/ffn_compute/ffndrop/pipeline.py \
        >> $FFNDROP_LOG_DIR/pipeline.log 2>&1 < /dev/null &
"""
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

from pathlib import Path

REPO = str(Path(__file__).resolve().parents[3])
HERE = os.path.dirname(os.path.abspath(__file__))
W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
STATE = os.environ.get("FFNDROP_STATE", f"{HERE}/pipeline_state.json")  # untracked; resumable
CPT_RUN, CPT_EX = "fdcpt-q35-4b-drop75l12-u1B", "01M3RG4EZ5KG4D17S6NM86SS4D"
BASE_TAG = "-bfdrop"
# (task, budget, arm): the settings where the base-model routed FFN sat clearly off the dense frontier
SFT = [("contradiction", "28M", "ffnmoe-t10"), ("nq", "32M", "ffnmoe-t10")]
EVAL_CLUSTER = "ai2/jupiter-cirrascale-2"
PY = sys.executable
CONDA_BIN = os.environ.get("FFNDROP_CONDA_BIN", "/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin")
ENV = dict(os.environ, PYTHONPATH=f"{REPO}/src:{REPO}/research/common", AWS_PROFILE="S3",
           PATH=f"{CONDA_BIN}:{os.path.expanduser('~/.local/bin')}:"
           + os.environ.get("PATH", ""))
sys.path.insert(0, f"{REPO}/research/ffn_compute/flop_scaling")
os.environ["FS35_BASE_TAG"] = BASE_TAG  # run_name() reads it at import
from orchestrate35 import EVAL_CFG  # noqa: E402
from launch_grid35 import run_name  # noqa: E402


def log(m):
    print(f"{datetime.now().strftime('%m-%d %H:%M:%S')} {m}", flush=True)


def sh(cmd, env=ENV, timeout=1200):
    try:
        r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def status(ex):
    """('S'|'R'|'F', exitCode) or ('?', None)."""
    rc, out = sh(["beaker", "experiment", "get", ex, "--format", "json"], timeout=120)
    try:
        j = json.loads(out)
        s = (j[0] if isinstance(j, list) else j)["jobs"][-1]["status"]
        if s.get("canceled"):
            return "F", "canceled"
        return ("F", s.get("exitCode")) if s.get("finalized") else (("R" if s.get("started") else "S"), None)
    except Exception:
        return "?", None


def parse_id(out):
    m = (re.search(r"SUBMITTED id=(\S+)", out) or re.search(r"beaker\.org/ex/([A-Z0-9]{26})", out)
         or re.search(r"rc=0: ([A-Z0-9]{26})", out) or re.search(r"ex/([A-Z0-9]{26})", out))
    return m.group(1) if m else None


def save(st):
    json.dump(st, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def main():
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"cpt": None, "devloss": None, "sft": {}, "evals": {}}
    log(f"pipeline start; state {json.dumps(st)}")
    while True:
        if st["cpt"] is None:
            s, code = status(CPT_EX)
            if s == "F":
                st["cpt"] = code
                save(st)
                log(f"CPT {CPT_RUN} finalized exit={code}")
                if code != 0:
                    log("!!! CPT did not exit 0 -- nothing downstream launched; check the job log and relaunch")
                    return
            else:
                time.sleep(600)
                continue
        base = f"{W}/ctc_suite/ckpts/{CPT_RUN}/model_and_optim"
        if st["devloss"] is None:
            rc, out = sh(["bash", "research/ffn_compute/ffndrop/eval_drop_devloss_beaker.sh"],
                         env=dict(ENV, CKPT=base, NAME=CPT_RUN))
            st["devloss"] = parse_id(out) or "LAUNCH-FAILED"
            save(st)
            log(f"devloss eval -> {st['devloss']}" + ("" if rc == 0 else f" rc={rc} {out[-300:]}"))
        for task, budget, arm in SFT:
            name = run_name(task, arm, budget)
            if name not in st["sft"]:
                rc, out = sh([PY, f"{REPO}/research/ffn_compute/flop_scaling/launch_grid35.py", "--tasks", task, "--budgets", budget,
                              "--arms", arm, "--wandb-group", "fdcpt-q35-4b-sft", "launch"],
                             env=dict(ENV, FS35_BASE=base, FS35_BASE_TAG=BASE_TAG))
                ex = parse_id(out)
                st["sft"][name] = {"ex": ex, "task": task, "state": "S" if ex else "LAUNCH-FAILED", "code": None}
                save(st)
                log(f"SFT {name} -> {ex or 'FAILED: ' + out[-400:]}")
        pending = 0
        for name, r in st["sft"].items():
            if r["state"] in ("S", "R"):
                s, code = status(r["ex"])
                if s == "F":
                    r["state"], r["code"] = "F", code
                    save(st)
                    log(f"SFT {name} finalized exit={code}")
                else:
                    r["state"] = s if s in ("S", "R") else r["state"]
                    pending += 1
                    continue
            if r["state"] == "F" and r["code"] == 0 and name not in st["evals"]:
                rungs, root, extra = EVAL_CFG[r["task"]]
                rc, out = sh([PY, "-u", f"{REPO}/research/common/beaker_native_lengthmix_eval.py", name, name,
                              "--ladder-tasks", r["task"], "--ladder-rungs", rungs, "--cluster", EVAL_CLUSTER,
                              "--eval500-root", root] + (extra.split() if extra else []))
                st["evals"][name] = {"ex": parse_id(out) or "LAUNCH-FAILED", "code": None}
                save(st)
                log(f"eval {name} -> {st['evals'][name]['ex']}" + ("" if rc == 0 else f" rc={rc} {out[-300:]}"))
        for name, e in st["evals"].items():
            if e["code"] is None and e["ex"] != "LAUNCH-FAILED":
                s, code = status(e["ex"])
                if s == "F":
                    e["code"] = code
                    save(st)
                    log(f"eval {name} finalized exit={code}")
                else:
                    pending += 1
        if pending == 0 and all(r["state"] == "F" or r["state"] == "LAUNCH-FAILED" for r in st["sft"].values()):
            log("PIPELINE DONE")
            return
        time.sleep(600)


if __name__ == "__main__":
    main()
