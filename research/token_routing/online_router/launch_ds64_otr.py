"""Online-router arms in the ds64 LONG-CONTEXT setting (short-heavy 2k-56k length mix, per-task pools), against the
ds64 dense curves (16M / 32M / 64M) and the ds64 5-rung eval (2k / 8k / 16k / 32k / 64k).

Same geometry as ds64's soft-token arms (``research/token_routing/ds64/launch_ds64.py``, imported read-only):
UNPACKED rows at seq 65536 (the soft-token path strips padding and length-sorts micro-batches), 128 rows/step,
flash_2, LR 5e-6, the repaired 4B base, shard / budget / --max-tokens from ``shard_for``. Differences:
  * micro-batch 1 (not 2): mid-stack runs blocks [0, L) on the FULL row and every 8th step is fully dense, so two
    56k rows in one micro-batch is the OOM the ds64 mixing arms already hit;
  * DS64_NGPU defaults to 4 here (ds64: "rows/step is unchanged, so accuracy and FLOPs are unaffected -- only the
    gradient-accumulation depth and wall-clock change"), because 4-GPU holes schedule on jupiter.
Evals: the ds64 orchestrator's 5-rung files and command (``orchestrate_ds64.RUNG_FILES``).

    python research/token_routing/online_router/launch_ds64_otr.py --tasks contradiction,nq --arms otr10M8,otr30M8 --budgets 64M launch
    python research/token_routing/online_router/launch_ds64_otr.py --runs ds64-contradiction-otr10M8-b128f-u64M eval
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

os.environ.setdefault("DS64_NGPU", "4")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "research", "token_routing", "ds64"))
sys.path.insert(0, HERE)
import launch_ds64 as L  # noqa: E402
from launch_otr import ARMS  # noqa: E402
from orchestrate_ds64 import EVAL_NGPU, RUNG_FILES, TASK_KEY, TOKENIZER  # noqa: E402

STATE = os.path.join(HERE, "state_ds64.json")
LAUNCHER = f"{REPO}/recipes/train/research_trainer/beaker_ctc_suite.py"
EVAL = f"{REPO}/recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py"
W = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
MICRO = 1


def sh(cmd, timeout=1500):
    r = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, PYTHONPATH=f"{REPO}/src"), capture_output=True, text=True,
                       timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def parse_id(out):
    ids = [l.split("id=")[1].split()[0] for l in out.splitlines() if "SUBMITTED id=" in l]
    return ids[-1] if ids else None


def load():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"runs": {}, "evals": {}}


def save(st):
    json.dump(st, open(STATE + ".tmp", "w"), indent=1)
    os.replace(STATE + ".tmp", STATE)


def do_launch(a, mode):
    st = load()
    for task in a.tasks.split(","):
        for arm in a.arms.split(","):
            for budget in a.budgets.split(","):
                name = L.run_name(task, arm, budget)
                if name in st["runs"] and mode == "launch" and not a.force:
                    print(f"[skip] {name}")
                    continue
                shard, cap = L.shard_for(budget)
                # 512 target samples per row (not 2048): 128 rows/step still give ~65k samples, and the pending
                # feature rows (kept until the step's refit) stay ~0.35 GB per rank at 32 micro-batches
                extra = ARMS[arm] + " --st-otr-samples-per-row 512" + (f" --max-tokens {cap}" if cap else "")
                cmd = [sys.executable, "-u", LAUNCHER, "--task", task, "--variant", "softtoken", "--model-family",
                       "qwen3_5", "--model-scale", "4b", "--data-root", f"{L.SHARDS}/{task}_u{shard}", "--run-name", name,
                       "--exact-run-name", "--num-nodes", "1", "--num-gpus", os.environ["DS64_NGPU"], "--epochs", "1",
                       "--lr", str(a.lr), "--cluster", L.CLUSTER, "--wandb-group", "ds64-otr", "--priority", "urgent",
                       "--no-follow", "--no-compile", "--experiment", "learned_router", "--seq-len", "65536",
                       "--global-batch", str(L.SOFT_GB), "--micro-batch-instances", str(MICRO),
                       "--base-checkpoint", L.BASE, "--extra-args", extra, mode]
                rc, out = sh(cmd)
                ex = parse_id(out)
                for ln in out.splitlines():
                    if ln.startswith("[beaker_ctc_suite] cmd=") or ln.startswith("[beaker_ctc_suite] data="):
                        print("  " + ln[:900])
                print(f"{name} -> rc={rc} {ex or out[-200:]}")
                if mode == "launch" and ex:
                    st["runs"][name] = {"ex": ex, "task": task, "arm": arm, "budget": budget, "lr": a.lr}
                    save(st)


def do_eval(a):
    st = load()
    for name in a.runs.split(","):
        r = st["runs"][name]
        if name in st["evals"] and not a.force:
            print(f"[skip] eval {name}")
            continue
        files = RUNG_FILES[r["task"]]
        cmd = [sys.executable, "-u", EVAL, name, L.CLUSTER, "--task", TASK_KEY[r["task"]], "--variant", "docchunk",
               "--ckpt", f"{W}/ctc_suite/ckpts/{name}", "--query-position", "after", "--cot-mode", "none",
               "--tokenizer", TOKENIZER, "--ngpu", EVAL_NGPU, "--max-test", "500", "--max-length", "70000",
               "--priority", "urgent", "--dc-rung-files", json.dumps(files),
               "--dc-rungs", ",".join(next(iter(files.values())).keys())]
        rc, out = sh(cmd)
        ex = parse_id(out)
        print(f"eval {name} -> {ex or 'FAILED: ' + out[-300:]}")
        if ex:
            st["evals"][name] = {"ex": ex, "train_run": name, "task": r["task"]}
            save(st)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="contradiction,nq")
    ap.add_argument("--arms", default="otr10M8,otr30M8")
    ap.add_argument("--budgets", default="64M")
    ap.add_argument("--lr", type=float, default=5e-6, help="ds64's LR for every arm")
    ap.add_argument("--runs", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("mode", choices=["launch", "dry_run", "eval"])
    a = ap.parse_args()
    do_eval(a) if a.mode == "eval" else do_launch(a, a.mode)


if __name__ == "__main__":
    main()
