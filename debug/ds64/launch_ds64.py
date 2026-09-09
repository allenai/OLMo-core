"""
Launch the short-heavy 2k-64k data-scaling grid (records/ds64-scaling-plan.md): dense vs the best
soft-token construction per task, Qwen3.5-4B (SCALE=4b, default) or 27B (SCALE=27b), on the
marker shards debug/ds64/build_ds64_data_beaker.sh wrote to weka (ds64/shards/<task>_u<B>).

Arms (all soft arms: detached slots -- attention K/V AND GDN writes -- no bias, GDN intact forward):
  dense            packed seq 65536, 8 rows/step, flash_2
  hdr03/08/17/33   contradiction: `Claim N:` headers real, gold + 1/36 .. 1/3 random docs real (keep ablation)
  runs03/08        contradiction: gold + random picks kept as +-1 runs (no header)
  ohdr33/17/08     oolong: Date/User/Instance headers real, 1/3 .. 1/12 of lines real (gold-blind)
  kv08/17/33       nq / outlier: gold + 1/12 .. 1/3 random docs real
Phase 1 = contradiction's full set + every task's dense; other soft arms via debug/ds64/soft_arms.json.
Soft arms train UNPACKED at seq 65536, global batch 16 x micro 2 (~ the dense 524k tokens/step at
a ~40k mean length), torch attention backend unless DS64_SOFT_BACKEND says otherwise.

    python debug/ds64/launch_ds64.py --tasks contradiction --budgets 32M --arms dense,hdr36 dry_run
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
LAUNCHER = f"{REPO}/src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py"
WEKA = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
SCALE = os.environ.get("DS64_SCALE", "4b")
BASE = {"4b": f"{WEKA}/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim",
        "27b": f"{WEKA}/ctc_suite/bases/q35-27b-base-markerfix/model_and_optim"}[SCALE]
SHARDS = f"{WEKA}/ds64/shards"
LEDGER = f"{REPO}/debug/ds64/LAUNCH_LEDGER.tsv"
BUDGETS = ["16M", "32M", "64M", "128M"]
# contradiction's pair pool cannot build the 128M arm (compose skipped it): its grid stops at 64M
TASK_BUDGETS = {"contradiction": ["16M", "32M", "64M"]}


def budgets_for(task):
    return TASK_BUDGETS.get(task, BUDGETS)
TASKS = ["contradiction", "oolong", "nq", "outlier"]
SOFT_BACKEND = os.environ.get("DS64_SOFT_BACKEND", "torch")
# per scale: nodes, GPUs/node, soft micro-batch, cluster
NODES = {"4b": 1, "27b": int(os.environ.get("DS64_NUM_NODES", "2"))}[SCALE]
GPUS = 8
SOFT_MICRO = {"4b": 2, "27b": 1}[SCALE]
CLUSTER = os.environ.get("DS64_CLUSTER", {"4b": "ai2/jupiter-cirrascale-2", "27b": "ai2/titan-cirrascale"}[SCALE])

# keep-ratio suffix as in the old grid: 03 = 1/36, 08 = 1/12, 17 = 1/6, 33 = 1/3
_HDR1 = "--st-keep-mode gold_plus_random --st-header-stop-id 25 --st-header-stop-count 1"
ARM_EXTRA = {
    # contradiction: `Claim N:` headers real + gold + a fraction of random docs (keep-ratio ablation first, Prasann 2026-09-08)
    "hdr03": f"--st-keep-frac 0.0278 {_HDR1}", "hdr08": f"--st-keep-frac 0.0833 {_HDR1}",
    "hdr17": f"--st-keep-frac 0.1667 {_HDR1}", "hdr33": f"--st-keep-frac 0.3333 {_HDR1}",
    # contradiction: leak-free neighbour runs, no header
    "runs03": "--st-keep-frac 0.0278 --st-keep-mode gold_plus_random --st-neighbour-runs 1",
    "runs08": "--st-keep-frac 0.0833 --st-keep-mode gold_plus_random --st-neighbour-runs 1",
    # oolong: Date/User/Instance headers real, a fraction of lines real (gold-blind)
    "ohdr33": "--st-gold-blind --st-keep-prob 0.3333 --st-header-stop-id 25 --st-header-stop-count 3",
    "ohdr17": "--st-gold-blind --st-keep-prob 0.1667 --st-header-stop-id 25 --st-header-stop-count 3",
    "ohdr08": "--st-gold-blind --st-keep-prob 0.0833 --st-header-stop-id 25 --st-header-stop-count 3",
    # nq / outlier: gold + a fraction of random docs
    "kv08": "--st-keep-frac 0.0833 --st-keep-mode gold_plus_random",
    "kv17": "--st-keep-frac 0.1667 --st-keep-mode gold_plus_random",
    "kv33": "--st-keep-frac 0.3333 --st-keep-mode gold_plus_random",
}
# Phase 1 (contradiction first): dense + the keep ablation. Other tasks: dense only until
# debug/ds64/soft_arms.json (read by the orchestrator every cycle) names their soft arms.
TASK_ARMS = {"contradiction": ["dense", "hdr03", "hdr08", "hdr17", "hdr33", "runs03", "runs08"],
             "oolong": ["dense"], "nq": ["dense"], "outlier": ["dense"]}


def run_name(task, arm, budget):
    return f"ds64{'' if SCALE == '4b' else '-' + SCALE}-{task}-{arm}-u{budget}"


def arm_args(task, arm, budget):
    data = f"{SHARDS}/{task}_u{budget}"
    if arm == "dense":
        return "full", data, ["--pack", "--seq-len", "65536", "--global-batch", "8", "--micro-batch-instances", "1", "--base-checkpoint", BASE], ""
    if arm in ARM_EXTRA:
        gb = 8 * NODES * SOFT_MICRO
        return "softtoken", data, ["--seq-len", "65536", "--global-batch", str(gb), "--micro-batch-instances", str(SOFT_MICRO), "--base-checkpoint", BASE], \
            f"{ARM_EXTRA[arm]} --attn-backend {SOFT_BACKEND}"
    raise SystemExit(f"unknown arm {arm}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--budgets", default=",".join(BUDGETS))
    ap.add_argument("--arms", default="", help="comma list; default = the task's arms")
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--wandb-group", default=f"ds64-q35-{SCALE}")
    ap.add_argument("--skip", default="")
    ap.add_argument("mode", choices=["launch", "dry_run"])
    args = ap.parse_args()
    skip = set(x for x in args.skip.split(",") if x)
    rows = []
    for task in args.tasks.split(","):
        for budget in (args.budgets.split(",") if args.budgets != ",".join(BUDGETS) else budgets_for(task)):
            for arm in (args.arms.split(",") if args.arms else TASK_ARMS[task]):
                variant, data, largs, extra = arm_args(task, arm, budget)
                name = run_name(task, arm, budget)
                if name in skip:
                    print(f"[skip] {name}"); continue
                cmd = [sys.executable, "-u", LAUNCHER, "--task", task, "--variant", variant,
                       "--model-family", "qwen3_5", "--model-scale", SCALE, "--data-root", data,
                       "--run-name", name, "--exact-run-name", "--num-nodes", str(NODES), "--num-gpus", str(GPUS),
                       "--epochs", "1", "--lr", str(args.lr), "--cluster", CLUSTER, "--wandb-group", args.wandb_group,
                       "--no-follow", "--no-compile"] + largs + (["--extra-args", extra] if extra else []) + [args.mode]
                print(" ".join(cmd), flush=True)
                res = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, PYTHONPATH=f"{REPO}/src"), capture_output=True, text=True)
                out = res.stdout + res.stderr
                os.makedirs(f"{REPO}/debug/ds64/launch_logs", exist_ok=True)
                open(f"{REPO}/debug/ds64/launch_logs/{name}.{args.mode}.log", "w").write(out)
                ids = [l.split("id=")[1].split()[0] for l in out.splitlines() if "SUBMITTED id=" in l]
                tail = (ids[-1] + " " if ids else "") + "\n".join(out.strip().splitlines()[-2:]).replace("\t", " ")[:200]
                print(f"  -> rc={res.returncode}: {tail[:300]}", flush=True)
                rows.append((name, task, arm, budget, res.returncode, tail))
    if args.mode == "launch":
        with open(LEDGER, "a") as f:
            w = csv.writer(f, delimiter="\t")
            for name, task, arm, budget, rc, tail in rows:
                w.writerow(["ds64", SCALE, task, arm, budget, name, CLUSTER, dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "LAUNCHED" if rc == 0 else "LAUNCH-FAILED", tail])


if __name__ == "__main__":
    main()
