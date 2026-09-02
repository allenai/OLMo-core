"""
Launch (or dry-run) the FLOP-scaling grid on Beaker (records/flop-scaling-ffn-kv-plan.md §4).

One training job per (task, budget, arm[, stage]) through ``beaker_ctc_suite.py``. Run names are
deterministic (``fs-<task>-<arm>-sh<B>[-s1]``) so stage 2 of the ffnmoe arm can warm-start from
stage 1's export at a known weka path; reusing a name RESUMES (fresh experiment => new name).

Batching (plan §6 / the smoke findings): dense + ffnmoe train PACKED (seq 65536, 8 packed rows per
step ~ 524k tokens); the soft-token arms need one example per row (per-row fingerprints) and run the
padded path at seq 40960 with a 160-example global batch, i.e. the same ~524k tokens per optimizer
step on average -- their compaction drops the padding before any compute.

    python debug/flop_scaling/launch_grid.py --tasks outlier --budgets 16M --arms dense,kv17 dry_run
    python debug/flop_scaling/launch_grid.py --tasks outlier,nq,oolong,contradiction \\
        --budgets 8M,16M,32M,64M,128M --arms dense,kv17,kv33,ffnmoe-s1 launch
    # after the s1 runs exported:
    python debug/flop_scaling/launch_grid.py ... --arms ffnmoe-s2 launch
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
WEKA_ROOT = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
SHARDS = f"{WEKA_ROOT}/flop_scaling/shards"
BASE = f"{WEKA_ROOT}/flop_scaling/bases/q4b-dense-cpt-fixmark/model_and_optim"
CKPTS = f"{WEKA_ROOT}/ctc_suite/ckpts"
LEDGER = f"{REPO}/debug/flop_scaling/LAUNCH_LEDGER.tsv"

# contradiction's pool caps its budgets (plan §3)
TASK_BUDGETS = {"contradiction": ["8M", "16M", "32M", "48M"]}
DEFAULT_BUDGETS = ["8M", "16M", "32M", "64M", "128M"]

FFN_LADDER = "1,16,64,256,1024,9728"
KV_FRAC = {"kv17": 1 / 6, "kv33": 1 / 3}


def run_name(task: str, arm: str, budget: str) -> str:
    return f"fs-{task}-{arm}-sh{budget}"


def arm_args(task: str, arm: str, budget: str) -> tuple[str, list[str], str]:
    """-> (variant, launcher args, extra train args)"""
    packed = ["--pack", "--seq-len", "65536", "--global-batch", "8", "--micro-batch-instances", "1"]
    if arm == "dense":
        return "full", packed, ""
    if arm == "ffnmoe-s1":
        extra = (
            f"--ffn-moe-start-layer 12 --ffn-moe-divisors {FFN_LADDER} --ffn-moe-width-multiple 1 "
            "--ffn-moe-target 0.01 --ffn-moe-target-anneal-frac 0.3 --ffn-moe-explore-anneal-frac 0.3"
        )
        return "ffnmoe", packed, extra
    if arm == "ffnmoe-s2":
        s1 = f"{CKPTS}/{run_name(task, 'ffnmoe-s1', budget)}/model_and_optim"
        extra = (
            f"--ffn-moe-start-layer 0 --ffn-moe-divisors {FFN_LADDER} --ffn-moe-width-multiple 1 "
            "--ffn-moe-target 0.02 --ffn-moe-target-anneal-frac 0.0 --ffn-moe-explore-anneal-frac 0.3"
        )
        return "ffnmoe", packed + ["--base-checkpoint", s1], extra
    if arm in KV_FRAC:
        frac = KV_FRAC[arm]
        padded = ["--seq-len", "40960", "--global-batch", "160", "--micro-batch-instances", "2"]
        if task == "oolong":  # no gold set: same fraction, gold-blind
            extra = f"--st-gold-blind --st-keep-prob {frac:.4f} --attn-backend flash_2"
        else:
            extra = f"--st-keep-frac {frac:.4f} --st-keep-mode gold_plus_random --attn-backend flash_2"
        return "softtoken", padded, extra
    raise SystemExit(f"unknown arm {arm}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="outlier,nq,oolong,contradiction")
    ap.add_argument("--budgets", default="")
    ap.add_argument("--arms", default="dense,kv17,kv33,ffnmoe-s1")
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6, help="outlier length-mix campaign's dense optimum")
    ap.add_argument("--wandb-group", default="flop-scaling")
    ap.add_argument("mode", choices=["launch", "dry_run"])
    args = ap.parse_args()

    tasks = args.tasks.split(",")
    arms = args.arms.split(",")
    rows = []
    for task in tasks:
        budgets = args.budgets.split(",") if args.budgets else TASK_BUDGETS.get(task, DEFAULT_BUDGETS)
        for budget in budgets:
            for arm in arms:
                variant, largs, extra = arm_args(task, arm, budget)
                name = run_name(task, arm, budget)
                cmd = [
                    sys.executable, "-u", LAUNCHER,
                    "--task", task, "--variant", variant, "--model-family", "qwen3", "--model-scale", "4b",
                    "--data-root", f"{SHARDS}/{task}_sh{budget}",
                    "--run-name", name, "--exact-run-name",
                    "--num-nodes", "1", "--num-gpus", str(args.num_gpus), "--epochs", "1",
                    "--lr", str(args.lr), "--cluster", args.cluster, "--wandb-group", args.wandb_group,
                    "--no-follow", "--no-compile",
                ] + largs
                if "--base-checkpoint" not in largs:
                    cmd += ["--base-checkpoint", BASE]
                if extra:
                    cmd += ["--extra-args", extra]
                cmd += [args.mode]
                print(" ".join(cmd), flush=True)
                env = dict(os.environ, PYTHONPATH=f"{REPO}/src")
                res = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
                out = res.stdout + res.stderr
                os.makedirs(f"{REPO}/debug/flop_scaling/launch_logs", exist_ok=True)
                with open(f"{REPO}/debug/flop_scaling/launch_logs/{name}.{args.mode}.log", "w") as lf:
                    lf.write(out)
                urls = [l.strip() for l in out.splitlines() if "beaker.org/ex/" in l]
                tail = urls[-1] if urls else "\n".join(out.strip().splitlines()[-3:])
                print(f"  -> rc={res.returncode}: {tail[:300]}", flush=True)
                rows.append((name, task, arm, budget, res.returncode, tail.replace("\t", " ")[:200]))
    if args.mode == "launch":
        with open(LEDGER, "a") as f:
            w = csv.writer(f, delimiter="\t")
            for name, task, arm, budget, rc, tail in rows:
                w.writerow(["beaker", task, arm, budget, arm.split("-")[-1] if "-" in arm else "-", name,
                            args.cluster, dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "LAUNCHED" if rc == 0 else "LAUNCH-FAILED", tail])
        print(f"ledger updated: {LEDGER}")


if __name__ == "__main__":
    main()
