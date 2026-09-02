"""
Qwen3.5-4B track of the FLOP-scaling study (Prasann 2026-09-02: "stick to 3.5 for everything"):
the method arms train on the SAME weka arms the prior dense campaigns used, with the same launch
settings, so those campaigns' dense points (debug/taskscale_lengthmix/points.json,
debug/outlier_lengthmix_scaling) are the baseline and nothing dense is retrained.

Dense reference settings (debug/taskscale_lengthmix/launch_arms.sh, outlier gate_*.sh):
  --model-family qwen3_5 --model-scale 4b, base q35-4b-base-markerfix, --seq-len 65536 --pack,
  --lr 5e-6, --global-batch 8 --micro-batch-instances 1, 1 epoch, 1 node x 8 GPU.

Arms:
  ffnmoe-s1 / ffnmoe-s2   same data as dense (no markers needed)
  kv17 / kv33             need MARKER-wrapped re-tokenizations of the same JSONL arms
                          (build_kv35_shards.sbatch / _beaker.sh -> flop_scaling35/shards/<arm>_mk),
                          padded path (per-row fingerprints), global batch 160 x micro 2.

    python debug/flop_scaling/launch_grid35.py --tasks outlier --budgets 16M --arms ffnmoe-s1 dry_run
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
BASE = f"{WEKA}/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"
CKPTS = f"{WEKA}/ctc_suite/ckpts"
LEDGER = f"{REPO}/debug/flop_scaling/LAUNCH_LEDGER.tsv"
FFN_LADDER = "1,16,64,256,1024,9728"  # Qwen3.5-4B hidden is also 9728
KV_FRAC = {"kv17": 1 / 6, "kv33": 1 / 3}

# (task -> budget -> dense-campaign arm dir on weka). Dense f1 for these exist in points.json.
ARMS = {
    "outlier": {b: f"outlier_lengthmix/arms/mix_s{b}" for b in ["16M", "32M", "64M", "160M", "320M"]},
    "nq": {b: f"outlier_lengthmix/arms/nmix_s{b}" for b in ["16M", "32M", "48M"]},
    "contradiction": {b: f"taskscale_lengthmix/arms_tokenized/contradiction_mix_s{b}" for b in ["14M", "28M", "56M"]},
    "oolong": {b: f"taskscale_lengthmix/arms_tokenized/oolong_mix_s{b}" for b in ["20M", "40M", "80M"]},
}
KV_SHARDS = f"{WEKA}/flop_scaling35/shards"  # <task>_s<B>_mk (marker re-tokenization)


def run_name(task, arm, budget):
    return f"fs35-{task}-{arm}-s{budget}"


def arm_args(task, arm, budget):
    packed = ["--pack", "--seq-len", "65536", "--global-batch", "8", "--micro-batch-instances", "1"]
    data = f"{WEKA}/{ARMS[task][budget]}"
    if arm == "ffnmoe-s1":
        extra = (f"--ffn-moe-start-layer 12 --ffn-moe-divisors {FFN_LADDER} --ffn-moe-width-multiple 1 "
                 "--ffn-moe-target 0.01 --ffn-moe-target-anneal-frac 0.3 --ffn-moe-explore-anneal-frac 0.3")
        return "ffnmoe", data, packed + ["--base-checkpoint", BASE], extra
    if arm == "ffnmoe-s2":
        s1 = f"{CKPTS}/{run_name(task, 'ffnmoe-s1', budget)}/model_and_optim"
        extra = (f"--ffn-moe-start-layer 0 --ffn-moe-divisors {FFN_LADDER} --ffn-moe-width-multiple 1 "
                 "--ffn-moe-target 0.02 --ffn-moe-target-anneal-frac 0.0 --ffn-moe-explore-anneal-frac 0.3")
        return "ffnmoe", data, packed + ["--base-checkpoint", s1], extra
    if arm in KV_FRAC:
        frac = KV_FRAC[arm]
        padded = ["--seq-len", "65536", "--global-batch", "160", "--micro-batch-instances", "2", "--base-checkpoint", BASE]
        kvdata = f"{KV_SHARDS}/{task}_s{budget}_mk"
        if task == "oolong":
            extra = f"--st-gold-blind --st-keep-prob {frac:.4f} --attn-backend flash_2"
        else:
            extra = f"--st-keep-frac {frac:.4f} --st-keep-mode gold_plus_random --attn-backend flash_2"
        return "softtoken", kvdata, padded, extra
    raise SystemExit(f"unknown arm {arm}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="outlier,nq,oolong,contradiction")
    ap.add_argument("--budgets", default="", help="comma list; default = every budget with a dense point")
    ap.add_argument("--arms", default="ffnmoe-s1")
    ap.add_argument("--cluster", default="ai2/jupiter-cirrascale-2")
    ap.add_argument("--num-gpus", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--wandb-group", default="flop-scaling-q35")
    ap.add_argument("--skip", default="")
    ap.add_argument("mode", choices=["launch", "dry_run"])
    args = ap.parse_args()
    skip = set(x for x in args.skip.split(",") if x)
    rows = []
    for task in args.tasks.split(","):
        budgets = args.budgets.split(",") if args.budgets else list(ARMS[task])
        for budget in budgets:
            for arm in args.arms.split(","):
                variant, data, largs, extra = arm_args(task, arm, budget)
                name = run_name(task, arm, budget)
                if name in skip:
                    print(f"[skip] {name}"); continue
                cmd = [sys.executable, "-u", LAUNCHER, "--task", task, "--variant", variant,
                       "--model-family", "qwen3_5", "--model-scale", "4b", "--data-root", data,
                       "--run-name", name, "--exact-run-name", "--num-nodes", "1", "--num-gpus", str(args.num_gpus),
                       "--epochs", "1", "--lr", str(args.lr), "--cluster", args.cluster, "--wandb-group", args.wandb_group,
                       "--no-follow", "--no-compile"] + largs + (["--extra-args", extra] if extra else []) + [args.mode]
                print(" ".join(cmd), flush=True)
                res = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, PYTHONPATH=f"{REPO}/src"), capture_output=True, text=True)
                out = res.stdout + res.stderr
                os.makedirs(f"{REPO}/debug/flop_scaling/launch_logs", exist_ok=True)
                open(f"{REPO}/debug/flop_scaling/launch_logs/{name}.{args.mode}.log", "w").write(out)
                ids = [l.split("id=")[1].split()[0] for l in out.splitlines() if "SUBMITTED id=" in l]
                tail = (ids[-1] + " " if ids else "") + "\n".join(out.strip().splitlines()[-2:]).replace("\t", " ")[:200]
                print(f"  -> rc={res.returncode}: {tail[:300]}", flush=True)
                rows.append((name, task, arm, budget, res.returncode, tail))
    if args.mode == "launch":
        with open(LEDGER, "a") as f:
            w = csv.writer(f, delimiter="\t")
            for name, task, arm, budget, rc, tail in rows:
                w.writerow(["beaker35", task, arm, budget, arm.split("-")[-1] if "-" in arm else "-", name, args.cluster,
                            dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "LAUNCHED" if rc == 0 else "LAUNCH-FAILED", tail])


if __name__ == "__main__":
    main()
