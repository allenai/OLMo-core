"""
Launch the soft-detach CPT comparison (records/softdetach-cpt-plan.md): dense vs two >=4x-compaction
soft-token arms on the marker-wrapped dolma3_longmino CPT shard, Qwen3.5-4B, matched rows/step, on
Beaker (urgent, unallocated) through beaker_ctc_suite.py -- the same path as debug/ds64/launch_ds64.py.

Arms (all soft arms: detached slot K/V + GDN writes, cent_cmean slot, gold-blind, no header rule):
  dense   every token real (8 rows x 64k / step)
  sd20    keep_prob 0.2: 20% of the 512-token pseudo-docs stay whole, 80% collapse to ONE slot  (~5x)
  sfl20   keep_prob 0 + first_last 20%: EVERY pseudo-doc keeps its first 51 + last 51 body tokens (~5x)

    python src/scripts/train/memexpress/cpt/softdetach/launch_softdetach_cpt.py dry_run
    python src/scripts/train/memexpress/cpt/softdetach/launch_softdetach_cpt.py launch --budgets 32M,64M,128M
"""
import argparse
import csv
import datetime as dt
import os
import subprocess
import sys

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
LAUNCHER = f"{REPO}/src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py"
WEKA = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
SHARD = f"{WEKA}/softdetach_cpt/shards/cpt_u128M"  # budgets <= 128M are --max-tokens prefixes of it
SHARD_1B = f"{WEKA}/softdetach_cpt/shards/cpt_u1B"  # budgets > 128M (same rows first: parts 0-27, dev part 28 held out)
BASE = f"{WEKA}/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"  # marker-repaired 4B base (ds64's)
TOKENIZER = f"{WEKA}/hf_tokenizers/Qwen3.5-0.8B-Base"
LEDGER = f"{REPO}/src/scripts/train/memexpress/cpt/softdetach/LAUNCH_LEDGER.tsv"
CLUSTER = os.environ.get("SDC_CLUSTER", "ai2/jupiter-cirrascale-2")
GPUS = int(os.environ.get("SDC_NGPU", "8"))
BACKEND = os.environ.get("SDC_BACKEND", "flash_2")
ROWS_PER_STEP = 8  # x 65025 tokens = 520k tokens/step for every arm (ds64 dense geometry)

_SOFT = f"--st-gold-blind --st-slot-mode cent_cmean --st-slot-tokenizer {TOKENIZER} --attn-backend {BACKEND}"
ARMS = {
    "dense": None,
    "sd20": f"--st-keep-prob 0.2 {_SOFT}",
    "sfl20": f"--st-keep-prob 0.0 --st-keep-token-rule first_last --st-keep-token-k 0.2 {_SOFT}",
    # learned slot: sd20 geometry, slot NOT detached and only the projector trains (frozen backbone) --
    # the cheapest "trainable summary" arm; scored afterwards with the dev-loss driver like every other scheme
    "lslot20": f"--st-keep-prob 0.2 --st-no-detach-soft-kv --freeze-backbone {_SOFT}",
}


def run_name(arm, budget):
    return f"sdcpt-q35-4b-{arm}-u{budget}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["launch", "dry_run"])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--budgets", default="32M,64M,128M")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--wandb-group", default="sdcpt-q35-4b")
    a = ap.parse_args()
    rows = []
    for arm in a.arms.split(","):
        for budget in a.budgets.split(","):
            cap = int(float(budget.rstrip("MmBb")) * (1_000_000_000 if budget[-1] in "Bb" else 1_000_000))
            shard = SHARD_1B if cap > 128_000_000 else SHARD
            name = run_name(arm, budget)
            variant = "full" if arm == "dense" else "softtoken"
            extra = f"--max-tokens {cap}" + (f" {ARMS[arm]}" if ARMS[arm] else "")
            cmd = [sys.executable, "-u", LAUNCHER, "--task", "cpt", "--variant", variant,
                   "--model-family", "qwen3_5", "--model-scale", "4b", "--data-root", shard,
                   "--run-name", name, "--exact-run-name", "--num-nodes", "1", "--num-gpus", str(GPUS),
                   "--epochs", "1", "--lr", str(a.lr), "--cluster", CLUSTER, "--wandb-group", a.wandb_group,
                   "--no-follow", "--no-compile", "--seq-len", "65536", "--global-batch", str(ROWS_PER_STEP),
                   "--micro-batch-instances", "1", "--base-checkpoint", BASE, "--extra-args", extra, a.mode]
            print(" ".join(cmd), flush=True)
            res = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, PYTHONPATH=f"{REPO}/src"), capture_output=True, text=True)
            out = res.stdout + res.stderr
            os.makedirs(f"{REPO}/src/scripts/train/memexpress/cpt/softdetach/launch_logs", exist_ok=True)
            open(f"{REPO}/src/scripts/train/memexpress/cpt/softdetach/launch_logs/{name}.{a.mode}.log", "w").write(out)
            ids = [l.split("id=")[1].split()[0] for l in out.splitlines() if "SUBMITTED id=" in l]
            tail = (ids[-1] + " " if ids else "") + "\n".join(out.strip().splitlines()[-2:]).replace("\t", " ")[:200]
            print(f"  -> rc={res.returncode}: {tail[:300]}", flush=True)
            rows.append((name, arm, budget, res.returncode, tail))
    if a.mode == "launch":
        new = not os.path.exists(LEDGER)
        with open(LEDGER, "a") as f:
            w = csv.writer(f, delimiter="\t")
            if new:
                w.writerow(["run", "arm", "budget", "cluster", "launched", "state", "note"])
            for name, arm, budget, rc, tail in rows:
                w.writerow([name, arm, budget, CLUSTER, dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "LAUNCHED" if rc == 0 else "LAUNCH-FAILED", tail])


if __name__ == "__main__":
    main()
