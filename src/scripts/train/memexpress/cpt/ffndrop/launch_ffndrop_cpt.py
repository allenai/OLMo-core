"""
Launch the drop-CPT comparison (records/ffndrop-cpt-plan.md): does continued pretraining with random
per-token FFN dropping (AdaMoE-style null expert, no router) make the routed-FFN SFT stage
compute-optimal against dense SFT? Qwen3.5-4B, the marker-wrapped dolma3_longmino CPT shard of the
soft-detach CPT study (same base, shard, rows/step, LR), Beaker urgent + unallocated through
beaker_ctc_suite.py.

Arms (the CPT stage; the SFT stage later starts from each export):
  dense      every FFN runs -- the control (CPT data alone may lift dense SFT too)
  drop75l12  each row draws r ~ U[0, 0.75]; every token skips each FFN of layers >= 1 with prob r,
             and each (row, layer) drops the FFN for the whole row with prob 0.125

    python src/scripts/train/memexpress/cpt/ffndrop/launch_ffndrop_cpt.py dry_run
    python src/scripts/train/memexpress/cpt/ffndrop/launch_ffndrop_cpt.py launch --budgets 1B
"""
import argparse
import csv
import datetime as dt
import os
import subprocess
import sys

REPO = "/accounts/projects/berkeleynlp/prasann/projects/OLMo-core"
HERE = f"{REPO}/src/scripts/train/memexpress/cpt/ffndrop"
LAUNCHER = f"{REPO}/src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py"
WEKA = "/weka/oe-training-default/ai2-llm/checkpoints/prasanns"
SHARD = f"{WEKA}/softdetach_cpt/shards/cpt_u128M"  # budgets <= 128M are --max-tokens prefixes of it
SHARD_1B = f"{WEKA}/softdetach_cpt/shards/cpt_u1B"  # parts 0-27; dev part 28 held out (cpt_dev)
BASE = f"{WEKA}/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim"  # marker-repaired 4B base
LEDGER = f"{HERE}/LAUNCH_LEDGER.tsv"
CLUSTER = os.environ.get("FDC_CLUSTER", "ai2/jupiter-cirrascale-2")
GPUS = int(os.environ.get("FDC_NGPU", "8"))
ROWS_PER_STEP = 8  # x 65025 tokens = 520k tokens/step, the sdcpt geometry
SAVE_INTERVAL = 480  # ~250M tokens: mid-CPT bases at 250M/500M/750M for a CPT-length axis + resume

ARMS = {
    "dense": "",
    "drop75l12": "--ffn-drop-max-rate 0.75 --ffn-drop-layer-prob 0.125 --ffn-drop-start-layer 1",
}


def run_name(arm, budget):
    return f"fdcpt-q35-4b-{arm}-u{budget}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["launch", "dry_run"])
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--budgets", default="1B")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--wandb-group", default="fdcpt-q35-4b")
    ap.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    ap.add_argument("--max-steps", type=int, default=None, help="hard stop (smoke runs)")
    ap.add_argument("--name-suffix", default="")
    a = ap.parse_args()
    rows = []
    for arm in a.arms.split(","):
        for budget in a.budgets.split(","):
            cap = int(float(budget.rstrip("MmBb")) * (1_000_000_000 if budget[-1] in "Bb" else 1_000_000))
            shard = SHARD_1B if cap > 128_000_000 else SHARD
            name = run_name(arm, budget) + a.name_suffix
            extra = f"--max-tokens {cap} --save-interval {a.save_interval} {ARMS[arm]}".strip()
            cmd = [sys.executable, "-u", LAUNCHER, "--task", "cpt", "--variant", "full",
                   "--model-family", "qwen3_5", "--model-scale", "4b", "--data-root", shard,
                   "--run-name", name, "--exact-run-name", "--num-nodes", "1", "--num-gpus", str(GPUS),
                   "--epochs", "1", "--lr", str(a.lr), "--cluster", CLUSTER, "--wandb-group", a.wandb_group,
                   "--no-follow", "--no-compile", "--seq-len", "65536", "--global-batch", str(ROWS_PER_STEP),
                   "--micro-batch-instances", "1", "--base-checkpoint", BASE]
            if a.max_steps:
                cmd += ["--max-steps", str(a.max_steps)]
            cmd += ["--extra-args", extra, a.mode]
            print(" ".join(cmd), flush=True)
            res = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, PYTHONPATH=f"{REPO}/src"), capture_output=True, text=True)
            out = res.stdout + res.stderr
            os.makedirs(f"{HERE}/launch_logs", exist_ok=True)
            open(f"{HERE}/launch_logs/{name}.{a.mode}.log", "w").write(out)
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
