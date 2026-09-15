"""Detached Gantry submission for the contradiction-only SFT pair and CPU prep.

Requires uvx; --ref pins a pushed snapshot of the training scripts. Training uses
2 x 8 GPUs on Jupiter, urgent priority, and minRuntime=0.
"""

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Queue contradiction-only SFT on two nodes, minRuntime=0"
)
parser.add_argument("arm", choices=["dense", "compressive", "prep"])
parser.add_argument("--ref", required=True, help="Full pushed commit SHA")
parser.add_argument("--run-name")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
mode, ref = args.arm, args.ref
if len(ref) != 40 or any(c not in "0123456789abcdef" for c in ref):
    parser.error("--ref must be a full 40-character commit SHA")
p = "src/scripts/train/memexpress/sft_xlong256k/"
name = args.run_name or f"q35-{mode}-contra-3ep-256k-{datetime.now():%Y%m%dT%H%M%S}"
cmd = [
    "uvx",
    "--from",
    "beaker-gantry@3.7.0",
    "gantry",
    "run",
    "--name",
    name,
    "--task-name",
    "prep" if mode == "prep" else "train",
    "--workspace",
    "ai2/flex2",
    "--budget",
    "ai2/oe-other",
    "--cluster",
    "ai2/jupiter-cirrascale-2",
    "--priority",
    "urgent",
    "--min-runtime",
    "0",
    "--beaker-image",
    "tylerr/olmo-core-tch291cu128-2025-11-25",
    "--gpus",
    "0" if mode == "prep" else "8",
    "--replicas",
    "1" if mode == "prep" else "2",
    "--shared-memory",
    "10GiB",
    "--weka",
    "oe-training-default:/weka/oe-training-default",
    "--branch",
    "amandab/contradiction-only-256k-20260914",
    "--ref",
    ref,
    "--env-secret",
    "BEAKER_TOKEN=amandab_BEAKER_TOKEN",
    "--env-secret",
    "WANDB_API_KEY=AMANDAB_WANDB_API_KEY",
    "--env-secret",
    "HF_TOKEN=amandab_HF_TOKEN",
    "--env",
    "OMP_NUM_THREADS=8",
    "--env",
    "NCCL_DEBUG=WARN",
    "--env",
    "PYTORCH_ALLOC_CONF=expandable_segments:True",
    "--env",
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
    "--env",
    "OLMO_SHARED_FS=1",
    "--env",
    "PYTHONPATH=src",
    "--system-python",
    "--install",
    "python -m pip install -e '.[beaker,eval,fla,wandb,transformers]' 'torch==2.9.1+cu128' 'triton==3.5.1'",
    "--allow-dirty",
    "--yes",
    "--no-logs",
    "--timeout",
    "0",
    "--save-spec",
    f"/tmp/{name}.yaml",
]
if mode == "prep":
    cmd += ["--cluster", "ai2/neptune*", "--cluster", "ai2/ceres*", "--cluster", "ai2/saturn*"]
if mode != "prep":
    cmd += ["--torchrun", "--host-networking", "--synchronized-start-timeout", "90m"]
if args.dry_run:
    cmd += ["--dry-run"]
if mode == "prep":
    cmd += ["--", "python", p + "prep_contradiction_256k.py"]
else:
    cmd += [
        "--",
        p + f"Qwen3.5-4B-{mode}-contradiction-3ep-256k-SFT.py",
        "train",
        name,
        "ai2/jupiter-cirrascale-2",
    ]
Path(f"/tmp/{name}-command.json").write_text(json.dumps(cmd, indent=2))
subprocess.run(cmd, check=True)
