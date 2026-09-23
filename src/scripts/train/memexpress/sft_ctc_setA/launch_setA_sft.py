"""
Every Beaker step of the setA dense SFT pair (pre-flight + training), in one file.

    # 1. CPU pre-flight: shards, drops per window, base ckpt + marker audit, packed windows/steps
    python src/scripts/train/memexpress/sft_ctc_setA/launch_setA_sft.py prep
    # 2. train (1 node each; 256k-2node = amandab's exact 2-node geometry, fallback)
    python src/scripts/train/memexpress/sft_ctc_setA/launch_setA_sft.py 32k
    python src/scripts/train/memexpress/sft_ctc_setA/launch_setA_sft.py 256k

Every job is pinned to a full pushed commit SHA (default: this checkout's HEAD, which must be on the
remote). The data are the setA shards as built on ``prasann/landmark`` (see _qwen35_setA_common.py).
All jobs: workspace ai2/flex2, budget ai2/oe-other (unallocated),
priority urgent, jupiter H100s. ``--dry-run`` prints the gantry command without submitting.
"""

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REL = "src/scripts/train/memexpress/sft_ctc_setA"
GANTRY = os.environ.get(
    "GANTRY", "/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/gantry"
)

IMAGE = "tylerr/olmo-core-tch291cu128-2025-11-25"  # == OLMoCoreBeakerImage.stable
TRAIN_INSTALL = (
    "python -m pip install -e '.[beaker,eval,fla,wandb,transformers]' "
    "'torch==2.9.1+cu128' 'triton==3.5.1'"
)  # amandab's validated training install
NODES = {"32k": 1, "256k": 1, "256k-2node": 2}
CPU_CLUSTERS = [
    "ai2/jupiter-cirrascale-2",
    "ai2/neptune-cirrascale",
    "ai2/ceres-cirrascale",
    "ai2/saturn-cirrascale",
]


def head_sha() -> str:
    sha = subprocess.check_output(["git", "-C", HERE, "rev-parse", "HEAD"], text=True).strip()
    remote = subprocess.run(
        ["git", "-C", HERE, "branch", "-r", "--contains", sha], capture_output=True, text=True
    ).stdout.strip()
    if not remote:
        sys.exit(
            f"HEAD {sha[:10]} is not on any remote branch -- push first (gantry runs "
            "pushed code only)"
        )
    return sha


def branch_of(sha: str) -> str:
    out = subprocess.check_output(["git", "-C", HERE, "branch", "-r", "--contains", sha], text=True)
    names = [b.strip().removeprefix("origin/") for b in out.splitlines() if "->" not in b]
    return "prasann/ctc-setA-sft" if "prasann/ctc-setA-sft" in names else names[0]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=["prep", *NODES])
    ap.add_argument("--ref", default=None, help="full pushed SHA for prep/train (default HEAD)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%dT%H%M")
    common = [
        "--workspace",
        "ai2/flex2",
        "--budget",
        "ai2/oe-other",
        "--priority",
        "urgent",
        "--beaker-image",
        IMAGE,
        "--weka",
        "oe-training-default:/weka/oe-training-default",
        "--system-python",
        "--allow-dirty",
        "--yes",
        "--timeout",
        "0",
    ]

    ref = args.ref or head_sha()
    if len(ref) != 40:
        sys.exit("--ref must be a full 40-character SHA")
    base = [
        GANTRY,
        "run",
        *common,
        "--branch",
        branch_of(ref),
        "--ref",
        ref,
        "--install",
        TRAIN_INSTALL,
        "--env-secret",
        "WANDB_API_KEY=PRASANNS_WANDB_API_KEY",
        "--env-secret",
        "BEAKER_TOKEN=PRASANNS_BEAKER_TOKEN",
        "--env",
        "PYTHONPATH=src",
        "--env",
        "OMP_NUM_THREADS=8",
        "--env",
        "NCCL_DEBUG=WARN",
        "--env",
        "OLMO_SHARED_FS=1",
        "--env",
        "PYTORCH_ALLOC_CONF=expandable_segments:True",
        "--env",
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
    ]
    if args.step == "prep":
        name = args.run_name or f"ctc-setA-sft-prep-{stamp}"
        cmd = [*base, "--name", name, "--gpus", "0", "--cpus", "32", "--shared-memory", "32GiB"]
        for c in CPU_CLUSTERS:
            cmd += ["--cluster", c]
        cmd += ["--", "python", f"{REL}/prep_setA_sft.py"]
    else:
        n = NODES[args.step]
        name = args.run_name or f"q35-4b-dense-ctc-setA-{args.step}-{stamp}"
        cmd = [
            *base,
            "--name",
            name,
            "--task-name",
            "train",
            "--cluster",
            "ai2/jupiter-cirrascale-2",
            "--gpus",
            "8",
            "--replicas",
            str(n),
            "--shared-memory",
            "10GiB",
            "--min-runtime",
            "1h",
            "--torchrun",
        ]
        if n > 1:
            cmd += ["--host-networking", "--synchronized-start-timeout", "90m"]
        cmd += [
            "--",
            f"{REL}/Qwen3.5-4B-dense-ctc-setA-{args.step}-SFT.py",
            "train",
            name,
            "ai2/jupiter-cirrascale-2",
        ]

    print(" ".join(shlex.quote(c) for c in cmd), flush=True)
    if args.dry_run:
        return
    log = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"{name}.gantry.log")
    # detached: gantry --timeout 0 returns once submitted; keep its output for the experiment URL
    with open(log, "w") as f:
        subprocess.run(
            cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=False
        )
    print(open(log).read()[-1500:])


if __name__ == "__main__":
    main()
