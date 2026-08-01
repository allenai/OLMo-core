#!/usr/bin/env python3
"""Launch HF-backed RULER evals for the completed 275M KDA Cx8 LCE model."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

OLMO_EVAL_REPO = Path("/weka/oe-adapt-default/jacobm/olmoe3/olmo-eval")
RULER_CACHE = Path(
    "/weka/oe-adapt-default/jacobm/olmoe3/cache/huggingface/datasets"
)
MODEL = Path(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/"
    "hf/long-context/lc-275m-geometry-hybrid-kda-ev2-neg-nope-gated-"
    "cx8-ptlr8e-4-mtlr1p6e-4-lclr8e-5-r1/step37991"
)
LENGTHS = (4096, 8192, 16384, 32768, 65536, 131072)
WORKSPACE = "ai2/OLMo-3-moe-experiments"
# The stock olmo-eval HF image currently carries a CUDA 12.3 runtime. Its FLA
# 0.4.1 KDA kernel is valid on Ceres' H100s but fails at launch on Holmes'
# B300s, which require the newer Blackwell runtime used by our training image.
CLUSTER = "ai2/ceres"
BUDGET = "ai2/oe-other"
GROUP = "olmoe3-v2-kda-lc-ruler"


def command(length: int, *, dry_run: bool) -> list[str]:
    cmd = [
        "uv",
        "run",
        "--project",
        str(OLMO_EVAL_REPO),
        "--no-group",
        "vllm",
        "olmo-eval",
        "beaker",
        "launch",
        "-H",
        "default",
        "-o",
        "provider.kind=hf",
        "-o",
        "provider.dtype=bfloat16",
        "-o",
        "provider.trust_remote_code=true",
        "-o",
        "batching.chunk_size=8",
        "-o",
        'provider.dependencies=["flash-linear-attention==0.4.1"]',
        "-n",
        f"lc-275m-kda-cx8-ruler-{length // 1024}k-hf",
        "-m",
        str(MODEL),
        "-t",
        f"ruler_all__{length}",
        "--gpus",
        "1",
        "--cluster",
        CLUSTER,
        "--workspace",
        WORKSPACE,
        "--budget",
        BUDGET,
        "--priority",
        "urgent",
        "--timeout",
        "24h",
        "--retries",
        "2",
        "--group",
        GROUP,
        "--secret-env",
        "HF_TOKEN:HF_TOKEN",
        "--env",
        f"HF_DATASETS_CACHE={RULER_CACHE}",
        "--no-save-requests",
        "--save-predictions",
        "--no-follow",
        "-y",
    ]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--length",
        action="append",
        type=int,
        choices=LENGTHS,
        help="Context length to launch; repeat to launch multiple (default: all).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    marker = MODEL / "conversion_complete.json"
    if not marker.is_file():
        raise RuntimeError(f"HF conversion is incomplete: missing {marker}")
    lengths = args.length or list(LENGTHS)
    for length in lengths:
        launch_command = command(length, dry_run=args.dry_run)
        print(" ".join(launch_command), flush=True)
        subprocess.run(launch_command, cwd=OLMO_EVAL_REPO, check=True)


if __name__ == "__main__":
    main()
