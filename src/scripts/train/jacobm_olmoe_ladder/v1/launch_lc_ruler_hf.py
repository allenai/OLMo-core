#!/usr/bin/env python3
"""Launch canonical one-H100 RULER-64K evals for converted V1 LC models."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from launch_lc_hf_conversions import MODELS, Model


OLMO_EVAL_REPO = Path("/weka/oe-adapt-default/jacobm/olmoe3/olmo-eval")
TASK = "ruler_all__65536"
MAX_MODEL_LEN = 131072
CLUSTER = "ai2/jupiter"
WORKSPACE = "ai2/OLMo-3-moe-experiments"
BUDGET = "ai2/oe-other"
GROUP = "olmoe3-v1-lc-ruler"


def command(model: Model) -> list[str]:
    overrides = [
        "provider.kind=vllm",
        "provider.num_instances=1",
        "provider.dtype=bfloat16",
        "provider.trust_remote_code=true",
        f"provider.max_model_len={MAX_MODEL_LEN}",
        "provider.kwargs.model_impl=transformers",
        "provider.kwargs.enforce_eager=true",
        "provider.kwargs.attention_backend=FLASH_ATTN",
        "provider.kwargs.add_bos_token=false",
        "provider.kwargs.gpu_memory_utilization=0.8",
    ]
    return [
        "uv",
        "run",
        "olmo-eval",
        "beaker",
        "launch",
        "-H",
        "default",
        *[item for value in overrides for item in ("-o", value)],
        "-e",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1",
        "-n",
        f"olmoe3-{model.key}-lc-ruler64k-vllm-1gpu",
        "-m",
        str(model.output),
        "-t",
        TASK,
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
        "1",
        "--group",
        GROUP,
        "--no-save-requests",
        "--save-predictions",
        "--no-follow",
        "-y",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=sorted(MODELS), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for key in args.model:
        model = MODELS[key]
        marker = model.output / "conversion_complete.json"
        if not marker.is_file():
            raise RuntimeError(f"HF conversion is incomplete: missing {marker}")

    for key in args.model:
        launch_command = command(MODELS[key])
        print(" ".join(launch_command), flush=True)
        if not args.dry_run:
            subprocess.run(launch_command, cwd=OLMO_EVAL_REPO, check=True)


if __name__ == "__main__":
    main()
