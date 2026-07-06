#!/usr/bin/env python3
"""Launch OLMoBase evals for converted ladder HF checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPO = Path("/weka/oe-adapt-default/jacobm/olmoe3/olmo-eval")
DEFAULT_MANIFEST = Path(__file__).with_name("eval_1p2b_integration_cx1_cx2_targets.jsonl")
GROUP_OLMO_INSTRUCT = "olmoe3-olmobase"
GROUP_NORMAL = "olmoe3-olmobase-main"
BUDGET = "ai2/oe-other"

TASKS = [
    "olmobase:mcqa_stem",
    "olmobase:mcqa_non_stem",
    "olmobase:gen",
    "olmobase:math",
    "olmobase:easy:qa:rc",
    "olmobase:easy:qa:bpb",
    "olmobase:easy:math:bpb",
    "olmobase:easy:code:bpb",
]

VARIANT_SLUGS = {
    "baseline 48E/top4": "baseline",
    "Qwen-like active matched 4.5d": "q3am",
    "Qwen-like true 3.0d + depth": "q3td",
    "integration wide 256E/top8": "int-wide",
    "integration deep 256E/top8": "int-deep",
}


def load_targets(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def cx_slug(target: dict) -> str:
    return str(target["cx"]).lower()


def launch_command(target: dict, *, dry_run: bool) -> list[str]:
    size = target["model_size"]
    slug = VARIANT_SLUGS[target["variant"]]
    name = f"olmoe3-{size}-{cx_slug(target)}-{slug}-olmobase"
    workspace = (
        "ai2/olmo-instruct"
        if size in {"275m", "480m"}
        else "ai2/OLMo-3-moe-experiments"
    )
    group = GROUP_OLMO_INSTRUCT if size in {"275m", "480m"} else GROUP_NORMAL

    cmd = [
        "uv",
        "run",
        "--frozen",
        "--extra",
        "beaker",
        "olmo-eval",
        "beaker",
        "launch",
        "--name",
        name,
        "--model",
        target["hf_checkpoint"],
        "--harness",
        "default",
        "--override",
        "provider.num_instances=8",
        "--override",
        "provider.kind=vllm",
        "--override",
        "provider.trust_remote_code=true",
        "--override",
        "provider.dtype=bfloat16",
        "--override",
        "provider.max_model_len=4096",
        "--override",
        "provider.kwargs.tensor_parallel_size=1",
        "--override",
        "provider.kwargs.gpu_memory_utilization=0.85",
        "--secret-env",
        "jacobm_HF_TOKEN:HF_TOKEN",
        "--cluster",
        "ai2/jupiter",
        "--image",
        "01KVTNKQXYACY9J3HEE265KJXA",
        "--workspace",
        workspace,
        "--priority",
        "urgent",
        "--preemptible" if size in {"275m", "480m"} else "--no-preemptible",
        "--gpus",
        "8",
        "--timeout",
        "24h",
        "--group",
        group,
        "--no-store",
        "--no-follow",
        "--yes",
    ]
    if size not in {"275m", "480m"}:
        cmd.extend(["--budget", BUDGET])
    for task in TASKS:
        cmd.extend(["--task", task])
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--size", action="append", default=[])
    args = parser.parse_args()

    targets = load_targets(args.manifest)
    if args.only:
        targets = [
            target
            for target in targets
            if any(s in target["train_name"] or s in target["variant"] for s in args.only)
        ]
    if args.size:
        targets = [target for target in targets if target["model_size"] in set(args.size)]

    print(f"targets {len(targets)}")
    for target in targets:
        out = Path(target["hf_checkpoint"])
        if not (out / "config.json").exists():
            raise FileNotFoundError(out / "config.json")
        has_single_file = (out / "model.safetensors").exists()
        has_sharded_index = (out / "model.safetensors.index.json").exists()
        if not (has_single_file or has_sharded_index):
            raise FileNotFoundError(f"{out}/model.safetensors or model.safetensors.index.json")

        cmd = launch_command(target, dry_run=not args.launch)
        print("+", " ".join(cmd))
        if args.launch:
            result = subprocess.run(
                cmd,
                cwd=REPO,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            interesting = [
                line
                for line in result.stdout.splitlines()
                if "beaker.org" in line or "Created group" in line or "Experiment" in line
            ]
            for line in interesting[-20:]:
                print(line)
            if result.returncode != 0:
                print("--- launch output tail ---")
                print("\n".join(result.stdout.splitlines()[-80:]))
                raise subprocess.CalledProcessError(result.returncode, cmd)


if __name__ == "__main__":
    main()
