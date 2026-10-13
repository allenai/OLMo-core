#!/usr/bin/env python3
"""Submit the initial materialization job for the 1,019B logical unique extension pool."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-experiment", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--workspace", default="ai2/flex2")
    parser.add_argument("--priority", default="urgent")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--cpu-count", type=float, default=64)
    parser.add_argument("--memory", default="512 GiB")
    parser.add_argument("--gpu-count", type=int, default=3)
    parser.add_argument("--cluster", default="ai2/jupiter")
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def get_base_spec(experiment: str) -> dict[str, Any]:
    result = subprocess.run(
        ["beaker", "experiment", "spec", experiment, "--format", "json"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def build_spec(
    base_spec: dict[str, Any],
    *,
    revision: str,
    priority: str,
    workers: int,
    cpu_count: float,
    memory: str,
    gpu_count: int,
    cluster: str,
) -> dict[str, Any]:
    spec = copy.deepcopy(base_spec)
    task = spec["tasks"][0]
    output_root = "/weka/oe-training-default/sewonm/icsl/data/dclm_0802_unique_extension_1019b"
    manifest_root = f"{output_root}/manifests"
    task["arguments"] = [
        "python",
        "scripts/data/create_dclm_unique_extension_pool.py",
        "--original-unique-manifest="
        "src/olmo_core/data/subsets/0802/dclm_0802_unique_train_5b.json",
        "--partition-manifest=src/olmo_core/data/subsets/0802/dclm_0802_partition.json",
        f"--pool-manifest={manifest_root}/dclm_0802_unique_extension_1019b.pool.json",
        f"--chunk-manifest={manifest_root}/dclm_0802_unique_extension_5b_to_16b.json",
        f"--materialized-output={output_root}/dclm_0802_unique_extension_5b_to_16b_uint32.npy",
        f"--candidate-dir={output_root}/candidates-5b-to-16b",
        "--chunk-name=unique_extension_5b_to_16b",
        "--logical-pool-tokens=1019000000000",
        "--chunk-tokens=11500781568",
        f"--workers={workers}",
        "--candidate-multiplier=1.5",
        "--alignment-tokens=4194304",
        "--data-root=/weka/oe-training-default/ai2-llm",
        "--manifest-base-dir=/weka/oe-training-default/ai2-llm",
    ]
    task["envVars"] = [
        item for item in task.get("envVars", []) if item.get("name") != "GANTRY_USE_TORCHRUN"
    ]
    for item in task["envVars"]:
        if item.get("name") == "GIT_REF":
            item["value"] = revision
            break
    else:
        task["envVars"].append({"name": "GIT_REF", "value": revision})
    task["resources"] = {
        "cpuCount": cpu_count,
        "gpuCount": gpu_count,
        "memory": memory,
        "sharedMemory": "16 GiB",
    }
    task["context"] = {"priority": priority, "minRuntime": "0s", "autoResume": False}
    task["constraints"] = {"cluster": [cluster]}
    task["propagateFailure"] = False
    task["propagatePreemption"] = False
    spec.pop("description", None)
    return spec


def main() -> None:
    args = parse_args()
    spec = build_spec(
        get_base_spec(args.base_experiment),
        revision=args.revision,
        priority=args.priority,
        workers=args.workers,
        cpu_count=args.cpu_count,
        memory=args.memory,
        gpu_count=args.gpu_count,
        cluster=args.cluster,
    )
    if args.print_only:
        json.dump(spec, sys.stdout, indent=2)
        print()
        return
    result = subprocess.run(
        [
            "beaker",
            "experiment",
            "create",
            "-",
            "--name",
            args.name,
            "--workspace",
            args.workspace,
            "--priority",
            args.priority,
        ],
        check=True,
        input=json.dumps(spec),
        text=True,
        stdout=subprocess.PIPE,
    )
    print(result.stdout, end="")


if __name__ == "__main__":
    main()
