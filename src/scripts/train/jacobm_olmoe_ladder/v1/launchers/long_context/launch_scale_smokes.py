#!/usr/bin/env python3
"""Validate, render, and optionally submit long-context scale smoke tests."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "manifests" / "integration_wide_scale_smokes.yaml"
DEFAULT_OUTPUT = SCRIPT_DIR / "generated" / "integration_wide_scale_smokes.yaml"


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, _data: Any) -> bool:
        return True


def env_var(name: str, value: Any) -> dict[str, str]:
    return {"name": name, "value": str(value)}


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as file:
        manifest = yaml.safe_load(file)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"{path}: expected schema_version: 1")
    return manifest


def validate(manifest: dict[str, Any], source_repo: Path) -> list[dict[str, Any]]:
    wrapper = source_repo / str(manifest["source"]["wrapper"])
    if not wrapper.is_file():
        raise ValueError(f"Missing source wrapper: {wrapper}")
    training = manifest["training"]
    sequence_length = int(training["sequence_length"])
    global_batch = int(training["global_batch_size"])
    if global_batch % sequence_length:
        raise ValueError("Global batch must contain a whole number of sequences")
    global_sequences = global_batch // sequence_length
    rows = manifest.get("smokes")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must contain at least one smoke")
    names: set[str] = set()
    run_names: set[str] = set()
    for row in rows:
        task_name = str(row["task_name"])
        run_name = str(row["run_name"])
        if task_name in names or run_name in run_names:
            raise ValueError(f"Duplicate task or run name: {task_name} / {run_name}")
        names.add(task_name)
        run_names.add(run_name)
        world_size = int(row["gpu_count"])
        ep_size = int(row["expert_parallel_size"])
        ep_path = str(row["expert_parallel_path"])
        if ep_path not in {"sync_1d", "rowwise_nvshmem"}:
            raise ValueError(f"{task_name}: unsupported EP path {ep_path!r}")
        if ep_size < 1 or world_size % ep_size:
            raise ValueError(f"{task_name}: EP={ep_size} must divide world={world_size}")
        expert_dp_degree = world_size // ep_size
        # OLMo-core shards the global batch across every rank even when EP is on.
        denominator = world_size * int(row["rank_microbatch_sequences"])
        if global_sequences % denominator:
            raise ValueError(
                f"{task_name}: {global_sequences} sequences is not divisible by "
                f"world({world_size})*MB({row['rank_microbatch_sequences']})"
            )
        row["expert_dp_degree"] = expert_dp_degree
        row["accumulation_steps"] = global_sequences // denominator
    return rows


def build_task(
    manifest: dict[str, Any], row: dict[str, Any], *, source_repo: Path
) -> dict[str, Any]:
    source = manifest["source"]
    beaker = manifest["beaker"]
    training = manifest["training"]
    arguments = (
        "set -euo pipefail\n"
        f"SOURCE_REPO={shlex.quote(str(source_repo))}\n"
        "WRAPPER=/tmp/jacobm_olmoe3_long_context_beaker.sh\n"
        f'cp "${{SOURCE_REPO}}/{source["wrapper"]}" "${{WRAPPER}}"\n'
        'bash "${WRAPPER}"\n'
    )
    env_vars = [
        {"name": str(name), "secret": str(secret)}
        for name, secret in manifest.get("secrets", {}).items()
    ]
    env_vars.extend(
        [
            env_var("OLMOE3_LC_RUN_NAME", row["run_name"]),
            env_var("OLMOE3_LC_MODEL_SIZE", row["model_size"]),
            env_var("OLMOE3_LC_FAMILY", row["family"]),
            env_var("OLMOE3_LC_LOAD_PATH", row["load_path"]),
            env_var("OLMOE3_LC_SAVE_ROOT", manifest["experiment"]["checkpoint_root"]),
            env_var("OLMOE3_LC_WORK_DIR", training["work_dir"]),
            env_var("OLMOE3_LC_SEQUENCE_LENGTH", training["sequence_length"]),
            env_var("OLMOE3_LC_GLOBAL_BATCH_SIZE", training["global_batch_size"]),
            env_var("OLMOE3_LC_RANK_MICROBATCH_SEQUENCES", row["rank_microbatch_sequences"]),
            env_var("OLMOE3_LC_WORLD_SIZE", row["gpu_count"]),
            env_var("OLMOE3_LC_EP_SIZE", row["expert_parallel_size"]),
            env_var("OLMOE3_LC_EP_PATH", row["expert_parallel_path"]),
            env_var("OLMOE3_LC_LR", training["learning_rate"]),
            env_var("OLMOE3_LC_WARMUP_STEPS", training["warmup_steps"]),
            env_var("OLMOE3_LC_HARD_STOP_STEPS", training["hard_stop_steps"]),
            env_var("OLMOE3_LC_SAVE_INTERVAL", training["save_interval"]),
            env_var("OLMOE3_LC_EPHEMERAL_SAVE_INTERVAL", training["ephemeral_save_interval"]),
            env_var("OLMOE3_LC_USE_COMPILE", int(bool(training["compile"]))),
            env_var("OLMOE3_LC_WANDB", int(bool(training["wandb"]))),
            env_var("OLMOE3_LC_EVALS", int(bool(training["evals"]))),
            env_var("OLMOE3_LC_ASYNC_BOOKKEEPING", 0),
        ]
    )
    datasets = [
        {"mountPath": item["mount_path"], "source": {"weka": item["weka"]}}
        for item in manifest.get("datasets", [])
    ]
    return {
        "name": row["task_name"],
        "image": {"beaker": str(source["image"])},
        "command": ["/bin/bash", "-lc"],
        "arguments": [arguments],
        "envVars": env_vars,
        "datasets": datasets,
        "result": {"path": str(beaker.get("result_path", "/results"))},
        "resources": {
            "gpuCount": int(row["gpu_count"]),
            "sharedMemory": str(beaker["shared_memory"]),
        },
        "context": {
            "priority": str(beaker["priority"]),
            "minRuntime": str(beaker["min_runtime"]),
            "autoResume": False,
        },
        "constraints": {"cluster": [str(beaker["cluster"])]},
        "hostNetworking": True,
        "propagateFailure": False,
        "propagatePreemption": False,
        "timeout": str(beaker["timeout"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--experiment-name")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest.resolve())
    source_repo = Path(os.environ.get("SOURCE_REPO", manifest["source"]["repo"])).resolve()
    rows = validate(manifest, source_repo)
    if args.task:
        wanted = set(args.task)
        available = {str(row["task_name"]) for row in rows}
        if missing := wanted - available:
            raise ValueError(f"Unknown --task values: {sorted(missing)}")
        rows = [row for row in rows if str(row["task_name"]) in wanted]
    spec = {
        "version": "v2",
        "description": f"{manifest['experiment']['description']} ({len(rows)} tasks)",
        "tasks": [build_task(manifest, row, source_repo=source_repo) for row in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        yaml.dump(spec, file, Dumper=NoAliasDumper, sort_keys=False)

    print("size  GPU  EP  EP path  EP-DP  rank MB  accum  global tokens  run name")
    for row in rows:
        print(
            f"{row['model_size']:<5} {row['gpu_count']:>4} {row['expert_parallel_size']:>3} "
            f"{row['expert_parallel_path']:>10} {row['expert_dp_degree']:>5} "
            f"{row['rank_microbatch_sequences']:>8} {row['accumulation_steps']:>6} "
            f"{int(manifest['training']['global_batch_size']):>14}  {row['run_name']}"
        )
    print(
        f"\nRendered {args.output}; peak concurrent allocation is "
        f"{sum(int(row['gpu_count']) for row in rows)} GPUs."
    )

    if not args.submit:
        print("Dry run only. Add --submit --experiment-name NAME to launch.")
        return
    if not args.experiment_name:
        parser.error("--experiment-name is required with --submit")
    root = Path(str(manifest["experiment"]["checkpoint_root"]))
    existing = [
        root / str(row["run_name"]) for row in rows if (root / str(row["run_name"])).exists()
    ]
    if existing:
        raise RuntimeError(
            "Refusing to submit existing checkpoint directories:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )
    if shutil.which("beaker") is None:
        raise RuntimeError("beaker CLI is not available")
    command = [
        "beaker",
        "experiment",
        "create",
        str(args.output),
        "--workspace",
        str(manifest["beaker"]["workspace"]),
        "--name",
        args.experiment_name,
    ]
    print(f"Submitting: {' '.join(command)}")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
