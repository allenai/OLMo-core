#!/usr/bin/env python3
"""Validate, render, and optionally submit v2 hybrid midtraining runs."""

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
DEFAULT_GENERATED_DIR = SCRIPT_DIR / "generated"


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, _data: Any) -> bool:
        return True


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as file:
        manifest = yaml.safe_load(file)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"{path}: expected schema_version: 1")
    return manifest


def env_var(name: str, value: Any) -> dict[str, str]:
    return {"name": name, "value": str(value)}


def validate(manifest: dict[str, Any], source_repo: Path) -> list[dict[str, Any]]:
    wrapper = source_repo / str(manifest["source"]["wrapper"])
    if not wrapper.is_file():
        raise ValueError(f"Missing source wrapper: {wrapper}")
    training = manifest["training"]
    sequence_length = int(training["sequence_length"])
    rows = manifest["runs"]
    for row in rows:
        load_path = Path(str(row["load_path"]))
        if not load_path.is_dir():
            raise ValueError(f"Missing source checkpoint: {load_path}")
        world = int(row["gpu_count"])
        ep = int(row["expert_parallel_size"])
        global_batch = int(row["global_batch_size"])
        microbatch = int(row["rank_microbatch_sequences"])
        if world % ep:
            raise ValueError(f"{row['task_name']}: EP={ep} must divide world={world}")
        if global_batch % sequence_length:
            raise ValueError(f"{row['task_name']}: global batch must contain whole sequences")
        global_sequences = global_batch // sequence_length
        if global_sequences % world:
            raise ValueError(f"{row['task_name']}: global sequences must divide across ranks")
        rank_sequences = global_sequences // world
        if rank_sequences % microbatch:
            raise ValueError(f"{row['task_name']}: rank sequences must divide by microbatch")
        row["global_sequences"] = global_sequences
        row["rank_sequences"] = rank_sequences
        row["accumulation_steps"] = rank_sequences // microbatch
    return rows


def build_task(
    manifest: dict[str, Any], row: dict[str, Any], *, source_repo: Path
) -> dict[str, Any]:
    source = manifest["source"]
    beaker = manifest["beaker"]
    training = manifest["training"]
    wrapper = source_repo / str(source["wrapper"])
    arguments = (
        "set -euo pipefail\n"
        f"SOURCE_REPO={shlex.quote(str(source_repo))}\n"
        f"bash {shlex.quote(str(wrapper))}\n"
    )
    env_vars = [
        {"name": str(name), "secret": str(secret)}
        for name, secret in manifest.get("secrets", {}).items()
    ]
    env_vars.extend(
        [
            env_var("OLMOE3_HYBRID_RUN_NAME", row["run_name"]),
            env_var("OLMOE3_HYBRID_MODEL_SIZE", row["model_size"]),
            env_var("OLMOE3_HYBRID_MODEL_VARIANT", row["model_variant"]),
            env_var("OLMOE3_HYBRID_LR", row["learning_rate"]),
            env_var("OLMOE3_HYBRID_GLOBAL_BATCH_SIZE", row["global_batch_size"]),
            env_var("OLMOE3_HYBRID_WORLD_SIZE", row["gpu_count"]),
            env_var("OLMOE3_HYBRID_EP_SIZE", row["expert_parallel_size"]),
            env_var(
                "OLMOE3_HYBRID_RANK_MICROBATCH_SEQUENCES",
                row["rank_microbatch_sequences"],
            ),
            env_var("OLMOE3_HYBRID_SEQUENCE_LENGTH", training["sequence_length"]),
            env_var("OLMOE3_HYBRID_HARD_STOP_STEPS", "0"),
            env_var("OLMOE3_HYBRID_SAVE_INTERVAL", training["save_interval"]),
            env_var(
                "OLMOE3_HYBRID_EPHEMERAL_SAVE_INTERVAL",
                training["ephemeral_save_interval"],
            ),
            env_var("OLMOE3_HYBRID_CHECKPOINT_REMOVAL", training["checkpoint_removal"]),
            env_var("OLMOE3_HYBRID_CHECKPOINTS", "1"),
            env_var("OLMOE3_HYBRID_EVALS", "0"),
            env_var("OLMOE3_HYBRID_EVAL_ON_FINISH", "0"),
            env_var("OLMOE3_HYBRID_USE_COMPILE", int(bool(training["compile"]))),
            env_var("OLMOE3_HYBRID_WANDB", "1"),
            env_var("OLMOE3_HYBRID_SAVE_ROOT", manifest["experiment"]["checkpoint_root"]),
            env_var("OLMOE3_MT_LOAD_PATH", row["load_path"]),
            env_var("OLMOE3_MT_MAX_TOKENS", training["max_tokens"]),
            env_var("OLMOE3_MT_WARMUP_STEPS", training["warmup_steps"]),
            env_var("OLMOE3_MT_SOURCE_PROCESSES", training["source_processes"]),
        ]
    )
    datasets = [
        {"mountPath": item["mount_path"], "source": {"weka": item["weka"]}}
        for item in manifest.get("datasets", [])
    ]
    return {
        "name": str(row["task_name"]),
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
            "autoResume": bool(beaker.get("auto_resume", True)),
        },
        "constraints": {"cluster": [str(beaker["cluster"])]},
        "hostNetworking": True,
        "propagateFailure": False,
        "propagatePreemption": False,
        "timeout": str(beaker["timeout"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--experiment-name")
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    source_repo = Path(os.environ.get("SOURCE_REPO", manifest["source"]["repo"])).resolve()
    rows = validate(manifest, source_repo)
    if args.task:
        wanted = set(args.task)
        rows = [row for row in rows if str(row["task_name"]) in wanted]
        missing = wanted - {str(row["task_name"]) for row in rows}
        if missing:
            raise ValueError(f"Unknown tasks: {sorted(missing)}")

    output = args.output or DEFAULT_GENERATED_DIR / f"{manifest_path.stem}.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = {
        "version": "v2",
        "description": f"{manifest['experiment']['description']} ({len(rows)} tasks)",
        "tasks": [build_task(manifest, row, source_repo=source_repo) for row in rows],
    }
    with output.open("w") as file:
        yaml.dump(spec, file, Dumper=NoAliasDumper, sort_keys=False)

    print("size  GPU  EP  global seq  rank seq  MB  accum  LR       run name")
    for row in rows:
        print(
            f"{row['model_size']:<5} {row['gpu_count']:>3} {row['expert_parallel_size']:>3} "
            f"{row['global_sequences']:>10} {row['rank_sequences']:>9} "
            f"{row['rank_microbatch_sequences']:>3} {row['accumulation_steps']:>6} "
            f"{str(row['learning_rate']):<8} {row['run_name']}"
        )
    print(f"Rendered: {output}")

    if not args.submit:
        print("Dry run only. Add --submit --experiment-name NAME to launch.")
        return
    if not args.experiment_name:
        parser.error("--experiment-name is required with --submit")
    checkpoint_root = Path(str(manifest["experiment"]["checkpoint_root"]))
    existing = [
        checkpoint_root / str(row["run_name"])
        for row in rows
        if (checkpoint_root / str(row["run_name"])).exists()
    ]
    if existing and not args.resume_existing:
        raise RuntimeError(
            "Refusing existing midtraining checkpoint directories:\n"
            + "\n".join(f"  - {path}" for path in existing)
        )
    if shutil.which("beaker") is None:
        raise RuntimeError("beaker CLI is not available")
    command = [
        "beaker",
        "experiment",
        "create",
        str(output),
        "--workspace",
        str(manifest["beaker"]["workspace"]),
        "--name",
        args.experiment_name,
    ]
    print(f"Submitting: {' '.join(command)}")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
