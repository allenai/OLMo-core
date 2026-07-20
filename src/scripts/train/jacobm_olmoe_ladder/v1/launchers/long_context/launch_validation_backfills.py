#!/usr/bin/env python3
"""Render and optionally submit final-checkpoint validation for V1 LC runs."""

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
    manifest = yaml.safe_load(path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"{path}: expected schema_version: 1")
    return manifest


def env_value(name: str, value: str | int) -> dict[str, str]:
    return {"name": name, "value": str(value)}


def validate(manifest: dict[str, Any], source_repo: Path) -> list[dict[str, Any]]:
    wrapper = source_repo / str(manifest["source"]["wrapper"])
    if not wrapper.is_file():
        raise ValueError(f"Missing source wrapper: {wrapper}")
    root = Path(str(manifest["experiment"]["checkpoint_root"]))
    rows = manifest["targets"]
    for row in rows:
        checkpoint = root / str(row["source_run"]) / str(row["step"])
        if not checkpoint.is_dir():
            raise ValueError(f"Missing final checkpoint: {checkpoint}")
        row["checkpoint"] = str(checkpoint)
        world = int(row["gpu_count"])
        ep = int(row.get("expert_parallel_size", 1))
        global_batch = int(manifest["evaluation"]["global_batch_size"])
        sequence_length = int(manifest["evaluation"]["sequence_length"])
        microbatch = int(row["rank_microbatch_sequences"])
        if world % ep:
            raise ValueError(f"{row['source_run']}: EP={ep} must divide world={world}")
        global_sequences = global_batch // sequence_length
        if global_batch % sequence_length or global_sequences % (world * microbatch):
            raise ValueError(f"{row['source_run']}: invalid batch/world/microbatch shape")
    return rows


def build_task(
    manifest: dict[str, Any], row: dict[str, Any], *, source_repo: Path
) -> dict[str, Any]:
    source = manifest["source"]
    beaker = manifest["beaker"]
    evaluation = manifest["evaluation"]
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
            env_value("OLMOE3_LC_RUN_NAME", f"val-{row['source_run']}"),
            env_value("OLMOE3_LC_SUBCOMMAND", "eval_checkpoints"),
            env_value("OLMOE3_LC_MODEL_SIZE", row["model_size"]),
            env_value("OLMOE3_LC_FAMILY", row["family"]),
            env_value("OLMOE3_LC_LOAD_PATH", row["checkpoint"]),
            env_value("OLMOE3_LC_EVAL_CHECKPOINT", row["checkpoint"]),
            env_value("OLMOE3_LC_SEQUENCE_LENGTH", evaluation["sequence_length"]),
            env_value("OLMOE3_LC_EVAL_SEQUENCE_LENGTH", evaluation["eval_sequence_length"]),
            env_value("OLMOE3_LC_GLOBAL_BATCH_SIZE", evaluation["global_batch_size"]),
            env_value("OLMOE3_LC_MAX_TOKENS", evaluation["max_tokens"]),
            env_value("OLMOE3_LC_WORLD_SIZE", row["gpu_count"]),
            env_value("OLMOE3_LC_EP_SIZE", row.get("expert_parallel_size", 1)),
            env_value("OLMOE3_LC_EP_PATH", row.get("expert_parallel_path", "sync_1d")),
            env_value(
                "OLMOE3_LC_RANK_MICROBATCH_SEQUENCES",
                row["rank_microbatch_sequences"],
            ),
            env_value("OLMOE3_LC_EVAL_TASK_SET", evaluation["task_set"]),
            env_value("OLMOE3_LC_EVALS", "1"),
            env_value("OLMOE3_LC_EVAL_ON_FINISH", "0"),
            env_value("OLMOE3_LC_USE_COMPILE", "0"),
            env_value("OLMOE3_LC_WANDB", "1"),
            env_value("OLMOE3_LC_ASYNC_BOOKKEEPING", "0"),
            env_value("OLMOE3_LC_SAVE_ROOT", evaluation["save_root"]),
            env_value("OLMOE3_LC_WORK_DIR", evaluation["work_dir"]),
            env_value("OLMOE3_LC_DATA_GLOB", evaluation["data_glob"]),
        ]
    )
    datasets = [
        {"mountPath": item["mount_path"], "source": {"weka": item["weka"]}}
        for item in manifest.get("datasets", [])
    ]
    return {
        "name": str(row["source_run"]).removeprefix("lc-"),
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--experiment-name")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    source_repo = Path(os.environ.get("SOURCE_REPO", manifest["source"]["repo"])).resolve()
    rows = validate(manifest, source_repo)
    output = args.output or DEFAULT_GENERATED_DIR / f"{manifest_path.stem}.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = {
        "version": "v2",
        "description": f"{manifest['experiment']['description']} ({len(rows)} tasks)",
        "tasks": [build_task(manifest, row, source_repo=source_repo) for row in rows],
    }
    output.write_text(yaml.dump(spec, Dumper=NoAliasDumper, sort_keys=False))

    for row in rows:
        print(
            f"{row['model_size']} {row['family']} EP{row.get('expert_parallel_size', 1)} "
            f"{Path(row['checkpoint']).name} {row['source_run']}"
        )
    print(f"Rendered: {output}")
    if not args.submit:
        print("Dry run only. Add --submit --experiment-name NAME to launch.")
        return
    if not args.experiment_name:
        parser.error("--experiment-name is required with --submit")
    if shutil.which("beaker") is None:
        raise RuntimeError("beaker CLI is not available")
    subprocess.run(
        [
            "beaker",
            "experiment",
            "create",
            str(output),
            "--workspace",
            str(manifest["beaker"]["workspace"]),
            "--name",
            args.experiment_name,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
