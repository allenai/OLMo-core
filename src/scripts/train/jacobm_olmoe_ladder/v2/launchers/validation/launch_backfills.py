#!/usr/bin/env python3
"""Validate, render, and optionally submit v2 final-checkpoint validation backfills."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_GENERATED_DIR = SCRIPT_DIR / "generated"


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, _data: Any) -> bool:
        return True


@dataclass(frozen=True)
class Target:
    source_run: str
    checkpoint: Path
    model_size: str
    variant: str
    cx: int
    lr: str
    global_batch_size: int
    expert_parallel_size: int
    expert_parallel_path: str
    rank_microbatch_sequences: int

    @property
    def eval_run(self) -> str:
        return f"val-{self.source_run}"

    @property
    def task_name(self) -> str:
        return self.source_run.removeprefix("pt-")


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as f:
        manifest = yaml.safe_load(f)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"{path}: expected schema_version: 1")
    for key in ("experiment", "source", "beaker", "evaluation", "targets"):
        if key not in manifest:
            raise ValueError(f"{path}: missing {key!r}")
    return manifest


def parse_targets(manifest: dict[str, Any], source_runs: set[str] | None = None) -> list[Target]:
    checkpoint_root = Path(str(manifest["experiment"]["checkpoint_root"]))
    batch_sizes = {int(k): int(v) for k, v in manifest["evaluation"]["batch_sizes"].items()}
    available = {str(raw["source_run"]) for raw in manifest["targets"]}
    if source_runs and (missing := source_runs - available):
        raise ValueError(f"Unknown source runs: {sorted(missing)}")
    targets: list[Target] = []
    for raw in manifest["targets"]:
        source_run = str(raw["source_run"])
        if source_runs and source_run not in source_runs:
            continue
        checkpoint = checkpoint_root / source_run / str(raw["step"])
        cx = int(raw["cx"])
        target = Target(
            source_run=source_run,
            checkpoint=checkpoint,
            model_size=str(raw.get("model_size", "275m")),
            variant=str(raw["variant"]),
            cx=cx,
            lr=str(raw["lr"]),
            global_batch_size=batch_sizes[cx],
            expert_parallel_size=int(raw.get("expert_parallel_size", 1)),
            expert_parallel_path=str(raw.get("expert_parallel_path", "sync_1d")),
            rank_microbatch_sequences=int(
                raw.get(
                    "rank_microbatch_sequences",
                    manifest["evaluation"]["rank_microbatch_sequences"],
                )
            ),
        )
        if target.variant not in {"integration_wide_gdn_ev1", "geometry_275m_gdn_ev2"}:
            raise ValueError(f"Unknown model variant for {source_run}: {target.variant}")
        if not target.checkpoint.is_dir():
            raise ValueError(f"Missing final checkpoint: {target.checkpoint}")
        targets.append(target)
    names = [target.source_run for target in targets]
    if len(names) != len(set(names)):
        raise ValueError("Manifest contains duplicate source runs")
    return targets


def env_value(name: str, value: str | int) -> dict[str, str]:
    return {"name": name, "value": str(value)}


def build_task(manifest: dict[str, Any], target: Target, *, source_repo: Path) -> dict[str, Any]:
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
            env_value("OLMOE3_HYBRID_RUN_NAME", target.eval_run),
            env_value("OLMOE3_HYBRID_MODEL_SIZE", target.model_size),
            env_value("OLMOE3_HYBRID_MODEL_VARIANT", target.variant),
            env_value("OLMOE3_HYBRID_SUBCOMMAND", "eval_checkpoints"),
            env_value("OLMOE3_HYBRID_EVAL_CHECKPOINT", target.checkpoint),
            env_value("OLMOE3_HYBRID_EVAL_TASK_SET", evaluation["task_set"]),
            env_value("OLMOE3_HYBRID_EVALS", "1"),
            env_value("OLMOE3_HYBRID_EVAL_ON_FINISH", "0"),
            env_value("OLMOE3_HYBRID_CHECKPOINTS", "0"),
            env_value("OLMOE3_HYBRID_USE_COMPILE", "0"),
            env_value("OLMOE3_HYBRID_WANDB", "1"),
            env_value("OLMOE3_HYBRID_SAVE_ROOT", evaluation["save_root"]),
            env_value("OLMOE3_HYBRID_LR", target.lr),
            env_value("OLMOE3_HYBRID_CHINCHILLA_MULTIPLE", target.cx),
            env_value("OLMOE3_HYBRID_GLOBAL_BATCH_SIZE", target.global_batch_size),
            env_value("OLMOE3_HYBRID_SEQUENCE_LENGTH", evaluation["sequence_length"]),
            env_value("OLMOE3_HYBRID_WORLD_SIZE", beaker["gpu_count"]),
            env_value("OLMOE3_HYBRID_EP_SIZE", target.expert_parallel_size),
            env_value("OLMOE3_HYBRID_EP_PATH", target.expert_parallel_path),
            env_value(
                "OLMOE3_HYBRID_RANK_MICROBATCH_SEQUENCES",
                target.rank_microbatch_sequences,
            ),
        ]
    )
    datasets = [
        {"mountPath": str(item["mount_path"]), "source": {"weka": str(item["weka"])}}
        for item in manifest.get("datasets", [])
    ]
    return {
        "name": target.task_name,
        "image": {"beaker": str(source["image"])},
        "command": ["/bin/bash", "-lc"],
        "arguments": [arguments],
        "envVars": env_vars,
        "datasets": datasets,
        "result": {"path": str(beaker.get("result_path", "/results"))},
        "resources": {
            "gpuCount": int(beaker["gpu_count"]),
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
    parser.add_argument("--source-run", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--experiment-name")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    targets = parse_targets(manifest, set(args.source_run) or None)
    source_repo = Path(os.environ.get("SOURCE_REPO", str(manifest["source"]["repo"]))).resolve()
    if not source_repo.is_dir():
        raise ValueError(f"Missing source repo: {source_repo}")

    spec = {
        "version": "v2",
        "description": f"{manifest['experiment']['description']} ({len(targets)} checkpoints)",
        "tasks": [build_task(manifest, target, source_repo=source_repo) for target in targets],
    }
    output = args.output or DEFAULT_GENERATED_DIR / f"{manifest_path.stem}.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        yaml.dump(spec, f, Dumper=NoAliasDumper, sort_keys=False)

    print("source run | variant | Cx | final checkpoint")
    for target in targets:
        print(
            f"{target.source_run} | {target.model_size} | {target.variant} | "
            f"{target.cx} | EP{target.expert_parallel_size} | {target.checkpoint.name}"
        )
    gpu_count = int(manifest["beaker"]["gpu_count"])
    print(f"\n{len(targets)} tasks, {gpu_count} GPUs/task, {len(targets) * gpu_count} GPUs total")
    print(f"Rendered: {output}")

    if not args.submit:
        print("Dry run only. Add --submit --experiment-name NAME to launch.")
        return
    if not args.experiment_name:
        parser.error("--experiment-name is required with --submit")
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
