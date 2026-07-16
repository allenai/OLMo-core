#!/usr/bin/env python3
"""Validate, render, and optionally submit a v2 pretraining LR sweep."""

from __future__ import annotations

import argparse
import math
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
REQUIRED_ENV_NAMES = {
    "run_name",
    "learning_rate",
    "chinchilla_multiple",
    "global_batch_size",
    "world_size",
    "rank_microbatch_sequences",
    "sequence_length",
}


class NoAliasDumper(yaml.SafeDumper):
    """Keep rendered Beaker tasks explicit and independently editable."""

    def ignore_aliases(self, _data: Any) -> bool:
        return True


@dataclass(frozen=True)
class SweepPoint:
    cx: int
    lr: str
    global_batch_size: int
    rank_microbatch_sequences: int
    chinchilla_multiple: str
    accumulation_steps: int
    run_name: str
    task_name: str


def find_repo_root() -> Path:
    for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents):
        if (parent / "src" / "olmo_core").is_dir():
            return parent
    raise RuntimeError("Could not find the OLMo-core repository root")


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as f:
        manifest = yaml.safe_load(f)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"{path}: expected schema_version: 1")
    for key in ("experiment", "source", "beaker", "training", "sweep"):
        if not isinstance(manifest.get(key), dict):
            raise ValueError(f"{path}: missing mapping '{key}'")
    return manifest


def lr_number(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"Invalid learning rate: {value!r}")
    return result


def lr_tag(value: str) -> str:
    normalized = value.strip().lower().replace("e+", "e")
    if "e" not in normalized:
        raise ValueError(f"LRs must use scientific notation for stable run names: {value!r}")
    return normalized.replace(".", "p")


def all_points(manifest: dict[str, Any]) -> list[SweepPoint]:
    experiment = manifest["experiment"]
    training = manifest["training"]
    sequence_length = int(training["sequence_length"])
    world_size = int(training["world_size"])
    replica = int(experiment.get("replica", 1))
    run_prefix = str(experiment["run_prefix"])

    if int(training.get("expert_parallel_size", 1)) != 1:
        raise ValueError("v2 intervention sweeps must default to expert_parallel_size=1")
    if sequence_length <= 0 or world_size <= 0:
        raise ValueError("sequence_length and world_size must be positive")
    if world_size != int(manifest["beaker"]["gpu_count"]):
        raise ValueError("The single-node DDP launcher requires world_size == beaker.gpu_count")

    points: list[SweepPoint] = []
    for cx_raw, settings in sorted(manifest["sweep"].items(), key=lambda item: int(item[0])):
        cx = int(cx_raw)
        if cx not in {1, 2, 4, 8}:
            raise ValueError(f"Unsupported canonical Cx: {cx}")
        global_batch_size = int(settings["global_batch_size"])
        rank_microbatch_sequences = int(settings["rank_microbatch_sequences"])
        denominator = sequence_length * world_size * rank_microbatch_sequences
        if global_batch_size % denominator:
            raise ValueError(
                f"Cx{cx}: global batch {global_batch_size} is not divisible by "
                f"sequence_length*world_size*rank_microbatch_sequences={denominator}"
            )
        accumulation_steps = global_batch_size // denominator
        if accumulation_steps < 1:
            raise ValueError(f"Cx{cx}: accumulation must be at least one")

        for lr in settings["lrs"]:
            lr = str(lr)
            lr_number(lr)
            tag = lr_tag(lr)
            points.append(
                SweepPoint(
                    cx=cx,
                    lr=lr,
                    global_batch_size=global_batch_size,
                    rank_microbatch_sequences=rank_microbatch_sequences,
                    chinchilla_multiple=str(settings.get("chinchilla_multiple", cx)),
                    accumulation_steps=accumulation_steps,
                    run_name=f"{run_prefix}-cx{cx}-lr{tag}-r{replica}",
                    task_name=f"cx{cx}-lr{tag}",
                )
            )

    run_names = [point.run_name for point in points]
    if len(run_names) != len(set(run_names)):
        raise ValueError("Manifest generates duplicate run names")
    return points


def parse_exact_points(values: list[str]) -> set[tuple[int, float]]:
    selected: set[tuple[int, float]] = set()
    for value in values:
        try:
            cx_text, lr_text = value.split(":", 1)
            selected.add((int(cx_text), lr_number(lr_text)))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid --point {value!r}; expected CX:LR, e.g. 4:3.2e-3") from exc
    return selected


def select_points(points: list[SweepPoint], args: argparse.Namespace) -> list[SweepPoint]:
    if args.point and (args.cx or args.lr):
        raise ValueError("--point cannot be combined with --cx or --lr")

    if args.point:
        wanted = parse_exact_points(args.point)
        available = {(point.cx, lr_number(point.lr)) for point in points}
        missing = wanted - available
        if missing:
            raise ValueError(f"Requested points are not in the manifest: {sorted(missing)}")
        return [point for point in points if (point.cx, lr_number(point.lr)) in wanted]

    selected_cxs = set(args.cx or [])
    selected_lrs = {lr_number(lr) for lr in (args.lr or [])}
    selected = [
        point
        for point in points
        if (not selected_cxs or point.cx in selected_cxs)
        and (not selected_lrs or lr_number(point.lr) in selected_lrs)
    ]
    if not selected:
        raise ValueError("Filters selected no sweep points")
    return selected


def env_var(name: str, value: str) -> dict[str, str]:
    return {"name": name, "value": str(value)}


def build_task(
    manifest: dict[str, Any],
    point: SweepPoint,
    *,
    source_repo: str,
    priority: str,
    cluster: str,
) -> dict[str, Any]:
    source = manifest["source"]
    beaker = manifest["beaker"]
    training = manifest["training"]
    env_names = training["env_names"]
    missing_env_names = REQUIRED_ENV_NAMES - set(env_names)
    if missing_env_names:
        raise ValueError(f"training.env_names is missing: {sorted(missing_env_names)}")

    wrapper = str(source["wrapper"])
    arguments = (
        "set -euo pipefail\n"
        f"SOURCE_REPO={shlex.quote(source_repo)}\n"
        f'bash "${{SOURCE_REPO}}/{wrapper}"\n'
    )
    env_vars = [
        {"name": str(name), "secret": str(secret)}
        for name, secret in manifest.get("secrets", {}).items()
    ]
    env_vars.extend(
        [
            env_var(env_names["run_name"], point.run_name),
            env_var(env_names["learning_rate"], point.lr),
            env_var(env_names["chinchilla_multiple"], point.chinchilla_multiple),
            env_var(env_names["global_batch_size"], point.global_batch_size),
            env_var(env_names["world_size"], training["world_size"]),
            env_var(env_names["rank_microbatch_sequences"], point.rank_microbatch_sequences),
            env_var(env_names["sequence_length"], training["sequence_length"]),
        ]
    )
    env_vars.extend(
        env_var(str(name), str(value)) for name, value in training.get("static_env", {}).items()
    )
    env_var_names = [item["name"] for item in env_vars]
    if len(env_var_names) != len(set(env_var_names)):
        raise ValueError(f"Duplicate environment variables in rendered task {point.task_name}")

    datasets = [
        {"mountPath": str(dataset["mount_path"]), "source": {"weka": str(dataset["weka"])}}
        for dataset in manifest.get("datasets", [])
    ]
    return {
        "name": point.task_name,
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
            "priority": priority,
            "minRuntime": str(beaker["min_runtime"]),
            "autoResume": bool(beaker.get("auto_resume", False)),
        },
        "constraints": {"cluster": [cluster]},
        "hostNetworking": True,
        "propagateFailure": False,
        "propagatePreemption": False,
        "timeout": str(beaker["timeout"]),
    }


def render_spec(
    manifest: dict[str, Any],
    points: list[SweepPoint],
    *,
    source_repo: str,
    priority: str,
    cluster: str,
) -> dict[str, Any]:
    return {
        "version": "v2",
        "description": f"{manifest['experiment']['description']} ({len(points)} sweep points)",
        "tasks": [
            build_task(
                manifest,
                point,
                source_repo=source_repo,
                priority=priority,
                cluster=cluster,
            )
            for point in points
        ],
    }


def write_spec(spec: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        yaml.dump(spec, f, Dumper=NoAliasDumper, sort_keys=False)


def print_summary(points: list[SweepPoint], manifest: dict[str, Any]) -> None:
    sequence_length = int(manifest["training"]["sequence_length"])
    world_size = int(manifest["training"]["world_size"])
    print("Cx  LR       global tokens  global seq  rank mb seq  accum  run name")
    for point in points:
        print(
            f"{point.cx:<3} {point.lr:<8} {point.global_batch_size:>13,} "
            f"{point.global_batch_size // sequence_length:>11} "
            f"{point.rank_microbatch_sequences:>12} {point.accumulation_steps:>6}  {point.run_name}"
        )
    print(
        f"\n{len(points)} tasks, {int(manifest['beaker']['gpu_count'])} GPUs/task, "
        f"{len(points) * int(manifest['beaker']['gpu_count'])} GPUs if all run concurrently, "
        f"world_size={world_size}"
    )


def existing_checkpoint_dirs(manifest: dict[str, Any], points: list[SweepPoint]) -> list[Path]:
    checkpoint_root = manifest["experiment"].get("checkpoint_root")
    if checkpoint_root is None:
        return []
    root = Path(str(checkpoint_root)).expanduser()
    return [root / point.run_name for point in points if (root / point.run_name).exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--point", action="append", default=[], help="Exact CX:LR point; repeat as needed"
    )
    parser.add_argument("--cx", type=int, nargs="+", help="Select complete Cx grids")
    parser.add_argument("--lr", nargs="+", help="Select matching LRs from selected/all Cx grids")
    parser.add_argument("--output", type=Path, help="Rendered Beaker YAML path")
    parser.add_argument("--priority", choices=("low", "normal", "high", "urgent", "immediate"))
    parser.add_argument("--cluster")
    parser.add_argument("--workspace")
    parser.add_argument("--experiment-name", help="Required with --submit")
    parser.add_argument(
        "--submit", action="store_true", help="Create the rendered Beaker experiment"
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Submit an intentional retry/resume when a selected run already has a checkpoint directory",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    points = select_points(all_points(manifest), args)
    source_repo = os.environ.get(
        "SOURCE_REPO", str(manifest["source"].get("repo", find_repo_root()))
    )
    source_repo_path = Path(source_repo)
    wrapper_path = source_repo_path / str(manifest["source"]["wrapper"])
    if not source_repo_path.is_dir() or not wrapper_path.is_file():
        raise ValueError(f"Source repo or wrapper is missing: {wrapper_path}")

    beaker = manifest["beaker"]
    priority = args.priority or str(beaker["priority"])
    cluster = args.cluster or str(beaker["cluster"])
    workspace = args.workspace or str(beaker["workspace"])
    output_path = args.output or DEFAULT_GENERATED_DIR / f"{manifest_path.stem}.yaml"
    spec = render_spec(
        manifest,
        points,
        source_repo=source_repo,
        priority=priority,
        cluster=cluster,
    )
    write_spec(spec, output_path)
    print_summary(points, manifest)
    print(f"\nRendered: {output_path}")
    print(f"Destination: workspace={workspace}, cluster={cluster}, priority={priority}")

    if not args.submit:
        print("Dry run only. Add --submit --experiment-name NAME to launch.")
        return
    if not args.experiment_name:
        parser.error("--experiment-name is required with --submit")

    existing = existing_checkpoint_dirs(manifest, points)
    if existing and not args.resume_existing:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise RuntimeError(
            "Refusing to submit run names with existing checkpoint directories:\n"
            f"{formatted}\nUse a new run name, resume the existing Beaker task, or explicitly pass "
            "--resume-existing."
        )
    if shutil.which("beaker") is None:
        raise RuntimeError("beaker CLI is not available")

    command = [
        "beaker",
        "experiment",
        "create",
        str(output_path),
        "--workspace",
        workspace,
        "--name",
        args.experiment_name,
    ]
    print(f"Submitting: {' '.join(command)}")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
