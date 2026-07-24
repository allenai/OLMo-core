#!/usr/bin/env python3
"""Validate, render, and optionally submit manifest-defined hybrid scale runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "manifests" / "hybrid_scale_mb_smokes.yaml"
DEFAULT_OUTPUT = SCRIPT_DIR / "generated" / "hybrid_scale_mb_smokes.yaml"


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
    sequence_length = int(manifest["training"]["sequence_length"])
    rows = manifest.get("runs", manifest.get("smokes"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must contain at least one run")
    removal = str(manifest["training"].get("checkpoint_removal", "never"))
    if removal not in {"never", "ephemeral_only", "all_non_permanent"}:
        raise ValueError(f"Unsupported checkpoint_removal={removal!r}")
    task_names: set[str] = set()
    run_names: set[str] = set()
    for row in rows:
        task_name = str(row["task_name"])
        run_name = str(row["run_name"])
        if task_name in task_names or run_name in run_names:
            raise ValueError(f"Duplicate task or run name: {task_name} / {run_name}")
        task_names.add(task_name)
        run_names.add(run_name)
        world_size = int(row["gpu_count"])
        ep_size = int(row["expert_parallel_size"])
        ep_path = str(row.get("expert_parallel_path", "rowwise_nvshmem"))
        if ep_path not in {"sync_1d", "rowwise_nvshmem"}:
            raise ValueError(f"{task_name}: unsupported expert_parallel_path={ep_path!r}")
        if world_size % ep_size:
            raise ValueError(f"{task_name}: EP={ep_size} must divide world={world_size}")
        global_batch = int(row["global_batch_size"])
        if global_batch % sequence_length:
            raise ValueError(f"{task_name}: global batch is not a whole number of sequences")
        global_sequences = global_batch // sequence_length
        data_dp_degree = world_size
        if global_sequences % data_dp_degree:
            raise ValueError(
                f"{task_name}: {global_sequences} sequences is not divisible by "
                f"the data-parallel world size ({data_dp_degree})"
            )
        rank_sequences = global_sequences // data_dp_degree
        rank_microbatch_cap = int(row["rank_microbatch_sequences"])
        effective_rank_microbatch = min(rank_sequences, rank_microbatch_cap)
        if rank_sequences % effective_rank_microbatch:
            raise ValueError(
                f"{task_name}: rank batch ({rank_sequences} sequences) is not divisible by "
                f"the effective rank microbatch ({effective_rank_microbatch} sequences)"
            )
        row["data_dp_degree"] = data_dp_degree
        row["ep_dp_degree"] = world_size // ep_size
        row["rank_sequences"] = rank_sequences
        row["effective_rank_microbatch_sequences"] = effective_rank_microbatch
        row["accumulation_steps"] = rank_sequences // effective_rank_microbatch
    return rows


def build_task(
    manifest: dict[str, Any],
    row: dict[str, Any],
    *,
    source_repo: Path,
    debug_gradients: bool = False,
) -> dict[str, Any]:
    source = manifest["source"]
    beaker = manifest["beaker"]
    training = manifest["training"]
    arguments = (
        "set -euo pipefail\n"
        f"SOURCE_REPO={shlex.quote(str(source_repo))}\n"
        "WRAPPER=/tmp/jacobm_olmoe3_hybrid_scale_beaker.sh\n"
        f'cp "${{SOURCE_REPO}}/{source["wrapper"]}" "${{WRAPPER}}"\n'
        'bash "${WRAPPER}"\n'
    )
    env_vars = [
        {"name": str(name), "secret": str(secret)}
        for name, secret in manifest.get("secrets", {}).items()
    ]
    env_vars.extend(
        [
            env_var("OLMOE3_HYBRID_RUN_NAME", row["run_name"]),
            env_var("OLMOE3_HYBRID_MODEL_SIZE", row["model_size"]),
            env_var(
                "OLMOE3_HYBRID_MODEL_VARIANT",
                row.get(
                    "model_variant",
                    training.get("model_variant", "integration_wide_gdn_ev1"),
                ),
            ),
            env_var("OLMOE3_HYBRID_LR", row["learning_rate"]),
            env_var("OLMOE3_HYBRID_CHINCHILLA_MULTIPLE", row["cx"]),
            env_var("OLMOE3_HYBRID_GLOBAL_BATCH_SIZE", row["global_batch_size"]),
            env_var("OLMOE3_HYBRID_WORLD_SIZE", row["gpu_count"]),
            env_var("OLMOE3_HYBRID_EP_SIZE", row["expert_parallel_size"]),
            env_var(
                "OLMOE3_HYBRID_EP_PATH",
                row.get("expert_parallel_path", "rowwise_nvshmem"),
            ),
            env_var(
                "OLMOE3_HYBRID_RANK_MICROBATCH_SEQUENCES",
                row["rank_microbatch_sequences"],
            ),
            env_var("OLMOE3_HYBRID_SEQUENCE_LENGTH", training["sequence_length"]),
            env_var("OLMOE3_HYBRID_HARD_STOP_STEPS", training["hard_stop_steps"]),
            env_var("OLMOE3_HYBRID_SAVE_INTERVAL", training["save_interval"]),
            env_var(
                "OLMOE3_HYBRID_EPHEMERAL_SAVE_INTERVAL",
                training["ephemeral_save_interval"],
            ),
            env_var(
                "OLMOE3_HYBRID_CHECKPOINT_REMOVAL",
                training.get("checkpoint_removal", "never"),
            ),
            env_var("OLMOE3_HYBRID_EVAL_INTERVAL", training["eval_interval"]),
            env_var("OLMOE3_HYBRID_EVAL_STEPS", training["eval_steps"]),
            env_var(
                "OLMOE3_HYBRID_EVAL_TASK_SET",
                training.get("eval_task_set", "hellaswag"),
            ),
            env_var(
                "OLMOE3_HYBRID_EVAL_ON_FINISH",
                int(bool(training.get("eval_on_finish", False))),
            ),
            env_var("OLMOE3_HYBRID_USE_COMPILE", int(bool(training["compile"]))),
            env_var("OLMOE3_HYBRID_WANDB", int(bool(training["wandb"]))),
            env_var(
                "OLMOE3_HYBRID_CHECKPOINTS",
                int(bool(training.get("checkpoints", True))),
            ),
            env_var(
                "OLMOE3_HYBRID_EVALS",
                int(bool(training.get("evals", False))),
            ),
            env_var("OLMOE3_HYBRID_SAVE_ROOT", manifest["experiment"]["checkpoint_root"]),
        ]
    )
    if debug_gradients:
        env_vars.extend(
            [
                env_var("OLMO_DDP_DEBUG_NONFINITE_GRAD", 1),
                env_var("OLMO_DDP_DEBUG_NONFINITE_GRAD_RANKS", "all"),
                env_var("OLMO_DDP_DEBUG_NONFINITE_GRAD_TOPK", 50),
                env_var("OLMO_DDP_DEBUG_GRAD_NORMS", 20),
                env_var("OLMO_DDP_DEBUG_GRAD_NORMS_RANKS", "all"),
                env_var("OLMO_DDP_DEBUG_GRAD_NORMS_MIN", 100),
            ]
        )
    datasets = [
        {"mountPath": item["mount_path"], "source": {"weka": item["weka"]}}
        for item in manifest.get("datasets", [])
    ]
    return {
        "name": (
            f"{row['task_name']}-grad-debug-r1" if debug_gradients else row["task_name"]
        ),
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
            "autoResume": bool(beaker.get("auto_resume", False)),
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
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Render one exact task_name; repeat to select multiple tasks",
    )
    parser.add_argument("--experiment-name")
    parser.add_argument(
        "--record",
        type=Path,
        help="Append the submitted Beaker experiment and task names to this JSON ledger",
    )
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Allow submission when selected run checkpoint directories already exist",
    )
    parser.add_argument(
        "--debug-gradients",
        action="store_true",
        help="Enable all-rank non-finite and large-gradient diagnostics on a resume.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest.resolve())
    source_repo = Path(os.environ.get("SOURCE_REPO", manifest["source"]["repo"])).resolve()
    rows = validate(manifest, source_repo)
    if args.task:
        wanted = set(args.task)
        available = {str(row["task_name"]) for row in rows}
        missing = wanted - available
        if missing:
            raise ValueError(f"Unknown --task values: {sorted(missing)}")
        rows = [row for row in rows if str(row["task_name"]) in wanted]
    spec = {
        "version": "v2",
        "description": (
            f"{manifest['experiment']['description']} ({len(rows)} tasks)"
            + (" (gradient-debug resume)" if args.debug_gradients else "")
        ),
        "tasks": [
            build_task(
                manifest,
                row,
                source_repo=source_repo,
                debug_gradients=args.debug_gradients,
            )
            for row in rows
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        yaml.dump(spec, file, Dumper=NoAliasDumper, sort_keys=False)

    print(
        "size  Cx  GPU  EP  EP path           data DP  EP DP  rank seq  "
        "MB cap  effective  accum  run name"
    )
    for row in rows:
        print(
            f"{row['model_size']:<5} {row['cx']:>2} {row['gpu_count']:>4} "
            f"{row['expert_parallel_size']:>3} "
            f"{row.get('expert_parallel_path', '-'):>17} {row['data_dp_degree']:>7} "
            f"{row['ep_dp_degree']:>6} {row['rank_sequences']:>9} "
            f"{row['rank_microbatch_sequences']:>7} "
            f"{row['effective_rank_microbatch_sequences']:>10} "
            f"{row['accumulation_steps']:>6}  "
            f"{row['run_name']}"
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
    if args.debug_gradients and not args.resume_existing:
        parser.error("--debug-gradients requires --resume-existing")
    root = Path(str(manifest["experiment"]["checkpoint_root"]))
    existing = [
        root / str(row["run_name"]) for row in rows if (root / str(row["run_name"])).exists()
    ]
    if existing and not args.resume_existing:
        raise RuntimeError(
            "Refusing to submit existing checkpoint directories:\n"
            + "\n".join(f"  - {path}" for path in existing)
            + "\nPass --resume-existing to launch the same run names and resume them."
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
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print(result.stdout, end="")
    print(result.stderr, end="")
    ids = re.findall(r"\b01[A-Z0-9]{24}\b", result.stdout + "\n" + result.stderr)
    if not ids:
        raise RuntimeError("Beaker created the experiment but its ID was not found in CLI output")
    experiment_id = ids[-1]
    if args.record is not None:
        workspace = str(manifest["beaker"]["workspace"])
        organization, workspace_name = workspace.split("/", maxsplit=1)
        record = {
            "experiment_name": args.experiment_name,
            "experiment_id": experiment_id,
            "tasks": [str(row["task_name"]) for row in rows],
            "debug_gradients": args.debug_gradients,
            "url": (
                f"https://beaker.org/orgs/{organization}/workspaces/{workspace_name}/work/"
                f"{experiment_id}"
            ),
        }
        args.record.parent.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        if args.record.is_file():
            records = json.loads(args.record.read_text())
        if any(item.get("experiment_id") == experiment_id for item in records):
            raise RuntimeError(f"Duplicate experiment ID in submission ledger: {experiment_id}")
        records.append(record)
        args.record.write_text(json.dumps(records, indent=2) + "\n")
        print(f"Recorded submission in {args.record}")


if __name__ == "__main__":
    main()
