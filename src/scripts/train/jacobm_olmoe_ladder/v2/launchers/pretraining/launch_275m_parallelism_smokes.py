#!/usr/bin/env python3
"""Validate, render, and optionally launch 275M parallelism throughput smokes."""

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
DEFAULT_MANIFEST = SCRIPT_DIR / "manifests" / "275m_rope_gated_parallelism_smokes.yaml"
DEFAULT_OUTPUT = SCRIPT_DIR / "generated" / "275m_rope_gated_parallelism_smokes.yaml"
DEFAULT_RECORD = SCRIPT_DIR / "generated" / "275m_rope_gated_parallelism_submissions.json"
ALLOWED_VARIANTS = {
    "geometry_275m_gdn_ev2_rope_gated",
    "geometry_275m_gdn2_ev2_rope_gated",
    "geometry_275m_swa_rope_gated",
}
EXPECTED_SEQUENCE_LENGTH = 8_192
ALLOWED_GLOBAL_BATCHES = {
    262_144,
    2_097_152,
    2_228_224,  # MB17 x 16 accumulation x 8,192 tokens.
    2_359_296,  # MB18 x 16 accumulation x 8,192 tokens.
    2_490_368,  # MB19 x 16 accumulation x 8,192 tokens.
    2_621_440,  # MB20 x 16 accumulation x 8,192 tokens.
    4_194_304,
}
ALLOWED_EP_PATHS = {"sync_1d", "rowwise_nvshmem"}
ALLOWED_BUCKET_CAPS_MB = {25, 100, 250, 500}


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, _data: Any) -> bool:
        return True


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as file:
        manifest = yaml.safe_load(file)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"{path}: expected schema_version: 1")
    return manifest


def validate(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    training = manifest["training"]
    if str(training["model_variant"]) not in ALLOWED_VARIANTS:
        raise ValueError(f"expected one of the audited model variants {sorted(ALLOWED_VARIANTS)}")
    if bool(training.get("gdn2_disable_recompute", False)) and str(
        training["model_variant"]
    ) != "geometry_275m_gdn2_ev2_rope_gated":
        raise ValueError("gdn2_disable_recompute is only valid for the GDN2 model variant")
    if int(training["sequence_length"]) != EXPECTED_SEQUENCE_LENGTH:
        raise ValueError("parallelism study must preserve the 8,192-token sequence length")
    if int(training["global_batch_size"]) not in ALLOWED_GLOBAL_BATCHES:
        raise ValueError(
            "parallelism study global batch must be one of "
            f"{sorted(ALLOWED_GLOBAL_BATCHES)} tokens"
        )
    if int(training["rank_microbatch_sequences"]) < 1:
        raise ValueError("rank microbatch must be positive")
    if bool(training.get("checkpoints", True)):
        raise ValueError("parallelism smokes must disable checkpoints")
    if bool(training.get("evals", True)):
        raise ValueError("parallelism smokes must disable evaluations")
    if int(training["hard_stop_steps"]) < 12:
        raise ValueError("parallelism smokes need at least 12 optimizer steps")

    rows = manifest["runs"]
    task_names: set[str] = set()
    run_names: set[str] = set()
    for row in rows:
        task_name = str(row["task_name"])
        run_name = str(row["run_name"])
        if task_name in task_names or run_name in run_names:
            raise ValueError(f"duplicate task/run name: {task_name} / {run_name}")
        task_names.add(task_name)
        run_names.add(run_name)

        world_size = int(row["gpu_count"])
        ep_size = int(row["expert_parallel_size"])
        use_code_defaults = bool(row.get("expert_parallel_use_code_defaults", False))
        ep_path_value = row.get("expert_parallel_path")
        ep_path = None if ep_path_value is None else str(ep_path_value)
        if world_size not in {1, 2, 4, 8}:
            raise ValueError(f"{task_name}: only 1-, 2-, 4-, and 8-GPU cells are permitted")
        if ep_size < 1 or world_size % ep_size:
            raise ValueError(f"{task_name}: EP={ep_size} must divide world={world_size}")
        if ep_path is not None and ep_path not in ALLOWED_EP_PATHS:
            raise ValueError(f"{task_name}: unsupported EP path {ep_path!r}")
        if use_code_defaults and ep_path is not None:
            raise ValueError(
                f"{task_name}: code-default EP must not specify expert_parallel_path"
            )
        if ep_size > 1 and ep_path is None and not use_code_defaults:
            raise ValueError(
                f"{task_name}: EP>1 requires a path or expert_parallel_use_code_defaults=true"
            )
        bucket_cap = row.get("data_parallel_bucket_cap_mb")
        if bucket_cap is not None and int(bucket_cap) not in ALLOWED_BUCKET_CAPS_MB:
            raise ValueError(
                f"{task_name}: unsupported DDP bucket cap {bucket_cap}; expected one of "
                f"{sorted(ALLOWED_BUCKET_CAPS_MB)} MiB"
            )
        global_batch_size = int(row.get("global_batch_size", training["global_batch_size"]))
        if global_batch_size not in ALLOWED_GLOBAL_BATCHES:
            raise ValueError(
                f"{task_name}: unsupported global batch {global_batch_size}; expected one of "
                f"{sorted(ALLOWED_GLOBAL_BATCHES)}"
            )
        global_sequences = global_batch_size // EXPECTED_SEQUENCE_LENGTH
        if global_sequences % world_size:
            raise ValueError(f"{task_name}: global sequence batch does not divide world size")
        rank_sequences = global_sequences // world_size
        rank_microbatch = int(
            row.get("rank_microbatch_sequences", training["rank_microbatch_sequences"])
        )
        if rank_microbatch < 1 or rank_microbatch > rank_sequences:
            raise ValueError(
                f"{task_name}: rank microbatch MB{rank_microbatch} exceeds rank batch "
                f"{rank_sequences}"
            )
        if rank_sequences % rank_microbatch:
            raise ValueError(
                f"{task_name}: rank batch {rank_sequences} does not divide MB{rank_microbatch}"
            )
        row["world_size"] = world_size
        row["resolved_ep_path"] = "code-default" if use_code_defaults else (ep_path or "n/a")
        row["global_batch_size"] = global_batch_size
        row["rank_sequences"] = rank_sequences
        row["effective_rank_microbatch"] = rank_microbatch
        row["accumulation_steps"] = rank_sequences // rank_microbatch

    return rows


def env_value(name: str, value: Any) -> dict[str, str]:
    return {"name": name, "value": str(value)}


def build_task(manifest: dict[str, Any], row: dict[str, Any], source_repo: str) -> dict[str, Any]:
    source = manifest["source"]
    beaker = manifest["beaker"]
    training = manifest["training"]
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
            env_value("OLMOE3_HYBRID_RUN_NAME", row["run_name"]),
            env_value("OLMOE3_HYBRID_MODEL_SIZE", training["model_size"]),
            env_value("OLMOE3_HYBRID_MODEL_VARIANT", training["model_variant"]),
            env_value("OLMOE3_HYBRID_LR", training["learning_rate"]),
            env_value("OLMOE3_HYBRID_CHINCHILLA_MULTIPLE", training["chinchilla_multiple"]),
            env_value("OLMOE3_HYBRID_GLOBAL_BATCH_SIZE", row["global_batch_size"]),
            env_value("OLMOE3_HYBRID_WORLD_SIZE", row["world_size"]),
            env_value("OLMOE3_HYBRID_NUM_NODES", 1),
            env_value("OLMOE3_HYBRID_EP_SIZE", row["expert_parallel_size"]),
            env_value(
                "OLMOE3_HYBRID_RANK_MICROBATCH_SEQUENCES",
                row["effective_rank_microbatch"],
            ),
            env_value("OLMOE3_HYBRID_SEQUENCE_LENGTH", training["sequence_length"]),
            env_value(
                "OLMOE3_HYBRID_HARD_STOP_STEPS",
                row.get("hard_stop_steps", training["hard_stop_steps"]),
            ),
            env_value("OLMOE3_HYBRID_CHECKPOINTS", int(bool(training["checkpoints"]))),
            env_value("OLMOE3_HYBRID_SAVE_INTERVAL", 999_999_999),
            env_value("OLMOE3_HYBRID_EPHEMERAL_SAVE_INTERVAL", 999_999_999),
            env_value("OLMOE3_HYBRID_CHECKPOINT_REMOVAL", "never"),
            env_value("OLMOE3_HYBRID_EVALS", int(bool(training["evals"]))),
            env_value("OLMOE3_HYBRID_EVAL_ON_FINISH", 0),
            env_value("OLMOE3_HYBRID_USE_COMPILE", int(bool(training["compile"]))),
            env_value("OLMOE3_HYBRID_WANDB", int(bool(training["wandb"]))),
            env_value("OLMOE3_HYBRID_SAVE_ROOT", manifest["experiment"]["checkpoint_root"]),
        ]
    )
    optional_env = {
        "OLMOE3_HYBRID_EP_PATH": row.get("expert_parallel_path"),
        "OLMOE3_HYBRID_EP_USE_CODE_DEFAULTS": int(
            bool(row.get("expert_parallel_use_code_defaults", False))
        ),
        "OLMOE3_HYBRID_EP_ROWWISE_GET_NBLOCKS": row.get("rowwise_get_nblocks"),
        "OLMOE3_HYBRID_EP_ROWWISE_PUT_NBLOCKS": row.get("rowwise_put_nblocks"),
        "OLMOE3_HYBRID_EP_ROWWISE_WEIGHTED_PUT_NBLOCKS": row.get(
            "rowwise_weighted_put_nblocks"
        ),
        "OLMOE3_HYBRID_DP_USE_REDUCE_SCATTER": int(
            bool(row.get("data_parallel_use_reduce_scatter", False))
        ),
        "OLMOE3_HYBRID_DP_BUCKET_CAP_MB": row.get("data_parallel_bucket_cap_mb"),
        "OLMOE3_HYBRID_GDN2_DISABLE_RECOMPUTE": (
            int(bool(training.get("gdn2_disable_recompute", False)))
            if "gdn2_disable_recompute" in training
            else None
        ),
    }
    env_vars.extend(
        env_value(name, value) for name, value in optional_env.items() if value is not None
    )
    datasets = [
        {"mountPath": str(item["mount_path"]), "source": {"weka": str(item["weka"])}}
        for item in manifest.get("datasets", [])
    ]
    return {
        "name": str(row["task_name"]),
        "image": {"beaker": str(source["image"])},
        "command": ["/bin/bash", "-lc"],
        "arguments": [arguments],
        "envVars": env_vars,
        "datasets": datasets,
        "result": {"path": str(beaker["result_path"])},
        "resources": {
            "gpuCount": int(row["gpu_count"]),
            "sharedMemory": str(beaker["shared_memory"]),
        },
        "context": {
            "priority": str(beaker["priority"]),
            "minRuntime": str(beaker["min_runtime"]),
            "autoResume": bool(beaker["auto_resume"]),
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
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--rank-microbatch-sequences", type=int)
    parser.add_argument("--hard-stop-steps", type=int)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument(
        "--experiment-name",
        default="jacobm-275m-rope-gated-parallelism-smokes-r1",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest.resolve())
    if args.rank_microbatch_sequences is not None:
        manifest["training"]["rank_microbatch_sequences"] = args.rank_microbatch_sequences
    if args.hard_stop_steps is not None:
        manifest["training"]["hard_stop_steps"] = args.hard_stop_steps
    rows = validate(manifest)
    if args.task:
        wanted = set(args.task)
        available = {str(row["task_name"]) for row in rows}
        missing = wanted - available
        if missing:
            raise ValueError(f"unknown selected tasks: {sorted(missing)}")
        rows = [row for row in rows if str(row["task_name"]) in wanted]

    source_repo = os.environ.get("SOURCE_REPO", str(manifest["source"]["repo"]))
    wrapper = Path(source_repo) / str(manifest["source"]["wrapper"])
    if not wrapper.is_file():
        raise ValueError(f"source wrapper is missing: {wrapper}")

    print("task                 batch GPU EP path              rank_seq MB accum run")
    for row in rows:
        print(
            f"{row['task_name']:<20} {row['global_batch_size']:>7} {row['gpu_count']:>3} "
            f"{row['expert_parallel_size']:>2} {row['resolved_ep_path']:<17} "
            f"{row['rank_sequences']:>8} {row['effective_rank_microbatch']:>2} "
            f"{row['accumulation_steps']:>5} {row['run_name']}"
        )
    print(f"\n{len(rows)} tasks; {sum(int(row['gpu_count']) for row in rows)} concurrent GPUs")

    spec = {
        "version": "v2",
        "description": f"{manifest['experiment']['description']} ({len(rows)} cells)",
        "tasks": [build_task(manifest, row, source_repo) for row in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as file:
        yaml.dump(spec, file, Dumper=NoAliasDumper, sort_keys=False)
    print(f"Rendered {args.output}")

    if not args.submit:
        print("Dry run only; pass --submit to launch")
        return
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
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    print(result.stdout, end="")
    print(result.stderr, end="")
    ids = re.findall(r"\b01[A-Z0-9]{24}\b", result.stdout + "\n" + result.stderr)
    if not ids:
        raise RuntimeError("Beaker created the experiment but its ID was not found in CLI output")
    experiment_id = ids[-1]
    record = {
        "experiment_name": args.experiment_name,
        "experiment_id": experiment_id,
        "tasks": [str(row["task_name"]) for row in rows],
        "url": (
            "https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/"
            f"{experiment_id}"
        ),
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    if args.record.is_file():
        records = json.loads(args.record.read_text())
    records.append(record)
    args.record.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Recorded submission in {args.record}")


if __name__ == "__main__":
    main()
