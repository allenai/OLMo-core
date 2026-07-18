#!/usr/bin/env python3
"""Validate and launch checkpoint-free larger-geometry NoPE smokes."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml
from gantry.api import GitRepoState, Recipe

from scripts.train.jacobm_olmoe_ladder.v2.models.geometry_matched_scale import (
    GEOMETRIES,
    build_geometry_matched_scale_model_config,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "manifests" / "geometry_matched_scale_nope_smokes.yaml"
DEFAULT_RECORD = SCRIPT_DIR / "generated" / "geometry_matched_scale_nope_smoke_submissions.json"


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open() as file:
        manifest = yaml.safe_load(file)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"{path}: expected schema_version: 1")
    return manifest


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def validate_remote_commit(remote: str, branch: str) -> str:
    commit = git_output("rev-parse", "HEAD")
    remote_head = subprocess.check_output(
        ["git", "ls-remote", remote, f"refs/heads/{branch}"],
        text=True,
    ).split()
    if not remote_head or remote_head[0] != commit:
        found = remote_head[0] if remote_head else "missing"
        raise RuntimeError(
            f"Local HEAD {commit} is not pushed at {remote} {branch} (found {found})"
        )
    return commit


def validate(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    training = manifest["training"]
    if bool(training.get("checkpoints", True)):
        raise ValueError("Smoke manifest must set training.checkpoints: false")
    if bool(training.get("evals", True)):
        raise ValueError("Smoke manifest must set training.evals: false")
    if int(training["hard_stop_steps"]) <= 0:
        raise ValueError("Smoke manifest must set a positive hard_stop_steps")
    if training["model_variant"] != "geometry_matched_gdn_ev2_nope":
        raise ValueError("Smoke manifest must select the larger-geometry NoPE variant")

    sequence_length = int(training["sequence_length"])
    task_names: set[str] = set()
    run_names: set[str] = set()
    rows = manifest["runs"]
    for row in rows:
        task_name = str(row["task_name"])
        run_name = str(row["run_name"])
        if task_name in task_names or run_name in run_names:
            raise ValueError(f"Duplicate task/run name: {task_name} / {run_name}")
        task_names.add(task_name)
        run_names.add(run_name)

        num_nodes = int(row["num_nodes"])
        gpus_per_node = int(row["gpus_per_node"])
        if num_nodes < 1 or gpus_per_node < 1 or gpus_per_node > 8:
            raise ValueError(f"{task_name}: invalid node/GPU shape")
        if num_nodes > 1 and gpus_per_node != 8:
            raise ValueError(f"{task_name}: multi-node smokes must use full B300 nodes")
        world_size = num_nodes * gpus_per_node
        ep_size = int(row["expert_parallel_size"])
        if world_size % ep_size:
            raise ValueError(f"{task_name}: EP={ep_size} does not divide world={world_size}")

        global_batch = int(row["global_batch_size"])
        if global_batch % sequence_length:
            raise ValueError(f"{task_name}: global batch is not whole sequences")
        global_sequences = global_batch // sequence_length
        if global_sequences % world_size:
            raise ValueError(
                f"{task_name}: {global_sequences} sequences does not divide world={world_size}"
            )
        rank_sequences = global_sequences // world_size
        microbatch = min(rank_sequences, int(row["rank_microbatch_sequences"]))
        if rank_sequences % microbatch:
            raise ValueError(
                f"{task_name}: rank batch {rank_sequences} does not divide MB={microbatch}"
            )
        row["world_size"] = world_size
        row["rank_sequences"] = rank_sequences
        row["effective_microbatch"] = microbatch
        row["accumulation_steps"] = rank_sequences // microbatch

    for model_size in sorted({str(row["model_size"]) for row in rows}):
        model = build_geometry_matched_scale_model_config(model_size, rope=False)
        expected = GEOMETRIES[model_size]
        counts = (
            model.num_active_params,
            model.num_active_non_embedding_params,
            model.num_params,
        )
        expected_counts = (
            expected.expected_active_params,
            expected.expected_active_non_embedding_params,
            expected.expected_total_params,
        )
        if counts != expected_counts:
            raise ValueError(
                f"{model_size}: expected parameter counts {expected_counts}, found {counts}"
            )
        print(
            f"{model_size}: active={counts[0]:,} "
            f"active_non_embedding={counts[1]:,} total={counts[2]:,}"
        )
    return rows


def recipe_for(
    manifest: dict[str, Any],
    row: dict[str, Any],
    *,
    commit: str,
) -> Recipe:
    source = manifest["source"]
    beaker = manifest["beaker"]
    training = manifest["training"]
    env_vars = [
        ("PYTHONPATH", "src"),
        ("PYTHONUNBUFFERED", "1"),
        ("CUDA_SCALE_LAUNCH_QUEUES", "4x"),
        ("OLMO_SHARED_FS", "1"),
        ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
        ("OMP_NUM_THREADS", "8"),
        ("NCCL_DEBUG", "INFO"),
        ("LOG_FILTER_TYPE", "local_rank0_only"),
        ("OLMOE3_HYBRID_RUN_NAME", str(row["run_name"])),
        ("OLMOE3_HYBRID_MODEL_SIZE", str(row["model_size"])),
        ("OLMOE3_HYBRID_MODEL_VARIANT", str(training["model_variant"])),
        ("OLMOE3_HYBRID_LR", str(row["learning_rate"])),
        ("OLMOE3_HYBRID_CHINCHILLA_MULTIPLE", str(row["cx"])),
        ("OLMOE3_HYBRID_GLOBAL_BATCH_SIZE", str(row["global_batch_size"])),
        ("OLMOE3_HYBRID_WORLD_SIZE", str(row["world_size"])),
        ("OLMOE3_HYBRID_NUM_NODES", str(row["num_nodes"])),
        ("OLMOE3_HYBRID_EP_SIZE", str(row["expert_parallel_size"])),
        (
            "OLMOE3_HYBRID_EP_PATH",
            str(row.get("expert_parallel_path", "rowwise_nvshmem")),
        ),
        (
            "OLMOE3_HYBRID_RANK_MICROBATCH_SEQUENCES",
            str(row["rank_microbatch_sequences"]),
        ),
        ("OLMOE3_HYBRID_SEQUENCE_LENGTH", str(training["sequence_length"])),
        ("OLMOE3_HYBRID_HARD_STOP_STEPS", str(training["hard_stop_steps"])),
        ("OLMOE3_HYBRID_CHECKPOINTS", "0"),
        ("OLMOE3_HYBRID_EVALS", "0"),
        ("OLMOE3_HYBRID_EVAL_ON_FINISH", "0"),
        ("OLMOE3_HYBRID_USE_COMPILE", str(int(bool(training["compile"])))),
        ("OLMOE3_HYBRID_WANDB", str(int(bool(training["wandb"])))),
        ("OLMOE3_HYBRID_SAVE_ROOT", str(manifest["experiment"]["checkpoint_root"])),
    ]
    env_secrets = [(str(name), str(secret)) for name, secret in manifest.get("secrets", {}).items()]
    weka = [(str(item["bucket"]), str(item["mount"])) for item in manifest.get("weka", [])]
    # Resolve file membership from the local immutable Git tree. The generated
    # Beaker entrypoint still clones source["remote"] at this exact pushed SHA,
    # while avoiding a dependency on local GitHub CLI authentication.
    git_repo = GitRepoState.from_env(
        ref=commit,
        branch=str(source["branch"]),
    )
    return Recipe(
        args=[
            "src/scripts/train/jacobm_olmoe3_hybrid_scale.py",
            "train",
            str(row["run_name"]),
            "local",
        ],
        name=str(row["run_name"]),
        description=str(manifest["experiment"]["description"]),
        workspace=str(beaker["workspace"]),
        task_name=str(row["task_name"]),
        git_repo=git_repo,
        allow_dirty=False,
        yes=True,
        clusters=[str(beaker["cluster"])],
        gpus=int(row["gpus_per_node"]),
        shared_memory=str(beaker["shared_memory"]),
        beaker_image=str(source["image"]),
        env_vars=env_vars,
        env_secrets=env_secrets,
        weka=weka,
        priority=str(beaker["priority"]),
        min_runtime=str(beaker["min_runtime"]),
        preemptible=bool(beaker["preemptible"]),
        auto_resume=bool(beaker["auto_resume"]),
        task_timeout=str(beaker["timeout"]),
        replicas=int(row["num_nodes"]) if int(row["num_nodes"]) > 1 else None,
        leader_selection=int(row["num_nodes"]) > 1,
        host_networking=True,
        propagate_failure=True if int(row["num_nodes"]) > 1 else None,
        propagate_preemption=True if int(row["num_nodes"]) > 1 else None,
        synchronized_start_timeout="90m" if int(row["num_nodes"]) > 1 else None,
        torchrun=True,
        # The image is already the tested training environment. Do not let
        # Gantry install this checkout into it, which can replace CUDA Python
        # packages underneath the image's prebuilt TransformerEngine binary.
        no_python=True,
        pre_setup="unset S3_PROFILE",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--wave", type=int, choices=(1, 2))
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest.resolve())
    rows = validate(manifest)
    if args.wave is not None:
        rows = [row for row in rows if int(row["wave"]) == args.wave]
    if args.task:
        wanted = set(args.task)
        found = {str(row["task_name"]) for row in rows}
        missing = wanted - found
        if missing:
            raise ValueError(f"Unknown selected tasks: {sorted(missing)}")
        rows = [row for row in rows if str(row["task_name"]) in wanted]

    print("\nwave size  Cx nodes GPU/node world EP rank_seq MB accum run")
    for row in rows:
        print(
            f"{row['wave']:>4} {row['model_size']:<5} {row['cx']:>2} "
            f"{row['num_nodes']:>5} {row['gpus_per_node']:>8} {row['world_size']:>5} "
            f"{row['expert_parallel_size']:>2} {row['rank_sequences']:>8} "
            f"{row['effective_microbatch']:>2} {row['accumulation_steps']:>5} "
            f"{row['run_name']}"
        )
    print(f"\nSelected {len(rows)} runs using {sum(row['world_size'] for row in rows)} GPUs.")
    if not args.submit:
        print("Dry run only; pass --submit to launch.")
        return

    source = manifest["source"]
    commit = validate_remote_commit(str(source["remote"]), str(source["branch"]))
    records: list[dict[str, Any]] = []
    for row in rows:
        workload = recipe_for(manifest, row, commit=commit).launch(show_logs=False)
        experiment = workload.experiment
        record = {
            "task_name": row["task_name"],
            "run_name": row["run_name"],
            "commit": commit,
            "experiment_id": experiment.id,
            "task_ids": [task.id for task in experiment.tasks],
            "url": (
                "https://beaker.org/orgs/ai2/workspaces/"
                f"OLMo-3-moe-experiments/work/{experiment.id}"
            ),
        }
        records.append(record)
        print(f"Submitted {row['task_name']}: {record['url']}")

    args.record.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if args.record.is_file():
        existing = json.loads(args.record.read_text())
    existing.extend(records)
    args.record.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"Recorded {len(records)} submissions in {args.record}")


if __name__ == "__main__":
    main()
