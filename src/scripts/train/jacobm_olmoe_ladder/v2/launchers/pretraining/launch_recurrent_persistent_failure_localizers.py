#!/usr/bin/env python3
"""Replay representative persistent GDN1/GDN2 failures with exact localization."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.train.jacobm_olmoe_ladder.v2.launchers.pretraining.launch_geometry_matched_scale_full import (
    recipe_for,
    validate,
)
from scripts.train.jacobm_olmoe_ladder.v2.launchers.pretraining.launch_geometry_matched_scale_nope_smokes import (
    load_manifest,
    validate_remote_commit,
)


SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST_DIR = SCRIPT_DIR / "manifests"
RECORD = SCRIPT_DIR / "generated" / "recurrent_persistent_failure_localizers.json"
DUMP_ROOT = Path(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/"
    "olmo-ddp/debug/recurrent-persistent-failure-localizers"
)
REMOTE = "https://github.com/allenai/OLMo-core"
BRANCH = "jacobm/moe-v2-core-gdn2"


@dataclass(frozen=True)
class Case:
    name: str
    manifest: str
    task_name: str
    checkpoint: str
    failure_step: int
    start_step: int
    stop_step: int
    kind: str
    small_manifest: bool = False


CASES = {
    case.name: case
    for case in (
        Case(
            name="original-gdn2-1p2b-cx4-step9059",
            manifest="geometry_matched_scale_gdn2_nope_gated_full_candidate.yaml",
            task_name="1p2b-cx4",
            checkpoint="step9000",
            failure_step=9_059,
            start_step=9_050,
            stop_step=9_070,
            kind="GDN2 expand_v=2 negative-eigenvalues; broad all-rank NaN gradients",
        ),
        Case(
            name="original-gdn2-275m-cx8-step36768",
            manifest="275m_geometry_gdn2_ev2_nope_gated.yaml",
            task_name="cx8-lr1p6e-3",
            checkpoint="step36500",
            failure_step=36_768,
            start_step=36_750,
            stop_step=36_780,
            kind="GDN2 expand_v=2 negative-eigenvalues; non-finite loss",
            small_manifest=True,
        ),
        Case(
            name="gdn1-1p2b-cx8-step17592",
            manifest="geometry_matched_scale_nope_full.yaml",
            task_name="1p2b-cx8",
            checkpoint="step17500",
            failure_step=17_592,
            start_step=17_580,
            stop_step=17_605,
            kind="GDN1 expand_v=2 negative-eigenvalues; broad all-rank NaN gradients",
        ),
    )
}


def _load_case(case: Case) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_manifest(MANIFEST_DIR / case.manifest)
    if not case.small_manifest:
        rows = validate(manifest)
        # All diagnostic hooks live on the migration branch, including replays
        # of checkpoints originally produced by the older GDN1 branch.
        manifest["source"]["remote"] = REMOTE
        manifest["source"]["branch"] = BRANCH
        return manifest, next(row for row in rows if str(row["task_name"]) == case.task_name)

    # The original 275M sweep predates the Gantry production manifests. Adapt
    # its one-node row without changing any training or parallelism settings.
    rows = manifest["runs"]
    source_row = next(row for row in rows if str(row["task_name"]) == case.task_name)
    row = deepcopy(source_row)
    row["num_nodes"] = 1
    row["gpus_per_node"] = int(row.pop("gpu_count"))
    row["world_size"] = int(row["gpus_per_node"])
    global_sequences = int(row["global_batch_size"]) // int(manifest["training"]["sequence_length"])
    row["rank_sequences"] = global_sequences // int(row["world_size"])
    row["accumulation_steps"] = row["rank_sequences"] // int(
        row["rank_microbatch_sequences"]
    )

    adapted = deepcopy(manifest)
    adapted["source"] = {
        "remote": REMOTE,
        "branch": BRANCH,
        "image": manifest["source"]["image"],
    }
    adapted["beaker"]["preemptible"] = False
    adapted["weka"] = [
        {"bucket": item["weka"], "mount": item["mount_path"]}
        for item in manifest.get("datasets", [])
    ]
    return adapted, row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=sorted(CASES), default=[])
    parser.add_argument(
        "--allocated",
        action="store_true",
        help="request a short allocated reservation instead of the zero-minimum unallocated pool",
    )
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    selected = [CASES[name] for name in args.case] if args.case else list(CASES.values())
    prepared: list[tuple[Case, dict[str, Any], dict[str, Any], Path]] = []
    for case in selected:
        manifest, row = _load_case(case)
        checkpoint_dir = Path(manifest["experiment"]["checkpoint_root"]) / str(row["run_name"])
        checkpoints = sorted(path.name for path in checkpoint_dir.glob("step*") if path.is_dir())
        if checkpoints != [case.checkpoint]:
            raise RuntimeError(
                f"{case.name}: expected only {case.checkpoint} in {checkpoint_dir}, found {checkpoints}"
            )
        output_dir = DUMP_ROOT / case.name
        if output_dir.exists() and any(output_dir.iterdir()):
            raise RuntimeError(f"{case.name}: refusing to overwrite {output_dir}")
        prepared.append((case, manifest, row, checkpoint_dir))
        print(
            f"{case.name}: {row['run_name']} {case.checkpoint} -> failure={case.failure_step} "
            f"window={case.start_step}..{case.stop_step} world={row['world_size']} "
            f"EP={row['expert_parallel_size']} MB={row['rank_microbatch_sequences']} "
            f"accum={row['accumulation_steps']} ({case.kind})"
        )

    if not args.submit:
        print("Dry run only; pass --submit to launch.")
        return

    for case, manifest, row, checkpoint_dir in prepared:
        if args.allocated:
            manifest["beaker"]["min_runtime"] = "10m"
        source = manifest["source"]
        commit = validate_remote_commit(str(source["remote"]), str(source["branch"]))
        debug_steps = ",".join(str(step) for step in range(case.start_step, case.stop_step + 1))
        workload = recipe_for(
            manifest,
            row,
            commit=commit,
            debug_gradients=True,
            runtime_env_overrides={
                "OLMOE3_HYBRID_HARD_STOP_STEPS": str(case.stop_step),
                "OLMOE3_HYBRID_CHECKPOINTS": "1",
                "OLMOE3_HYBRID_CHECKPOINT_WRITES": "0",
                "OLMOE3_HYBRID_WANDB": "0",
                "OLMOE3_HYBRID_GDN2_LOCALIZE_NONFINITE": "1",
                "OLMOE3_HYBRID_GDN2_LOCALIZE_START_STEP": str(case.start_step),
                "OLMOE3_HYBRID_GDN2_LOCALIZE_END_STEP": str(case.stop_step),
                "OLMOE3_HYBRID_GDN2_LOCALIZE_DUMP_ROOT": str(DUMP_ROOT),
                "OLMOE3_HYBRID_GDN2_LOCALIZE_RUN_ID": case.name,
                "OLMO_DEBUG_CHECK_LOCAL_LOSS": "1",
                "OLMO_DEBUG_DUMP_DIR": str(DUMP_ROOT),
                "OLMO_DEBUG_RUN_ID": case.name,
                "OLMO_DEBUG_DUMP_STEPS": debug_steps,
            },
            recipe_suffix_override=f"-{case.name}-localizer-r1",
            description_suffix=(
                f"read-only exact recurrent failure replay; {case.kind}; forward, "
                "backward-input, backward-output, and pre-reduction loss localization"
            ),
        ).launch(show_logs=False)
        experiment = workload.experiment
        record = {
            "case": case.name,
            "kind": case.kind,
            "run_name": row["run_name"],
            "commit": commit,
            "source_checkpoint": str(checkpoint_dir / case.checkpoint),
            "expected_failure_step": case.failure_step,
            "localize_start_step": case.start_step,
            "hard_stop_step": case.stop_step,
            "world_size": row["world_size"],
            "expert_parallel_size": row["expert_parallel_size"],
            "rank_microbatch_sequences": row["rank_microbatch_sequences"],
            "checkpoint_writes": False,
            "wandb": False,
            "allocated": args.allocated,
            "dump_dir": str(DUMP_ROOT / case.name),
            "experiment_id": experiment.id,
            "task_ids": [task.id for task in experiment.tasks],
            "url": (
                "https://beaker.org/orgs/ai2/workspaces/"
                f"OLMo-3-moe-experiments/work/{experiment.id}"
            ),
        }
        existing: list[dict[str, Any]] = []
        if RECORD.is_file():
            existing = json.loads(RECORD.read_text())
        existing.append(record)
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        RECORD.write_text(json.dumps(existing, indent=2) + "\n")
        print(f"Submitted {case.name}: {record['url']}")
        print(f"Recorded submission in {RECORD}")


if __name__ == "__main__":
    main()
