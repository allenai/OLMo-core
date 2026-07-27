#!/usr/bin/env python3
"""Replay the deterministic 1.2B GDN2 failure with the FLA v0.5.2 release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.train.jacobm_olmoe_ladder.v2.launchers.pretraining.launch_geometry_matched_scale_full import (
    recipe_for,
    validate,
)
from scripts.train.jacobm_olmoe_ladder.v2.launchers.pretraining.launch_geometry_matched_scale_nope_smokes import (
    load_manifest,
    validate_remote_commit,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MANIFEST = SCRIPT_DIR / "manifests" / "geometry_matched_scale_gdn2_nope_gated_full_candidate.yaml"
RECORD = SCRIPT_DIR / "generated" / "gdn2_v052_release_deterministic_replay_submissions.json"
TARGET_TASK = "1p2b-cx4"
EXPECTED_CHECKPOINT = "step9000"
EXPECTED_FAILURE_STEP = 9_059
LOCALIZE_START_STEP = 9_050
HARD_STOP_STEP = 9_075
FLA_OVERLAY = "/tmp/fla-gdn2-v0.5.2-release"
FLA_SPEC = (
    "flash-linear-attention[cuda] @ git+https://github.com/fla-org/"
    "flash-linear-attention.git@v0.5.2"
)
FLA_COMMIT = "9c8e42e762fce087c27b673af4922795d9edb85e"
DUMP_ROOT = Path(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/"
    "olmo-ddp/debug/gdn2-v052-release-deterministic-replay"
)
RUN_ID = "1p2b-gdn2-ev2-neg-cx4-step9059-v052-release-r1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(MANIFEST)
    manifest["experiment"]["description"] = (
        "Read-only deterministic 1.2B GDN2 expand_v=2 negative-eigenvalue "
        "checkpoint replay with the FLA v0.5.2 release tag"
    )
    manifest["beaker"]["auto_resume"] = False
    rows = validate(manifest)
    row = next(row for row in rows if str(row["task_name"]) == TARGET_TASK)

    checkpoint_dir = Path(manifest["experiment"]["checkpoint_root"]) / str(row["run_name"])
    checkpoints = sorted(path for path in checkpoint_dir.glob("step*") if path.is_dir())
    if [path.name for path in checkpoints] != [EXPECTED_CHECKPOINT]:
        raise RuntimeError(
            f"Expected only {EXPECTED_CHECKPOINT} in {checkpoint_dir}, found "
            f"{[path.name for path in checkpoints]}"
        )
    output_dir = DUMP_ROOT / RUN_ID
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite existing diagnostic output {output_dir}")

    source = manifest["source"]
    commit = validate_remote_commit(str(source["remote"]), str(source["branch"]))
    print(
        f"Replay {row['run_name']} from {EXPECTED_CHECKPOINT}: FLA tag=v0.5.2 "
        f"commit={FLA_COMMIT} world={row['world_size']} "
        f"EP={row['expert_parallel_size']} MB={row['rank_microbatch_sequences']} "
        f"global_batch={row['global_batch_size']} "
        f"expected_failure={EXPECTED_FAILURE_STEP} hard_stop={HARD_STOP_STEP} "
        "checkpoint_writes=False wandb=False"
    )
    if not args.submit:
        print("Dry run only; pass --submit to launch.")
        return

    workload = recipe_for(
        manifest,
        row,
        commit=commit,
        debug_gradients=True,
        runtime_env_overrides={
            "OLMOE3_HYBRID_HARD_STOP_STEPS": str(HARD_STOP_STEP),
            "OLMOE3_HYBRID_CHECKPOINTS": "1",
            "OLMOE3_HYBRID_CHECKPOINT_WRITES": "0",
            "OLMOE3_HYBRID_WANDB": "0",
            "OLMOE3_HYBRID_GDN2_LOCALIZE_NONFINITE": "1",
            "OLMOE3_HYBRID_GDN2_LOCALIZE_START_STEP": str(LOCALIZE_START_STEP),
            "OLMOE3_HYBRID_GDN2_LOCALIZE_END_STEP": str(HARD_STOP_STEP),
            "OLMOE3_HYBRID_GDN2_LOCALIZE_DUMP_ROOT": str(DUMP_ROOT),
            "OLMOE3_HYBRID_GDN2_LOCALIZE_RUN_ID": RUN_ID,
            "OLMO_DEBUG_CHECK_LOCAL_LOSS": "1",
            "OLMO_DEBUG_DUMP_DIR": str(DUMP_ROOT),
            "OLMO_DEBUG_RUN_ID": RUN_ID,
        },
        recipe_suffix_override="-fla-v0p5p2-release-deterministic-replay-r1",
        description_suffix=(
            "same checkpoint, optimizer, data position, LR, batch, architecture, "
            "and parallelism as the deterministic step9059 replay; only FLA changes "
            "from cbb0a72 to release v0.5.2"
        ),
        gdn2_fla_overlay=FLA_OVERLAY,
        gdn2_fla_spec=FLA_SPEC,
        gdn2_fla_expected_commit=FLA_COMMIT,
    ).launch(show_logs=False)

    experiment = workload.experiment
    record = {
        "task_name": TARGET_TASK,
        "run_name": row["run_name"],
        "source_checkpoint": str(checkpoint_dir / EXPECTED_CHECKPOINT),
        "source_commit": commit,
        "fla_tag": "v0.5.2",
        "fla_commit": FLA_COMMIT,
        "previous_fla_commit": "cbb0a72efb55c18ca0ef4f298298317573ad2cb3",
        "expected_failure_step": EXPECTED_FAILURE_STEP,
        "localize_start_step": LOCALIZE_START_STEP,
        "hard_stop_step": HARD_STOP_STEP,
        "world_size": row["world_size"],
        "expert_parallel_size": row["expert_parallel_size"],
        "rank_microbatch_sequences": row["rank_microbatch_sequences"],
        "checkpoint_writes": False,
        "wandb": False,
        "dump_dir": str(output_dir),
        "experiment_id": experiment.id,
        "task_ids": [task.id for task in experiment.tasks],
        "url": (
            f"https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/{experiment.id}"
        ),
    }
    existing: list[dict[str, object]] = []
    if RECORD.is_file():
        existing = json.loads(RECORD.read_text())
    existing.append(record)
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"Submitted: {record['url']}")
    print(f"Recorded submission in {RECORD}")


if __name__ == "__main__":
    main()
