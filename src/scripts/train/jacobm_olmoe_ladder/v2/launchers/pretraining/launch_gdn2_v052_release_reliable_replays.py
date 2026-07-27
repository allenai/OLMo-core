#!/usr/bin/env python3
"""Replay every remaining reliably failing GDN2 checkpoint with FLA v0.5.2."""

from __future__ import annotations

import argparse
import json
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
RECORD = SCRIPT_DIR / "generated" / "gdn2_v052_release_reliable_replays.json"
FLA_OVERLAY = "/tmp/fla-gdn2-v0.5.2-release"
FLA_SPEC = (
    "flash-linear-attention[cuda] @ git+https://github.com/fla-org/"
    "flash-linear-attention.git@v0.5.2"
)
FLA_COMMIT = "9c8e42e762fce087c27b673af4922795d9edb85e"
PREVIOUS_FLA_COMMIT = "cbb0a72efb55c18ca0ef4f298298317573ad2cb3"
DUMP_ROOT = Path(
    "/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/"
    "olmo-ddp/debug/gdn2-v052-release-reliable-replays"
)


@dataclass(frozen=True)
class Case:
    name: str
    manifest: str
    task_name: str
    checkpoint: str
    expected_failure_steps: tuple[int, ...]
    localize_start_step: int
    hard_stop_step: int
    evidence: str


# These cases repeatedly failed from the same saved checkpoint under the old
# pinned FLA commit. The 275M Cx8 and 1.2B Cx4 cases are omitted because their
# actual-release replays were already submitted separately. Moving failures
# (for example original 1.2B Cx2 and fresh 810M Cx2 at its latest checkpoint)
# are deliberately not classified as reliable checkpoint replays.
CASES = {
    case.name: case
    for case in (
        Case(
            name="canonical-480m-cx2-step24668",
            manifest="geometry_matched_scale_gdn2_ev1_noneg_nope_gated_full.yaml",
            task_name="480m-cx2",
            checkpoint="step24500",
            expected_failure_steps=(24_668,),
            localize_start_step=24_640,
            hard_stop_step=24_690,
            evidence="failed at step24668 in seven resumes and the exact localizer",
        ),
        Case(
            name="original-810m-cx1-step10039",
            manifest="geometry_matched_scale_gdn2_nope_gated_full_candidate.yaml",
            task_name="810m-cx1",
            checkpoint="step10000",
            expected_failure_steps=(10_039,),
            localize_start_step=10_025,
            hard_stop_step=10_055,
            evidence="production and diagnostic attempts both failed at step10039",
        ),
        Case(
            name="original-810m-cx2-step56755",
            manifest="geometry_matched_scale_gdn2_nope_gated_full_candidate.yaml",
            task_name="810m-cx2",
            checkpoint="step56500",
            expected_failure_steps=(56_755,),
            localize_start_step=56_735,
            hard_stop_step=56_775,
            evidence="multiple checkpoint-local attempts failed at step56755",
        ),
        Case(
            name="original-1p2b-cx1-step8029",
            manifest="geometry_matched_scale_gdn2_nope_gated_full_candidate.yaml",
            task_name="1p2b-cx1",
            checkpoint="step8000",
            expected_failure_steps=(8_029,),
            localize_start_step=8_015,
            hard_stop_step=8_045,
            evidence="production and diagnostic attempts both failed at step8029",
        ),
        Case(
            name="original-1p2b-cx8-step7073-or7125",
            manifest="geometry_matched_scale_gdn2_nope_gated_full_candidate.yaml",
            task_name="1p2b-cx8",
            checkpoint="step7000",
            expected_failure_steps=(7_073, 7_125),
            localize_start_step=7_060,
            hard_stop_step=7_140,
            evidence=(
                "every checkpoint-local retry failed at one of two repeat points, "
                "step7073 or step7125"
            ),
        ),
    )
}


def _prepare(case: Case) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    manifest = load_manifest(MANIFEST_DIR / case.manifest)
    manifest["experiment"]["description"] = (
        "Read-only reliably failing GDN2 checkpoint replay with the actual "
        "FLA v0.5.2 release tag"
    )
    manifest["beaker"]["auto_resume"] = False
    rows = validate(manifest)
    row = next(row for row in rows if str(row["task_name"]) == case.task_name)

    checkpoint_dir = Path(manifest["experiment"]["checkpoint_root"]) / str(row["run_name"])
    checkpoints = sorted(path.name for path in checkpoint_dir.glob("step*") if path.is_dir())
    if checkpoints != [case.checkpoint]:
        raise RuntimeError(
            f"{case.name}: expected only {case.checkpoint} in {checkpoint_dir}, "
            f"found {checkpoints}"
        )
    output_dir = DUMP_ROOT / case.name
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"{case.name}: refusing to overwrite {output_dir}")

    source = manifest["source"]
    commit = validate_remote_commit(str(source["remote"]), str(source["branch"]))
    return manifest, row, checkpoint_dir, commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=sorted(CASES), default=[])
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    selected = [CASES[name] for name in args.case] if args.case else list(CASES.values())
    prepared = []
    for case in selected:
        manifest, row, checkpoint_dir, commit = _prepare(case)
        prepared.append((case, manifest, row, checkpoint_dir, commit))
        expected = "/".join(str(step) for step in case.expected_failure_steps)
        print(
            f"{case.name}: {row['run_name']}/{case.checkpoint} expected={expected} "
            f"window={case.localize_start_step}..{case.hard_stop_step} "
            f"world={row['world_size']} EP={row['expert_parallel_size']} "
            f"MB={row['rank_microbatch_sequences']} accum={row['accumulation_steps']} "
            f"FLA={FLA_COMMIT[:9]} ({case.evidence})"
        )

    if not args.submit:
        print("Dry run only; pass --submit to launch.")
        return

    existing: list[dict[str, Any]] = []
    if RECORD.is_file():
        existing = json.loads(RECORD.read_text())

    for case, manifest, row, checkpoint_dir, commit in prepared:
        workload = recipe_for(
            manifest,
            row,
            commit=commit,
            debug_gradients=True,
            runtime_env_overrides={
                "OLMOE3_HYBRID_HARD_STOP_STEPS": str(case.hard_stop_step),
                "OLMOE3_HYBRID_CHECKPOINTS": "1",
                "OLMOE3_HYBRID_CHECKPOINT_WRITES": "0",
                "OLMOE3_HYBRID_WANDB": "0",
                "OLMOE3_HYBRID_GDN2_LOCALIZE_NONFINITE": "1",
                "OLMOE3_HYBRID_GDN2_LOCALIZE_START_STEP": str(case.localize_start_step),
                "OLMOE3_HYBRID_GDN2_LOCALIZE_END_STEP": str(case.hard_stop_step),
                "OLMOE3_HYBRID_GDN2_LOCALIZE_DUMP_ROOT": str(DUMP_ROOT),
                "OLMOE3_HYBRID_GDN2_LOCALIZE_RUN_ID": case.name,
                "OLMO_DEBUG_CHECK_LOCAL_LOSS": "1",
                "OLMO_DEBUG_DUMP_DIR": str(DUMP_ROOT),
                "OLMO_DEBUG_RUN_ID": case.name,
            },
            recipe_suffix_override=f"-fla-v0p5p2-release-{case.name}-r1",
            description_suffix=(
                f"actual-release replay of reliable checkpoint failure: {case.evidence}; "
                "checkpoint writes and W&B disabled"
            ),
            gdn2_fla_overlay=FLA_OVERLAY,
            gdn2_fla_spec=FLA_SPEC,
            gdn2_fla_expected_commit=FLA_COMMIT,
        ).launch(show_logs=False)

        experiment = workload.experiment
        record = {
            "case": case.name,
            "run_name": row["run_name"],
            "source_checkpoint": str(checkpoint_dir / case.checkpoint),
            "source_commit": commit,
            "fla_tag": "v0.5.2",
            "fla_commit": FLA_COMMIT,
            "previous_fla_commit": PREVIOUS_FLA_COMMIT,
            "prior_evidence": case.evidence,
            "expected_failure_steps": list(case.expected_failure_steps),
            "localize_start_step": case.localize_start_step,
            "hard_stop_step": case.hard_stop_step,
            "world_size": row["world_size"],
            "expert_parallel_size": row["expert_parallel_size"],
            "rank_microbatch_sequences": row["rank_microbatch_sequences"],
            "checkpoint_writes": False,
            "wandb": False,
            "dump_dir": str(DUMP_ROOT / case.name),
            "experiment_id": experiment.id,
            "task_ids": [task.id for task in experiment.tasks],
            "url": (
                "https://beaker.org/orgs/ai2/workspaces/"
                f"OLMo-3-moe-experiments/work/{experiment.id}"
            ),
        }
        existing.append(record)
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        RECORD.write_text(json.dumps(existing, indent=2) + "\n")
        print(f"Submitted {case.name}: {record['url']}")

    print(f"Recorded submissions in {RECORD}")


if __name__ == "__main__":
    main()
