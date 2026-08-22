"""Replay one direct SSMax perception step and produce all-rank CPU health evidence.

The immutable direct manifest selects the sole run and checkpoint.  This producer reconstructs
the unpacked loader for every saved rank, replays from step 0, and binds optimizer/non-finite
events to the checkpoint-native resume-safe health ledger.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from olmo_core.eval import vision_alignment_ssmax_bridge as bridge
from olmo_core.eval import vision_alignment_ssmax_perception as paired
from olmo_core.eval.vision_alignment_ssmax_perception_direct import (
    DIRECT_RUN_IDENTITIES,
    HEALTH_PRODUCER,
    HEALTH_RECEIPT_FORMAT,
    REQUIRED_STEPS,
    SOURCES,
    TRAINING_ROLE,
    SSMaxPerceptionDirectEvidenceError,
    canonical_sha256,
    load_json,
    load_manifest,
    manifest_reference,
    sha256_file,
    validate_manifest_producer_source,
    write_json_once,
)
from olmo_core.train.callbacks import (
    SSMaxHealthLedgerError,
    extract_ssmax_health_ledgers,
)
from scripts.eval import vision_alignment_ssmax_perception_health as paired_runner

_DIRECT_SERIALIZED_CHECKPOINTER = {
    "_CLASS_": "olmo_core.train.callbacks.checkpointer.CheckpointerCallback",
    "enabled": True,
    "ephemeral_save_interval": 400,
    "fixed_steps": [500, 1000, 2000, 3000, 4000],
    "max_checkpoints": 6,
    "pre_train_checkpoint": True,
    "remove": "ephemeral_only",
    "save_async": False,
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--step", type=int, choices=REQUIRED_STEPS, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefetch-workers", type=int, default=0)
    parser.add_argument("--checkpoint-hash-workers", type=int, default=8)
    parser.add_argument("--created-at")
    return parser.parse_args(argv)


def _materialize_training_recipe(manifest: Mapping[str, Any], *, work_dir: Path) -> Path:
    """Materialize the manifest-bound training-ref recipe without using evidence-tree bytes."""

    reference = manifest["training_recipe"]
    relative = str(reference["repo_relative_path"])
    training_ref = str(manifest["training_git"]["ref"])
    repository_root = Path(__file__).resolve().parents[3]

    def read_blob() -> bytes:
        return subprocess.check_output(
            ["git", "show", f"{training_ref}:{relative}"],
            cwd=repository_root,
            stderr=subprocess.PIPE,
        )

    try:
        raw = read_blob()
    except (OSError, subprocess.CalledProcessError):
        try:
            subprocess.check_call(
                ["git", "fetch", "--no-tags", "--depth", "1", "origin", training_ref],
                cwd=repository_root,
                stderr=subprocess.PIPE,
            )
            raw = read_blob()
        except (OSError, subprocess.CalledProcessError) as error:
            raise SSMaxPerceptionDirectEvidenceError(
                "Could not fetch and read the manifest-bound training recipe Git blob"
            ) from error
    expected = str(reference["sha256"])
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SSMaxPerceptionDirectEvidenceError(
            "Training recipe Git blob differs from the manifest's raw SHA-256"
        )
    destination_dir = work_dir.expanduser().resolve() / "training-recipe"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{expected}.py"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination_dir,
            prefix=f".{expected}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            pass
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    if not destination.is_file() or sha256_file(destination) != expected:
        raise SSMaxPerceptionDirectEvidenceError(
            "Materialized training recipe differs from its manifest pin"
        )
    return destination


def _load_training_recipe(manifest: Mapping[str, Any], *, work_dir: Path) -> Any:
    materialized = _materialize_training_recipe(manifest, work_dir=work_dir)
    recipe = paired_runner._load_recipe(materialized)
    # The exact training-ref module has repository-relative dependency checks based on
    # ``__file__``.  Execute the content-addressed materialization, then restore only that path
    # anchor so those unchanged, evidence-diff-allowlisted-out dependencies resolve canonically.
    relative = str(manifest["training_recipe"]["repo_relative_path"])
    recipe.__file__ = str((Path(__file__).resolve().parents[3] / relative).resolve())
    return recipe


def _summarize_optimizer_guard(ledger: Mapping[str, Any], *, step: int) -> dict[str, Any]:
    """Use the exact v2 compatibility policy consumed by the translated core validator."""

    return paired.summarize_optimizer_guard_trajectory(
        ledger,
        policy=paired._locked_promotion_policy(paired.PERCEPTION_V2_SCHEMA_VERSION),
        step=step,
    )


def _exact_json_value(value: Any, expected: Any) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(value) == set(expected) and all(
            _exact_json_value(value[field], expected[field]) for field in expected
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _exact_json_value(item, expected_item)
            for item, expected_item in zip(value, expected, strict=True)
        )
    return bool(value == expected)


def _hydrate_direct_saved_config(
    raw_config: Mapping[str, Any], *, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Restore one serialization-omitted runtime ``None`` after exact contract checks."""

    model_variant = manifest.get("model_variant")
    identity = DIRECT_RUN_IDENTITIES.get(str(model_variant))
    run = manifest.get("run")
    if identity is None or not isinstance(run, Mapping):
        raise SSMaxPerceptionDirectEvidenceError("Direct hydration lineage identity is invalid")
    expected_identity = {
        "model_variant": model_variant,
        "phase": "perception",
        "perception_trainability_arm": TRAINING_ROLE,
        "required_run_name": identity["run_name"],
        "reviewed_profile_path": identity["profile"],
        "reviewed_profile_sha256": identity["profile_sha256"],
    }
    if run.get("run_name") != identity["run_name"] or any(
        not _exact_json_value(raw_config.get(field), expected)
        for field, expected in expected_identity.items()
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct saved config does not have the exact reviewed profile identity"
        )
    trainer = raw_config.get("trainer")
    callbacks = trainer.get("callbacks") if isinstance(trainer, Mapping) else None
    checkpointer = callbacks.get("checkpointer") if isinstance(callbacks, Mapping) else None
    if not isinstance(checkpointer, Mapping):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct saved config lacks its reviewed checkpointer"
        )
    if "save_interval" in checkpointer:
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct serialized checkpointer must omit runtime-null save_interval"
        )
    if not _exact_json_value(checkpointer, _DIRECT_SERIALIZED_CHECKPOINTER):
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct serialized checkpointer differs from the reviewed contract"
        )
    hydrated = copy.deepcopy(dict(raw_config))
    hydrated["trainer"]["callbacks"]["checkpointer"]["save_interval"] = None
    return hydrated


def _validate_trainer_cursor(
    trainer_state: Mapping[str, Any], *, step: int, world_size: int, rank: int
) -> tuple[dict[str, Any], int | None]:
    """Validate and return one direct checkpoint's exact saved loader cursor."""

    try:
        saved = paired_runner._jsonable(trainer_state["data_loader"])
    except (KeyError, paired.SSMaxPerceptionEvidenceError) as error:
        raise SSMaxPerceptionDirectEvidenceError(
            f"Trainer rank{rank} cursor is incompatible"
        ) from error
    if (
        type(trainer_state.get("global_step")) is not int
        or trainer_state["global_step"] != step
        or type(trainer_state.get("world_size")) is not int
        or trainer_state["world_size"] != world_size
        or type(saved.get("batches_processed")) is not int
        or saved["batches_processed"] != step
        or "packing_state" in saved
    ):
        raise SSMaxPerceptionDirectEvidenceError(f"Trainer rank{rank} cursor is incompatible")
    if "epoch" not in saved:
        raise SSMaxPerceptionDirectEvidenceError(f"Trainer rank{rank} epoch is invalid")
    epoch = saved["epoch"]
    if step == 0:
        if epoch is not None:
            raise SSMaxPerceptionDirectEvidenceError(f"Trainer rank{rank} epoch is invalid")
    elif type(epoch) is not int or epoch <= 0:
        raise SSMaxPerceptionDirectEvidenceError(f"Trainer rank{rank} epoch is invalid")
    return saved, epoch


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.prefetch_workers < 0 or args.checkpoint_hash_workers <= 0:
        raise ValueError("Worker counts must be non-negative (hash workers positive)")
    manifest_path = args.manifest.expanduser().resolve()
    if sha256_file(manifest_path) != args.expected_manifest_sha256:
        raise SSMaxPerceptionDirectEvidenceError(
            "Direct manifest differs from its explicit CLI pin"
        )
    manifest = load_manifest(manifest_path, verify_live=False)
    producer_source = validate_manifest_producer_source(
        manifest,
        producer=HEALTH_PRODUCER,
        source_path=Path(__file__),
    )
    run = manifest["run"]
    try:
        candidate = bridge.validate_checkpoint_reference(
            run["checkpoints"][str(args.step)],
            expected_step=args.step,
            workers=args.checkpoint_hash_workers,
            verify_live=True,
        )
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable health receipt {output}")
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    recipe = _load_training_recipe(manifest, work_dir=work_dir)
    raw_config = load_json(Path(str(candidate["path"])) / "config.json")
    if not isinstance(raw_config, Mapping):
        raise SSMaxPerceptionDirectEvidenceError("Candidate config must contain an object")
    config = recipe.ExperimentConfig.from_dict(
        _hydrate_direct_saved_config(raw_config, manifest=manifest)
    )
    if (
        str(config.phase) != "perception"
        or str(config.model_variant) != manifest["model_variant"]
        or str(config.perception_trainability_arm) != TRAINING_ROLE
        or config.required_run_name != run["run_name"]
        or config.data.pack_sequences is not False
    ):
        raise SSMaxPerceptionDirectEvidenceError(
            "Candidate is not the manifest's direct perception run"
        )
    tokenizer, token_ids = recipe._load_tokenizer(config.artifacts)
    try:
        rank_states, rank_inventory = paired_runner._trainer_rank_states(
            Path(str(candidate["path"]))
        )
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    world_size = int(manifest["topology"]["world_size"])
    if len(rank_states) != world_size:
        raise SSMaxPerceptionDirectEvidenceError("Trainer rank-state count differs from topology")
    try:
        ledger_summary = extract_ssmax_health_ledgers(
            rank_states,
            expected_model_variant=str(manifest["model_variant"]),
            expected_phase="perception",
            expected_run_name=str(run["run_name"]),
            expected_step=args.step,
            expected_world_size=world_size,
        )
    except SSMaxHealthLedgerError as error:
        raise SSMaxPerceptionDirectEvidenceError(
            f"Checkpoint health ledgers are invalid: {error}"
        ) from error
    stats = paired_runner._empty_stats()
    rank_receipts: list[dict[str, Any]] = []
    dataset_fingerprints: Any = None
    data_errors = 0
    health_ledgers: list[Mapping[str, Any]] = []
    for rank, trainer_state in enumerate(rank_states):
        saved, epoch = _validate_trainer_cursor(
            trainer_state, step=args.step, world_size=world_size, rank=rank
        )
        ledger = ledger_summary["rank_ledgers"][rank]
        health_ledgers.append(ledger)
        try:
            loader = paired_runner._build_loader(
                recipe,
                config,
                tokenizer,
                token_ids,
                rank=rank,
                world_size=world_size,
                work_dir=work_dir,
                prefetch_workers=args.prefetch_workers,
            )
            current_fingerprints = paired_runner._jsonable(loader.dataset_fingerprints)
        except paired.SSMaxPerceptionEvidenceError as error:
            raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
        if dataset_fingerprints is None:
            dataset_fingerprints = current_fingerprints
        elif current_fingerprints != dataset_fingerprints:
            raise SSMaxPerceptionDirectEvidenceError("Dataset fingerprints differ across ranks")
        if epoch is not None:
            loader.reshuffle(epoch=epoch)
        iterator = iter(loader)
        try:
            for _ in range(args.step):
                paired_runner._accumulate_batch(stats, next(iterator))
            replayed = paired_runner._jsonable(loader.state_dict())
        except paired.SSMaxPerceptionEvidenceError as error:
            raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        if replayed != paired_runner._jsonable(loader.state_dict()) or replayed != saved:
            raise SSMaxPerceptionDirectEvidenceError(f"Trainer rank{rank} replay cursor differs")
        rank_receipts.append(
            {
                "rank": rank,
                "global_step": args.step,
                "batches_processed": int(saved["batches_processed"]),
                "data_loader_state_sha256": canonical_sha256(saved),
                "trainer_state_sha256": rank_inventory[rank]["sha256"],
                "trainer_state_size_bytes": rank_inventory[rank]["size"],
                "health_ledger": dict(ledger),
            }
        )
        data_errors += int(saved.get("total_data_errors", 0))

    targets = dict(config.train_module.source_loss_mass_targets or {})
    if targets != manifest["loss_mass_targets"]:
        raise SSMaxPerceptionDirectEvidenceError("Runtime source loss-mass targets differ")
    sources = {
        source: {
            "examples": int(stats[source]["examples"]),
            "tokens": int(stats[source]["tokens"]),
            "positive_tokens": int(stats[source]["positive_tokens"]),
            "loss_weight": stats[source]["loss_weight"],
            "active_loss_weight": stats[source]["active_loss_weight"],
            "target_loss_mass": float(targets[source]),
        }
        for source in SOURCES
    }
    counters = dict(ledger_summary["counters"])
    if counters["data_errors"] != data_errors:
        raise SSMaxPerceptionDirectEvidenceError("Ledger and replay data-error totals differ")
    total_loss = sum(item["loss_weight"] for item in sources.values())
    total_active = sum(item["active_loss_weight"] for item in sources.values())
    within_mass = args.step == 0 or all(
        total > 0
        and all(
            abs(sources[source][field] / total - targets[source])
            <= manifest["policy"]["loss_mass_share_tolerance"]
            for source in SOURCES
        )
        for field, total in (
            ("loss_weight", total_loss),
            ("active_loss_weight", total_active),
        )
    )
    try:
        guard_summary = _summarize_optimizer_guard(health_ledgers[0], step=args.step)
    except paired.SSMaxPerceptionEvidenceError as error:
        raise SSMaxPerceptionDirectEvidenceError(str(error)) from error
    status_ok = (
        within_mass
        and guard_summary["passed"]
        and counters["data_errors"] <= manifest["policy"]["maximum_data_errors"]
        and counters["optimizer_guard_skips"] <= manifest["policy"]["maximum_optimizer_guard_skips"]
        and counters["nonfinite_losses"] <= manifest["policy"]["maximum_nonfinite_losses"]
        and counters["nonfinite_gradients"] <= manifest["policy"]["maximum_nonfinite_gradients"]
    )
    receipt: dict[str, Any] = {
        "format": HEALTH_RECEIPT_FORMAT,
        "version": manifest["version"],
        "status": "passed" if status_ok else "failed",
        "created_at": args.created_at or datetime.now(timezone.utc).isoformat(),
        "manifest": manifest_reference(manifest_path, manifest),
        "run_id": manifest["run_id"],
        "model_variant": manifest["model_variant"],
        "step": args.step,
        "checkpoint": dict(candidate),
        "rank_states": rank_receipts,
        "sources": sources,
        "run_counters": counters,
        "evidence": {
            "training_recipe": dict(manifest["training_recipe"]),
            "producer": producer_source,
        },
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    write_json_once(output, receipt)


if __name__ == "__main__":
    main()
