"""Replay one SSMax perception arm/step and produce all-rank cursor/loss-mass health evidence.

The immutable pair manifest selects the run and checkpoint.  This CPU producer reconstructs the
unpacked perception loader for every saved rank, replays exactly from step 0, and requires each
cursor to equal its trainer state. Every online optimizer/non-finite event comes from the
checkpoint-native, resume-safe SSMax health ledger saved inside that same trainer-rank file.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from olmo_core.data.multimodal import MixtureDataLoader
from olmo_core.eval import vision_alignment_ssmax_bridge as bridge
from olmo_core.eval.vision_alignment_ssmax_perception import (
    ARMS,
    HEALTH_PRODUCER,
    HEALTH_RECEIPT_FORMAT,
    REQUIRED_STEPS,
    SCHEMA_VERSION,
    SOURCES,
    SSMaxPerceptionEvidenceError,
    canonical_sha256,
    load_json,
    load_manifest,
    manifest_reference,
    sha256_file,
    validate_artifact_reference,
    validate_manifest_producer_source,
    write_json_once,
)
from olmo_core.train.callbacks import (
    SSMaxHealthLedgerError,
    extract_ssmax_health_ledgers,
)

_METRICS = ("examples", "tokens", "positive_tokens", "loss_weight", "active_loss_weight")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--step", type=int, choices=REQUIRED_STEPS, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefetch-workers", type=int, default=0)
    parser.add_argument("--checkpoint-hash-workers", type=int, default=8)
    parser.add_argument("--created-at")
    return parser.parse_args(argv)


def _load_recipe(path: Path) -> Any:
    module_name = "_vision_alignment_ssmax_perception_health_recipe"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import pinned recipe {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.detach().cpu().tolist()
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise SSMaxPerceptionEvidenceError(f"Loader state contains unsupported value {value!r}")


def _trainer_rank_states(
    checkpoint: Path,
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    paths = sorted(
        checkpoint.joinpath("train").glob("rank*.pt"),
        key=lambda path: int(path.stem.removeprefix("rank")),
    )
    if not paths:
        raise SSMaxPerceptionEvidenceError("Checkpoint has no trainer rank states")
    states: list[Mapping[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for rank, path in enumerate(paths):
        if path.name != f"rank{rank}.pt":
            raise SSMaxPerceptionEvidenceError("Trainer rank-state inventory is not contiguous")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("data_loader"), Mapping):
            raise SSMaxPerceptionEvidenceError(f"Trainer rank{rank} state lacks a data loader")
        states.append(payload)
        inventory.append(
            {
                "rank": rank,
                "path": path.relative_to(checkpoint).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return states, inventory


def _empty_stats() -> dict[str, dict[str, float]]:
    return {source: {metric: 0.0 for metric in _METRICS} for source in SOURCES}


def _accumulate_batch(stats: dict[str, dict[str, float]], batch: Mapping[str, Any]) -> None:
    required = {"source_names", "router_token_mask", "loss_masks", "labels"}
    if missing := required - set(batch):
        raise SSMaxPerceptionEvidenceError(f"Replayed batch omits {sorted(missing)}")
    names = batch["source_names"]
    labels = batch["labels"]
    if not isinstance(names, list) or len(names) != int(labels.shape[0]):
        raise SSMaxPerceptionEvidenceError("Unpacked source metadata differs from batch rows")
    for row, source in enumerate(names):
        if source not in stats:
            raise SSMaxPerceptionEvidenceError(f"Replay observed unknown source {source!r}")
        positions = batch["router_token_mask"][row].bool()
        active = positions & (labels[row] != -100)
        values = stats[source]
        values["examples"] += 1
        values["tokens"] += float(positions.sum(dtype=torch.long))
        values["positive_tokens"] += float(((batch["loss_masks"][row] > 0) & active).sum())
        values["loss_weight"] += float((batch["loss_masks"][row] * positions).sum())
        values["active_loss_weight"] += float((batch["loss_masks"][row] * active).sum())


def _build_loader(
    recipe: Any,
    config: Any,
    tokenizer: Any,
    token_ids: Any,
    *,
    rank: int,
    world_size: int,
    work_dir: Path,
    prefetch_workers: int,
) -> MixtureDataLoader:
    datasets, weights, names = recipe._build_mixture_sources(tokenizer, token_ids, config)
    if tuple(names) != SOURCES:
        raise SSMaxPerceptionEvidenceError(f"Perception source order differs: {names!r}")
    return MixtureDataLoader(
        datasets,
        weights,
        config.collator.build(),
        work_dir=work_dir / f"rank{rank}",
        global_batch_size=config.global_batch_size,
        seed=config.data_seed,
        pack=False,
        pack_max_crops=None,
        pack_buffer_size=0,
        prefetch_workers=prefetch_workers,
        dataset_names=names,
        allow_legacy_state_without_dataset_fingerprints=False,
        dp_world_size=world_size,
        dp_rank=rank,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.prefetch_workers < 0 or args.checkpoint_hash_workers <= 0:
        raise ValueError("Worker counts must be non-negative (hash workers positive)")
    manifest_path = args.manifest.expanduser().resolve()
    if sha256_file(manifest_path) != args.expected_manifest_sha256:
        raise SSMaxPerceptionEvidenceError("Pair manifest differs from its explicit CLI pin")
    manifest = load_manifest(manifest_path, verify_live=False)
    producer_source = validate_manifest_producer_source(
        manifest,
        producer=HEALTH_PRODUCER,
        source_path=Path(__file__),
    )
    arm_manifest = manifest["arms"][args.arm]
    try:
        candidate = bridge.validate_checkpoint_reference(
            arm_manifest["checkpoints"][str(args.step)],
            expected_step=args.step,
            workers=args.checkpoint_hash_workers,
            verify_live=True,
        )
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxPerceptionEvidenceError(str(error)) from error
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable health receipt {output}")
    recipe_path = validate_artifact_reference(manifest["recipe"], name="training recipe")
    recipe = _load_recipe(recipe_path)
    raw_config = load_json(Path(str(candidate["path"])) / "config.json")
    if not isinstance(raw_config, Mapping):
        raise SSMaxPerceptionEvidenceError("Candidate config must contain an object")
    config = recipe.ExperimentConfig.from_dict(raw_config)
    if (
        str(config.phase) != "perception"
        or str(config.model_variant) != manifest["model_variant"]
        or str(config.perception_trainability_arm) != args.arm
        or config.required_run_name != arm_manifest["run_name"]
        or config.data.pack_sequences is not False
    ):
        raise SSMaxPerceptionEvidenceError("Candidate is not the manifest perception arm")
    tokenizer, token_ids = recipe._load_tokenizer(config.artifacts)
    rank_states, rank_inventory = _trainer_rank_states(Path(str(candidate["path"])))
    world_size = int(manifest["topology"]["world_size"])
    if len(rank_states) != world_size:
        raise SSMaxPerceptionEvidenceError("Trainer rank-state count differs from topology")
    try:
        ledger_summary = extract_ssmax_health_ledgers(
            rank_states,
            expected_model_variant=str(manifest["model_variant"]),
            expected_phase="perception",
            expected_run_name=str(arm_manifest["run_name"]),
            expected_step=args.step,
            expected_world_size=world_size,
        )
    except SSMaxHealthLedgerError as error:
        raise SSMaxPerceptionEvidenceError(
            f"Checkpoint health ledgers are invalid: {error}"
        ) from error
    stats = _empty_stats()
    rank_receipts: list[dict[str, Any]] = []
    dataset_fingerprints: Any = None
    data_errors = 0
    health_ledgers: list[Mapping[str, Any]] = []
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    for rank, trainer_state in enumerate(rank_states):
        saved = _jsonable(trainer_state["data_loader"])
        if (
            trainer_state.get("global_step") != args.step
            or trainer_state.get("world_size") != world_size
            or saved.get("batches_processed") != args.step
            or "packing_state" in saved
        ):
            raise SSMaxPerceptionEvidenceError(f"Trainer rank{rank} cursor is incompatible")
        ledger = ledger_summary["rank_ledgers"][rank]
        health_ledgers.append(ledger)
        loader = _build_loader(
            recipe,
            config,
            tokenizer,
            token_ids,
            rank=rank,
            world_size=world_size,
            work_dir=work_dir,
            prefetch_workers=args.prefetch_workers,
        )
        current_fingerprints = _jsonable(loader.dataset_fingerprints)
        if dataset_fingerprints is None:
            dataset_fingerprints = current_fingerprints
        elif current_fingerprints != dataset_fingerprints:
            raise SSMaxPerceptionEvidenceError("Dataset fingerprints differ across ranks")
        epoch = saved.get("epoch")
        if type(epoch) is not int or epoch <= 0:
            raise SSMaxPerceptionEvidenceError(f"Trainer rank{rank} epoch is invalid")
        loader.reshuffle(epoch=epoch)
        iterator = iter(loader)
        try:
            for _ in range(args.step):
                _accumulate_batch(stats, next(iterator))
            replayed = _jsonable(loader.state_dict())
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        if replayed != _jsonable(loader.state_dict()) or replayed != saved:
            raise SSMaxPerceptionEvidenceError(f"Trainer rank{rank} replay cursor differs")
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
        raise SSMaxPerceptionEvidenceError("Runtime source loss-mass targets differ")
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
        raise SSMaxPerceptionEvidenceError("Ledger and replay data-error totals differ")
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
    status_ok = (
        within_mass
        and counters["data_errors"] <= manifest["policy"]["maximum_data_errors"]
        and counters["optimizer_guard_skips"] <= manifest["policy"]["maximum_optimizer_guard_skips"]
        and counters["nonfinite_losses"] == 0
        and counters["nonfinite_gradients"] == 0
    )
    receipt: dict[str, Any] = {
        "format": HEALTH_RECEIPT_FORMAT,
        "version": SCHEMA_VERSION,
        "status": "passed" if status_ok else "failed",
        "created_at": args.created_at or datetime.now(timezone.utc).isoformat(),
        "manifest": manifest_reference(manifest_path, manifest),
        "pair_id": manifest["pair_id"],
        "model_variant": manifest["model_variant"],
        "arm": args.arm,
        "step": args.step,
        "checkpoint": dict(candidate),
        "rank_states": rank_receipts,
        "sources": sources,
        "run_counters": counters,
        "evidence": {
            "recipe": dict(manifest["recipe"]),
            "producer": producer_source,
        },
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    write_json_once(output, receipt)


if __name__ == "__main__":
    main()
