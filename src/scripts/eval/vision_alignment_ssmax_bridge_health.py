"""Replay an SSMax bridge loader and emit a manifest-bound data-health receipt.

This CPU audit rebuilds the exact pinned recipe and un-packed GatedDeltaNet-safe mixture for every
data-parallel rank.  It replays from step 0 to one manifest-selected checkpoint, requires the
reconstructed loader cursor to equal every saved trainer state, and reports cumulative delivered
supervised-loss mass and data errors.  Arbitrary checkpoint and recipe overrides are absent.
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
from olmo_core.eval.vision_alignment_ssmax_bridge import (
    HEALTH_PRODUCER,
    HEALTH_RECEIPT_FORMAT,
    REQUIRED_STEPS,
    SCHEMA_VERSION,
    SOURCES,
    SSMaxBridgeEvidenceError,
    canonical_sha256,
    load_json,
    load_manifest,
    manifest_reference,
    sha256_file,
    validate_artifact_reference,
    validate_checkpoint_reference,
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
    parser.add_argument("--step", type=int, choices=REQUIRED_STEPS, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefetch-workers", type=int, default=0)
    parser.add_argument("--checkpoint-hash-workers", type=int, default=8)
    parser.add_argument("--created-at")
    return parser.parse_args(argv)


def _load_recipe(path: Path):
    module_name = "_vision_alignment_ssmax_bridge_health_recipe"
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
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise SSMaxBridgeEvidenceError(f"Loader state contains unsupported value {value!r}")


def _trainer_rank_states(checkpoint: Path) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    paths = sorted(
        checkpoint.joinpath("train").glob("rank*.pt"),
        key=lambda path: int(path.stem.removeprefix("rank")),
    )
    if not paths:
        raise SSMaxBridgeEvidenceError("Checkpoint has no trainer rank states")
    states = []
    inventory = []
    for rank, path in enumerate(paths):
        if path.name != f"rank{rank}.pt":
            raise SSMaxBridgeEvidenceError("Trainer rank-state inventory is not contiguous")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("data_loader"), Mapping):
            raise SSMaxBridgeEvidenceError(f"Trainer rank{rank} state lacks a data loader")
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


def _validate_trainer_cursor(
    trainer_state: Mapping[str, Any], *, step: int, world_size: int, rank: int
) -> tuple[dict[str, Any], int | None]:
    """Validate and return the exact saved loader cursor and its epoch."""

    saved = _jsonable(trainer_state["data_loader"])
    if (
        trainer_state.get("global_step") != step
        or trainer_state.get("world_size") != world_size
        or saved.get("batches_processed") != step
        or "packing_state" in saved
    ):
        raise SSMaxBridgeEvidenceError(f"Trainer rank{rank} cursor is incompatible")
    epoch = saved.get("epoch")
    if step == 0:
        if epoch is not None:
            raise SSMaxBridgeEvidenceError(f"Trainer rank{rank} has an invalid epoch")
    elif type(epoch) is not int or epoch <= 0:
        raise SSMaxBridgeEvidenceError(f"Trainer rank{rank} has an invalid epoch")
    return saved, epoch


def _empty_stats() -> dict[str, dict[str, float]]:
    return {source: {name: 0.0 for name in _METRICS} for source in SOURCES}


def _accumulate_batch(stats: dict[str, dict[str, float]], batch: Mapping[str, Any]) -> None:
    """Accumulate the exact unpacked source fields consumed by online telemetry."""

    required = {"source_names", "router_token_mask", "loss_masks", "labels"}
    missing = required - set(batch)
    if missing:
        raise SSMaxBridgeEvidenceError(f"Replayed unpacked batch omits {sorted(missing)}")
    source_names = batch["source_names"]
    token_mask = batch["router_token_mask"]
    loss_masks = batch["loss_masks"]
    labels = batch["labels"]
    if not isinstance(source_names, list) or len(source_names) != int(labels.shape[0]):
        raise SSMaxBridgeEvidenceError("Unpacked source metadata does not match batch rows")
    for row, source in enumerate(source_names):
        if source not in stats:
            raise SSMaxBridgeEvidenceError(f"Replay observed unknown source {source!r}")
        positions = token_mask[row].bool()
        active = positions & (labels[row] != -100)
        source_stats = stats[source]
        source_stats["examples"] += 1.0
        source_stats["tokens"] += float(positions.sum(dtype=torch.long))
        source_stats["positive_tokens"] += float(((loss_masks[row] > 0) & active).sum())
        source_stats["loss_weight"] += float((loss_masks[row] * positions).sum())
        source_stats["active_loss_weight"] += float((loss_masks[row] * active).sum())


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
        raise SSMaxBridgeEvidenceError(f"Bridge source order differs: {names!r}")
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
    """Replay one saved-step cursor and publish its immutable loss-mass/health receipt."""

    args = _parse_args(argv)
    if args.prefetch_workers < 0 or args.checkpoint_hash_workers <= 0:
        raise ValueError("Worker counts must be non-negative (hash workers must be positive)")
    manifest_path = args.manifest.expanduser().resolve()
    if sha256_file(manifest_path) != args.expected_manifest_sha256:
        raise SSMaxBridgeEvidenceError("Run manifest differs from its explicit CLI pin")
    manifest = load_manifest(manifest_path, verify_live=False)
    producer_source = validate_manifest_producer_source(
        manifest,
        producer=HEALTH_PRODUCER,
        source_path=Path(__file__),
    )
    candidate = validate_checkpoint_reference(
        manifest["checkpoints"][str(args.step)],
        expected_step=args.step,
        workers=args.checkpoint_hash_workers,
        verify_live=True,
    )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable health receipt {output}")
    recipe_path = validate_artifact_reference(manifest["recipe"], name="training recipe")
    recipe = _load_recipe(recipe_path)
    raw_config = load_json(Path(str(candidate["path"])) / "config.json")
    if not isinstance(raw_config, Mapping):
        raise SSMaxBridgeEvidenceError("Candidate config must contain an object")
    config = recipe.ExperimentConfig.from_dict(raw_config)
    if (
        str(config.phase) != "bridge"
        or str(config.model_variant) != manifest["model_variant"]
        or config.required_run_name != manifest["run_name"]
        or config.data.pack_sequences is not False
    ):
        raise SSMaxBridgeEvidenceError("Candidate is not the manifest's unpacked SSMax bridge")
    tokenizer, token_ids = recipe._load_tokenizer(config.artifacts)
    rank_states, rank_inventory = _trainer_rank_states(Path(str(candidate["path"])))
    world_size = int(manifest["topology"]["world_size"])
    if len(rank_states) != world_size:
        raise SSMaxBridgeEvidenceError("Trainer rank-state count differs from manifest topology")
    try:
        ledger_summary = extract_ssmax_health_ledgers(
            rank_states,
            expected_model_variant=str(manifest["model_variant"]),
            expected_phase="bridge",
            expected_run_name=str(manifest["run_name"]),
            expected_step=args.step,
            expected_world_size=world_size,
        )
    except SSMaxHealthLedgerError as error:
        raise SSMaxBridgeEvidenceError(f"Checkpoint health ledgers are invalid: {error}") from error
    stats = _empty_stats()
    checkpoint_states = []
    replayed_states = []
    initial_states = []
    dataset_fingerprints: Any = None
    total_data_errors = 0
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    for rank, trainer_state in enumerate(rank_states):
        saved, epoch = _validate_trainer_cursor(
            trainer_state, step=args.step, world_size=world_size, rank=rank
        )
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
            raise SSMaxBridgeEvidenceError("Dataset fingerprints differ across replay ranks")
        if epoch is not None:
            loader.reshuffle(epoch=epoch)
        initial_states.append({"rank": rank, "state": _jsonable(loader.state_dict())})
        iterator = iter(loader)
        try:
            for _ in range(args.step):
                batch = next(iterator)
                _accumulate_batch(stats, batch)
            replayed = _jsonable(loader.state_dict())
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        if replayed != _jsonable(loader.state_dict()) or replayed != saved:
            raise SSMaxBridgeEvidenceError(f"Trainer rank{rank} replay cursor differs")
        checkpoint_states.append({"rank": rank, "state": saved})
        replayed_states.append({"rank": rank, "state": replayed})
        total_data_errors += int(saved.get("total_data_errors", 0))

    targets = dict(config.train_module.source_loss_mass_targets or {})
    if set(targets) != set(SOURCES):
        raise SSMaxBridgeEvidenceError("Bridge source loss-mass targets differ from policy")
    total_weight = sum(stats[source]["loss_weight"] for source in SOURCES)
    total_active = sum(stats[source]["active_loss_weight"] for source in SOURCES)
    delivery_observed = args.step > 0
    if delivery_observed and (total_weight <= 0 or total_active <= 0):
        raise SSMaxBridgeEvidenceError("Nonzero bridge step has empty supervised loss mass")
    sources: dict[str, Any] = {}
    tolerance = float(manifest["policy"]["loss_mass_share_tolerance"])
    for source in SOURCES:
        values = stats[source]
        target = float(targets[source])
        share = values["loss_weight"] / total_weight if delivery_observed else target
        active_share = values["active_loss_weight"] / total_active if delivery_observed else target
        sources[source] = {
            "delivery_observed": delivery_observed,
            "examples": int(values["examples"]),
            "tokens": int(values["tokens"]),
            "positive_tokens": int(values["positive_tokens"]),
            "loss_weight": values["loss_weight"],
            "active_loss_weight": values["active_loss_weight"],
            "loss_mass_share": share,
            "active_loss_mass_share": active_share,
            "target_loss_mass": target,
            "absolute_error": abs(share - target),
            "active_absolute_error": abs(active_share - target),
        }
    within_tolerance = all(
        sources[source][field] <= tolerance
        for source in SOURCES
        for field in ("absolute_error", "active_absolute_error")
    )
    maximum_errors = int(manifest["policy"]["maximum_data_errors"])
    ledger_counters = dict(ledger_summary["counters"])
    if ledger_counters["data_errors"] != total_data_errors:
        raise SSMaxBridgeEvidenceError("Ledger and replay data-error totals differ")
    counters_ok = (
        ledger_counters["data_errors"] <= maximum_errors
        and ledger_counters["optimizer_guard_skips"] == 0
        and ledger_counters["nonfinite_losses"] == 0
        and ledger_counters["nonfinite_gradients"] == 0
    )
    created_at = args.created_at or datetime.now(timezone.utc).isoformat()
    receipt: dict[str, Any] = {
        "format": HEALTH_RECEIPT_FORMAT,
        "version": SCHEMA_VERSION,
        "status": ("passed" if within_tolerance and counters_ok else "failed"),
        "created_at": created_at,
        "manifest": manifest_reference(manifest_path, manifest),
        "pair_id": manifest["pair_id"],
        "arm": manifest["arm"],
        "model_variant": manifest["model_variant"],
        "step": args.step,
        "checkpoint": dict(candidate),
        "protocol": {
            "name": "exact-unpacked-loader-cumulative-loss-mass-v1",
            "start_step": 0,
            "end_step": args.step,
            "share_tolerance": tolerance,
            "packing": False,
        },
        "loader": {
            "data_contract_sha256": config.vision_alignment.data_contract_sha256,
            "dataset_fingerprints_sha256": canonical_sha256(dataset_fingerprints),
            "initial_state_sha256": canonical_sha256(initial_states),
            "checkpoint_final_state_sha256": canonical_sha256(checkpoint_states),
            "replayed_final_state_sha256": canonical_sha256(replayed_states),
            "rank_state_inventory_sha256": canonical_sha256(rank_inventory),
            "rank_state_count": len(rank_states),
            "rank_states_global_step": args.step,
            "rank_states_batches_processed": args.step,
            "dp_world_size": world_size,
            "batches_replayed": args.step,
            "total_data_errors": total_data_errors,
        },
        "sources": sources,
        "health_ledger": ledger_summary,
        "summary": {
            "delivery_observed": delivery_observed,
            "total_loss_weight": total_weight,
            "total_active_loss_weight": total_active,
            "share_sum": sum(sources[source]["loss_mass_share"] for source in SOURCES),
            "active_share_sum": sum(
                sources[source]["active_loss_mass_share"] for source in SOURCES
            ),
            "within_tolerance": within_tolerance,
            "within_data_error_budget": total_data_errors <= maximum_errors,
            "zero_optimizer_guard_skips": ledger_counters["optimizer_guard_skips"] == 0,
            "finite_losses": ledger_counters["nonfinite_losses"] == 0,
            "finite_gradients": ledger_counters["nonfinite_gradients"] == 0,
        },
        "evidence": {
            "recipe": dict(manifest["recipe"]),
            "producer": producer_source,
            "rank_state_inventory": rank_inventory,
        },
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    write_json_once(output, receipt)


if __name__ == "__main__":
    main()
