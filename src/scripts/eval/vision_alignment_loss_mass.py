"""Replay the exact bridge loader and emit a cumulative supervised-loss-mass receipt.

This is a CPU-only audit. It reconstructs the pinned bridge datasets through the production
recipe, replays every data-parallel rank from step 0 through the candidate step, and requires
the resulting version-5 packed-loader cursor to equal every saved trainer rank state. The
receipt sums the same per-source fields as the train module's runtime telemetry.

The replay is intentionally exhaustive and can take hours because it formats and preprocesses
the original images. Existing output files are never replaced.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from olmo_core.data.multimodal import MixtureDataLoader
from olmo_core.eval.vision_alignment_promotion import (
    LOSS_MASS_RECEIPT_FORMAT,
    PromotionValidationError,
    artifact_reference,
    candidate_from_matched_receipt,
    canonical_sha256,
    load_json,
    sha256_file,
    validate_loss_mass_receipt,
)

SOURCE_NAMES = ("pixmo_caption", "pixmo_transcript")
METRIC_NAMES = ("examples", "tokens", "positive_tokens", "loss_weight", "active_loss_weight")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--matched-step500", type=Path, required=True)
    parser.add_argument("--expected-matched-step500-sha256", required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "train" / "Vision-Alignment.py",
    )
    parser.add_argument("--expected-recipe-sha256", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefetch-workers", type=int)
    parser.add_argument("--created-at")
    return parser.parse_args(argv)


def _load_recipe(path: Path, expected_sha256: str):
    path = path.expanduser().resolve()
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise PromotionValidationError("Vision Alignment recipe differs from its explicit pin")
    module_name = "_vision_alignment_recipe_for_loss_mass_replay"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Vision Alignment recipe from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    # The production checkpoint was written while the recipe ran as ``__main__``. Dataset
    # type is part of the version-5 cursor fingerprint, so reproduce that stable identity.
    module._AuditedDataset.__module__ = "__main__"
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
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise PromotionValidationError("Loader state contains a non-finite float")
        return value
    raise PromotionValidationError(f"Loader state contains unsupported value {type(value)!r}")


def _empty_stats() -> dict[str, dict[str, float]]:
    return {source: {metric: 0.0 for metric in METRIC_NAMES} for source in SOURCE_NAMES}


def _accumulate_batch(stats: dict[str, dict[str, float]], batch: Mapping[str, Any]) -> None:
    """Accumulate the exact fields used by ``_record_source_data_metrics`` on CPU."""
    required = {"pack_source_names", "example_ids", "router_token_mask", "loss_masks", "labels"}
    missing = required - set(batch)
    if missing:
        raise PromotionValidationError(f"Replayed packed batch omits {sorted(missing)}")
    packed_sources = batch["pack_source_names"]
    example_ids = batch["example_ids"]
    token_mask = batch["router_token_mask"]
    loss_masks = batch["loss_masks"]
    labels = batch["labels"]
    if not isinstance(packed_sources, list) or len(packed_sources) != int(example_ids.shape[0]):
        raise PromotionValidationError("Packed source metadata does not match the replay batch")
    for row, source_names in enumerate(packed_sources):
        if not isinstance(source_names, list):
            raise PromotionValidationError("Packed source metadata must contain source lists")
        for example_id, source_name in enumerate(source_names):
            if source_name not in stats:
                raise PromotionValidationError(f"Replay observed unknown source {source_name!r}")
            positions = (example_ids[row] == example_id) & token_mask[row]
            active_positions = positions & (labels[row] != -100)
            source_stats = stats[source_name]
            source_stats["examples"] += 1.0
            source_stats["tokens"] += float(positions.sum(dtype=torch.long).item())
            source_stats["positive_tokens"] += float(
                ((loss_masks[row] > 0) & active_positions).sum(dtype=torch.long).item()
            )
            source_stats["loss_weight"] += float((loss_masks[row] * positions).sum().item())
            source_stats["active_loss_weight"] += float(
                (loss_masks[row] * active_positions).sum().item()
            )


def _trainer_rank_states(checkpoint: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = sorted(
        checkpoint.joinpath("train").glob("rank*.pt"),
        key=lambda path: int(path.stem.removeprefix("rank")),
    )
    if not paths:
        raise PromotionValidationError("Candidate checkpoint has no trainer rank states")
    states: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for expected_rank, path in enumerate(paths):
        rank = int(path.stem.removeprefix("rank"))
        if rank != expected_rank:
            raise PromotionValidationError("Trainer rank-state inventory is not contiguous")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("data_loader"), Mapping):
            raise PromotionValidationError(f"Trainer rank{rank} state lacks its data loader")
        states.append(dict(payload))
        inventory.append({"rank": rank, "path": str(path.resolve()), "sha256": sha256_file(path)})
    if any(state.get("world_size") != len(states) for state in states):
        raise PromotionValidationError("Trainer rank states disagree on their world size")
    return states, inventory


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite immutable receipt {path}") from error
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> None:
    """Replay every rank and write one immutable cumulative loss-mass receipt."""
    args = _parse_args(argv)
    if args.prefetch_workers is not None and args.prefetch_workers < 0:
        raise ValueError("--prefetch-workers must be non-negative")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable receipt {output}")
    matched_path = args.matched_step500.expanduser().resolve()
    if sha256_file(matched_path) != args.expected_matched_step500_sha256:
        raise PromotionValidationError("Primary step500 receipt differs from its explicit pin")
    matched = load_json(matched_path)
    if not isinstance(matched, Mapping):
        raise PromotionValidationError("Primary step500 receipt must be an object")
    checkpoint = args.checkpoint.expanduser().resolve()
    candidate = candidate_from_matched_receipt(checkpoint, matched)

    recipe = _load_recipe(args.recipe, args.expected_recipe_sha256)
    raw_config = load_json(checkpoint / "config.json")
    if not isinstance(raw_config, Mapping):
        raise PromotionValidationError("Candidate config must be an object")
    config = recipe.ExperimentConfig.from_dict(raw_config)
    if str(config.phase) != "bridge" or config.global_batch_size <= 0:
        raise PromotionValidationError("Loss-mass replay requires the bridge configuration")
    if config.vision_alignment.data_contract_sha256 != candidate["data_contract_sha256"]:
        raise PromotionValidationError("Live recipe data contract differs from the candidate")

    tokenizer, token_ids = recipe._load_tokenizer(config.artifacts)
    datasets, weights, names = recipe._build_mixture_sources(tokenizer, token_ids, config)
    if tuple(names) != SOURCE_NAMES:
        raise PromotionValidationError(f"Bridge replay sources differ: {names!r}")
    collator = config.collator.build()
    rank_states, rank_inventory = _trainer_rank_states(checkpoint)
    dp_world_size = len(rank_states)
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    totals = _empty_stats()
    initial_states: list[dict[str, Any]] = []
    checkpoint_states: list[dict[str, Any]] = []
    replayed_states: list[dict[str, Any]] = []
    dataset_fingerprints: Any = None
    total_data_errors = 0

    for rank, trainer_state in enumerate(rank_states):
        saved_loader = _jsonable(trainer_state["data_loader"])
        if (
            trainer_state.get("global_step") != candidate["global_step"]
            or saved_loader.get("batches_processed") != candidate["global_step"]
        ):
            raise PromotionValidationError(f"Trainer rank{rank} is not at candidate step500")
        packing = saved_loader.get("packing_state")
        if not isinstance(packing, Mapping) or packing.get("version") != 5:
            raise PromotionValidationError(f"Trainer rank{rank} lacks an exact v5 packing cursor")
        if packing.get("dp_rank") != rank or packing.get("dp_world_size") != dp_world_size:
            raise PromotionValidationError(f"Trainer rank{rank} packing topology differs")
        loader = MixtureDataLoader(
            datasets,
            weights,
            collator,
            work_dir=work_dir / f"rank{rank}",
            global_batch_size=config.global_batch_size,
            seed=config.data_seed,
            pack=config.data.pack_sequences,
            pack_max_crops=config.data.pack_max_crops,
            pack_buffer_size=config.data.pack_buffer_size,
            prefetch_workers=(
                config.data.prefetch_workers
                if args.prefetch_workers is None
                else args.prefetch_workers
            ),
            dataset_names=names,
            allow_legacy_state_without_dataset_fingerprints=False,
            dp_world_size=dp_world_size,
            dp_rank=rank,
        )
        current_fingerprints = _jsonable(loader.dataset_fingerprints)
        if dataset_fingerprints is None:
            dataset_fingerprints = current_fingerprints
        elif current_fingerprints != dataset_fingerprints:
            raise PromotionValidationError("Dataset fingerprints differ across replay ranks")
        if current_fingerprints != _jsonable(packing.get("dataset_fingerprints")):
            raise PromotionValidationError(f"Trainer rank{rank} dataset fingerprints differ")
        loader.reshuffle(epoch=int(saved_loader["epoch"]))
        initial_states.append({"rank": rank, "state": _jsonable(loader.state_dict())})
        iterator = iter(loader)
        try:
            for step in range(candidate["global_step"]):
                batch = next(iterator)
                _accumulate_batch(totals, batch)
                del batch
                if rank == 0 and (step == 0 or (step + 1) % 50 == 0):
                    print(
                        f"rank0 replayed {step + 1}/{candidate['global_step']} batches", flush=True
                    )
            replayed = _jsonable(loader.state_dict())
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        if replayed != _jsonable(loader.state_dict()):
            raise PromotionValidationError(f"Trainer rank{rank} cursor changed while closing")
        if replayed != saved_loader:
            raise PromotionValidationError(
                f"Trainer rank{rank} replay cursor differs from checkpoint"
            )
        checkpoint_states.append({"rank": rank, "state": saved_loader})
        replayed_states.append({"rank": rank, "state": replayed})
        total_data_errors += int(replayed["total_data_errors"])
        print(f"completed exact loader replay for rank {rank}/{dp_world_size - 1}", flush=True)

    targets = dict(config.train_module.source_loss_mass_targets or {})
    if set(targets) != set(SOURCE_NAMES):
        raise PromotionValidationError("Bridge source-loss-mass targets differ from policy")
    total_loss_weight = sum(totals[source]["loss_weight"] for source in SOURCE_NAMES)
    total_active_loss_weight = sum(totals[source]["active_loss_weight"] for source in SOURCE_NAMES)
    if total_loss_weight <= 0 or total_active_loss_weight <= 0:
        raise PromotionValidationError("Replayed bridge loss mass is empty")
    sources: dict[str, Any] = {}
    for source in SOURCE_NAMES:
        metrics = totals[source]
        share = metrics["loss_weight"] / total_loss_weight
        active_share = metrics["active_loss_weight"] / total_active_loss_weight
        target = float(targets[source])
        sources[source] = {
            "examples": int(metrics["examples"]),
            "tokens": int(metrics["tokens"]),
            "positive_tokens": int(metrics["positive_tokens"]),
            "loss_weight": metrics["loss_weight"],
            "active_loss_weight": metrics["active_loss_weight"],
            "loss_mass_share": share,
            "active_loss_mass_share": active_share,
            "target_loss_mass": target,
            "absolute_error": abs(share - target),
            "active_absolute_error": abs(active_share - target),
        }

    created_at = args.created_at or datetime.now(timezone.utc).isoformat()
    receipt: dict[str, Any] = {
        "format": LOSS_MASS_RECEIPT_FORMAT,
        "version": 1,
        "status": "passed",
        "created_at": created_at,
        "candidate": {
            "checkpoint": candidate["checkpoint"],
            "global_step": candidate["global_step"],
            "checkpoint_config_sha256": candidate["checkpoint_config_sha256"],
            "checkpoint_identity_sha256": candidate["checkpoint_identity_sha256"],
        },
        "protocol": {
            "name": "exact-packed-loader-cumulative-loss-mass-v1",
            "start_step": 0,
            "end_step": candidate["global_step"],
            "share_tolerance": 0.02,
            "exact_packing_cursor": True,
        },
        "loader": {
            "data_contract_sha256": candidate["data_contract_sha256"],
            "dataset_fingerprints_sha256": canonical_sha256(dataset_fingerprints),
            "initial_state_sha256": canonical_sha256(initial_states),
            "checkpoint_final_state_sha256": canonical_sha256(checkpoint_states),
            "replayed_final_state_sha256": canonical_sha256(replayed_states),
            "rank_state_inventory_sha256": canonical_sha256(rank_inventory),
            "rank_state_count": len(rank_states),
            "rank_states_global_step": candidate["global_step"],
            "rank_states_batches_processed": candidate["global_step"],
            "dp_world_size": dp_world_size,
            "batches_replayed": candidate["global_step"],
            "total_data_errors": total_data_errors,
        },
        "evidence": {
            "recipe": artifact_reference(args.recipe.expanduser().resolve()),
            "producer": artifact_reference(Path(__file__).resolve()),
            "rank_state_inventory": rank_inventory,
        },
        "sources": sources,
        "summary": {
            "total_loss_weight": total_loss_weight,
            "total_active_loss_weight": total_active_loss_weight,
            "share_sum": sum(sources[source]["loss_mass_share"] for source in SOURCE_NAMES),
            "active_share_sum": sum(
                sources[source]["active_loss_mass_share"] for source in SOURCE_NAMES
            ),
            "within_tolerance": all(
                sources[source][field] <= 0.02
                for source in SOURCE_NAMES
                for field in ("absolute_error", "active_absolute_error")
            ),
        },
    }
    validate_loss_mass_receipt(receipt, candidate=candidate)
    _write_json_once(output, receipt)
    print(
        json.dumps(
            {"path": str(output), "sha256": sha256_file(output), "sources": sources},
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
