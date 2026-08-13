"""Replay the paired perception loader and emit immutable cumulative loss-mass evidence.

The two causal arms must contain byte-semantic identical version-5 loader cursors on all 16
data-parallel ranks.  After proving that equality, this CPU-only audit reconstructs the exact
pinned production recipe under its original ``__main__`` identity and replays one shared arm
from step 0 through step 4000.  Existing receipt paths are never replaced.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.machinery
import io
import json
import math
import os
import sys
import types
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from olmo_core.data.multimodal import MixtureDataLoader
from olmo_core.eval.vision_alignment_perception_promotion import (
    ARMS,
    CONTROL_ARM,
    LOSS_MASS_ABSOLUTE_TOLERANCE,
    LOSS_MASS_PAIR_RECEIPT_FORMAT,
    LOSS_MASS_TARGETS,
    PRIMARY_STEP,
    RECEIPT_VERSION,
    SOURCES,
    TREATMENT_ARM,
    PromotionValidationError,
    artifact_reference,
    candidate_from_outcome_receipt,
    canonical_sha256,
    load_json_pinned,
    sha256_file,
    validate_loss_mass_pair_receipt,
)

METRIC_NAMES = ("examples", "tokens", "positive_tokens", "loss_weight", "active_loss_weight")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--treatment-checkpoint", type=Path, required=True)
    parser.add_argument("--perception-outcome", type=Path, required=True)
    parser.add_argument("--expected-perception-outcome-sha256", required=True)
    parser.add_argument("--pair-contract", type=Path, required=True)
    parser.add_argument("--expected-pair-contract-sha256", required=True)
    parser.add_argument(
        "--recipe",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "train" / "Vision-Alignment.py",
    )
    parser.add_argument("--expected-recipe-sha256", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefetch-workers", type=int, default=0)
    parser.add_argument("--created-at")
    return parser.parse_args(argv)


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If) or node.orelse:
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


@contextmanager
def _recipe_main_module(recipe: types.ModuleType) -> Iterator[None]:
    previous = sys.modules.get("__main__")
    sys.modules["__main__"] = recipe
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("__main__", None)
        else:
            sys.modules["__main__"] = previous


def _load_recipe(path: Path, expected_sha256: str) -> types.ModuleType:
    """Execute the exact pinned recipe definitions using production script semantics."""
    path = path.expanduser().resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PromotionValidationError("Vision Alignment recipe differs from its explicit pin")
    tree = ast.parse(raw, filename=str(path))
    guards = [index for index, node in enumerate(tree.body) if _is_main_guard(node)]
    if guards != [len(tree.body) - 1]:
        raise PromotionValidationError(
            "Pinned recipe must contain exactly one final __main__ CLI guard"
        )
    tree.body.pop()
    module = types.ModuleType("__main__")
    module.__file__ = str(path)
    module.__loader__ = importlib.machinery.SourceFileLoader("__main__", str(path))
    module.__package__ = None
    module.__spec__ = None
    with _recipe_main_module(module):
        exec(compile(tree, str(path), "exec"), module.__dict__)  # noqa: S102
    for symbol in ("ExperimentConfig", "_load_tokenizer", "_build_mixture_sources"):
        if not hasattr(module, symbol):
            raise PromotionValidationError(f"Pinned recipe omits {symbol}")
    return module


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _jsonable(value.item())
        return _jsonable(value.detach().cpu().tolist())
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise PromotionValidationError("Loader state contains a non-finite float")
        return value
    raise PromotionValidationError(f"Loader state contains unsupported value {type(value)!r}")


def _empty_stats() -> dict[str, dict[str, float]]:
    return {source: {metric: 0.0 for metric in METRIC_NAMES} for source in SOURCES}


def _accumulate_batch(stats: dict[str, dict[str, float]], batch: Mapping[str, Any]) -> None:
    """Accumulate the exact per-source fields emitted by the production train module."""
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


def _exact_rank_state_paths(checkpoint: Path) -> list[Path]:
    train = checkpoint.expanduser().resolve() / "train"
    expected = [train / f"rank{rank}.pt" for rank in range(16)]
    observed = sorted(train.glob("rank*.pt"), key=lambda path: path.name)
    if set(observed) != set(expected) or any(path.is_symlink() for path in observed):
        raise PromotionValidationError(
            "Perception rank states must be the exact checkpoint/train/rank{0..15}.pt files"
        )
    return expected


def _load_trainer_state_and_sha256(path: Path) -> tuple[Mapping[str, Any], str]:
    """Safely decode and hash one trainer state from the same immutable byte buffer."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PromotionValidationError(f"Could not read trainer state {path}: {error}") from error
    digest = hashlib.sha256(raw).hexdigest()
    allowed_globals = [
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        type(np.dtype("uint32")),
        type(np.dtype("int64")),
        type(np.dtype("float64")),
        type(np.dtype("bool")),
    ]
    try:
        with torch.serialization.safe_globals(allowed_globals):
            value = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as error:
        raise PromotionValidationError(
            f"Could not safely load trainer state {path}: {error}"
        ) from error
    if not isinstance(value, Mapping):
        raise PromotionValidationError(f"Trainer state {path} must be an object")
    return value, digest


def _rank_states(checkpoint: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = _exact_rank_state_paths(checkpoint)
    if len(paths) != 16:  # Defensive: keep the invariant local if the helper changes.
        raise PromotionValidationError("Perception checkpoint must contain 16 trainer states")
    states: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for expected_rank, path in enumerate(paths):
        rank = int(path.stem.removeprefix("rank"))
        if rank != expected_rank:
            raise PromotionValidationError("Trainer rank-state inventory is not contiguous")
        payload, digest = _load_trainer_state_and_sha256(path)
        loader = payload.get("data_loader") if isinstance(payload, Mapping) else None
        if not isinstance(loader, Mapping):
            raise PromotionValidationError(f"Rank{rank} trainer state lacks its data loader")
        states.append(dict(payload))
        inventory.append({"rank": rank, "path": str(path.resolve()), "sha256": digest})
    return states, inventory


def _loader_state_inventory(states: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"rank": rank, "state": _jsonable(state["data_loader"])}
        for rank, state in enumerate(states)
    ]


def _prove_arm_cursor_equality(
    control_states: Sequence[Mapping[str, Any]],
    treatment_states: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Prove rankwise semantic equality and return the canonical shared cursor inventory."""
    control = _loader_state_inventory(control_states)
    treatment = _loader_state_inventory(treatment_states)
    if control != treatment:
        differing = [
            rank
            for rank, (left, right) in enumerate(zip(control, treatment, strict=True))
            if left != right
        ]
        raise PromotionValidationError(
            f"Control and treatment saved loader cursors differ on ranks {differing}"
        )
    return control, canonical_sha256(control)


def _candidate_fields(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": candidate["checkpoint"],
        "global_step": candidate["global_step"],
        "checkpoint_config_sha256": candidate["checkpoint_config_sha256"],
        "checkpoint_identity_sha256": candidate["checkpoint_identity_sha256"],
    }


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


def _decode_config(recipe: types.ModuleType, checkpoint: Path, *, expected_sha256: str) -> Any:
    raw = load_json_pinned(
        checkpoint / "config.json",
        expected_sha256,
        name=f"{checkpoint.name} checkpoint config",
    )
    if not isinstance(raw, Mapping):
        raise PromotionValidationError("Checkpoint config must be a JSON object")
    with _recipe_main_module(recipe):
        config = recipe.ExperimentConfig.from_dict(raw)
    if str(config.phase) != "perception" or config.global_batch_size <= 0:
        raise PromotionValidationError("Loss-mass replay requires a perception checkpoint")
    return config


def _validate_saved_state(
    state: Mapping[str, Any], *, rank: int, world_size: int
) -> Mapping[str, Any]:
    loader = _jsonable(state["data_loader"])
    packing = loader.get("packing_state")
    if (
        state.get("global_step") != PRIMARY_STEP
        or state.get("world_size") != world_size
        or loader.get("batches_processed") != PRIMARY_STEP
        or loader.get("total_data_errors") != 0
        or not isinstance(packing, Mapping)
        or packing.get("version") != 5
        or packing.get("dp_rank") != rank
        or packing.get("dp_world_size") != world_size
    ):
        raise PromotionValidationError(f"Trainer rank{rank} packing cursor is incompatible")
    return loader


def main(argv: Sequence[str] | None = None) -> None:
    """Replay all 16 shared data-loader ranks and write the canonical paired receipt."""
    args = _parse_args(argv)
    if args.prefetch_workers < 0:
        raise ValueError("--prefetch-workers must be non-negative")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable receipt {output}")
    outcome_path = args.perception_outcome.expanduser().resolve()
    pair_path = args.pair_contract.expanduser().resolve()
    if sha256_file(pair_path) != args.expected_pair_contract_sha256:
        raise PromotionValidationError("Pair contract differs from its explicit SHA-256 pin")
    outcome = load_json_pinned(
        outcome_path,
        args.expected_perception_outcome_sha256,
        name="perception outcome",
    )
    if not isinstance(outcome, Mapping):
        raise PromotionValidationError("Perception outcome must be a JSON object")
    control = candidate_from_outcome_receipt(
        args.control_checkpoint, outcome, role="control", verify_live_contents=True
    )
    treatment = candidate_from_outcome_receipt(
        args.treatment_checkpoint, outcome, role="treatment", verify_live_contents=True
    )

    control_states, control_inventory = _rank_states(Path(control["checkpoint"]))
    treatment_states, treatment_inventory = _rank_states(Path(treatment["checkpoint"]))
    saved_inventory, saved_cursor_sha = _prove_arm_cursor_equality(control_states, treatment_states)
    world_size = len(control_states)
    saved_loaders = [
        _validate_saved_state(state, rank=rank, world_size=world_size)
        for rank, state in enumerate(control_states)
    ]
    if canonical_sha256(saved_inventory) != saved_cursor_sha:
        raise AssertionError("Shared saved cursor identity changed during validation")

    recipe = _load_recipe(args.recipe, args.expected_recipe_sha256)
    config = _decode_config(
        recipe,
        Path(control["checkpoint"]),
        expected_sha256=str(control["checkpoint_config_sha256"]),
    )
    treatment_config = _decode_config(
        recipe,
        Path(treatment["checkpoint"]),
        expected_sha256=str(treatment["checkpoint_config_sha256"]),
    )
    if (
        config.data_seed != treatment_config.data_seed
        or config.global_batch_size != treatment_config.global_batch_size
        or config.data.as_config_dict() != treatment_config.data.as_config_dict()
        or config.collator.as_config_dict() != treatment_config.collator.as_config_dict()
        or config.train_module.source_loss_mass_targets
        != treatment_config.train_module.source_loss_mass_targets
    ):
        raise PromotionValidationError("Causal arms differ in their loader-replay configuration")
    if config.vision_alignment.data_contract_sha256 != control["data_contract_sha256"]:
        raise PromotionValidationError("Live recipe data contract differs from the outcome")
    with _recipe_main_module(recipe):
        tokenizer, token_ids = recipe._load_tokenizer(config.artifacts)
        datasets, weights, names = recipe._build_mixture_sources(tokenizer, token_ids, config)
    if tuple(names) != SOURCES:
        raise PromotionValidationError(f"Perception replay sources differ: {names!r}")
    targets = dict(config.train_module.source_loss_mass_targets or {})
    if targets != LOSS_MASS_TARGETS:
        raise PromotionValidationError("Perception source-loss-mass targets differ from policy")
    collator = config.collator.build()
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    totals = _empty_stats()
    replayed_states: list[dict[str, Any]] = []
    dataset_fingerprints: Any = None

    for rank, saved_loader in enumerate(saved_loaders):
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
            prefetch_workers=args.prefetch_workers,
            dataset_names=names,
            allow_legacy_state_without_dataset_fingerprints=False,
            dp_world_size=world_size,
            dp_rank=rank,
        )
        current_fingerprints = _jsonable(loader.dataset_fingerprints)
        packing = saved_loader["packing_state"]
        if dataset_fingerprints is None:
            dataset_fingerprints = current_fingerprints
        elif current_fingerprints != dataset_fingerprints:
            raise PromotionValidationError("Dataset fingerprints differ across replay ranks")
        if current_fingerprints != _jsonable(packing.get("dataset_fingerprints")):
            raise PromotionValidationError(f"Trainer rank{rank} dataset fingerprints differ")
        loader.reshuffle(epoch=int(saved_loader["epoch"]))
        iterator = iter(loader)
        try:
            for step in range(PRIMARY_STEP):
                batch = next(iterator)
                _accumulate_batch(totals, batch)
                del batch
                if rank == 0 and (step == 0 or (step + 1) % 100 == 0):
                    print(f"rank0 replayed {step + 1}/{PRIMARY_STEP} batches", flush=True)
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
        replayed_states.append({"rank": rank, "state": replayed})
        print(f"completed exact loader replay for rank {rank}/{world_size - 1}", flush=True)

    replayed_cursor_sha = canonical_sha256(replayed_states)
    if replayed_cursor_sha != saved_cursor_sha:
        raise PromotionValidationError("Replayed final cursor differs from both causal arms")
    total_loss_weight = sum(totals[source]["loss_weight"] for source in SOURCES)
    total_active_loss_weight = sum(totals[source]["active_loss_weight"] for source in SOURCES)
    if total_loss_weight <= 0 or total_active_loss_weight <= 0:
        raise PromotionValidationError("Replayed perception loss mass is empty")
    sources: dict[str, Any] = {}
    for source in SOURCES:
        metrics = totals[source]
        share = metrics["loss_weight"] / total_loss_weight
        active_share = metrics["active_loss_weight"] / total_active_loss_weight
        target = LOSS_MASS_TARGETS[source]
        sources[source] = {
            "examples": int(metrics["examples"]),
            "tokens": int(metrics["tokens"]),
            "positive_tokens": int(metrics["positive_tokens"]),
            "loss_weight": metrics["loss_weight"],
            "active_loss_weight": metrics["active_loss_weight"],
            "target_loss_mass": target,
            "loss_mass_share": share,
            "active_loss_mass_share": active_share,
            "absolute_error": abs(share - target),
            "active_absolute_error": abs(active_share - target),
        }
    within_tolerance = all(
        sources[source][field] <= LOSS_MASS_ABSOLUTE_TOLERANCE
        for source in SOURCES
        for field in ("absolute_error", "active_absolute_error")
    )
    if not within_tolerance:
        raise PromotionValidationError("Cumulative source loss mass exceeds the locked tolerance")

    arm_inventories = {
        CONTROL_ARM: control_inventory,
        TREATMENT_ARM: treatment_inventory,
    }
    arm_loader = {
        arm: {
            "rank_states_global_step": PRIMARY_STEP,
            "rank_states_batches_processed": PRIMARY_STEP,
            "rank_state_inventory_sha256": canonical_sha256(arm_inventories[arm]),
            "checkpoint_final_state_sha256": saved_cursor_sha,
            "replayed_final_state_sha256": replayed_cursor_sha,
        }
        for arm in ARMS
    }
    receipt: dict[str, Any] = {
        "format": LOSS_MASS_PAIR_RECEIPT_FORMAT,
        "version": RECEIPT_VERSION,
        "status": "passed",
        "created_at": args.created_at or datetime.now(timezone.utc).isoformat(),
        "producer": artifact_reference(Path(__file__).resolve()),
        "pair_contract": artifact_reference(pair_path),
        "candidate": _candidate_fields(treatment),
        "comparator": _candidate_fields(control),
        "protocol": {
            "name": "exact-packed-loader-paired-cumulative-loss-mass-v1",
            "start_step": 0,
            "end_step": PRIMARY_STEP,
            "exact_packing_cursor": True,
            "share_tolerance": LOSS_MASS_ABSOLUTE_TOLERANCE,
            "arm_cursor_equality": True,
        },
        "loader": {
            "dp_world_size": world_size,
            "batches_replayed": PRIMARY_STEP,
            "rank_state_count": world_size,
            "total_data_errors": 0,
            "dataset_fingerprints_sha256": canonical_sha256(dataset_fingerprints),
            "replayed_final_state_sha256": replayed_cursor_sha,
            "arms": arm_loader,
        },
        "evidence": {
            "recipe": artifact_reference(args.recipe.expanduser().resolve()),
            "producer": artifact_reference(Path(__file__).resolve()),
            "rank_state_inventory": arm_inventories,
        },
        "sources": sources,
        "summary": {
            "total_loss_weight": total_loss_weight,
            "total_active_loss_weight": total_active_loss_weight,
            "share_sum": sum(sources[source]["loss_mass_share"] for source in SOURCES),
            "active_share_sum": sum(
                sources[source]["active_loss_mass_share"] for source in SOURCES
            ),
            "within_tolerance": within_tolerance,
            "arm_final_cursor_equal": True,
        },
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    validate_loss_mass_pair_receipt(receipt, candidate=treatment, comparator=control)
    _write_json_once(output, receipt)
    print(
        json.dumps(
            {
                "path": str(output),
                "sha256": sha256_file(output),
                "final_cursor_sha256": replayed_cursor_sha,
                "sources": sources,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
