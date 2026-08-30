"""Replay SSMax joint data cursors and emit run-health/loss-mass evidence.

This CPU producer rebuilds the unpacked nine-source loader for every saved rank and advances it
from step 0 to the selected permanent checkpoint.  Cursor equality and absence of packed or
multi-branch rows are hard requirements.  Observed loss-mass shares and optimizer skips are
reported descriptively; data and non-finite counters are hard collapse invariants.  Every counter
comes from the resume-safe hash-chained health ledger inside the raw-pinned trainer states; W&B
history and hand-authored summaries are not accepted.
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
from olmo_core.eval.vision_alignment_ssmax_joint import (
    HEALTH_RECEIPT_FORMAT,
    REQUIRED_STEPS,
    SCHEMA_VERSION,
    TRAIN_SOURCES,
    SSMaxJointEvidenceError,
    artifact_reference,
    canonical_sha256,
    load_json,
    load_manifest,
    manifest_reference,
    resolve_repository_artifact,
    sha256_file,
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


def _load_recipe(path: Path) -> Any:
    name = "_vision_alignment_ssmax_joint_health_recipe"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import pinned recipe {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
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
    raise SSMaxJointEvidenceError(f"loader state contains unsupported value {value!r}")


def _trainer_rank_states(checkpoint: Path) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    paths = sorted(
        checkpoint.joinpath("train").glob("rank*.pt"),
        key=lambda path: int(path.stem.removeprefix("rank")),
    )
    if not paths:
        raise SSMaxJointEvidenceError("checkpoint has no trainer rank states")
    states = []
    inventory = []
    for rank, path in enumerate(paths):
        if path.name != f"rank{rank}.pt":
            raise SSMaxJointEvidenceError("trainer rank-state inventory is not contiguous")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or not isinstance(payload.get("data_loader"), Mapping):
            raise SSMaxJointEvidenceError(f"trainer rank{rank} lacks a data-loader state")
        states.append(payload)
        inventory.append({"rank": rank, "sha256": sha256_file(path)})
    return states, inventory


def _empty_stats() -> dict[str, dict[str, float]]:
    return {source: {metric: 0.0 for metric in _METRICS} for source in TRAIN_SOURCES}


def _validate_one_branch(batch: Mapping[str, Any], *, row: int, source: str) -> None:
    if "subsegment_ids" in batch or "example_ids" in batch or "pack_source_names" in batch:
        raise SSMaxJointEvidenceError("joint replay contains packed sequence metadata")
    labels = batch["labels"][row]
    weights = batch["loss_masks"][row]
    if labels.ndim != 1 or weights.shape != labels.shape:
        raise SSMaxJointEvidenceError(f"{source} replay labels/loss masks differ")
    if not bool((weights > 0).any()):
        raise SSMaxJointEvidenceError(f"{source} replay row has no supervised response")


def _accumulate_batch(stats: dict[str, dict[str, float]], batch: Mapping[str, Any]) -> None:
    required = {"source_names", "router_token_mask", "loss_masks", "labels"}
    if missing := required - set(batch):
        raise SSMaxJointEvidenceError(f"replayed batch omits {sorted(missing)}")
    names = batch["source_names"]
    labels = batch["labels"]
    if not isinstance(names, list) or len(names) != int(labels.shape[0]):
        raise SSMaxJointEvidenceError("unpacked source metadata differs from batch rows")
    for row, source in enumerate(names):
        if source not in stats:
            raise SSMaxJointEvidenceError(f"replay observed unknown source {source!r}")
        _validate_one_branch(batch, row=row, source=source)
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
    if set(names) != set(TRAIN_SOURCES) or len(names) != len(TRAIN_SOURCES):
        raise SSMaxJointEvidenceError(f"joint source set differs: {names!r}")
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
        raise ValueError("worker counts must be non-negative (hash workers positive)")
    manifest_path = args.manifest.expanduser().resolve()
    if sha256_file(manifest_path) != args.expected_manifest_sha256:
        raise SSMaxJointEvidenceError("joint manifest differs from its CLI pin")
    manifest = load_manifest(manifest_path, verify_live=False)
    try:
        candidate = bridge.validate_checkpoint_reference(
            manifest["checkpoints"][str(args.step)],
            expected_step=args.step,
            workers=args.checkpoint_hash_workers,
            verify_live=True,
        )
    except bridge.SSMaxBridgeEvidenceError as error:
        raise SSMaxJointEvidenceError(str(error)) from error
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable health receipt {output}")
    recipe_path = resolve_repository_artifact(manifest["recipe"], name="training recipe")
    recipe = _load_recipe(recipe_path)
    raw_config = load_json(Path(str(candidate["path"])) / "config.json")
    if not isinstance(raw_config, Mapping):
        raise SSMaxJointEvidenceError("candidate config must be an object")
    config = recipe.ExperimentConfig.from_dict(raw_config)
    if (
        str(config.phase) != "joint"
        or str(config.model_variant) != manifest["model_variant"]
        or config.required_run_name != manifest["run_name"]
        or config.data.pack_sequences is not False
    ):
        raise SSMaxJointEvidenceError("candidate is not the manifest joint run")
    tokenizer, token_ids = recipe._load_tokenizer(config.artifacts)
    states, inventory = _trainer_rank_states(Path(str(candidate["path"])))
    world = int(manifest["topology"]["world_size"])
    if len(states) != world:
        raise SSMaxJointEvidenceError("trainer rank-state count differs from topology")
    try:
        ledger_summary = extract_ssmax_health_ledgers(
            states,
            expected_model_variant=str(manifest["model_variant"]),
            expected_phase="joint",
            expected_run_name=str(manifest["run_name"]),
            expected_step=args.step,
            expected_world_size=world,
        )
    except SSMaxHealthLedgerError as error:
        raise SSMaxJointEvidenceError(f"checkpoint health ledgers are invalid: {error}") from error
    stats = _empty_stats()
    rank_receipts = []
    dataset_fingerprints: Any = None
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    for rank, trainer_state in enumerate(states):
        saved = _jsonable(trainer_state["data_loader"])
        if (
            trainer_state.get("global_step") != args.step
            or trainer_state.get("world_size") != world
            or saved.get("batches_processed") != args.step
            or "packing_state" in saved
        ):
            raise SSMaxJointEvidenceError(f"trainer rank{rank} cursor is incompatible")
        ledger = ledger_summary["rank_ledgers"][rank]
        loader = _build_loader(
            recipe,
            config,
            tokenizer,
            token_ids,
            rank=rank,
            world_size=world,
            work_dir=work_dir,
            prefetch_workers=args.prefetch_workers,
        )
        current_fingerprints = _jsonable(loader.dataset_fingerprints)
        if dataset_fingerprints is None:
            dataset_fingerprints = current_fingerprints
        elif current_fingerprints != dataset_fingerprints:
            raise SSMaxJointEvidenceError("dataset fingerprints differ across ranks")
        epoch = saved.get("epoch")
        if args.step == 0:
            if epoch is not None:
                raise SSMaxJointEvidenceError(
                    f"pre-train checkpoint rank{rank} unexpectedly has an epoch cursor"
                )
            replayed = _jsonable(loader.state_dict())
        else:
            if type(epoch) is not int or epoch <= 0:
                raise SSMaxJointEvidenceError(f"trainer rank{rank} epoch is invalid")
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
        if replayed != saved:
            raise SSMaxJointEvidenceError(f"trainer rank{rank} replay cursor differs")
        rank_receipts.append(
            {
                "rank": rank,
                "global_step": args.step,
                "batches_processed": int(saved["batches_processed"]),
                "data_loader_state_sha256": canonical_sha256(saved),
                "trainer_state_sha256": inventory[rank]["sha256"],
                "health_ledger": dict(ledger),
            }
        )
    targets = dict(config.train_module.source_loss_mass_targets or {})
    if targets != manifest["loss_mass_targets"]:
        raise SSMaxJointEvidenceError("runtime source loss-mass targets differ")
    sources = {
        source: {
            "examples": int(stats[source]["examples"]),
            "tokens": int(stats[source]["tokens"]),
            "positive_tokens": int(stats[source]["positive_tokens"]),
            "loss_weight": stats[source]["loss_weight"],
            "active_loss_weight": stats[source]["active_loss_weight"],
            "target_loss_mass": float(targets[source]),
        }
        for source in TRAIN_SOURCES
    }
    counters = dict(ledger_summary["counters"])
    policy = manifest["policy"]
    status_ok = (
        counters["data_errors"] <= policy["maximum_data_errors"]
        and counters["optimizer_guard_skips"] <= policy["maximum_optimizer_guard_skips"]
        and counters["nonfinite_losses"] <= policy["maximum_nonfinite_losses"]
        and counters["nonfinite_gradients"] <= policy["maximum_nonfinite_gradients"]
    )
    receipt: dict[str, Any] = {
        "format": HEALTH_RECEIPT_FORMAT,
        "version": SCHEMA_VERSION,
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
            "recipe": artifact_reference(recipe_path),
            "producer": artifact_reference(Path(__file__).resolve()),
        },
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    write_json_once(output, receipt)


if __name__ == "__main__":
    main()
