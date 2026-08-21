"""Produce one manifest-bound SSMax perception outcome/state/text receipt.

Run this with the exact distributed topology in the finalized causal-pair manifest.  Checkpoint
and arm overrides are deliberately absent: ``--arm`` and ``--step`` select only a member of the
closed manifest.  The producer performs a strict generic model-only DCP load, hashes logical
model surfaces relative to that arm's step 0, evaluates immutable correct/exact-geometry-wrong
rows for all eight perception sources, and runs the fixed native-text sentinel.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from olmo_core.data.multimodal import MultimodalCollator, MultimodalDataLoader
from olmo_core.data.multimodal.ssmax_single_response import (
    SSMAX_SINGLE_RESPONSE_PROJECTION_SEED,
    SSMaxSingleResponseDataset,
)
from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    build_selected_perception_dataset,
    load_perception_provenance_manifest,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    load_pinned_vision_alignment_tokenizer,
)
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.eval import (
    MultimodalFixedValidationDataset,
    MultimodalMatchedWrongImageDataset,
)
from olmo_core.eval.vision_alignment_ssmax_perception import (
    ARMS,
    CONTROL_ARM,
    EVALUATION_PRODUCER,
    EVALUATION_RECEIPT_FORMAT,
    IMAGE_TOKEN_ROWS,
    REQUIRED_STEPS,
    SOURCES,
    WINDOWS,
    SSMaxPerceptionEvidenceError,
    canonical_sha256,
    load_json,
    load_manifest,
    manifest_reference,
    sha256_file,
    validate_artifact_reference,
    validate_manifest_producer_source,
)
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.vision import MultimodalLM
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.utils import gc_cuda, move_to_device
from scripts.eval import vision_alignment_ssmax_bridge as bridge_runner


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--step", type=int, choices=REQUIRED_STEPS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-load-threads", type=int, default=8)
    parser.add_argument("--checkpoint-hash-workers", type=int, default=8)
    return parser.parse_args(argv)


class _ModelInputDataset:
    """Drop diagnostic metadata while rejecting unknown non-model fields."""

    required = frozenset(
        {
            "input_ids",
            "labels",
            "loss_masks",
            "position_ids",
            "token_type_ids",
            "images",
            "pooled_patches_idx",
        }
    )
    optional: frozenset[str] = frozenset()

    def __init__(self, dataset: Any):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get(index, 0)

    def get(self, index: int, epoch: int = 0) -> dict[str, Any]:
        get = getattr(self.dataset, "get", None)
        example = get(index, epoch) if callable(get) else self.dataset[index]
        if not isinstance(example, Mapping):
            raise TypeError(f"Perception validation row {index} is not a mapping")
        missing = self.required - set(example)
        unknown = set(example) - self.required - self.optional - {"metadata"}
        if missing or unknown:
            raise ValueError(
                f"Perception validation row {index} fields differ: "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if "subsegment_ids" in example or "example_ids" in example:
            raise ValueError(f"Perception validation row {index} contains packed metadata")
        return {
            field: example[field]
            for field in (*sorted(self.required), *sorted(self.optional))
            if field in example
        }


def _inventory(descriptors: Mapping[str, Mapping[str, Any]], *, protocol: str) -> dict[str, Any]:
    rows = [dict(descriptors[name]) for name in sorted(descriptors)]
    return {
        "protocol": protocol,
        "tensor_count": len(rows),
        "inventory_sha256": canonical_sha256(rows),
    }


def _state_descriptors(model: MultimodalLM) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    all_tensors: dict[str, dict[str, Any]] = {}
    frozen_lm: dict[str, dict[str, Any]] = {}
    vision: dict[str, dict[str, Any]] = {}
    for name, tensor in sorted(parameters.items()):
        descriptor = bridge_runner._logical_tensor_descriptor(name, "parameter", tensor)
        all_tensors[name] = descriptor
        if name.startswith("lm.") and name != "lm.embeddings.weight":
            frozen_lm[name] = descriptor
        if name.startswith("vision."):
            vision[name] = descriptor
    for name, tensor in sorted(buffers.items()):
        key = f"buffer:{name}"
        descriptor = bridge_runner._logical_tensor_descriptor(key, "buffer", tensor)
        all_tensors[key] = descriptor
        if name.startswith("lm."):
            frozen_lm[key] = descriptor
        if name.startswith("vision."):
            vision[key] = descriptor
    embedding = parameters.get("lm.embeddings.weight")
    if embedding is None:
        raise RuntimeError("Could not locate the SSMax input embedding table")
    non_image = bridge_runner._replicated_row_descriptor(
        "lm.embeddings.weight[non_image_rows]",
        "non_image_embedding_rows",
        embedding,
        IMAGE_TOKEN_ROWS,
        invert=True,
    )
    return {
        "full_model": _inventory(all_tensors, protocol="logical-model-tensor-inventory-sha256-v1"),
        "frozen_lm": frozen_lm,
        "non_image_embedding_rows": {non_image["name"]: non_image},
        "vision": vision,
    }


def _compare_surface(
    reference: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(reference) != set(candidate):
        raise RuntimeError("Reference/candidate tensor surfaces differ")
    mismatches = 0
    for name in sorted(reference):
        left = reference[name]
        right = candidate[name]
        for field in ("name", "kind", "dtype", "shape", "numel"):
            if left[field] != right[field]:
                raise RuntimeError(f"Tensor surface metadata differs for {name!r}")
        mismatches += int(left["sha256"] != right["sha256"])
    return {
        "protocol": "logical-tensor-comparison-sha256-v1",
        "tensor_count": len(reference),
        "reference_inventory_sha256": canonical_sha256(
            [dict(reference[name]) for name in sorted(reference)]
        ),
        "candidate_inventory_sha256": canonical_sha256(
            [dict(candidate[name]) for name in sorted(candidate)]
        ),
        "mismatch_count": mismatches,
    }


def _state_receipt(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "full_model": dict(candidate["full_model"]),
        "frozen_lm": _compare_surface(reference["frozen_lm"], candidate["frozen_lm"]),
        "non_image_embedding_rows": _compare_surface(
            reference["non_image_embedding_rows"], candidate["non_image_embedding_rows"]
        ),
        "vision": _compare_surface(reference["vision"], candidate["vision"]),
    }


def _assert_batches_match(correct: Mapping[str, Any], wrong: Mapping[str, Any]) -> None:
    if set(correct) != set(wrong):
        raise RuntimeError("Correct and wrong-image batches expose different fields")
    for name in correct:
        if name == "images":
            continue
        left, right = correct[name], wrong[name]
        same = (
            left.dtype == right.dtype and torch.equal(left, right)
            if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
            else left == right
        )
        if not same:
            raise RuntimeError(f"Wrong-image batch changes recipient field {name!r}")
    if correct["images"].shape != wrong["images"].shape:
        raise RuntimeError("Correct/wrong images have different geometry")


def _response_ce_by_example(
    batch: Mapping[str, torch.Tensor], logits: torch.Tensor
) -> list[dict[str, Any]]:
    labels = batch["labels"]
    weights = batch["loss_masks"].float()
    mask = weights > 0
    counts = mask.sum(dim=1)
    if bool(torch.any(counts <= 0)) or bool(torch.any(labels.masked_select(mask) == -100)):
        raise RuntimeError("Every matched row must contain supervised response tokens")
    selected_labels = labels.masked_select(mask)
    selected_weights = weights.masked_select(mask)
    if logits.ndim == 3:
        selected_logits = logits.reshape(-1, logits.shape[-1])[mask.reshape(-1)]
    elif logits.ndim == 2 and logits.shape[0] == selected_labels.numel():
        selected_logits = logits
    else:
        raise RuntimeError("Response-only logits do not align with supervised tokens")
    token_ce = F.cross_entropy(selected_logits.float(), selected_labels, reduction="none")
    limits = {"first_1": 1, "first_8": 8, "first_32": 32, "all": None}
    records = []
    offset = 0
    for count_tensor in counts:
        count = int(count_tensor.item())
        ce = token_ce[offset : offset + count]
        row_weights = selected_weights[offset : offset + count]
        windows = {}
        for window in WINDOWS:
            limit = limits[window]
            width = count if limit is None else min(count, limit)
            windows[window] = float(
                ((ce[:width] * row_weights[:width]).sum() / row_weights[:width].sum())
                .detach()
                .cpu()
            )
        records.append({"response_tokens": count, "windows": windows})
        offset += count
    return records


def _evaluate_source(
    train_module: Any,
    dataset: Any,
    pairing: Mapping[str, Any],
    *,
    source: str,
    pairing_sha256: str,
    collator: MultimodalCollator,
    work_dir: Path,
    rank_batch_instances: int,
) -> dict[str, Any]:
    correct_dataset = MultimodalFixedValidationDataset(
        dataset, pairing=pairing, pairing_sha256=pairing_sha256
    )
    wrong_dataset = MultimodalMatchedWrongImageDataset(
        dataset, pairing=pairing, pairing_sha256=pairing_sha256
    )
    group = train_module.dp_process_group
    if group is None:
        raise RuntimeError("SSMax perception evaluation requires a DP group")
    world_size = get_world_size(group)
    rank = get_rank(group)
    global_instances = world_size * rank_batch_instances
    if len(correct_dataset) % global_instances:
        raise RuntimeError("Matched population is not divisible by the global batch")
    sequence_length = collator.pad_sequence_length
    assert sequence_length is not None

    def loader(name: str, selected: Any) -> MultimodalDataLoader:
        return MultimodalDataLoader(
            selected,
            collator,
            work_dir=work_dir / source / name,
            global_batch_size=global_instances * sequence_length,
            seed=int(pairing["seed"]),
            shuffle=False,
            dp_world_size=world_size,
            dp_rank=rank,
        )

    correct_loader = loader("correct", correct_dataset)
    wrong_loader = loader("wrong", wrong_dataset)
    correct_loader.reshuffle(epoch=1)
    wrong_loader.reshuffle(epoch=1)
    pairs = list(pairing["pairs"])
    local_records: list[dict[str, Any]] = []
    started = time.monotonic()
    for batch_index, (correct_batch, wrong_batch) in enumerate(
        zip(correct_loader, wrong_loader, strict=True)
    ):
        _assert_batches_match(correct_batch, wrong_batch)
        local_start = batch_index * global_instances + rank * rank_batch_instances
        local_pairs = pairs[local_start : local_start + rank_batch_instances]
        correct_device = move_to_device(correct_batch, train_module.device)
        wrong_device = move_to_device(wrong_batch, train_module.device)
        correct_output = train_module.eval_batch(correct_device, return_response_logits=True)
        wrong_output = train_module.eval_batch(wrong_device, return_response_logits=True)
        if correct_output.logits is None or wrong_output.logits is None:
            raise RuntimeError("Matched perception forwards did not return response logits")
        correct_ce = _response_ce_by_example(correct_device, correct_output.logits)
        wrong_ce = _response_ce_by_example(wrong_device, wrong_output.logits)
        if len(local_pairs) != len(correct_ce) or len(correct_ce) != len(wrong_ce):
            raise RuntimeError("Pairing order and local response rows diverged")
        for offset, pair in enumerate(local_pairs):
            if correct_ce[offset]["response_tokens"] != wrong_ce[offset]["response_tokens"]:
                raise RuntimeError("Correct/wrong response-token counts differ")
            local_records.append(
                {
                    "pairing_position": local_start + offset,
                    "recipient_index": int(pair["recipient"]),
                    "donor_index": int(pair["donor"]),
                    "response_tokens": correct_ce[offset]["response_tokens"],
                    "correct_ce": correct_ce[offset]["windows"],
                    "wrong_ce": wrong_ce[offset]["windows"],
                    "ce_gap_wrong_minus_correct": {
                        window: wrong_ce[offset]["windows"][window]
                        - correct_ce[offset]["windows"][window]
                        for window in WINDOWS
                    },
                }
            )
        del correct_device, wrong_device, correct_output, wrong_output
        gc_cuda()
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_records, group=group)
    records = [row for rank_rows in gathered for row in rank_rows]
    records.sort(key=lambda row: row["pairing_position"])
    if [row["pairing_position"] for row in records] != list(range(len(correct_dataset))):
        raise RuntimeError("Distributed matched rows are incomplete or duplicated")
    return {
        "pairing_sha256": pairing_sha256,
        "examples": len(records),
        "per_example": records,
        "elapsed_seconds": time.monotonic() - started,
    }


def _text_sentinel(train_module: Any, path: Path, expected_sha256: str) -> dict[str, Any]:
    if sha256_file(path) != expected_sha256:
        raise SSMaxPerceptionEvidenceError("Native text sentinel differs from its manifest pin")
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise SSMaxPerceptionEvidenceError("Native text sentinel must be an object")
    input_ids = torch.tensor([payload["input_ids"]], dtype=torch.long, device=train_module.device)
    labels = torch.tensor([payload["labels"]], dtype=torch.long, device=train_module.device)
    output = train_module.eval_batch({"input_ids": input_ids, "labels": labels})
    if not isinstance(output, LMOutputWithLoss) or output.logits is None:
        raise RuntimeError("Native text sentinel did not return logits and CE")
    result = {
        "artifact_sha256": expected_sha256,
        "input_sha256": hashlib.sha256(bridge_runner._tensor_bytes(input_ids)).hexdigest(),
        "labels_sha256": hashlib.sha256(bridge_runner._tensor_bytes(labels)).hexdigest(),
        "token_count": int((labels != -100).sum()),
        "logits_sha256": hashlib.sha256(bridge_runner._tensor_bytes(output.logits)).hexdigest(),
        "ce_sha256": hashlib.sha256(bridge_runner._tensor_bytes(output.ce_loss)).hexdigest(),
    }
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, result)
    if any(item != result for item in gathered):
        raise RuntimeError("Native text sentinel output differs across ranks")
    return result


def _load_perception_datasets(
    raw_config: Mapping[str, Any], manifest: Mapping[str, Any]
) -> tuple[Any, Mapping[str, Any], Mapping[str, tuple[str, ...]]]:
    artifacts = raw_config["artifacts"]
    tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
        identifier=artifacts["tokenizer_id"],
        revision=artifacts["tokenizer_revision"],
        expected_fingerprint=artifacts["tokenizer_fingerprint"],
        cache_dir=artifacts["hf_cache_dir"],
    )
    provenance_ref = manifest["perception_provenance"]
    provenance = load_perception_provenance_manifest(
        provenance_ref["path"],
        expected_sha256=provenance_ref["sha256"],
        verify_finevision_materialization=False,
        load_image_path_signatures=False,
    )
    projection = raw_config["data"].get("ssmax_single_response_projection")
    if not isinstance(projection, Mapping):
        raise SSMaxPerceptionEvidenceError(
            "SSMax perception checkpoint lacks its single-response projection contract"
        )
    seed = projection.get("seed")
    loss_token_weighting = raw_config["data"].get("loss_token_weighting")
    if seed != SSMAX_SINGLE_RESPONSE_PROJECTION_SEED:
        raise SSMaxPerceptionEvidenceError("SSMax projection seed is non-canonical")
    datasets = {
        source: _ModelInputDataset(
            SSMaxSingleResponseDataset(
                build_selected_perception_dataset(
                    provenance,
                    tokenizer,
                    token_ids,
                    source,
                    logical_split="validation",
                    validate_required_annotations=True,
                    verify_finevision_materialization=False,
                ),
                source_name=source,
                logical_split="validation",
                seed=seed,
                loss_token_weighting=loss_token_weighting,
            )
        )
        for source in SOURCES
    }
    if any(len(dataset) != 512 for dataset in datasets.values()):
        raise RuntimeError("Perception provenance does not expose 512 held-out rows per source")
    content_ids = {
        source: tuple(provenance.selection(source, "validation").row_image_content_sha256)
        for source in SOURCES
    }
    return tokenizer, datasets, content_ids


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.checkpoint_load_threads <= 0 or args.checkpoint_hash_workers <= 0:
        raise ValueError("Checkpoint thread/worker counts must be positive")
    manifest_path = args.manifest.expanduser().resolve()
    if sha256_file(manifest_path) != args.expected_manifest_sha256:
        raise SSMaxPerceptionEvidenceError("Pair manifest differs from its explicit CLI pin")
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable receipt {output_path}")
    manifest = load_manifest(manifest_path, verify_live=False)
    evaluator_source = validate_manifest_producer_source(
        manifest,
        producer=EVALUATION_PRODUCER,
        source_path=Path(__file__),
    )
    expected_world = int(manifest["topology"]["world_size"])
    if int(os.environ.get("WORLD_SIZE", "1")) != expected_world:
        raise ValueError(f"Manifest requires WORLD_SIZE={expected_world}")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prepare_training_environment()
    try:
        arm_manifest = manifest["arms"][args.arm]
        reference = bridge_runner._validate_checkpoint_distributed(
            arm_manifest["checkpoints"]["0"],
            expected_step=0,
            workers=args.checkpoint_hash_workers,
        )
        candidate = (
            reference
            if args.step == 0
            else bridge_runner._validate_checkpoint_distributed(
                arm_manifest["checkpoints"][str(args.step)],
                expected_step=args.step,
                workers=args.checkpoint_hash_workers,
            )
        )
        raw_config = load_json(Path(str(candidate["path"])) / "config.json")
        if not isinstance(raw_config, Mapping) or (
            raw_config.get("model_variant") != manifest["model_variant"]
            or raw_config.get("phase") != "perception"
            or raw_config.get("perception_trainability_arm") != args.arm
            or raw_config.get("required_run_name") != arm_manifest["run_name"]
            or raw_config.get("data", {}).get("pack_sequences") is not False
        ):
            raise SSMaxPerceptionEvidenceError("Selected checkpoint violates pair identity")
        rank_batch = int(manifest["evaluation"]["rank_batch_instances"])
        model, train_module = bridge_runner._build_model_and_module(
            raw_config, rank_batch_instances=rank_batch
        )
        reference_load = bridge_runner._strict_load(
            train_module,
            Path(str(reference["path"])),
            threads=args.checkpoint_load_threads,
        )
        reference_state = _state_descriptors(model)
        if args.step == 0:
            candidate_load = reference_load
            candidate_state = reference_state
        else:
            candidate_load = bridge_runner._strict_load(
                train_module,
                Path(str(candidate["path"])),
                threads=args.checkpoint_load_threads,
            )
            candidate_state = _state_descriptors(model)
        state = _state_receipt(reference_state, candidate_state)

        tokenizer, datasets, content_ids = _load_perception_datasets(raw_config, manifest)
        pairings = {
            source: load_json(
                validate_artifact_reference(manifest["pairings"][source], name=f"{source} pairing")
            )
            for source in SOURCES
        }
        sequence_length = int(raw_config["data"]["sequence_length"])
        collator = MultimodalCollator(
            pad_token_id=int(tokenizer.pad_token_id),
            label_ignore_index=-100,
            pad_sequence_length=sequence_length,
        )
        results = {
            source: _evaluate_source(
                train_module,
                datasets[source],
                pairings[source],
                source=source,
                pairing_sha256=manifest["pairings"][source]["sha256"],
                collator=collator,
                work_dir=args.work_dir.expanduser().resolve(),
                rank_batch_instances=rank_batch,
            )
            for source in SOURCES
        }
        for result in results.values():
            result.pop("elapsed_seconds")
        attention_probe_path = validate_artifact_reference(
            manifest["attention_probe"], name="SSMax attention probe"
        )
        attention_diagnostics = bridge_runner._run_attention_probe(
            train_module,
            datasets["pixmo_caption"],
            content_ids=content_ids["pixmo_caption"],
            probe_path=attention_probe_path,
            probe_sha256=manifest["attention_probe"]["sha256"],
            collator=collator,
            checkpoint_identity=candidate,
        )
        sentinel_path = validate_artifact_reference(
            manifest["text_sentinel"], name="native text sentinel"
        )
        text_result = _text_sentinel(
            train_module, sentinel_path, manifest["text_sentinel"]["sha256"]
        )
        status_ok = (
            state["frozen_lm"]["mismatch_count"] == 0
            and state["non_image_embedding_rows"]["mismatch_count"] == 0
            and (args.arm != CONTROL_ARM or state["vision"]["mismatch_count"] == 0)
        )
        payload: dict[str, Any] = {
            "format": EVALUATION_RECEIPT_FORMAT,
            "version": manifest["version"],
            "status": "passed" if status_ok else "failed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "manifest": manifest_reference(manifest_path, manifest),
            "pair_id": manifest["pair_id"],
            "model_variant": manifest["model_variant"],
            "arm": args.arm,
            "step": args.step,
            "checkpoint": dict(candidate),
            "strict_generic_dcp_load": candidate_load,
            "state": state,
            "text_sentinel": text_result,
            "attention_diagnostics": attention_diagnostics,
            "pairings": {source: dict(manifest["pairings"][source]) for source in SOURCES},
            "results": results,
            "evaluator": evaluator_source,
        }
        payload["content_sha256"] = canonical_sha256(payload)
        bridge_runner._write_distributed(output_path, payload)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
