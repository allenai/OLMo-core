"""Evaluate one manifest-bound SSMax bridge checkpoint with fixed matched-wrong rows.

Launch this script with the exact distributed world pinned by the finalized per-model manifest.
The only checkpoint selector is ``--step``; arbitrary checkpoint paths are intentionally absent.
For each selected step the evaluator:

* rehashes the step-0 reference and selected checkpoint against the manifest;
* builds the generic :class:`MultimodalLM` and HSDP eval module from the saved config;
* proves exact DCP model key/shape/dtype coverage and performs a strict optimizer-free load;
* hashes all frozen LM/vision tensors and every non-image input-embedding row against step 0;
* evaluates the same immutable correct/exact-geometry-wrong rows and records per-example CE.
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
from torch.distributed.tensor import DTensor

from olmo_core.data.multimodal import MultimodalCollator, MultimodalDataLoader
from olmo_core.distributed.checkpoint import get_checkpoint_metadata
from olmo_core.distributed.checkpoint import (
    load_state_dict as load_distributed_state_dict,
)
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.eval import (
    MultimodalFixedValidationDataset,
    MultimodalMatchedWrongImageDataset,
)
from olmo_core.eval.ssmax_attention_diagnostics import (
    SSMaxAttentionDiagnosticsCollector,
    SSMaxProbeManifest,
    capture_ssmax_probe_batches,
    iter_ssmax_probe_batches,
)
from olmo_core.eval.vision_alignment_ssmax_bridge import (
    IMAGE_TOKEN_ROWS,
    MATCHED_STATE_PRODUCER,
    MATCHED_STATE_RECEIPT_FORMAT,
    REQUIRED_STEPS,
    SCHEMA_VERSION,
    SOURCES,
    WINDOWS,
    SSMaxBridgeEvidenceError,
    aggregate_matched_records,
    canonical_sha256,
    load_json,
    load_manifest,
    manifest_reference,
    sha256_file,
    validate_artifact_reference,
    validate_checkpoint_reference,
    validate_manifest_producer_source,
    verify_generic_dcp_load_inventory,
    write_json_once,
)
from olmo_core.eval.vision_alignment_ssmax_data import (
    build_validation_datasets,
    load_fixed_pairing,
)
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.vision import MultimodalLM, MultimodalLMConfig
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.train_module.transformer import (
    MultimodalTransformerTrainModuleConfig,
)
from olmo_core.utils import gc_cuda, move_to_device


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--step", type=int, choices=REQUIRED_STEPS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-load-threads", type=int, default=8)
    parser.add_argument("--checkpoint-hash-workers", type=int, default=8)
    return parser.parse_args(argv)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    local = local.detach().cpu().contiguous()
    return local.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")


def _local_tensor_descriptor(tensor: torch.Tensor) -> dict[str, Any]:
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return {
        "local_shape": list(local.shape),
        "sha256": hashlib.sha256(_tensor_bytes(tensor)).hexdigest(),
    }


def _logical_tensor_descriptor(name: str, kind: str, tensor: torch.Tensor) -> dict[str, Any]:
    local = {
        "name": name,
        "kind": kind,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "local": _local_tensor_descriptor(tensor),
    }
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local)
    for rank, packet in enumerate(gathered):
        if not isinstance(packet, Mapping):
            raise RuntimeError(f"Frozen-state rank {rank} descriptor is malformed")
        for field in ("name", "kind", "dtype", "shape", "numel"):
            if packet.get(field) != local[field]:
                raise RuntimeError(f"Frozen-state tensor {name!r} differs across ranks")
    rank_shards = [{"rank": rank, **dict(packet["local"])} for rank, packet in enumerate(gathered)]
    return {
        "name": name,
        "kind": kind,
        "dtype": local["dtype"],
        "shape": local["shape"],
        "numel": local["numel"],
        "sha256": canonical_sha256(rank_shards),
    }


def _replicated_row_descriptor(
    name: str,
    kind: str,
    tensor: torch.Tensor,
    row_ids: Sequence[int],
    *,
    invert: bool,
) -> dict[str, Any]:
    full = tensor.full_tensor() if isinstance(tensor, DTensor) else tensor
    full = full.detach()
    keep = torch.zeros(full.shape[0], dtype=torch.bool, device=full.device)
    keep[list(row_ids)] = True
    if invert:
        keep.logical_not_()
    selected = full[keep]
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        packet[0] = {
            "name": name,
            "kind": kind,
            "dtype": str(selected.dtype),
            "shape": list(selected.shape),
            "numel": selected.numel(),
            "sha256": hashlib.sha256(_tensor_bytes(selected)).hexdigest(),
        }
    dist.broadcast_object_list(packet, src=0)
    descriptor = packet[0]
    if not isinstance(descriptor, Mapping):
        raise RuntimeError(f"Could not broadcast row descriptor {name!r}")
    return dict(descriptor)


def _frozen_state_descriptors(model: MultimodalLM) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    embedding_name = "lm.embeddings.weight"
    embedding = parameters.get(embedding_name)
    if embedding is None:
        raise RuntimeError("Could not locate the LM input embedding table")
    unexpected_trainable = sorted(
        name
        for name, parameter in parameters.items()
        if (name.startswith("vision.") or (name.startswith("lm.") and name != embedding_name))
        and parameter.requires_grad
    )
    if unexpected_trainable:
        raise RuntimeError(
            f"Bridge frozen-state surface has trainable LM/vision tensors: {unexpected_trainable}"
        )
    descriptors: dict[str, dict[str, Any]] = {}
    for name, tensor in sorted(parameters.items()):
        if name == embedding_name or not name.startswith(("lm.", "vision.")):
            continue
        descriptors[name] = _logical_tensor_descriptor(name, "frozen_parameter", tensor)
    for name, tensor in sorted(buffers.items()):
        if not name.startswith(("lm.", "vision.")):
            continue
        descriptors[f"buffer:{name}"] = _logical_tensor_descriptor(
            f"buffer:{name}", "frozen_buffer", tensor
        )
    non_image_name = "lm.embeddings.weight[non_image_rows]"
    descriptors[non_image_name] = _replicated_row_descriptor(
        non_image_name,
        "non_image_embedding_rows",
        embedding,
        IMAGE_TOKEN_ROWS,
        invert=True,
    )
    image_name = "lm.embeddings.weight[image_rows]"
    image_rows = _replicated_row_descriptor(
        image_name,
        "image_embedding_rows",
        embedding,
        IMAGE_TOKEN_ROWS,
        invert=False,
    )
    connector: dict[str, dict[str, Any]] = {}
    for name, tensor in sorted(parameters.items()):
        if name.startswith("connector."):
            connector[name] = _logical_tensor_descriptor(name, "connector_parameter", tensor)
    for name, tensor in sorted(buffers.items()):
        if name.startswith("connector."):
            qualified = f"buffer:{name}"
            connector[qualified] = _logical_tensor_descriptor(qualified, "connector_buffer", tensor)
    return {"frozen": descriptors, "image_rows": image_rows, "connector": connector}


def _component_state(descriptors: Mapping[str, Any]) -> dict[str, Any]:
    frozen = descriptors["frozen"]
    vision = [dict(value) for name, value in sorted(frozen.items()) if "vision." in name]
    connector = [dict(value) for _, value in sorted(descriptors["connector"].items())]
    image_rows = dict(descriptors["image_rows"])
    payload = {
        "protocol": "same-topology-rank-shard-sha256-v1",
        "vision": {
            "tensor_count": len(vision),
            "inventory_sha256": canonical_sha256(vision),
            "tensors": vision,
        },
        "connector": {
            "tensor_count": len(connector),
            "inventory_sha256": canonical_sha256(connector),
            "tensors": connector,
        },
        "image_embedding_rows": image_rows,
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def _compare_frozen_state(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    left = reference["frozen"]
    right = candidate["frozen"]
    if set(left) != set(right):
        raise RuntimeError("Reference and candidate frozen tensor inventories differ")
    comparisons = []
    for name in sorted(left):
        reference_descriptor = left[name]
        candidate_descriptor = right[name]
        for field in ("name", "kind", "dtype", "shape", "numel"):
            if reference_descriptor[field] != candidate_descriptor[field]:
                raise RuntimeError(f"Frozen tensor metadata differs for {name!r}")
        comparisons.append(
            {
                "name": name,
                "kind": reference_descriptor["kind"],
                "dtype": reference_descriptor["dtype"],
                "shape": reference_descriptor["shape"],
                "numel": reference_descriptor["numel"],
                "reference_sha256": reference_descriptor["sha256"],
                "candidate_sha256": candidate_descriptor["sha256"],
            }
        )
    mismatches = [
        comparison
        for comparison in comparisons
        if comparison["reference_sha256"] != comparison["candidate_sha256"]
    ]
    payload = {
        "complete": True,
        "protocol": "same-topology-rank-shard-sha256-v1",
        "image_embedding_rows": list(IMAGE_TOKEN_ROWS),
        "expected_frozen_tensor_count": len(comparisons),
        "compared_frozen_tensor_count": len(comparisons),
        "mismatch_count": len(mismatches),
        "comparison_inventory_sha256": canonical_sha256(comparisons),
        "reference_image_rows_sha256": reference["image_rows"]["sha256"],
        "candidate_image_rows_sha256": candidate["image_rows"]["sha256"],
        "comparisons": comparisons,
    }
    payload["sha256"] = canonical_sha256(payload)
    return payload


def _build_model_and_module(
    raw_config: Mapping[str, Any], *, rank_batch_instances: int
) -> tuple[MultimodalLM, Any]:
    model_config = MultimodalLMConfig.from_dict(raw_config["model"])
    model = model_config.build(init_device="meta")
    if not isinstance(model, MultimodalLM):
        raise TypeError("SSMax bridge checkpoint did not build a generic MultimodalLM")
    sequence_length = int(raw_config["data"]["sequence_length"])
    module_config = MultimodalTransformerTrainModuleConfig.from_dict(raw_config["train_module"])
    module_config.rank_microbatch_size = rank_batch_instances * sequence_length
    module_config.max_sequence_length = sequence_length
    module_config.compile_model = False
    module_config.vision_activation_checkpointing = False
    module_config.connector_activation_checkpointing = False
    module_config.response_logits_only = True
    module_config.diagnostics_interval = None
    train_module = module_config.build(model, eval_only=True)
    return model, train_module


def _strict_load(train_module: Any, checkpoint: Path, *, threads: int) -> dict[str, Any]:
    state_dir = checkpoint / "model_and_optim"
    metadata = get_checkpoint_metadata(state_dir)
    state = train_module.state_dict_to_load(metadata, optim=False)
    model = train_module.multimodal_model
    inventory = verify_generic_dcp_load_inventory(
        metadata=metadata,
        state_dict_to_load=state,
        parameter_names=tuple(name for name, _ in model.named_parameters()),
        buffer_names=tuple(name for name, _ in model.named_buffers()),
    ).as_dict()
    load_distributed_state_dict(
        state_dir,
        state,
        process_group=dist.group.WORLD,
        thread_count=threads,
    )
    train_module.load_state_dict(state)
    inventory.pop("sha256")
    inventory["load_completed"] = True
    inventory["sha256"] = canonical_sha256(inventory)
    return inventory


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
        raise RuntimeError("Correct and wrong image tensors have different geometry")


def _response_ce_by_example(
    batch: Mapping[str, torch.Tensor], logits: torch.Tensor
) -> list[dict[str, Any]]:
    labels = batch["labels"]
    weights = batch["loss_masks"].float()
    mask = weights > 0
    counts = mask.sum(dim=1)
    if bool(torch.any(counts <= 0)) or bool(torch.any(labels.masked_select(mask) == -100)):
        raise RuntimeError("Every matched row must contain valid supervised response tokens")
    selected_labels = labels.masked_select(mask)
    selected_weights = weights.masked_select(mask)
    if logits.ndim == 3:
        selected_logits = logits.reshape(-1, logits.shape[-1])[mask.reshape(-1)]
    elif logits.ndim == 2 and logits.shape[0] == selected_labels.numel():
        selected_logits = logits
    else:
        raise RuntimeError("Response-only logits do not align with supervised tokens")
    token_ce = F.cross_entropy(selected_logits.float(), selected_labels, reduction="none")
    records: list[dict[str, Any]] = []
    offset = 0
    limits = {"first_8": 8, "first_32": 32, "all": None}
    for count_tensor in counts:
        count = int(count_tensor.item())
        ce = token_ce[offset : offset + count]
        row_weights = selected_weights[offset : offset + count]
        windows: dict[str, float] = {}
        for window in WINDOWS:
            limit = limits[window]
            width = count if limit is None else min(count, limit)
            value = (ce[:width] * row_weights[:width]).sum() / row_weights[:width].sum()
            windows[window] = float(value.detach().cpu())
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
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    correct_dataset = MultimodalFixedValidationDataset(
        dataset, pairing=pairing, pairing_sha256=pairing_sha256
    )
    wrong_dataset = MultimodalMatchedWrongImageDataset(
        dataset, pairing=pairing, pairing_sha256=pairing_sha256
    )
    group = train_module.dp_process_group
    if group is None:
        raise RuntimeError("SSMax evaluation requires an explicit data-parallel group")
    dp_world_size = get_world_size(group)
    dp_rank = get_rank(group)
    global_instances = rank_batch_instances * dp_world_size
    if len(correct_dataset) % global_instances:
        raise RuntimeError("Matched population is not divisible by the global instance batch")
    configured_sequence_length = collator.pad_sequence_length
    if configured_sequence_length is None:
        raise RuntimeError("Matched evaluation collator must pin its sequence length")
    sequence_length = int(configured_sequence_length)

    def loader(name: str, selected: Any) -> MultimodalDataLoader:
        return MultimodalDataLoader(
            selected,
            collator,
            work_dir=work_dir / source / name,
            global_batch_size=global_instances * sequence_length,
            seed=int(pairing["seed"]),
            shuffle=False,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )

    correct_loader = loader("correct", correct_dataset)
    wrong_loader = loader("wrong", wrong_dataset)
    correct_loader.reshuffle(epoch=1)
    wrong_loader.reshuffle(epoch=1)
    pair_rows = list(pairing["pairs"])
    local_records: list[dict[str, Any]] = []
    started = time.monotonic()
    for batch_index, (correct_batch, wrong_batch) in enumerate(
        zip(correct_loader, wrong_loader, strict=True)
    ):
        _assert_batches_match(correct_batch, wrong_batch)
        local_start = batch_index * global_instances + dp_rank * rank_batch_instances
        local_pairs = pair_rows[local_start : local_start + rank_batch_instances]
        correct_device = move_to_device(correct_batch, train_module.device)
        correct_output = train_module.eval_batch(correct_device, return_response_logits=True)
        wrong_device = move_to_device(wrong_batch, train_module.device)
        wrong_output = train_module.eval_batch(wrong_device, return_response_logits=True)
        if (
            not isinstance(correct_output, LMOutputWithLoss)
            or correct_output.logits is None
            or not isinstance(wrong_output, LMOutputWithLoss)
            or wrong_output.logits is None
        ):
            raise RuntimeError("Matched forwards did not return response-only logits")
        correct_ce = _response_ce_by_example(correct_device, correct_output.logits)
        wrong_ce = _response_ce_by_example(wrong_device, wrong_output.logits)
        if len(local_pairs) != len(correct_ce) or len(correct_ce) != len(wrong_ce):
            raise RuntimeError("Pairing order and local response rows diverged")
        for offset, pair in enumerate(local_pairs):
            if correct_ce[offset]["response_tokens"] != wrong_ce[offset]["response_tokens"]:
                raise RuntimeError("Correct/wrong forwards retained different response rows")
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
    gathered: list[Any] = [None for _ in range(dp_world_size)]
    dist.all_gather_object(gathered, local_records, group=group)
    records = [record for rank_records in gathered for record in rank_records]
    records.sort(key=lambda row: row["pairing_position"])
    if [row["pairing_position"] for row in records] != list(range(len(correct_dataset))):
        raise RuntimeError("Distributed matched rows are incomplete or duplicated")
    return {
        "pairing_sha256": pairing_sha256,
        "examples": len(records),
        "elapsed_seconds": time.monotonic() - started,
        "metrics": aggregate_matched_records(
            records,
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        ),
        "per_example": records,
    }


def _run_attention_probe(
    train_module: Any,
    dataset: Any,
    *,
    content_ids: Sequence[str],
    probe_path: Path,
    probe_sha256: str,
    collator: MultimodalCollator,
    checkpoint_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    probe = SSMaxProbeManifest.load(
        probe_path,
        expected_sha256=probe_sha256,
        verify_validation_manifest=True,
    )
    population = probe.payload.get("population")
    if not isinstance(population, Mapping) or (
        population.get("source") != "pixmo_caption"
        or population.get("split") != "validation"
        or population.get("epoch") != 0
    ):
        raise SSMaxBridgeEvidenceError("Attention probe population is incompatible")
    world_size = dist.get_world_size(train_module.dp_process_group)
    rank = dist.get_rank(train_module.dp_process_group)
    if len(probe.rows_by_sample_id) < world_size:
        raise SSMaxBridgeEvidenceError("Attention probe has fewer rows than DP ranks")
    with SSMaxAttentionDiagnosticsCollector(
        train_module.multimodal_model.lm,
        probe,
        query_chunk_size=8,
    ) as collector:
        batches = iter_ssmax_probe_batches(
            dataset,
            probe,
            content_ids=content_ids,
            collate=lambda examples: move_to_device(collator(examples), train_module.device),
            rank=rank,
            world_size=world_size,
            batch_size=(len(probe.rows_by_sample_id) + world_size - 1) // world_size,
        )

        def forward_batch(batch: Mapping[str, Any]) -> None:
            output = train_module.eval_batch(dict(batch), return_response_logits=False)
            if not isinstance(output, LMOutputWithLoss):
                raise RuntimeError("Attention probe forward did not return a loss output")

        local_state = capture_ssmax_probe_batches(
            collector,
            batches,
            forward_batch=forward_batch,
        )
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_state, group=train_module.dp_process_group)
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        packet[0] = SSMaxAttentionDiagnosticsCollector.finalize_states(
            probe,
            gathered,
            checkpoint_identity=checkpoint_identity,
        )
    dist.broadcast_object_list(packet, src=0)
    report = packet[0]
    if not isinstance(report, Mapping):
        raise RuntimeError("Could not broadcast finalized attention diagnostics")
    gc_cuda()
    return report


def _write_distributed(path: Path, payload: Mapping[str, Any]) -> None:
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            write_json_once(path, payload)
            packet[0] = {"ok": True}
        except Exception as error:  # noqa: BLE001 - propagate every rank-zero write failure.
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Could not publish immutable eval receipt: {detail}")


def _validate_checkpoint_distributed(
    reference: Mapping[str, Any], *, expected_step: int, workers: int
) -> Mapping[str, Any]:
    """Rehash one full checkpoint on rank zero and share the verified identity.

    Native checkpoints contain many large DCP shards. Having every evaluator rank hash every
    shard would multiply shared-filesystem traffic by the world size without adding independent
    evidence, since all ranks see the same Weka namespace. Rank zero performs the complete live
    byte verification and broadcasts either the exact identity or the failure to every rank.
    """

    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            packet[0] = {
                "ok": True,
                "identity": dict(
                    validate_checkpoint_reference(
                        reference,
                        expected_step=expected_step,
                        workers=workers,
                        verify_live=True,
                    )
                ),
            }
        except Exception as error:  # noqa: BLE001 - propagate every verification failure.
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Could not verify step{expected_step} checkpoint: {detail}")
    identity = result.get("identity")
    if not isinstance(identity, Mapping):
        raise RuntimeError(f"Verified step{expected_step} checkpoint identity is malformed")
    return identity


def main(argv: Sequence[str] | None = None) -> None:
    """Run one strict generic-DCP matched/state evaluation selected by manifest step."""

    args = _parse_args(argv)
    if args.checkpoint_load_threads <= 0 or args.checkpoint_hash_workers <= 0:
        raise ValueError("Checkpoint thread/worker counts must be positive")
    manifest_path = args.manifest.expanduser().resolve()
    if sha256_file(manifest_path) != args.expected_manifest_sha256:
        raise SSMaxBridgeEvidenceError("Run manifest differs from its explicit CLI pin")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable eval receipt {output}")
    manifest = load_manifest(manifest_path, verify_live=False)
    evaluator_source = validate_manifest_producer_source(
        manifest,
        producer=MATCHED_STATE_PRODUCER,
        source_path=Path(__file__),
    )
    expected_world = int(manifest["topology"]["world_size"])
    if int(os.environ.get("WORLD_SIZE", "1")) != expected_world:
        raise ValueError(f"Manifest requires WORLD_SIZE={expected_world}")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prepare_training_environment()
    try:
        reference = _validate_checkpoint_distributed(
            manifest["checkpoints"]["0"],
            expected_step=0,
            workers=args.checkpoint_hash_workers,
        )
        candidate = (
            reference
            if args.step == 0
            else _validate_checkpoint_distributed(
                manifest["checkpoints"][str(args.step)],
                expected_step=args.step,
                workers=args.checkpoint_hash_workers,
            )
        )
        config_path = Path(str(candidate["path"])) / "config.json"
        raw_config = load_json(config_path)
        if not isinstance(raw_config, Mapping):
            raise SSMaxBridgeEvidenceError("Checkpoint config must contain an object")
        if (
            raw_config.get("model_variant") != manifest["model_variant"]
            or raw_config.get("required_run_name") != manifest["run_name"]
            or raw_config.get("phase") != "bridge"
            or raw_config.get("data", {}).get("pack_sequences") is not False
        ):
            raise SSMaxBridgeEvidenceError("Checkpoint config violates the SSMax bridge identity")
        evaluation = manifest["evaluation"]
        rank_batch = int(evaluation["rank_batch_instances"])
        model, train_module = _build_model_and_module(raw_config, rank_batch_instances=rank_batch)

        reference_load = _strict_load(
            train_module, Path(str(reference["path"])), threads=args.checkpoint_load_threads
        )
        reference_state = _frozen_state_descriptors(model)
        if args.step == 0:
            candidate_load = dict(reference_load)
            candidate_state = reference_state
        else:
            candidate_load = _strict_load(
                train_module, Path(str(candidate["path"])), threads=args.checkpoint_load_threads
            )
            candidate_state = _frozen_state_descriptors(model)
        frozen_state = _compare_frozen_state(reference_state, candidate_state)
        component_state = _component_state(candidate_state)

        validation_path = validate_artifact_reference(
            manifest["validation"], name="validation manifest"
        )
        tokenizer, _, datasets, content_ids, validation_identity = build_validation_datasets(
            raw_config,
            manifest_path=validation_path,
            manifest_sha256=manifest["validation"]["sha256"],
        )
        pairings: dict[str, Mapping[str, Any]] = {}
        for source in SOURCES:
            pairing_reference = manifest["pairings"][source]
            pairings[source] = load_fixed_pairing(
                Path(pairing_reference["path"]),
                expected_sha256=pairing_reference["sha256"],
                dataset_size=len(datasets[source]),
                examples=int(evaluation["examples_per_source"]),
                seed=int(evaluation["pairing_seed"]),
                content_ids=content_ids,
            )
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
                bootstrap_seed=int(evaluation["bootstrap_seed"])
                + SOURCES.index(source) * 1_000_000,
                bootstrap_samples=int(evaluation["bootstrap_samples"]),
            )
            for source in SOURCES
        }
        attention_probe_path = validate_artifact_reference(
            manifest["attention_probe"], name="SSMax attention probe"
        )
        attention_diagnostics = _run_attention_probe(
            train_module,
            datasets["pixmo_caption"],
            content_ids=content_ids,
            probe_path=attention_probe_path,
            probe_sha256=manifest["attention_probe"]["sha256"],
            collator=collator,
            checkpoint_identity=candidate,
        )
        protocol = {
            "name": "vision-alignment-ssmax-fixed-matched-wrong-v1",
            "population": "fixed_exact_geometry_matched_validation_rows",
            "gap_sign": "wrong_ce - correct_ce; positive is a correct-image win",
            "windows": list(WINDOWS),
            "bootstrap": {
                "method": "deterministic iid paired-example percentile bootstrap",
                "confidence": 0.95,
                "seed": int(evaluation["bootstrap_seed"]),
                "samples": int(evaluation["bootstrap_samples"]),
            },
            "world_size": expected_world,
            "data_parallel": manifest["topology"]["data_parallel"],
            "rank_batch_instances": rank_batch,
            "sequence_length": sequence_length,
        }
        protocol["sha256"] = canonical_sha256(protocol)
        payload: dict[str, Any] = {
            "format": MATCHED_STATE_RECEIPT_FORMAT,
            "version": SCHEMA_VERSION,
            "status": "passed" if frozen_state["mismatch_count"] == 0 else "failed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "manifest": manifest_reference(manifest_path, manifest),
            "pair_id": manifest["pair_id"],
            "arm": manifest["arm"],
            "model_variant": manifest["model_variant"],
            "step": args.step,
            "checkpoint": dict(candidate),
            "step0_checkpoint": dict(reference),
            "strict_generic_dcp_load": candidate_load,
            "step0_strict_generic_dcp_load": reference_load,
            "frozen_state": frozen_state,
            "component_state": component_state,
            "validation": validation_identity,
            "pairings": {source: dict(manifest["pairings"][source]) for source in SOURCES},
            "protocol": protocol,
            "results": results,
            "attention_diagnostics": attention_diagnostics,
            "evaluator": evaluator_source,
        }
        payload["content_sha256"] = canonical_sha256(payload)
        _write_distributed(output, payload)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
