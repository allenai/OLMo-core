"""Produce frozen-state and image-free text-retention receipts for bridge promotion.

Run on one native EP8 node. The evaluator hashes both checkpoints before model construction,
loads each through the same native checkpoint path used by matched-wrong evaluation, compares
every frozen parameter plus all non-image input-embedding rows, and replays the pinned text
sentinel under identical topology/backend settings.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import importlib.util
import os
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from olmo_core.eval.vision_alignment_promotion import (
    FROZEN_STATE_RECEIPT_FORMAT,
    IMAGE_TOKEN_ROWS,
    TEXT_RETENTION_RECEIPT_FORMAT,
    PromotionValidationError,
    artifact_reference,
    candidate_from_matched_receipt,
    canonical_sha256,
    load_json,
    sha256_file,
    validate_frozen_state_receipt,
    validate_text_retention_receipt,
    validate_text_sentinel,
)
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.utils import gc_cuda, move_to_device

WORLD_SIZE = 8


def _load_matched_evaluator():
    path = Path(__file__).with_name("vision_alignment_matched_wrong.py")
    spec = importlib.util.spec_from_file_location("_vision_alignment_matched_wrong_native", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load native matched evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--matched-step500", type=Path, required=True)
    parser.add_argument("--expected-matched-step500-sha256", required=True)
    parser.add_argument("--text-sentinel", type=Path, required=True)
    parser.add_argument("--expected-text-sentinel-sha256", required=True)
    parser.add_argument("--frozen-output", type=Path, required=True)
    parser.add_argument("--text-output", type=Path, required=True)
    parser.add_argument("--text-batch-size", type=int, default=4)
    parser.add_argument("--checkpoint-load-threads", type=int, default=8)
    parser.add_argument("--checkpoint-hash-workers", type=int, default=8)
    return parser.parse_args(argv)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach()
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    tensor = tensor.cpu().contiguous()
    return tensor.view(torch.uint8).reshape(-1).numpy().tobytes(order="C")


def _local_descriptor(tensor: torch.Tensor) -> dict[str, Any]:
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return {
        "local_shape": list(local.shape),
        "sha256": hashlib.sha256(_tensor_bytes(local)).hexdigest(),
    }


def _logical_descriptors(
    local: Mapping[str, Mapping[str, Any]], *, group
) -> dict[str, dict[str, Any]]:
    gathered: list[Any] = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(gathered, dict(local), group=group)
    if any(not isinstance(packet, Mapping) for packet in gathered):
        raise RuntimeError("Frozen-state rank descriptor is malformed")
    name_sets = [set(packet) for packet in gathered]
    if any(names != name_sets[0] for names in name_sets[1:]):
        raise RuntimeError("Frozen-state parameter names differ across ranks")
    output: dict[str, dict[str, Any]] = {}
    for name in sorted(name_sets[0]):
        first = gathered[0][name]
        for packet in gathered[1:]:
            for field in ("kind", "dtype", "shape", "numel"):
                if packet[name][field] != first[field]:
                    raise RuntimeError(f"Frozen-state descriptor {name!r} differs across ranks")
        rank_shards = [
            {"rank": rank, **packet[name]["local"]} for rank, packet in enumerate(gathered)
        ]
        output[name] = {
            "kind": first["kind"],
            "dtype": first["dtype"],
            "shape": first["shape"],
            "numel": first["numel"],
            "sha256": canonical_sha256(rank_shards),
        }
    return output


def _model_state_descriptors(train_module: Any, freeze_patterns: Sequence[str]) -> dict[str, Any]:
    local: dict[str, dict[str, Any]] = {}
    seen_parameters: set[int] = set()
    embedding: torch.Tensor | None = None
    for model_part in train_module.model_parts:
        for name, parameter in model_part.named_parameters():
            if id(parameter) in seen_parameters:
                continue
            seen_parameters.add(id(parameter))
            qualified = name
            if name == "lm.embeddings.weight":
                embedding = parameter
            if not any(fnmatch.fnmatch(qualified, pattern) for pattern in freeze_patterns):
                continue
            if qualified in local:
                raise RuntimeError(f"Frozen parameter name {qualified!r} is not unique")
            local[qualified] = {
                "kind": "frozen_tensor",
                "dtype": str(parameter.dtype),
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "local": _local_descriptor(parameter),
            }
    if embedding is None:
        raise RuntimeError("Could not locate the language-model input embedding table")
    embedding_local = embedding.to_local() if isinstance(embedding, DTensor) else embedding
    if embedding_local.shape[0] != embedding.shape[0]:
        raise RuntimeError("Input embedding vocabulary rows unexpectedly use row sharding")
    keep = torch.ones(embedding_local.shape[0], dtype=torch.bool, device=embedding_local.device)
    keep[list(IMAGE_TOKEN_ROWS)] = False
    non_image = embedding_local[keep]
    local["lm.embeddings.weight[non_image_rows]"] = {
        "kind": "non_image_embedding_rows",
        "dtype": str(embedding.dtype),
        "shape": [int(non_image.shape[0]), int(non_image.shape[1])],
        "numel": non_image.numel(),
        "local": _local_descriptor(non_image),
    }
    buffers = train_module._persistent_model_buffer_state_dict()
    for name, tensor in buffers.items():
        local[f"buffer:{name}"] = {
            "kind": "frozen_tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
            "local": _local_descriptor(tensor),
        }
    return _logical_descriptors(local, group=dist.group.WORLD)


def _evaluate_text(
    train_module: Any, sentinel: Mapping[str, Any], *, batch_size: int
) -> dict[str, Any]:
    summary = validate_text_sentinel(sentinel)
    input_ids = summary["input_ids"]
    labels = summary["labels"]
    if len(input_ids) != len(labels) or not input_ids:
        raise RuntimeError("Text sentinel inputs and labels are not aligned")
    sequence_length = len(input_ids[0])
    if sequence_length <= 0 or any(len(row) != sequence_length for row in [*input_ids, *labels]):
        raise RuntimeError("Text sentinel rows do not have one fixed positive sequence length")
    group = train_module.dp_process_group
    if group is None:
        raise RuntimeError("Text sentinel evaluation requires an explicit DP process group")
    dp_world_size = dist.get_world_size(group)
    dp_rank = dist.get_rank(group)
    if dp_world_size != WORLD_SIZE:
        raise RuntimeError(
            f"Text sentinel DP process group must contain all {WORLD_SIZE} EP8 ranks, "
            f"got {dp_world_size}"
        )
    global_batch_size = batch_size * dp_world_size
    if len(input_ids) % global_batch_size:
        raise ValueError(
            f"Text sentinel example count {len(input_ids)} must be divisible by global batch "
            f"{global_batch_size} ({batch_size} per rank x {dp_world_size} DP ranks)"
        )

    token_ce: list[torch.Tensor] = []
    argmax: list[torch.Tensor] = []
    for global_start in range(0, len(input_ids), global_batch_size):
        local_start = global_start + dp_rank * batch_size
        local_end = local_start + batch_size
        batch_input = torch.tensor(input_ids[local_start:local_end], dtype=torch.long)
        batch_labels = torch.tensor(labels[local_start:local_end], dtype=torch.long)
        batch = {
            "input_ids": batch_input,
            "labels": batch_labels,
            "loss_masks": torch.ones_like(batch_input, dtype=torch.float32),
        }
        device_batch = move_to_device(batch, train_module.device)
        output = train_module.eval_batch(device_batch, return_response_logits=True)
        logits = output.logits
        if logits is None or logits.ndim != 2 or logits.shape[0] != batch_labels.numel():
            raise RuntimeError("Text sentinel forward did not return one logit row per token")
        flat_labels = device_batch["labels"].reshape(-1)
        batch_ce = (
            F.cross_entropy(logits.float(), flat_labels, reduction="none")
            .detach()
            .cpu()
            .reshape(batch_size, sequence_length)
        )
        batch_argmax = logits.argmax(dim=-1).detach().cpu().reshape(batch_size, sequence_length)
        local_packet = {
            "global_start": local_start,
            "token_ce": batch_ce,
            "argmax": batch_argmax,
        }
        gathered: list[Any] = [None for _ in range(dp_world_size)]
        dist.all_gather_object(gathered, local_packet, group=group)
        gathered_ce, gathered_argmax = _reconstruct_text_batch(
            gathered,
            global_start=global_start,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        token_ce.append(gathered_ce)
        argmax.append(gathered_argmax)
        del device_batch, output, logits
        gc_cuda()
    return {"token_ce": torch.cat(token_ce), "argmax": torch.cat(argmax)}


def _reconstruct_text_batch(
    gathered: Sequence[Any],
    *,
    global_start: int,
    batch_size: int,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and concatenate one global text batch in DP-rank order."""
    expected_fields = {"global_start", "token_ce", "argmax"}
    token_ce: list[torch.Tensor] = []
    argmax: list[torch.Tensor] = []
    expected_shape = (batch_size, sequence_length)
    for rank, packet in enumerate(gathered):
        if not isinstance(packet, Mapping) or set(packet) != expected_fields:
            fields = sorted(packet) if isinstance(packet, Mapping) else type(packet).__name__
            raise RuntimeError(f"Text sentinel rank {rank} returned malformed fields: {fields}")
        expected_start = global_start + rank * batch_size
        packet_start = packet["global_start"]
        if type(packet_start) is not int or packet_start != expected_start:
            raise RuntimeError(
                f"Text sentinel rank {rank} returned global_start={packet_start!r}; "
                f"expected {expected_start}"
            )
        packet_ce = packet["token_ce"]
        packet_argmax = packet["argmax"]
        if (
            not isinstance(packet_ce, torch.Tensor)
            or packet_ce.device.type != "cpu"
            or packet_ce.dtype != torch.float32
            or tuple(packet_ce.shape) != expected_shape
        ):
            raise RuntimeError(
                f"Text sentinel rank {rank} returned malformed token CE: "
                f"{_tensor_description(packet_ce)}; expected CPU float32 {expected_shape}"
            )
        if (
            not isinstance(packet_argmax, torch.Tensor)
            or packet_argmax.device.type != "cpu"
            or packet_argmax.dtype != torch.int64
            or tuple(packet_argmax.shape) != expected_shape
        ):
            raise RuntimeError(
                f"Text sentinel rank {rank} returned malformed argmax: "
                f"{_tensor_description(packet_argmax)}; expected CPU int64 {expected_shape}"
            )
        token_ce.append(packet_ce)
        argmax.append(packet_argmax)
    return torch.cat(token_ce).reshape(-1), torch.cat(argmax).reshape(-1)


def _tensor_description(value: Any) -> str:
    if not isinstance(value, torch.Tensor):
        return type(value).__name__
    return f"device={value.device}, dtype={value.dtype}, shape={tuple(value.shape)}"


def _assert_rank_summary_consensus(
    local: Mapping[str, Any], *, world_size: int = WORLD_SIZE
) -> None:
    """Require exact receipt-summary identity and report every differing rank."""
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, dict(local))
    failures = [
        f"rank {rank}: {summary!r}"
        for rank, summary in enumerate(gathered)
        if summary != gathered[0]
    ]
    if failures:
        raise RuntimeError(
            f"State/text audit summary differs across EP ranks; rank 0: {gathered[0]!r}; "
            f"differences: {failures}"
        )


def _checkpoint_reference(identity: Mapping[str, Any], *, step: int) -> dict[str, Any]:
    return {
        "checkpoint": identity["root"],
        "global_step": step,
        "checkpoint_config_sha256": identity["config_sha256"],
        "checkpoint_identity_sha256": identity["identity_sha256"],
    }


def _candidate_reference(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": candidate["checkpoint"],
        "global_step": candidate["global_step"],
        "checkpoint_config_sha256": candidate["checkpoint_config_sha256"],
        "checkpoint_identity_sha256": candidate["checkpoint_identity_sha256"],
    }


def _write_outputs(
    module: Any,
    *,
    frozen_output: Path,
    frozen_receipt: Mapping[str, Any],
    text_output: Path,
    text_receipt: Mapping[str, Any],
) -> None:
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            if frozen_output.exists() or text_output.exists():
                raise FileExistsError("Refusing to overwrite an existing promotion receipt")
            module._write_json_atomic(frozen_output, frozen_receipt)
            module._write_json_atomic(text_output, text_receipt)
            packet[0] = {"ok": True}
        except Exception as error:  # noqa: BLE001 - rank-zero persistence must reach every rank.
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    if not isinstance(packet[0], Mapping) or packet[0].get("ok") is not True:
        detail = packet[0].get("error") if isinstance(packet[0], Mapping) else repr(packet[0])
        raise RuntimeError(f"Could not persist state/text promotion receipts: {detail}")


def main(argv: Sequence[str] | None = None) -> None:
    """Run native EP8 frozen-state and text-retention audits."""
    args = _parse_args(argv)
    if int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE:
        raise ValueError(f"State/text audit requires torchrun WORLD_SIZE={WORLD_SIZE}")
    if (
        args.text_batch_size <= 0
        or args.checkpoint_load_threads <= 0
        or args.checkpoint_hash_workers <= 0
    ):
        raise ValueError("Text batch size and checkpoint worker counts must be positive")
    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    module = _load_matched_evaluator()
    prepare_training_environment()
    try:
        reference = args.reference_checkpoint.expanduser().resolve()
        checkpoint = args.checkpoint.expanduser().resolve()
        frozen_output = args.frozen_output.expanduser().resolve()
        text_output = args.text_output.expanduser().resolve()
        if frozen_output == text_output:
            raise ValueError("Frozen-state and text-retention outputs must be distinct")
        packet: list[Any] = [None]
        if dist.get_rank() == 0:
            try:
                if frozen_output.exists() or text_output.exists():
                    raise FileExistsError("Promotion receipt output already exists")
                matched_path = args.matched_step500.expanduser().resolve()
                if sha256_file(matched_path) != args.expected_matched_step500_sha256:
                    raise PromotionValidationError("Primary step500 receipt SHA-256 differs")
                matched = load_json(matched_path)
                sentinel_path = args.text_sentinel.expanduser().resolve()
                if sha256_file(sentinel_path) != args.expected_text_sentinel_sha256:
                    raise PromotionValidationError("Text sentinel SHA-256 differs")
                sentinel = load_json(sentinel_path)
                packet[0] = {"ok": True, "matched": matched, "sentinel": sentinel}
            except Exception as error:  # noqa: BLE001
                packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(packet, src=0)
        if not isinstance(packet[0], Mapping) or packet[0].get("ok") is not True:
            raise RuntimeError(f"State/text artifact preflight failed: {packet[0]}")
        matched = packet[0]["matched"]
        sentinel = packet[0]["sentinel"]
        if not isinstance(matched, Mapping) or not isinstance(sentinel, Mapping):
            raise TypeError("Broadcast promotion inputs are malformed")
        # The next operation hashes the live DCP once on rank zero and broadcasts its exact
        # identity. Avoid independently reading all shards from every EP rank here.
        candidate = candidate_from_matched_receipt(checkpoint, matched, verify_live_contents=False)
        sentinel_summary = validate_text_sentinel(
            sentinel,
            expected_raw_sha256=args.expected_text_sentinel_sha256,
            path=args.text_sentinel.expanduser().resolve(),
        )

        candidate_identity = module._checkpoint_identity_distributed(
            checkpoint,
            checkpoint / "config.json",
            hash_workers=args.checkpoint_hash_workers,
        )
        if candidate_identity["identity_sha256"] != candidate["checkpoint_identity_sha256"]:
            raise RuntimeError("Live candidate identity differs from the matched receipt")
        reference_identity = module._checkpoint_identity_distributed(
            reference,
            reference / "config.json",
            hash_workers=args.checkpoint_hash_workers,
        )
        if reference.name != "step0" or reference.parent != checkpoint.parent:
            raise ValueError("Frozen/text reference must be step0 from the candidate lineage")
        if reference_identity["config_sha256"] != candidate["checkpoint_config_sha256"]:
            raise RuntimeError("Reference and candidate configs differ")

        raw_config = load_json(checkpoint / "config.json")
        if not isinstance(raw_config, Mapping):
            raise TypeError("Checkpoint config must be an object")
        training_sequence_length = int(raw_config["data"]["sequence_length"])
        text_sequence_length = int(sentinel_summary["sequence_length"])
        if text_sequence_length > training_sequence_length:
            raise ValueError("Text sentinel exceeds the checkpoint's maximum sequence length")
        if int(sentinel_summary["examples"]) % args.text_batch_size:
            raise ValueError("Text sentinel example count must be divisible by --text-batch-size")
        model, module_config = module._build_model_and_module(
            raw_config,
            sequence_length=text_sequence_length,
            rank_batch_instances=args.text_batch_size,
        )
        train_module = module_config.build(model, eval_only=True)
        candidate_coverage = module._native_checkpoint_load_coverage_distributed(
            train_module, module._checkpoint_state_dir(checkpoint)
        )
        reference_coverage = module._native_checkpoint_load_coverage_distributed(
            train_module, module._checkpoint_state_dir(reference)
        )
        if candidate_coverage != reference_coverage:
            raise RuntimeError("Reference and candidate native-load coverage differ")

        train_module.load_state_dict_direct(
            module._checkpoint_state_dir(reference),
            process_group=dist.group.WORLD,
            thread_count=args.checkpoint_load_threads,
            load_optim_state=False,
        )
        freeze_patterns = raw_config["train_module"]["freeze_params"]
        reference_state = _model_state_descriptors(train_module, freeze_patterns)
        reference_text = _evaluate_text(train_module, sentinel, batch_size=args.text_batch_size)

        train_module.load_state_dict_direct(
            module._checkpoint_state_dir(checkpoint),
            process_group=dist.group.WORLD,
            thread_count=args.checkpoint_load_threads,
            load_optim_state=False,
        )
        candidate_state = _model_state_descriptors(train_module, freeze_patterns)
        candidate_text = _evaluate_text(train_module, sentinel, batch_size=args.text_batch_size)
        if set(reference_state) != set(candidate_state):
            raise RuntimeError("Reference and candidate comparison surfaces differ")
        comparisons = []
        for name in sorted(reference_state):
            left, right = reference_state[name], candidate_state[name]
            if any(left[field] != right[field] for field in ("kind", "dtype", "shape", "numel")):
                raise RuntimeError(f"State descriptor {name!r} changed shape or dtype")
            comparisons.append(
                {
                    "name": name,
                    "kind": left["kind"],
                    "dtype": left["dtype"],
                    "shape": left["shape"],
                    "numel": left["numel"],
                    "reference_sha256": left["sha256"],
                    "candidate_sha256": right["sha256"],
                }
            )
        mismatch_count = sum(
            comparison["reference_sha256"] != comparison["candidate_sha256"]
            for comparison in comparisons
        )
        frozen_count = sum(comparison["kind"] == "frozen_tensor" for comparison in comparisons)
        evaluator = artifact_reference(Path(__file__).resolve())
        frozen_receipt: dict[str, Any] = {
            "format": FROZEN_STATE_RECEIPT_FORMAT,
            "version": 1,
            "status": "passed" if mismatch_count == 0 else "failed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "evaluator": evaluator,
            "candidate": _candidate_reference(candidate),
            "reference_checkpoint": _checkpoint_reference(reference_identity, step=0),
            "protocol": {
                "name": "logical-tensor-sha256-v1",
                "hash_algorithm": "sha256",
                "tensor_encoding": "dtype-shape-contiguous-little-endian-v1",
                "image_embedding_rows": list(IMAGE_TOKEN_ROWS),
            },
            "comparisons": comparisons,
            "summary": {
                "complete": mismatch_count == 0,
                "expected_frozen_tensor_count": candidate_coverage["frozen_state_key_count"]
                + candidate_coverage["persistent_buffer_count"],
                "compared_frozen_tensor_count": frozen_count,
                "non_image_embedding_row_count": candidate["vocab_size"]
                - len(candidate["image_embedding_rows"]),
                "mismatch_count": mismatch_count,
                "comparison_inventory_sha256": canonical_sha256(
                    sorted(comparisons, key=lambda item: (item["kind"], item["name"]))
                ),
            },
        }

        reference_ce = reference_text["token_ce"].float()
        candidate_ce = candidate_text["token_ce"].float()
        absolute = (candidate_ce - reference_ce).abs()
        relative = absolute / reference_ce.abs().clamp_min(torch.finfo(torch.float32).tiny)
        all_finite = bool(
            torch.isfinite(reference_ce).all().item() and torch.isfinite(candidate_ce).all().item()
        )
        argmax_matches = int((reference_text["argmax"] == candidate_text["argmax"]).sum())
        text_receipt: dict[str, Any] = {
            "format": TEXT_RETENTION_RECEIPT_FORMAT,
            "version": 1,
            "status": "passed",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "evaluator": evaluator,
            "candidate": _candidate_reference(candidate),
            "reference_checkpoint": _checkpoint_reference(reference_identity, step=0),
            "dataset": {
                "path": str(args.text_sentinel.expanduser().resolve()),
                "sha256": args.expected_text_sentinel_sha256,
                "fingerprint": sentinel_summary["fingerprint"],
                "examples": sentinel_summary["examples"],
                "supervised_tokens": sentinel_summary["supervised_tokens"],
                "input_ids_sha256": sentinel_summary["input_ids_sha256"],
                "labels_sha256": sentinel_summary["labels_sha256"],
                "image_token_count": 0,
                "image_tensor_count": 0,
            },
            "protocol": {
                "name": "per-token-nll-and-argmax-v1",
                "atol": 1e-6,
                "rtol": 1e-6,
                "same_topology": True,
                "same_backend": True,
                "image_free": True,
            },
            "metrics": {
                "all_finite": all_finite,
                "reference_mean_ce": float(reference_ce.mean()),
                "candidate_mean_ce": float(candidate_ce.mean()),
                "max_abs_token_ce_delta": float(absolute.max()),
                "max_rel_token_ce_delta": float(relative.max()),
                "argmax_matches": argmax_matches,
                "argmax_total": reference_ce.numel(),
            },
        }
        validate_frozen_state_receipt(
            frozen_receipt,
            candidate=candidate,
            expected_frozen_tensor_count=candidate_coverage["frozen_state_key_count"]
            + candidate_coverage["persistent_buffer_count"],
        )
        validate_text_retention_receipt(text_receipt, candidate=candidate)

        local_summary = {
            "frozen_inventory_sha256": frozen_receipt["summary"]["comparison_inventory_sha256"],
            "text_metrics": text_receipt["metrics"],
        }
        _assert_rank_summary_consensus(local_summary)
        _write_outputs(
            module,
            frozen_output=frozen_output,
            frozen_receipt=frozen_receipt,
            text_output=text_output,
            text_receipt=text_receipt,
        )
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
