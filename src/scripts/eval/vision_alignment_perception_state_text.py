"""Produce native frozen-state, initialization, and image-free text-retention receipts.

Run on one native EP8 node. The evaluator hashes both checkpoints before model construction,
loads each through the same native checkpoint path used by matched-wrong evaluation, compares
every frozen parameter plus all non-image input-embedding rows, and replays the pinned text
sentinel under identical topology/backend settings. Perception promotion may additionally bind
the candidate to a paired outcome receipt and compare every model tensor in the bridge parent
with the arm's freshly initialized step-0 checkpoint.
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
    IMAGE_TOKEN_ROWS,
    artifact_reference,
    canonical_sha256,
    sha256_file,
    validate_text_sentinel,
)
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.utils import gc_cuda, move_to_device

WORLD_SIZE = 8
_NATIVE_HELPER_RECEIPT_FORMATS = frozenset(
    {
        "vision_alignment_perception_initialization_parity_receipt",
        "vision_alignment_perception_frozen_state_receipt",
        "vision_alignment_perception_text_retention_receipt",
    }
)


def _load_matched_evaluator():
    path = Path(__file__).with_name("vision_alignment_matched_wrong.py")
    spec = importlib.util.spec_from_file_location("_vision_alignment_matched_wrong_native", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load native matched evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_snapshot_evaluator():
    path = Path(__file__).with_name("vision_alignment_perception_matched_wrong.py")
    spec = importlib.util.spec_from_file_location(
        "_vision_alignment_perception_snapshot_native", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load perception snapshot helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _with_native_helper(
    receipt: Mapping[str, Any], module: Any, snapshot_module: Any
) -> dict[str, Any]:
    """Bind a receipt to the exact native-load and private-snapshot helpers."""
    if receipt.get("format") not in _NATIVE_HELPER_RECEIPT_FORMATS:
        raise ValueError("Native-helper evidence cannot be attached to this receipt format")
    if "native_helper" in receipt or "snapshot_helper" in receipt:
        raise ValueError("Receipt already contains native/snapshot-helper evidence")
    raw_path = getattr(module, "__file__", None)
    if not isinstance(raw_path, str) or not raw_path:
        raise RuntimeError("Loaded native-checkpoint helper does not expose __file__")
    helper_path = Path(raw_path).expanduser().resolve()
    expected_path = Path(__file__).with_name("vision_alignment_matched_wrong.py").resolve()
    if helper_path != expected_path:
        raise RuntimeError(
            f"Loaded native-checkpoint helper {helper_path} differs from {expected_path}"
        )
    output = dict(receipt)
    output["native_helper"] = artifact_reference(helper_path)
    snapshot_raw_path = getattr(snapshot_module, "__file__", None)
    if not isinstance(snapshot_raw_path, str) or not snapshot_raw_path:
        raise RuntimeError("Loaded checkpoint-snapshot helper does not expose __file__")
    snapshot_path = Path(snapshot_raw_path).expanduser().resolve()
    expected_snapshot_path = (
        Path(__file__).with_name("vision_alignment_perception_matched_wrong.py").resolve()
    )
    if snapshot_path != expected_snapshot_path:
        raise RuntimeError(
            f"Loaded checkpoint-snapshot helper {snapshot_path} differs from "
            f"{expected_snapshot_path}"
        )
    output["snapshot_helper"] = artifact_reference(snapshot_path)
    return output


def _load_private_checkpoint(
    train_module: Any,
    native_module: Any,
    snapshot_module: Any,
    identity: Mapping[str, Any],
    *,
    snapshot_base: Path,
    checkpoint_load_threads: int,
) -> Mapping[str, Any]:
    """Load a model only from a private byte-for-byte copy of an attested checkpoint."""
    state_dir = snapshot_module._materialize_checkpoint_snapshot_distributed(
        identity, base_dir=snapshot_base
    )
    try:
        coverage = native_module._native_checkpoint_load_coverage_distributed(
            train_module, state_dir
        )
        train_module.load_state_dict_direct(
            state_dir,
            process_group=dist.group.WORLD,
            thread_count=checkpoint_load_threads,
            load_optim_state=False,
        )
    finally:
        snapshot_module._remove_checkpoint_snapshot_distributed(state_dir)
    return coverage


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--perception-outcome", type=Path, required=True)
    parser.add_argument("--expected-perception-outcome-sha256", required=True)
    parser.add_argument(
        "--perception-role",
        choices=("control", "treatment"),
        required=True,
        help="Arm selected from --perception-outcome.",
    )
    parser.add_argument("--initialization-reference-checkpoint", type=Path)
    parser.add_argument("--initialization-output", type=Path)
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


def _model_state_descriptors(
    train_module: Any,
    freeze_patterns: Sequence[str],
    *,
    include_non_image_embedding_rows: bool = True,
) -> dict[str, Any]:
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
    if include_non_image_embedding_rows:
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


def _state_comparisons(
    reference_state: Mapping[str, Mapping[str, Any]],
    candidate_state: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Build an exact logical-tensor comparison inventory."""
    if set(reference_state) != set(candidate_state):
        missing = sorted(set(reference_state) - set(candidate_state))
        extra = sorted(set(candidate_state) - set(reference_state))
        raise RuntimeError(
            f"Reference and candidate comparison surfaces differ: missing={missing[:10]}, "
            f"extra={extra[:10]}"
        )
    comparisons: list[dict[str, Any]] = []
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
    return comparisons, mismatch_count


def _write_outputs(
    module: Any,
    *,
    frozen_output: Path,
    frozen_receipt: Mapping[str, Any],
    text_output: Path,
    text_receipt: Mapping[str, Any],
    initialization_output: Path | None = None,
    initialization_receipt: Mapping[str, Any] | None = None,
) -> None:
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            outputs = [frozen_output, text_output]
            if (initialization_output is None) != (initialization_receipt is None):
                raise ValueError("Initialization output and receipt must be supplied together")
            if initialization_output is not None:
                outputs.append(initialization_output)
            if len(set(outputs)) != len(outputs):
                raise ValueError("State/text receipt outputs must be distinct")
            if any(path.exists() for path in outputs):
                raise FileExistsError("Refusing to overwrite an existing promotion receipt")
            module._write_json_atomic(frozen_output, frozen_receipt)
            module._write_json_atomic(text_output, text_receipt)
            if initialization_output is not None and initialization_receipt is not None:
                module._write_json_atomic(initialization_output, initialization_receipt)
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
    if (args.initialization_reference_checkpoint is None) != (args.initialization_output is None):
        raise ValueError(
            "--initialization-reference-checkpoint and --initialization-output are paired"
        )
    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    module = _load_matched_evaluator()
    snapshot_module = _load_snapshot_evaluator()
    prepare_training_environment()
    try:
        reference = args.reference_checkpoint.expanduser().resolve()
        checkpoint = args.checkpoint.expanduser().resolve()
        frozen_output = args.frozen_output.expanduser().resolve()
        text_output = args.text_output.expanduser().resolve()
        initialization_output = (
            args.initialization_output.expanduser().resolve()
            if args.initialization_output is not None
            else None
        )
        if len({path for path in (frozen_output, text_output, initialization_output) if path}) != (
            3 if initialization_output is not None else 2
        ):
            raise ValueError("Frozen-state and text-retention outputs must be distinct")
        packet: list[Any] = [None]
        if dist.get_rank() == 0:
            try:
                if (
                    frozen_output.exists()
                    or text_output.exists()
                    or (initialization_output is not None and initialization_output.exists())
                ):
                    raise FileExistsError("Promotion receipt output already exists")
                primary_path = args.perception_outcome.expanduser().resolve()
                expected_primary_sha = args.expected_perception_outcome_sha256
                primary_receipt, _ = snapshot_module._load_json_bytes(
                    primary_path,
                    expected_sha256=expected_primary_sha,
                    name="primary outcome receipt",
                )
                sentinel_path = args.text_sentinel.expanduser().resolve()
                sentinel, _ = snapshot_module._load_json_bytes(
                    sentinel_path,
                    expected_sha256=args.expected_text_sentinel_sha256,
                    name="text sentinel",
                )
                packet[0] = {
                    "ok": True,
                    "primary_receipt": primary_receipt,
                    "sentinel": sentinel,
                }
            except Exception as error:  # noqa: BLE001
                packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(packet, src=0)
        if not isinstance(packet[0], Mapping) or packet[0].get("ok") is not True:
            raise RuntimeError(f"State/text artifact preflight failed: {packet[0]}")
        primary_receipt = packet[0]["primary_receipt"]
        sentinel = packet[0]["sentinel"]
        if not isinstance(primary_receipt, Mapping) or not isinstance(sentinel, Mapping):
            raise TypeError("Broadcast promotion inputs are malformed")
        # The next operation hashes the live DCP once on rank zero and broadcasts its exact
        # identity. Avoid independently reading all shards from every EP rank here.
        from olmo_core.eval.vision_alignment_perception_promotion import (
            candidate_from_outcome_receipt,
        )

        candidate = candidate_from_outcome_receipt(
            checkpoint,
            primary_receipt,
            role=args.perception_role,
            verify_live_contents=False,
        )
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
            raise RuntimeError("Live candidate identity differs from the perception outcome")
        reference_identity = module._checkpoint_identity_distributed(
            reference,
            reference / "config.json",
            hash_workers=args.checkpoint_hash_workers,
        )
        if reference.name != "step0" or reference.parent != checkpoint.parent:
            raise ValueError("Frozen/text reference must be step0 from the candidate lineage")
        if reference_identity["config_sha256"] != candidate["checkpoint_config_sha256"]:
            raise RuntimeError("Reference and candidate configs differ")

        raw_config, _ = snapshot_module._load_json_bytes(
            checkpoint / "config.json",
            expected_sha256=candidate_identity["config_sha256"],
            name="candidate checkpoint config",
        )
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
        snapshot_base = (
            Path(os.environ.get("RESULTS_DIR", "/tmp"))
            / "vision-alignment-perception-state-text"
            / "checkpoint-snapshots"
        )

        initialization_receipt: dict[str, Any] | None = None
        initialization_coverage: Mapping[str, Any] | None = None
        if args.initialization_reference_checkpoint is not None:
            initialization_reference = (
                args.initialization_reference_checkpoint.expanduser().resolve()
            )
            initialization_identity = module._checkpoint_identity_distributed(
                initialization_reference,
                initialization_reference / "config.json",
                hash_workers=args.checkpoint_hash_workers,
            )
            initialization_coverage = _load_private_checkpoint(
                train_module,
                module,
                snapshot_module,
                initialization_identity,
                snapshot_base=snapshot_base,
                checkpoint_load_threads=args.checkpoint_load_threads,
            )
            initialization_state = _model_state_descriptors(
                train_module,
                ("*",),
                include_non_image_embedding_rows=False,
            )

        reference_coverage = _load_private_checkpoint(
            train_module,
            module,
            snapshot_module,
            reference_identity,
            snapshot_base=snapshot_base,
            checkpoint_load_threads=args.checkpoint_load_threads,
        )
        if initialization_coverage is not None and initialization_coverage != reference_coverage:
            raise RuntimeError("Initialization and step0 native-load coverage differ")
        freeze_patterns = raw_config["train_module"]["freeze_params"]
        reference_state = _model_state_descriptors(train_module, freeze_patterns)
        reference_text = _evaluate_text(train_module, sentinel, batch_size=args.text_batch_size)
        if args.initialization_reference_checkpoint is not None:
            step0_initialization_state = _model_state_descriptors(
                train_module,
                ("*",),
                include_non_image_embedding_rows=False,
            )
            initialization_comparisons, initialization_mismatches = _state_comparisons(
                initialization_state,
                step0_initialization_state,
            )
            initialization_comparisons = [
                {
                    **{
                        key: value
                        for key, value in comparison.items()
                        if key not in {"kind", "candidate_sha256"}
                    },
                    "kind": (
                        "persistent_buffer"
                        if str(comparison["name"]).startswith("buffer:")
                        else "parameter"
                    ),
                    "step0_sha256": comparison["candidate_sha256"],
                }
                for comparison in initialization_comparisons
            ]
            arm = str(raw_config.get("perception_trainability_arm"))
            initialization_receipt = _with_native_helper(
                {
                    "format": "vision_alignment_perception_initialization_parity_receipt",
                    "version": 1,
                    "status": "passed" if initialization_mismatches == 0 else "failed",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "producer": artifact_reference(Path(__file__).resolve()),
                    "arm": arm,
                    "reference_checkpoint": dict(initialization_identity),
                    "perception_step0": dict(reference_identity),
                    "protocol": {
                        "name": "logical-all-model-tensor-sha256-v1",
                        "hash_algorithm": "sha256",
                        "tensor_encoding": "dtype-shape-contiguous-little-endian-v1",
                    },
                    "comparisons": initialization_comparisons,
                    "summary": {
                        "complete": initialization_mismatches == 0,
                        "expected_tensor_count": len(initialization_comparisons),
                        "compared_tensor_count": len(initialization_comparisons),
                        "mismatch_count": initialization_mismatches,
                        "comparison_inventory_sha256": canonical_sha256(
                            sorted(
                                initialization_comparisons,
                                key=lambda item: (str(item["kind"]), str(item["name"])),
                            )
                        ),
                    },
                    "content_sha256": "",
                },
                module,
                snapshot_module,
            )
            initialization_receipt["content_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in initialization_receipt.items()
                    if key != "content_sha256"
                }
            )
            from olmo_core.eval.vision_alignment_perception_promotion import (
                validate_initialization_parity_receipt,
            )

            validate_initialization_parity_receipt(
                initialization_receipt,
                candidate=candidate,
                expected_arm=arm,
            )

        candidate_coverage = _load_private_checkpoint(
            train_module,
            module,
            snapshot_module,
            candidate_identity,
            snapshot_base=snapshot_base,
            checkpoint_load_threads=args.checkpoint_load_threads,
        )
        if candidate_coverage != reference_coverage:
            raise RuntimeError("Reference and candidate native-load coverage differ")
        candidate_state = _model_state_descriptors(train_module, freeze_patterns)
        candidate_text = _evaluate_text(train_module, sentinel, batch_size=args.text_batch_size)
        closing_candidate_identity = module._checkpoint_identity_distributed(
            checkpoint,
            checkpoint / "config.json",
            hash_workers=args.checkpoint_hash_workers,
        )
        closing_reference_identity = module._checkpoint_identity_distributed(
            reference,
            reference / "config.json",
            hash_workers=args.checkpoint_hash_workers,
        )
        if (
            closing_candidate_identity != candidate_identity
            or closing_reference_identity != reference_identity
        ):
            raise RuntimeError("State/text checkpoint identity changed during evaluation")
        if args.initialization_reference_checkpoint is not None:
            closing_initialization_identity = module._checkpoint_identity_distributed(
                initialization_reference,
                initialization_reference / "config.json",
                hash_workers=args.checkpoint_hash_workers,
            )
            if closing_initialization_identity != initialization_identity:
                raise RuntimeError("Initialization reference checkpoint changed during evaluation")
        if (
            sha256_file(args.text_sentinel.expanduser().resolve())
            != args.expected_text_sentinel_sha256
        ):
            raise RuntimeError("Text sentinel changed during evaluation")
        comparisons, mismatch_count = _state_comparisons(reference_state, candidate_state)
        frozen_count = sum(comparison["kind"] == "frozen_tensor" for comparison in comparisons)
        evaluator = artifact_reference(Path(__file__).resolve())
        from olmo_core.eval.vision_alignment_perception_promotion import (
            PERCEPTION_FROZEN_STATE_RECEIPT_FORMAT,
            PERCEPTION_TEXT_RETENTION_RECEIPT_FORMAT,
            validate_perception_frozen_state_receipt,
            validate_perception_text_retention_receipt,
        )

        frozen_receipt: dict[str, Any] = _with_native_helper(
            {
                "format": PERCEPTION_FROZEN_STATE_RECEIPT_FORMAT,
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
            },
            module,
            snapshot_module,
        )

        reference_ce = reference_text["token_ce"].float()
        candidate_ce = candidate_text["token_ce"].float()
        absolute = (candidate_ce - reference_ce).abs()
        relative = absolute / reference_ce.abs().clamp_min(torch.finfo(torch.float32).tiny)
        all_finite = bool(
            torch.isfinite(reference_ce).all().item() and torch.isfinite(candidate_ce).all().item()
        )
        argmax_matches = int((reference_text["argmax"] == candidate_text["argmax"]).sum())
        text_receipt: dict[str, Any] = _with_native_helper(
            {
                "format": PERCEPTION_TEXT_RETENTION_RECEIPT_FORMAT,
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
            },
            module,
            snapshot_module,
        )
        validate_perception_frozen_state_receipt(
            frozen_receipt,
            candidate=candidate,
            expected_frozen_tensor_count=candidate_coverage["frozen_state_key_count"]
            + candidate_coverage["persistent_buffer_count"],
        )
        validate_perception_text_retention_receipt(text_receipt, candidate=candidate)

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
            initialization_output=initialization_output,
            initialization_receipt=initialization_receipt,
        )
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
