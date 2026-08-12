"""Replay exact matched-wrong-image CE on a native Vision Alignment checkpoint.

Run with exactly eight GPUs so the saved MoE language model uses EP8::

    torchrun --nproc-per-node=8 \
        src/scripts/eval/vision_alignment_matched_wrong.py \
        --checkpoint /path/to/checkpoint/step250 \
        --output /path/to/results.json

The evaluator never exports or converts model weights. It restores the native distributed
checkpoint, including its ``frozen_model.*`` tensors, and evaluates the same explicit recipient
set once with its correct image and once with a distinct exact-geometry donor image.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from olmo_core.data.multimodal import MultimodalCollator, MultimodalDataLoader
from olmo_core.data.multimodal.vision_alignment_sources import (
    VisionAlignmentSourceSpec,
    build_vision_alignment_dataset_config,
    load_pinned_vision_alignment_tokenizer,
    pixmo_row_path_inventory,
    runtime_dataset_fingerprint,
    vision_alignment_source_registry_sha256,
)
from olmo_core.distributed.checkpoint import get_checkpoint_metadata
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.eval import (
    MultimodalFixedValidationDataset,
    MultimodalMatchedWrongImageDataset,
    build_matched_wrong_image_pairing,
    matched_wrong_image_pairing_sha256,
    serialize_matched_wrong_image_pairing,
    validate_matched_wrong_image_pairing,
)
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.ddp import OLMoDDPTransformerBlockConfig
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
from olmo_core.nn.transformer import OLMoDDPModelConfig
from olmo_core.nn.vision import MultimodalLMConfig
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.train_module.transformer import MultimodalOLMoDDPTrainModuleConfig
from olmo_core.utils import gc_cuda, move_to_device

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = (
    "/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/checkpoints/"
    "vision-alignment-bridge-real-canary-v1/step250"
)
SOURCE_NAMES = ("pixmo_caption", "pixmo_transcript")
WINDOWS: tuple[tuple[str, int | None], ...] = (
    ("all", None),
    ("first_1", 1),
    ("first_8", 8),
    ("first_32", 32),
)
WORLD_SIZE = 8
EP_DEGREE = 8
SCHEMA_VERSION = 3


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", help="Config JSON (defaults to CHECKPOINT/config.json).")
    parser.add_argument("--sources", nargs="+", choices=SOURCE_NAMES, default=list(SOURCE_NAMES))
    parser.add_argument("--examples", type=int, default=512)
    parser.add_argument(
        "--pairing",
        action="append",
        default=[],
        metavar="SOURCE=PATH",
        help=(
            "Load or create this explicit source pairing. May be repeated. Unspecified source "
            "paths are derived from --pairing-dir or --output."
        ),
    )
    parser.add_argument("--pairing-dir", help="Directory for unspecified pairing JSON files.")
    parser.add_argument(
        "--expected-pairing-sha256",
        action="append",
        default=[],
        metavar="SOURCE=SHA256",
        help=(
            "Require this exact pairing artifact SHA-256. May be repeated. Every existing "
            "pairing file must have an explicit source pin."
        ),
    )
    parser.add_argument(
        "--exclude-pairing",
        action="append",
        default=[],
        metavar="SOURCE=PATH",
        help=(
            "When creating or reusing a pairing, exclude or validate against every recipient "
            "and donor in this pinned primary pairing. May be repeated by source."
        ),
    )
    parser.add_argument(
        "--expected-exclude-pairing-sha256",
        action="append",
        default=[],
        metavar="SOURCE=SHA256",
        help="Require the exact raw/canonical SHA-256 of each --exclude-pairing artifact.",
    )
    parser.add_argument(
        "--pairing-seed",
        type=int,
        help="Pairing seed (defaults to the checkpoint's intrinsic-evaluation seed).",
    )
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--rank-batch-instances",
        type=int,
        help="Per-rank instances (defaults to the checkpoint's intrinsic-evaluation batch).",
    )
    parser.add_argument("--work-dir", help="Evaluator data-loader work directory.")
    parser.add_argument("--output", help="Atomic result JSON path.")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Explicitly allow replacement of an existing result JSON.",
    )
    parser.add_argument("--checkpoint-load-threads", type=int, default=8)
    parser.add_argument(
        "--checkpoint-hash-workers",
        type=int,
        default=8,
        help="Workers used to SHA-256 every distributed-checkpoint state file.",
    )
    return parser.parse_args(argv)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _strict_json_loads(raw: str, *, source: Path) -> Any:
    def object_pairs_hook(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON file {source} repeats key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"JSON file {source} contains non-finite constant {value}")

    return json.loads(
        raw,
        object_pairs_hook=object_pairs_hook,
        parse_constant=reject_constant,
    )


def _load_json(path: Path) -> Any:
    return _strict_json_loads(path.read_text(), source=path)


def _write_bytes_atomic(path: Path, payload: bytes, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as file_handle:
            file_handle.write(payload)
            file_handle.flush()
            os.fsync(file_handle.fileno())
        if overwrite:
            temporary.replace(path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"Refusing to overwrite existing artifact {path}; choose a new path or "
                    "explicitly allow replacement"
                ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _write_bytes_atomic(path, raw, overwrite=overwrite)


def _checkpoint_state_dir(checkpoint: Path) -> Path:
    if checkpoint.name == "model_and_optim":
        return checkpoint
    nested = checkpoint / "model_and_optim"
    return nested if nested.is_dir() else checkpoint


def _checkpoint_root(checkpoint: Path) -> Path:
    return checkpoint.parent if checkpoint.name == "model_and_optim" else checkpoint


def _config_path(checkpoint: Path, explicit: str | None) -> Path:
    return (
        Path(explicit).expanduser().resolve()
        if explicit
        else _checkpoint_root(checkpoint) / "config.json"
    )


def _default_output(checkpoint: Path) -> Path:
    root = _checkpoint_root(checkpoint)
    return root / "eval" / "vision-alignment-matched-wrong-image.json"


def _parse_pairing_paths(
    values: Sequence[str], *, option_name: str = "--pairing"
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option_name} must be SOURCE=PATH, got {value!r}")
        source, raw_path = value.split("=", 1)
        if source not in SOURCE_NAMES or not raw_path or source in paths:
            raise ValueError(f"Invalid or duplicate {option_name} value {value!r}")
        paths[source] = Path(raw_path).expanduser().resolve()
    return paths


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_expected_pairing_sha256(values: Sequence[str], sources: Sequence[str]) -> dict[str, str]:
    pins: dict[str, str] = {}
    enabled_sources = set(sources)
    for value in values:
        if "=" not in value:
            raise ValueError(f"--expected-pairing-sha256 must be SOURCE=SHA256, got {value!r}")
        source, digest = value.split("=", 1)
        if source not in SOURCE_NAMES or source not in enabled_sources or source in pins:
            raise ValueError(f"Invalid or duplicate pairing SHA-256 pin {value!r}")
        if not _is_sha256(digest):
            raise ValueError(
                f"Pairing SHA-256 pin for {source!r} must be 64 lowercase hex characters"
            )
        pins[source] = digest
    return pins


def _resolve_pairing_paths(
    args: argparse.Namespace, output: Path, sources: Sequence[str]
) -> dict[str, Path]:
    paths = _parse_pairing_paths(args.pairing)
    unexpected = sorted(set(paths) - set(sources))
    if unexpected:
        raise ValueError(f"Pairing paths were supplied for disabled sources: {unexpected}")
    root = (
        Path(args.pairing_dir).expanduser().resolve()
        if args.pairing_dir
        else output.parent / f"{output.stem}-pairings"
    )
    for source in sources:
        paths.setdefault(source, root / f"{source}.json")
    return paths


def _validate_args(args: argparse.Namespace) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != WORLD_SIZE:
        raise ValueError(
            f"Vision Alignment matched-wrong evaluation requires WORLD_SIZE={WORLD_SIZE}, "
            f"got {world_size}; launch with torchrun --nproc-per-node={WORLD_SIZE}"
        )
    if args.examples <= 0:
        raise ValueError("--examples must be positive")
    if args.pairing_seed is not None and args.pairing_seed < 0:
        raise ValueError("--pairing-seed must be non-negative")
    if args.bootstrap_seed is not None and args.bootstrap_seed < 0:
        raise ValueError("--bootstrap-seed must be non-negative")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    if args.rank_batch_instances is not None and args.rank_batch_instances <= 0:
        raise ValueError("--rank-batch-instances must be positive")
    if args.checkpoint_load_threads <= 0:
        raise ValueError("--checkpoint-load-threads must be positive")
    if args.checkpoint_hash_workers <= 0:
        raise ValueError("--checkpoint-hash-workers must be positive")


def _preflight_artifact_paths_distributed(
    *,
    output: Path,
    overwrite_output: bool,
    pairing_paths: Mapping[str, Path],
    expected_pairing_sha256: Mapping[str, str],
    excluded_pairing_paths: Mapping[str, Path] | None = None,
    expected_excluded_pairing_sha256: Mapping[str, str] | None = None,
) -> None:
    """Fail before checkpoint hashing when output or reused-pairing policy is violated."""
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            if output.exists() and not overwrite_output:
                raise FileExistsError(
                    f"Refusing to overwrite existing result {output}; pass --overwrite-output "
                    "only when replacement is intentional"
                )
            for source, path in pairing_paths.items():
                expected = expected_pairing_sha256.get(source)
                if path.exists() and expected is None:
                    raise ValueError(
                        f"Existing {source} pairing {path} requires "
                        f"--expected-pairing-sha256={source}=SHA256"
                    )
                if path.exists():
                    actual = _sha256_file(path)
                    if actual != expected:
                        raise ValueError(
                            f"Existing {source} pairing SHA-256 differs: expected {expected}, "
                            f"got {actual}"
                        )
            excluded_pairing_paths = excluded_pairing_paths or {}
            expected_excluded_pairing_sha256 = expected_excluded_pairing_sha256 or {}
            if set(excluded_pairing_paths) != set(expected_excluded_pairing_sha256):
                raise ValueError(
                    "Every excluded primary pairing requires one exact source-specific SHA-256"
                )
            for source, path in excluded_pairing_paths.items():
                if not path.is_file():
                    raise FileNotFoundError(f"Excluded primary {source} pairing is missing: {path}")
                actual = _sha256_file(path)
                expected = expected_excluded_pairing_sha256[source]
                if actual != expected:
                    raise ValueError(
                        f"Excluded primary {source} pairing SHA-256 differs: "
                        f"expected {expected}, got {actual}"
                    )
            packet[0] = {"ok": True}
        except Exception as error:  # noqa: BLE001 - every rank-zero failure must be propagated.
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Artifact preflight failed: {detail}")


def _configure_lm_for_eval(lm_config: OLMoDDPModelConfig) -> None:
    blocks = [
        cast(OLMoDDPTransformerBlockConfig, lm_config.block),
        *(
            cast(OLMoDDPTransformerBlockConfig, block)
            for block in (lm_config.block_overrides or {}).values()
        ),
    ]
    for block in blocks:
        if isinstance(block.sequence_mixer, AttentionConfig):
            block.sequence_mixer.backend = AttentionBackendName.flex
        if block.ep is not None:
            block.ep.path = ExpertParallelPath.rowwise_nvshmem
    lm_config.recompute_each_block = False
    lm_config.recompute_all_blocks_by_chunk = False
    lm_config.two_batch_overlap = False


def _build_model_and_module(
    raw_config: Mapping[str, Any], *, sequence_length: int, rank_batch_instances: int
):
    model_config = MultimodalLMConfig.from_dict(raw_config["model"])
    if not isinstance(model_config.lm, OLMoDDPModelConfig):
        raise TypeError("Vision Alignment checkpoint does not contain an OLMoDDP LM config")
    _configure_lm_for_eval(model_config.lm)
    model = model_config.build(init_device="meta")

    # Preserve the checkpoint's exact freeze surface. Direct checkpoint loading uses this to
    # resolve ``frozen_model.*`` versus optimizer-master ``*.main`` keys.
    module_config = MultimodalOLMoDDPTrainModuleConfig.from_dict(raw_config["train_module"])
    if module_config.ep_config is None:
        raise ValueError("Vision Alignment checkpoint is missing expert-parallel configuration")
    module_config.ep_config.degree = EP_DEGREE
    module_config.rank_microbatch_size = rank_batch_instances * sequence_length
    module_config.max_sequence_length = sequence_length
    module_config.compile_model = False
    module_config.vision_activation_checkpointing = False
    module_config.connector_activation_checkpointing = False
    module_config.response_logits_only = True
    module_config.diagnostics_interval = None
    return model, module_config


def _load_validation_manifest(
    raw_config: Mapping[str, Any], dataset_path: str
) -> tuple[Mapping[str, Any], tuple[str, ...], dict[str, Any]]:
    evaluation = raw_config["evaluation"]
    manifest_path_value = evaluation.get("validation_manifest_path")
    expected_manifest_sha = evaluation.get("validation_manifest_sha256")
    if not isinstance(manifest_path_value, str) or not manifest_path_value:
        raise ValueError("Checkpoint does not pin a validation manifest path")
    manifest_path = Path(manifest_path_value).expanduser().resolve()
    actual_manifest_sha = _sha256_file(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError(
            "Validation manifest SHA mismatch: "
            f"expected {expected_manifest_sha}, got {actual_manifest_sha}"
        )
    manifest = _load_json(manifest_path)
    if (
        manifest.get("format") != "vision_alignment_validation_manifest"
        or manifest.get("version") != 3
    ):
        raise ValueError("Validation manifest has an incompatible format")
    try:
        builder = manifest["builder"]
        output = manifest["output"]
        validation = output["splits"]["validation"]
        content_path_value = validation["row_image_content_path"]
        expected_content_sha = validation["row_image_content_sha256"]
        expected_examples = validation["examples"]
        expected_fingerprint = validation["dataset_fingerprint"]
        expected_path_sha = validation["row_image_paths_sha256"]
        expected_unique_paths = validation["unique_image_paths"]
        row_path_algorithm = builder["row_image_paths_algorithm"]
        manifest_dataset_path = output["dataset_path"]
    except (KeyError, TypeError) as error:
        raise ValueError("Validation manifest is missing pinned output identities") from error
    if (
        not isinstance(expected_examples, int)
        or isinstance(expected_examples, bool)
        or expected_examples <= 0
        or not isinstance(expected_unique_paths, int)
        or isinstance(expected_unique_paths, bool)
        or not 0 < expected_unique_paths <= expected_examples
        or not isinstance(expected_fingerprint, str)
        or not expected_fingerprint
        or not isinstance(expected_path_sha, str)
        or len(expected_path_sha) != 64
        or not isinstance(row_path_algorithm, str)
        or not row_path_algorithm
    ):
        raise ValueError("Validation manifest contains malformed live-dataset identities")

    expected_dataset = (manifest_path.parent / manifest_dataset_path).resolve()
    if Path(dataset_path).expanduser().resolve() != expected_dataset:
        raise ValueError(
            f"Checkpoint data path {dataset_path!r} differs from manifest output "
            f"{str(expected_dataset)!r}"
        )
    content_relative = Path(content_path_value)
    if content_relative.is_absolute():
        raise ValueError("Validation row-content path must be relative")
    content_path = (manifest_path.parent / content_relative).resolve()
    if not content_path.is_relative_to(manifest_path.parent):
        raise ValueError("Validation row-content path escapes the artifact directory")
    raw_content = content_path.read_bytes()
    actual_content_sha = _sha256_bytes(raw_content)
    if actual_content_sha != expected_content_sha:
        raise ValueError(
            f"Validation row-content SHA mismatch: expected {expected_content_sha}, "
            f"got {actual_content_sha}"
        )
    content_ids = tuple(raw_content.decode("utf-8").splitlines())
    if (
        not raw_content.endswith(b"\n")
        or len(content_ids) != expected_examples
        or any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in content_ids
        )
    ):
        raise ValueError("Validation row-content identities are malformed")
    return (
        manifest,
        content_ids,
        {
            "manifest_path": str(manifest_path),
            "manifest_sha256": actual_manifest_sha,
            "row_content_path": str(content_path),
            "row_content_sha256": actual_content_sha,
            "expected_live_dataset": {
                "dataset_fingerprint": expected_fingerprint,
                "examples": expected_examples,
                "row_image_paths_algorithm": row_path_algorithm,
                "row_image_paths_sha256": expected_path_sha,
                "unique_image_paths": expected_unique_paths,
            },
        },
    )


def _validate_live_validation_dataset(dataset: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a live validation wrapper to the pinned v3 Arrow split identity."""
    try:
        validation = manifest["output"]["splits"]["validation"]
        algorithm = manifest["builder"]["row_image_paths_algorithm"]
    except (KeyError, TypeError) as error:
        raise ValueError("Validation manifest is missing live split identities") from error
    fingerprint = runtime_dataset_fingerprint(dataset)
    inventory = pixmo_row_path_inventory(dataset)
    expected = {
        "dataset_fingerprint": validation["dataset_fingerprint"],
        "examples": validation["examples"],
        "row_image_paths_algorithm": algorithm,
        "row_image_paths_sha256": validation["row_image_paths_sha256"],
        "unique_image_paths": validation["unique_image_paths"],
    }
    actual = {
        "dataset_fingerprint": fingerprint,
        "examples": len(dataset),
        "row_image_paths_algorithm": inventory["algorithm"],
        "row_image_paths_sha256": inventory["sha256"],
        "unique_image_paths": inventory["unique_paths"],
    }
    if actual != expected:
        differing = sorted(name for name in expected if actual[name] != expected[name])
        raise ValueError(
            "Live validation dataset differs from the pinned v3 manifest in fields "
            f"{differing}: expected {expected!r}, got {actual!r}"
        )
    return actual


def _content_ids_sha256(content_ids: Sequence[str]) -> str:
    return _sha256_bytes("".join(f"{value}\n" for value in content_ids).encode())


def _source_spec(raw_config: Mapping[str, Any]) -> VisionAlignmentSourceSpec:
    data = raw_config["data"]
    artifacts = raw_config["artifacts"]
    metadata = raw_config["vision_alignment"]
    return VisionAlignmentSourceSpec(
        phase=str(raw_config["phase"]),
        pixmo_cap_path=str(data["pixmo_cap_path"]),
        sequence_length=int(data["sequence_length"]),
        max_crops=int(data["max_crops"]),
        message_format=str(data["message_format"]),
        loss_token_weighting=str(data["loss_token_weighting"]),
        caption_prompt=str(data["caption_prompt"]),
        transcript_prompt=str(data["transcript_prompt"]),
        require_transcript=bool(data["require_transcript"]),
        tokenizer_id=str(artifacts["tokenizer_id"]),
        tokenizer_revision=str(artifacts["tokenizer_revision"]),
        tokenizer_fingerprint=str(artifacts["tokenizer_fingerprint"]),
        native_text_replay_fingerprint=data.get("native_text_replay_fingerprint"),
        recipe_version=int(metadata["recipe_version"]),
        formatter_version=str(metadata["formatter_version"]),
    )


def _load_or_build_pairing(
    dataset: Any,
    *,
    path: Path,
    source_name: str,
    expected_sha256: str | None,
    examples: int,
    seed: int,
    content_ids: Sequence[str],
    excluded_pairing_path: Path | None = None,
    expected_excluded_pairing_sha256: str | None = None,
) -> tuple[Mapping[str, Any], str, str, dict[str, Any] | None]:
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            exclusion_identity = None
            excluded_indices: list[int] = []
            if excluded_pairing_path is not None:
                if expected_excluded_pairing_sha256 is None:
                    raise ValueError(
                        f"Excluded primary {source_name} pairing requires a SHA-256 pin"
                    )
                excluded_payload = _load_json(excluded_pairing_path)
                validate_matched_wrong_image_pairing(
                    excluded_payload,
                    dataset_size=len(dataset),
                    content_ids_sha256=_content_ids_sha256(content_ids),
                )
                actual_excluded_sha = matched_wrong_image_pairing_sha256(excluded_payload)
                if (
                    _sha256_file(excluded_pairing_path) != expected_excluded_pairing_sha256
                    or actual_excluded_sha != expected_excluded_pairing_sha256
                ):
                    raise ValueError(f"Excluded primary {source_name} pairing differs from its pin")
                excluded_indices = sorted(
                    {
                        int(index)
                        for pair in excluded_payload["pairs"]
                        for index in (pair["recipient"], pair["donor"])
                    }
                )
                exclusion_identity = {
                    "path": str(excluded_pairing_path),
                    "sha256": actual_excluded_sha,
                    "excluded_recipient_and_donor_count": len(excluded_indices),
                    "excluded_indices_sha256": _canonical_sha256(excluded_indices),
                }
            if path.exists():
                if expected_sha256 is None:
                    raise ValueError(
                        f"Existing {source_name} pairing {path} requires an explicit SHA-256 pin"
                    )
                payload = _load_json(path)
                provenance = "loaded"
                persist_pairing = False
            else:
                payload = build_matched_wrong_image_pairing(
                    dataset,
                    recipient_count=examples,
                    seed=seed,
                    content_ids=content_ids,
                    epoch=0,
                    excluded_selection_indices=excluded_indices,
                )
                provenance = "built"
                persist_pairing = True
            validate_matched_wrong_image_pairing(
                payload,
                dataset_size=len(dataset),
                recipient_count=examples,
                seed=seed,
                epoch=0,
                content_ids_sha256=_content_ids_sha256(content_ids),
            )
            selected_indices = {
                int(index)
                for pair in payload["pairs"]
                for index in (pair["recipient"], pair["donor"])
            }
            overlap = selected_indices & set(excluded_indices)
            if overlap:
                raise ValueError(
                    f"{source_name} pairing overlaps its excluded primary population: "
                    f"{sorted(overlap)[:10]}"
                )
            pairing_sha256 = matched_wrong_image_pairing_sha256(payload)
            if expected_sha256 is not None and pairing_sha256 != expected_sha256:
                raise ValueError(
                    f"{source_name} pairing SHA-256 differs: expected {expected_sha256}, "
                    f"got {pairing_sha256}"
                )
            if persist_pairing:
                _write_bytes_atomic(path, serialize_matched_wrong_image_pairing(payload))
            packet[0] = {
                "ok": True,
                "payload": payload,
                "sha256": pairing_sha256,
                "provenance": provenance,
                "exclusion_identity": exclusion_identity,
            }
        except Exception as error:  # noqa: BLE001 - every rank-zero failure must be propagated.
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Could not load or build wrong-image pairing: {detail}")
    payload = result["payload"]
    pairing_sha = result["sha256"]
    validate_matched_wrong_image_pairing(
        payload,
        dataset_size=len(dataset),
        recipient_count=examples,
        seed=seed,
        epoch=0,
        content_ids_sha256=_content_ids_sha256(content_ids),
    )
    if matched_wrong_image_pairing_sha256(payload) != pairing_sha:
        raise RuntimeError("Broadcast pairing SHA-256 differs from its payload")
    if expected_sha256 is not None and pairing_sha != expected_sha256:
        raise RuntimeError(f"Broadcast {source_name} pairing SHA-256 differs from its expected pin")
    return (
        payload,
        pairing_sha,
        str(result["provenance"]),
        result.get("exclusion_identity"),
    )


def _assert_batches_match(correct: Mapping[str, Any], wrong: Mapping[str, Any]) -> None:
    if set(correct) != set(wrong):
        raise ValueError("Correct and wrong-image batches expose different fields")
    for name in correct:
        if name == "images":
            continue
        left, right = correct[name], wrong[name]
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            equal = left.dtype == right.dtype and torch.equal(left, right)
        else:
            equal = left == right
        if not equal:
            raise ValueError(f"Wrong-image pairing changed recipient batch field {name!r}")
    if not isinstance(correct["images"], torch.Tensor) or not isinstance(
        wrong["images"], torch.Tensor
    ):
        raise TypeError("Collated multimodal images must be tensors")
    if correct["images"].shape != wrong["images"].shape:
        raise ValueError("Correct and wrong-image batches have different image shapes")


def _response_ce_by_example(
    batch: Mapping[str, torch.Tensor], logits: torch.Tensor
) -> list[dict[str, Any]]:
    labels = batch["labels"]
    loss_masks = batch["loss_masks"]
    response_mask = loss_masks > 0
    if bool(torch.any(labels.masked_select(response_mask) == -100)):
        raise ValueError("A supervised response position has an ignored label")
    counts = response_mask.sum(dim=1)
    if bool(torch.any(counts <= 0)):
        raise ValueError("Every evaluated recipient must contain supervised response tokens")
    response_labels = labels.masked_select(response_mask)
    response_weights = loss_masks.masked_select(response_mask).float()
    if logits.ndim == 3:
        response_logits = logits.reshape(-1, logits.shape[-1])[response_mask.reshape(-1)]
    elif logits.ndim == 2 and logits.shape[0] == response_labels.numel():
        response_logits = logits
    else:
        raise ValueError(
            "Expected response-only logits with one row per supervised token, got "
            f"{tuple(logits.shape)} for {response_labels.numel()} tokens"
        )
    token_ce = F.cross_entropy(response_logits.float(), response_labels, reduction="none")
    records: list[dict[str, Any]] = []
    offset = 0
    for count_tensor in counts:
        count = int(count_tensor.item())
        ce = token_ce[offset : offset + count]
        weights = response_weights[offset : offset + count]
        windows: dict[str, float] = {}
        for name, limit in WINDOWS:
            width = count if limit is None else min(limit, count)
            selected_weights = weights[:width]
            value = (ce[:width] * selected_weights).sum() / selected_weights.sum()
            windows[name] = float(value.detach().cpu().item())
        records.append({"response_tokens": count, "windows": windows})
        offset += count
    if offset != response_labels.numel():
        raise RuntimeError("Response-logit rows were not partitioned exactly by example")
    return records


def _bootstrap_interval(values: Sequence[float], *, seed: int, samples: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Bootstrap values must be a non-empty finite vector")
    rng = np.random.RandomState(seed)
    means = np.empty(samples, dtype=np.float64)
    chunk = max(1, min(samples, 2048))
    for start in range(0, samples, chunk):
        end = min(start + chunk, samples)
        indices = rng.randint(0, len(array), size=(end - start, len(array)))
        means[start:end] = array[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {"confidence": 0.95, "low": float(low), "high": float(high)}


def _aggregate_records(
    records: Sequence[Mapping[str, Any]], *, bootstrap_seed: int, bootstrap_samples: int
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot aggregate an empty record set")
    output: dict[str, Any] = {}
    for window_index, (name, limit) in enumerate(WINDOWS):
        correct = np.asarray([row["correct_ce"][name] for row in records], dtype=np.float64)
        wrong = np.asarray([row["wrong_ce"][name] for row in records], dtype=np.float64)
        gaps = wrong - correct
        wins = (gaps > 0).astype(np.float64)
        seed = bootstrap_seed + 10_000 * window_index
        output[name] = {
            "token_limit": limit,
            "examples": len(records),
            "correct_ce_mean": float(correct.mean()),
            "wrong_ce_mean": float(wrong.mean()),
            "gap_wrong_minus_correct_mean": float(gaps.mean()),
            "gap_median": float(np.median(gaps)),
            "win_rate": float(wins.mean()),
            "tie_rate": float((gaps == 0).mean()),
            "mean_gap_bootstrap_ci": _bootstrap_interval(
                gaps, seed=seed, samples=bootstrap_samples
            ),
            "win_rate_bootstrap_ci": _bootstrap_interval(
                wins, seed=seed + 1, samples=bootstrap_samples
            ),
        }
    return output


def _evaluate_source(
    train_module,
    dataset: Any,
    *,
    source_name: str,
    pairing: Mapping[str, Any],
    pairing_sha256: str,
    collator: MultimodalCollator,
    work_dir: Path,
    sequence_length: int,
    rank_batch_instances: int,
    dp_world_size: int,
    dp_rank: int,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    correct_dataset = MultimodalFixedValidationDataset(
        dataset, pairing=pairing, pairing_sha256=pairing_sha256
    )
    wrong_dataset = MultimodalMatchedWrongImageDataset(
        dataset, pairing=pairing, pairing_sha256=pairing_sha256
    )
    global_instances = rank_batch_instances * dp_world_size
    if len(correct_dataset) % global_instances:
        raise ValueError(
            f"{source_name} recipient count {len(correct_dataset)} is not divisible by "
            f"global batch {global_instances}"
        )

    def loader_for(name: str, selected: Any) -> MultimodalDataLoader:
        return MultimodalDataLoader(
            selected,
            collator,
            work_dir=work_dir / source_name / name,
            global_batch_size=global_instances * sequence_length,
            seed=int(pairing["seed"]),
            shuffle=False,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )

    correct_loader = loader_for("correct", correct_dataset)
    wrong_loader = loader_for("wrong", wrong_dataset)
    correct_loader.reshuffle(epoch=1)
    wrong_loader.reshuffle(epoch=1)
    local_records: list[dict[str, Any]] = []
    started = time.monotonic()
    pair_rows = list(pairing["pairs"])
    for batch_index, (correct_batch, wrong_batch) in enumerate(
        zip(correct_loader, wrong_loader, strict=True)
    ):
        _assert_batches_match(correct_batch, wrong_batch)
        local_start = batch_index * global_instances + dp_rank * rank_batch_instances
        local_pairs = pair_rows[local_start : local_start + rank_batch_instances]
        if len(local_pairs) != int(correct_batch["input_ids"].shape[0]):
            raise RuntimeError("Pairing order and rank-local batch size diverged")

        correct_device = move_to_device(correct_batch, train_module.device)
        output = train_module.eval_batch(dict(correct_device), return_response_logits=True)
        if not isinstance(output, LMOutputWithLoss) or output.logits is None:
            raise TypeError("Correct-image forward did not return response-only logits")
        correct_ce = _response_ce_by_example(correct_device, output.logits)
        del output

        wrong_device = move_to_device(wrong_batch, train_module.device)
        output = train_module.eval_batch(dict(wrong_device), return_response_logits=True)
        if not isinstance(output, LMOutputWithLoss) or output.logits is None:
            raise TypeError("Wrong-image forward did not return response-only logits")
        wrong_ce = _response_ce_by_example(wrong_device, output.logits)
        del output, correct_device, wrong_device

        for offset, pair in enumerate(local_pairs):
            if correct_ce[offset]["response_tokens"] != wrong_ce[offset]["response_tokens"]:
                raise RuntimeError("Correct and wrong forwards retained different response rows")
            local_records.append(
                {
                    "pairing_position": local_start + offset,
                    "recipient_index": int(pair["recipient"]),
                    "donor_index": int(pair["donor"]),
                    "response_tokens": correct_ce[offset]["response_tokens"],
                    "correct_ce": correct_ce[offset]["windows"],
                    "wrong_ce": wrong_ce[offset]["windows"],
                    "ce_gap_wrong_minus_correct": {
                        name: wrong_ce[offset]["windows"][name]
                        - correct_ce[offset]["windows"][name]
                        for name, _ in WINDOWS
                    },
                }
            )
        if get_rank() == 0 and (batch_index == 0 or (batch_index + 1) % 10 == 0):
            log.info(
                "[%s] batch %d/%d",
                source_name,
                batch_index + 1,
                len(correct_dataset) // global_instances,
            )
        gc_cuda()

    gathered: list[Any] = [None for _ in range(dp_world_size)]
    dist.all_gather_object(gathered, local_records, group=train_module.dp_process_group)
    records = [record for rank_records in gathered for record in rank_records]
    records.sort(key=lambda row: row["pairing_position"])
    if len(records) != len(correct_dataset) or [row["pairing_position"] for row in records] != list(
        range(len(correct_dataset))
    ):
        raise RuntimeError("Distributed per-example results are incomplete or duplicated")
    return {
        "pairing_sha256": pairing_sha256,
        "examples": len(records),
        "elapsed_seconds": time.monotonic() - started,
        "metrics": _aggregate_records(
            records,
            bootstrap_seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        ),
        "per_example": records,
    }


def _checkpoint_identity(
    checkpoint: Path, config_path: Path, *, hash_workers: int = 8
) -> dict[str, Any]:
    root = _checkpoint_root(checkpoint)
    state_dir = _checkpoint_state_dir(checkpoint)
    marker = root / ".metadata.json"
    dcp_metadata = state_dir / ".metadata"
    if not marker.is_file() or not dcp_metadata.is_file():
        raise ValueError("Checkpoint marker or distributed-checkpoint metadata is missing")
    if hash_workers <= 0:
        raise ValueError("Checkpoint hash worker count must be positive")
    state_files = [path for path in sorted(state_dir.iterdir()) if path.is_file()]
    if not state_files:
        raise ValueError("Distributed checkpoint does not contain any state files")
    with ThreadPoolExecutor(max_workers=min(hash_workers, len(state_files))) as executor:
        state_hashes = list(executor.map(_sha256_file, state_files))
    inventory = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": digest,
        }
        for path, digest in zip(state_files, state_hashes, strict=True)
    ]
    identity = {
        "root": str(root.resolve()),
        "state_dir": str(state_dir.resolve()),
        "config_sha256": _sha256_file(config_path),
        "checkpoint_marker_sha256": _sha256_file(marker),
        "dcp_metadata_sha256": _sha256_file(dcp_metadata),
        "state_file_hash_algorithm": "sha256",
        "state_file_inventory_sha256": _canonical_sha256(inventory),
        "state_file_inventory": inventory,
    }
    identity["identity_sha256"] = _canonical_sha256(identity)
    return identity


def _checkpoint_identity_distributed(
    checkpoint: Path, config_path: Path, *, hash_workers: int
) -> Mapping[str, Any]:
    """Hash checkpoint contents once on rank zero and broadcast the verified identity."""
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            log.info("SHA-256 hashing every checkpoint state file under %s", checkpoint)
            packet[0] = {
                "ok": True,
                "identity": _checkpoint_identity(
                    checkpoint, config_path, hash_workers=hash_workers
                ),
            }
        except Exception as error:  # noqa: BLE001 - every rank-zero failure must be propagated.
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Could not identify checkpoint contents: {detail}")
    identity = result.get("identity")
    if not isinstance(identity, Mapping):
        raise TypeError("Broadcast checkpoint identity is malformed")
    return identity


def _native_checkpoint_load_coverage(train_module: Any, state_dir: Path) -> dict[str, Any]:
    """Prove that the eval-only native load maps every model parameter and buffer."""
    metadata = get_checkpoint_metadata(state_dir)
    checkpoint_keys = set(metadata.state_dict_metadata)
    required_methods = {
        "eval state": "_get_model_state_dict_for_eval_load",
        "checkpoint-key resolver": "_resolve_model_checkpoint_key",
        "frozen parameters": "_frozen_checkpoint_model_param_state_dict_for_load",
        "frozen tensors": "_frozen_checkpoint_param_state_dict_for_load",
        "persistent buffers": "_persistent_model_buffer_state_dict",
    }
    methods: dict[str, Any] = {}
    for label, name in required_methods.items():
        method = getattr(train_module, name, None)
        if not callable(method):
            raise TypeError(f"Native checkpoint load lacks the required {label} API {name}")
        methods[name] = method

    eval_state = methods["_get_model_state_dict_for_eval_load"](metadata)
    frozen_parameters = methods["_frozen_checkpoint_model_param_state_dict_for_load"](
        checkpoint_keys
    )
    frozen_tensors = methods["_frozen_checkpoint_param_state_dict_for_load"](checkpoint_keys)
    if set(frozen_parameters) != set(frozen_tensors):
        raise RuntimeError("Native frozen-parameter and frozen-tensor load keys differ")
    persistent_buffers = methods["_persistent_model_buffer_state_dict"]()
    missing_buffers = sorted(set(persistent_buffers) - checkpoint_keys)
    if missing_buffers:
        raise RuntimeError(
            "Native checkpoint is missing persistent model buffers: " f"{missing_buffers[:10]}"
        )

    for label, state in (
        ("eval", eval_state),
        ("frozen", frozen_tensors),
        ("buffer", persistent_buffers),
    ):
        for key, target in state.items():
            tensor_metadata = metadata.state_dict_metadata.get(key)
            if not isinstance(tensor_metadata, TensorStorageMetadata):
                raise TypeError(f"Native {label} load target {key!r} lacks tensor metadata")
            if tuple(target.size()) != tuple(tensor_metadata.size):
                raise RuntimeError(
                    f"Native {label} load target {key!r} shape {tuple(target.size())} differs "
                    f"from checkpoint shape {tuple(tensor_metadata.size)}"
                )
            metadata_numel = math.prod(int(size) for size in tensor_metadata.size)
            if target.numel() != metadata_numel:
                raise RuntimeError(
                    f"Native {label} load target {key!r} numel {target.numel()} differs "
                    f"from checkpoint numel {metadata_numel}"
                )

    model_parts = getattr(train_module, "model_parts", None)
    if not isinstance(model_parts, Sequence) or not model_parts:
        raise RuntimeError("Native checkpoint load does not expose non-empty model_parts")
    frozen_keys_by_parameter: dict[int, list[str]] = {}
    for key, parameter in frozen_parameters.items():
        frozen_keys_by_parameter.setdefault(id(parameter), []).append(key)

    parameter_names: dict[int, list[str]] = {}
    parameter_by_id: dict[int, Any] = {}
    for part_index, model_part in enumerate(model_parts):
        for name, parameter in model_part.named_parameters():
            parameter_names.setdefault(id(parameter), []).append(f"part{part_index}.{name}")
            parameter_by_id[id(parameter)] = parameter
    if not parameter_by_id:
        raise RuntimeError("Native checkpoint load model has no parameters")
    orphan_frozen_keys = sorted(
        key for key, parameter in frozen_parameters.items() if id(parameter) not in parameter_by_id
    )
    if orphan_frozen_keys:
        raise RuntimeError(
            "Native checkpoint frozen load targets are absent from model_parts: "
            f"{orphan_frozen_keys[:10]}"
        )

    covered_keys: set[str] = set()
    parameter_ids_by_checkpoint_key: dict[str, set[int]] = {}
    assignments: list[dict[str, Any]] = []
    missing_parameters: list[str] = []
    resolver = methods["_resolve_model_checkpoint_key"]
    for parameter_id, parameter in parameter_by_id.items():
        names = parameter_names[parameter_id]
        resolved_keys = {
            key
            for name in names
            if (key := resolver(name.split(".", 1)[1], checkpoint_keys)) is not None
        }
        resolved_keys &= set(eval_state)
        frozen_keys = set(frozen_keys_by_parameter.get(id(parameter), ()))
        if not resolved_keys and not frozen_keys:
            missing_parameters.extend(names)
            continue
        authoritative_keys = resolved_keys | frozen_keys
        if len(authoritative_keys) != 1:
            raise RuntimeError(
                f"Native model parameter {names} resolves ambiguously to "
                f"{sorted(authoritative_keys)}"
            )
        for key in authoritative_keys:
            parameter_ids_by_checkpoint_key.setdefault(key, set()).add(parameter_id)
        covered_keys.update(authoritative_keys)
        assignments.append(
            {
                "parameter_names": sorted(names),
                "checkpoint_keys": sorted(authoritative_keys),
            }
        )
    if missing_parameters:
        raise RuntimeError(
            "Native checkpoint load does not cover every model parameter; missing "
            f"{missing_parameters[:10]}"
        )
    multiply_mapped = sorted(
        key
        for key, parameter_ids in parameter_ids_by_checkpoint_key.items()
        if len(parameter_ids) > 1
    )
    if multiply_mapped:
        raise RuntimeError(
            "Native checkpoint keys resolve to multiple distinct model parameters: "
            f"{multiply_mapped[:10]}"
        )

    prepared_model_keys = set(eval_state) | set(frozen_tensors)
    unused_prepared_keys = sorted(prepared_model_keys - covered_keys)
    if unused_prepared_keys:
        raise RuntimeError(
            "Native checkpoint prepares model tensors not assigned to current parameters: "
            f"{unused_prepared_keys[:10]}"
        )

    def model_bearing_key(key: str) -> bool:
        return key.endswith(".main") or key.startswith(("frozen_model.", "model_buffer.", "model."))

    def logical_model_name(key: str) -> str:
        if key.startswith("frozen_model."):
            return key.removeprefix("frozen_model.")
        if key.endswith(".main"):
            key = key.removesuffix(".main")
            return key.removeprefix("module.")
        return key

    consumed_keys = covered_keys | set(persistent_buffers)
    unused_model_bearing = {
        key for key in checkpoint_keys - consumed_keys if model_bearing_key(key)
    }
    authoritative_main_names = {
        logical_model_name(key) for key in covered_keys if key.endswith(".main")
    }
    shadowed_frozen_keys = {
        key
        for key in unused_model_bearing
        if key.startswith("frozen_model.") and logical_model_name(key) in authoritative_main_names
    }
    unexpected_unused_model_keys = sorted(unused_model_bearing - shadowed_frozen_keys)
    if unexpected_unused_model_keys:
        raise RuntimeError(
            "Native checkpoint contains unused model-bearing keys: "
            f"{unexpected_unused_model_keys[:10]}"
        )

    report = {
        "complete": True,
        "checkpoint_key_count": len(checkpoint_keys),
        "model_parameter_count": len(parameter_by_id),
        "model_parameter_checkpoint_key_count": len(covered_keys),
        "model_parameter_checkpoint_keys_sha256": _canonical_sha256(sorted(covered_keys)),
        "model_parameter_assignments_sha256": _canonical_sha256(
            sorted(assignments, key=lambda assignment: assignment["parameter_names"])
        ),
        "eval_state_key_count": len(eval_state),
        "frozen_state_key_count": len(frozen_parameters),
        "persistent_buffer_count": len(persistent_buffers),
        "persistent_buffer_keys_sha256": _canonical_sha256(sorted(persistent_buffers)),
        "shadowed_frozen_key_count": len(shadowed_frozen_keys),
        "shadowed_frozen_keys_sha256": _canonical_sha256(sorted(shadowed_frozen_keys)),
        "unused_model_bearing_key_count": 0,
        "prepared_load_key_count": len(
            set(eval_state) | set(frozen_tensors) | set(persistent_buffers)
        ),
    }
    report["sha256"] = _canonical_sha256(report)
    return report


def _native_checkpoint_load_coverage_distributed(
    train_module: Any, state_dir: Path
) -> dict[str, Any]:
    """Require every rank to produce the same complete native-load coverage report."""
    try:
        local: dict[str, Any] = {
            "ok": True,
            "report": _native_checkpoint_load_coverage(train_module, state_dir),
        }
    except Exception as error:  # noqa: BLE001 - all ranks must receive every local failure.
        local = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    gathered: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local)
    failures = [
        f"rank {rank}: {packet.get('error')}"
        if isinstance(packet, Mapping)
        else f"rank {rank}: malformed report {packet!r}"
        for rank, packet in enumerate(gathered)
        if not isinstance(packet, Mapping) or packet.get("ok") is not True
    ]
    if failures:
        raise RuntimeError(f"Native checkpoint load coverage failed: {failures}")
    reports = [packet["report"] for packet in gathered]
    if any(report != reports[0] for report in reports[1:]):
        raise RuntimeError("Native checkpoint load coverage differs across ranks")
    return reports[0]


def _git_identity() -> dict[str, Any]:
    def command(*args: str) -> bytes | None:
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            return None

    revision = command("rev-parse", "HEAD")
    status = command("status", "--porcelain=v1", "--untracked-files=all")
    diff = command("diff", "--binary", "HEAD")
    return {
        "revision": revision.decode().strip() if revision is not None else None,
        "dirty": bool(status.strip()) if status is not None else None,
        "status_sha256": _sha256_bytes(status) if status is not None else None,
        "tracked_diff_sha256": _sha256_bytes(diff) if diff is not None else None,
    }


def _write_result_distributed(
    output: Path, payload: Mapping[str, Any], *, overwrite: bool = False
) -> None:
    """Atomically write on rank zero and propagate persistence failure to every rank."""
    packet: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            _write_json_atomic(output, payload, overwrite=overwrite)
            packet[0] = {"ok": True}
        except Exception as error:  # noqa: BLE001 - every rank-zero failure must be propagated.
            packet[0] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(packet, src=0)
    result = packet[0]
    if not isinstance(result, Mapping) or result.get("ok") is not True:
        detail = result.get("error") if isinstance(result, Mapping) else repr(result)
        raise RuntimeError(f"Could not persist matched-wrong-image results: {detail}")
    if dist.get_rank() == 0:
        log.info("Wrote matched-wrong-image results to %s", output)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the native distributed matched-wrong-image evaluation."""
    args = _parse_args(argv)
    _validate_args(args)
    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prepare_training_environment()
    try:
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        config_path = _config_path(checkpoint, args.config)
        raw_config = _load_json(config_path)
        data_config = raw_config["data"]
        evaluation_config = raw_config["evaluation"]
        artifacts = raw_config["artifacts"]
        sequence_length = int(data_config["sequence_length"])
        if data_config["message_format"] != "document":
            raise ValueError("This evaluator requires the pretrained-native document format")
        rank_batch_instances = (
            args.rank_batch_instances
            if args.rank_batch_instances is not None
            else int(evaluation_config["rank_batch_instances"])
        )
        pairing_seed = (
            args.pairing_seed if args.pairing_seed is not None else int(evaluation_config["seed"])
        )
        bootstrap_seed = (
            args.bootstrap_seed if args.bootstrap_seed is not None else pairing_seed + 1_000_003
        )
        output = (
            Path(args.output).expanduser().resolve() if args.output else _default_output(checkpoint)
        )
        pairing_paths = _resolve_pairing_paths(args, output, args.sources)
        expected_pairing_sha256 = _parse_expected_pairing_sha256(
            args.expected_pairing_sha256, args.sources
        )
        excluded_pairing_paths = _parse_pairing_paths(
            args.exclude_pairing, option_name="--exclude-pairing"
        )
        unexpected_exclusions = sorted(set(excluded_pairing_paths) - set(args.sources))
        if unexpected_exclusions:
            raise ValueError(
                f"Excluded pairings were supplied for disabled sources: {unexpected_exclusions}"
            )
        expected_excluded_pairing_sha256 = _parse_expected_pairing_sha256(
            args.expected_exclude_pairing_sha256, args.sources
        )
        _preflight_artifact_paths_distributed(
            output=output,
            overwrite_output=args.overwrite_output,
            pairing_paths=pairing_paths,
            expected_pairing_sha256=expected_pairing_sha256,
            excluded_pairing_paths=excluded_pairing_paths,
            expected_excluded_pairing_sha256=expected_excluded_pairing_sha256,
        )
        checkpoint_identity = _checkpoint_identity_distributed(
            checkpoint,
            config_path,
            hash_workers=args.checkpoint_hash_workers,
        )
        work_dir = (
            Path(args.work_dir).expanduser().resolve()
            if args.work_dir
            else Path(os.environ.get("RESULTS_DIR", "/tmp")) / "vision-alignment-matched-wrong"
        )

        tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
            identifier=artifacts["tokenizer_id"],
            revision=artifacts["tokenizer_revision"],
            expected_fingerprint=artifacts["tokenizer_fingerprint"],
            cache_dir=artifacts["hf_cache_dir"],
        )
        if int(raw_config["model"]["image_patch_token_id"]) != token_ids.im_patch_id:
            raise ValueError("Pinned tokenizer image-patch ID differs from the checkpoint")
        if tokenizer.pad_token_id is None:
            raise ValueError("Pinned Dolma tokenizer does not define a pad token")
        validation_manifest, content_ids, manifest_identity = _load_validation_manifest(
            raw_config, str(data_config["pixmo_cap_path"])
        )
        spec = _source_spec(raw_config)
        datasets = {
            source: build_vision_alignment_dataset_config(
                spec, token_ids, source, split="validation"
            ).build(tokenizer)
            for source in args.sources
        }
        live_dataset_identities = {
            source: _validate_live_validation_dataset(dataset, validation_manifest)
            for source, dataset in datasets.items()
        }
        manifest_identity["live_datasets"] = live_dataset_identities

        pairing_payloads: dict[str, Mapping[str, Any]] = {}
        pairings: dict[str, Any] = {}
        for source in args.sources:
            pairing, pairing_sha, provenance, exclusion_identity = _load_or_build_pairing(
                datasets[source],
                path=pairing_paths[source],
                source_name=source,
                expected_sha256=expected_pairing_sha256.get(source),
                examples=args.examples,
                seed=pairing_seed,
                content_ids=content_ids,
                excluded_pairing_path=excluded_pairing_paths.get(source),
                expected_excluded_pairing_sha256=expected_excluded_pairing_sha256.get(source),
            )
            pairing_payloads[source] = pairing
            coverage = pairing.get("coverage")
            if not isinstance(coverage, Mapping):
                raise TypeError(f"{source} pairing does not report matched-eligibility coverage")
            pairings[source] = {
                "path": str(pairing_paths[source]),
                "sha256": pairing_sha,
                "expected_sha256": expected_pairing_sha256.get(source),
                "provenance": provenance,
                "population": "matched_eligible_validation_subset",
                "pairing_schema_version": pairing["version"],
                "coverage": coverage,
                "recipient_indices_sha256": _canonical_sha256(
                    [pair["recipient"] for pair in pairing["pairs"]]
                ),
                "donor_indices_sha256": _canonical_sha256(
                    [pair["donor"] for pair in pairing["pairs"]]
                ),
                "excluded_primary_pairing": exclusion_identity,
            }

        model, module_config = _build_model_and_module(
            raw_config,
            sequence_length=sequence_length,
            rank_batch_instances=rank_batch_instances,
        )
        train_module = module_config.build(model, eval_only=True)
        state_dir = _checkpoint_state_dir(checkpoint)
        native_load_coverage = _native_checkpoint_load_coverage_distributed(
            train_module,
            state_dir,
        )
        log.info("Loading native Vision Alignment checkpoint from %s", state_dir)
        train_module.load_state_dict_direct(
            state_dir,
            process_group=dist.group.WORLD,
            thread_count=args.checkpoint_load_threads,
            load_optim_state=False,
        )
        native_load_coverage["load_completed"] = True
        native_load_coverage["sha256"] = _canonical_sha256(
            {key: value for key, value in native_load_coverage.items() if key != "sha256"}
        )
        dp_world_size = get_world_size(train_module.dp_process_group)
        dp_rank = get_rank(train_module.dp_process_group)
        global_instances = rank_batch_instances * dp_world_size
        if args.examples % global_instances:
            raise ValueError(
                f"--examples ({args.examples}) must divide global batch {global_instances}"
            )

        collator = MultimodalCollator(
            pad_token_id=int(tokenizer.pad_token_id),
            label_ignore_index=-100,
            pad_sequence_length=sequence_length,
        )
        results: dict[str, Any] = {}
        for source_index, source in enumerate(args.sources):
            pairing = pairing_payloads[source]
            pairing_sha = pairings[source]["sha256"]
            source_result = _evaluate_source(
                train_module,
                datasets[source],
                source_name=source,
                pairing=pairing,
                pairing_sha256=pairing_sha,
                collator=collator,
                work_dir=work_dir,
                sequence_length=sequence_length,
                rank_batch_instances=rank_batch_instances,
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
                bootstrap_seed=bootstrap_seed + source_index * 1_000_000,
                bootstrap_samples=args.bootstrap_samples,
            )
            source_result["population"] = "matched_eligible_validation_subset"
            source_result["coverage"] = pairing["coverage"]
            results[source] = source_result

        protocol = {
            "name": "vision-alignment-native-matched-wrong-image-v3",
            "sources": list(args.sources),
            "dataset_split": "validation",
            "evaluation_population": "matched_eligible_validation_subset",
            "population_definition": (
                "deterministically selected validation rows that have a distinct donor with "
                "the exact same image tensor geometry and byte-identical pooled_patches_idx; "
                "this is conditional on matching eligibility, not the full validation split"
            ),
            "examples_per_source": args.examples,
            "source_epoch": 0,
            "pairing_seed": pairing_seed,
            "pairing_sha256": {source: pairings[source]["sha256"] for source in args.sources},
            "pairing_pin_policy": (
                "every existing pairing file requires an exact source-specific CLI SHA-256 pin"
            ),
            "pairing_rule": (
                "distinct pinned content and materialized pixels; identical image tensor shape "
                "and byte-identical pooled_patches_idx; explicit unique donors; when an "
                "excluded primary pairing is supplied, none of its recipients or donors may "
                "be selected"
            ),
            "recipient_replay": "correct and wrong forwards use exactly the same recipients",
            "response_logits": "only positive-loss-mask positions are materialized",
            "per_example_ce": (
                "loss-mask-weighted mean over all or the first min(K,N) supervised response "
                "tokens in flattened sequence order; EOS is included when supervised"
            ),
            "gap_sign": "wrong_ce - correct_ce; positive is a correct-image win",
            "bootstrap": {
                "method": "deterministic iid example bootstrap percentile interval",
                "confidence": 0.95,
                "samples": args.bootstrap_samples,
                "seed": bootstrap_seed,
            },
            "windows": {name: limit for name, limit in WINDOWS},
            "message_format": data_config["message_format"],
            "loss_token_weighting": data_config["loss_token_weighting"],
            "sequence_length": sequence_length,
            "rank_batch_instances": rank_batch_instances,
            "global_batch_instances": global_instances,
            "world_size": WORLD_SIZE,
            "ep_degree": EP_DEGREE,
            "dp_process_group_size": dp_world_size,
            "source_registry_sha256": vision_alignment_source_registry_sha256(),
            "tokenizer": {
                "id": artifacts["tokenizer_id"],
                "revision": artifacts["tokenizer_revision"],
                "fingerprint": artifacts["tokenizer_fingerprint"],
                "token_ids": token_ids.as_config_dict(),
            },
        }
        protocol["sha256"] = _canonical_sha256(protocol)
        script_path = Path(__file__).resolve()
        pairing_implementation_path_value = inspect.getsourcefile(build_matched_wrong_image_pairing)
        if pairing_implementation_path_value is None:
            raise RuntimeError("Could not resolve matched-wrong pairing implementation source")
        pairing_implementation_path = Path(pairing_implementation_path_value).resolve()
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": checkpoint_identity,
            "native_checkpoint_load": native_load_coverage,
            "config_path": str(config_path),
            "git": _git_identity(),
            "evaluator": {
                "path": str(script_path),
                "sha256": _sha256_file(script_path),
                "pairing_implementation_path": str(pairing_implementation_path),
                "pairing_implementation_sha256": _sha256_file(pairing_implementation_path),
            },
            "validation": manifest_identity,
            "pairings": pairings,
            "artifact_policy": {
                "existing_pairing_requires_sha256_pin": True,
                "expected_pairing_sha256": expected_pairing_sha256,
                "expected_excluded_pairing_sha256": expected_excluded_pairing_sha256,
                "output_overwrite_enabled": args.overwrite_output,
            },
            "protocol": protocol,
            "results": results,
        }
        payload["config_and_protocol_sha256"] = _canonical_sha256(
            {
                "checkpoint_config_sha256": payload["checkpoint"]["config_sha256"],
                "protocol_sha256": protocol["sha256"],
                "pairing_sha256": {source: pairings[source]["sha256"] for source in args.sources},
            }
        )
        _write_result_distributed(output, payload, overwrite=args.overwrite_output)
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
