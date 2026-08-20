"""Pinned validation-data helpers for SSMax bridge evidence producers."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from olmo_core.data.multimodal.vision_alignment_sources import (
    VisionAlignmentSourceSpec,
    build_vision_alignment_dataset_config,
    load_pinned_vision_alignment_tokenizer,
    pixmo_row_path_inventory,
    runtime_dataset_fingerprint,
)
from olmo_core.eval.matched_wrong_image import (
    build_matched_wrong_image_pairing,
    matched_wrong_image_pairing_sha256,
    serialize_matched_wrong_image_pairing,
    validate_matched_wrong_image_pairing,
)

from .vision_alignment_ssmax_bridge import (
    SOURCES,
    SSMaxBridgeEvidenceError,
    load_json,
    sha256_file,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def content_ids_sha256(content_ids: Sequence[str]) -> str:
    """Hash newline-delimited validation image-content IDs."""

    return _sha256_bytes("".join(f"{value}\n" for value in content_ids).encode())


def load_validation_manifest(
    raw_config: Mapping[str, Any], *, expected_path: Path, expected_sha256: str
) -> tuple[Mapping[str, Any], tuple[str, ...], dict[str, Any]]:
    """Load the exact validation manifest named by config and outer run manifest."""

    evaluation = raw_config.get("evaluation")
    data = raw_config.get("data")
    if not isinstance(evaluation, Mapping) or not isinstance(data, Mapping):
        raise SSMaxBridgeEvidenceError("Checkpoint config lacks evaluation/data mappings")
    configured_path = evaluation.get("validation_manifest_path")
    configured_sha = evaluation.get("validation_manifest_sha256")
    manifest_path = expected_path.expanduser().resolve()
    if (
        not isinstance(configured_path, str)
        or Path(configured_path).expanduser().resolve() != manifest_path
        or configured_sha != expected_sha256
        or sha256_file(manifest_path) != expected_sha256
    ):
        raise SSMaxBridgeEvidenceError(
            "Checkpoint, outer manifest, and live validation-manifest identities differ"
        )
    manifest = load_json(manifest_path)
    if not isinstance(manifest, Mapping) or (
        manifest.get("format") != "vision_alignment_validation_manifest"
        or manifest.get("version") != 3
    ):
        raise SSMaxBridgeEvidenceError("Validation manifest has an incompatible format")
    try:
        builder = manifest["builder"]
        output = manifest["output"]
        validation = output["splits"]["validation"]
        content_value = validation["row_image_content_path"]
        expected_content_sha = validation["row_image_content_sha256"]
        expected_examples = validation["examples"]
        expected_fingerprint = validation["dataset_fingerprint"]
        expected_paths_sha = validation["row_image_paths_sha256"]
        expected_unique_paths = validation["unique_image_paths"]
        paths_algorithm = builder["row_image_paths_algorithm"]
        dataset_value = output["dataset_path"]
    except (KeyError, TypeError) as error:
        raise SSMaxBridgeEvidenceError(
            "Validation manifest is missing pinned output identities"
        ) from error
    dataset_path = Path(str(data.get("pixmo_cap_path", ""))).expanduser().resolve()
    if dataset_path != (manifest_path.parent / dataset_value).resolve():
        raise SSMaxBridgeEvidenceError("Checkpoint PixMo path differs from validation manifest")
    content_relative = Path(content_value)
    if content_relative.is_absolute():
        raise SSMaxBridgeEvidenceError("Validation row-content path must be relative")
    content_path = (manifest_path.parent / content_relative).resolve()
    if not content_path.is_relative_to(manifest_path.parent):
        raise SSMaxBridgeEvidenceError("Validation row-content path escapes its artifact root")
    raw_content = content_path.read_bytes()
    if _sha256_bytes(raw_content) != expected_content_sha:
        raise SSMaxBridgeEvidenceError("Validation row-content bytes differ from their pin")
    content_ids = tuple(raw_content.decode("utf-8").splitlines())
    if (
        not raw_content.endswith(b"\n")
        or type(expected_examples) is not int
        or len(content_ids) != expected_examples
        or any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in content_ids
        )
    ):
        raise SSMaxBridgeEvidenceError("Validation row-content identities are malformed")
    identity = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": expected_sha256,
        "row_content_path": str(content_path),
        "row_content_sha256": expected_content_sha,
        "expected_live_dataset": {
            "dataset_fingerprint": expected_fingerprint,
            "examples": expected_examples,
            "row_image_paths_algorithm": paths_algorithm,
            "row_image_paths_sha256": expected_paths_sha,
            "unique_image_paths": expected_unique_paths,
        },
    }
    return manifest, content_ids, identity


def source_spec(raw_config: Mapping[str, Any]) -> VisionAlignmentSourceSpec:
    """Reconstruct the exact importable source specification saved by training."""

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


def build_validation_datasets(
    raw_config: Mapping[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
) -> tuple[Any, Any, Mapping[str, Any], tuple[str, ...], dict[str, Any]]:
    """Build and content-validate both fixed bridge validation source wrappers."""

    artifacts = raw_config["artifacts"]
    tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
        identifier=artifacts["tokenizer_id"],
        revision=artifacts["tokenizer_revision"],
        expected_fingerprint=artifacts["tokenizer_fingerprint"],
        cache_dir=artifacts["hf_cache_dir"],
    )
    if tokenizer.pad_token_id is None:
        raise SSMaxBridgeEvidenceError("Pinned tokenizer has no pad token")
    if int(raw_config["model"]["image_patch_token_id"]) != token_ids.im_patch_id:
        raise SSMaxBridgeEvidenceError("Tokenizer and checkpoint image-patch IDs differ")
    validation_manifest, content_ids, identity = load_validation_manifest(
        raw_config,
        expected_path=manifest_path,
        expected_sha256=manifest_sha256,
    )
    spec = source_spec(raw_config)
    datasets = {
        source: build_vision_alignment_dataset_config(
            spec, token_ids, source, split="validation"
        ).build(tokenizer)
        for source in SOURCES
    }
    identity["live_datasets"] = {
        source: validate_live_validation_dataset(dataset, validation_manifest)
        for source, dataset in datasets.items()
    }
    return tokenizer, token_ids, datasets, content_ids, identity


def validate_live_validation_dataset(dataset: Any, manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a live validation wrapper to its Arrow fingerprint and row-path inventory."""

    validation = manifest["output"]["splits"]["validation"]
    algorithm = manifest["builder"]["row_image_paths_algorithm"]
    inventory = pixmo_row_path_inventory(dataset)
    expected = {
        "dataset_fingerprint": validation["dataset_fingerprint"],
        "examples": validation["examples"],
        "row_image_paths_algorithm": algorithm,
        "row_image_paths_sha256": validation["row_image_paths_sha256"],
        "unique_image_paths": validation["unique_image_paths"],
    }
    actual = {
        "dataset_fingerprint": runtime_dataset_fingerprint(dataset),
        "examples": len(dataset),
        "row_image_paths_algorithm": inventory["algorithm"],
        "row_image_paths_sha256": inventory["sha256"],
        "unique_image_paths": inventory["unique_paths"],
    }
    if actual != expected:
        differing = sorted(name for name in expected if expected[name] != actual[name])
        raise SSMaxBridgeEvidenceError(f"Live validation dataset differs in fields {differing}")
    return actual


def load_fixed_pairing(
    path: Path,
    *,
    expected_sha256: str,
    dataset_size: int,
    examples: int,
    seed: int,
    content_ids: Sequence[str],
) -> Mapping[str, Any]:
    """Load one immutable pairing; evaluation never creates or replaces it."""

    path = path.expanduser().resolve()
    if sha256_file(path) != expected_sha256:
        raise SSMaxBridgeEvidenceError(f"Pairing {path} differs from its raw-byte pin")
    payload = load_json(path)
    if not isinstance(payload, Mapping):
        raise SSMaxBridgeEvidenceError(f"Pairing {path} must contain an object")
    validate_matched_wrong_image_pairing(
        payload,
        dataset_size=dataset_size,
        recipient_count=examples,
        seed=seed,
        epoch=0,
        content_ids_sha256=content_ids_sha256(content_ids),
    )
    if matched_wrong_image_pairing_sha256(payload) != expected_sha256:
        raise SSMaxBridgeEvidenceError(f"Pairing {path} canonical SHA-256 differs")
    return payload


def create_or_validate_pairing(
    dataset: Any,
    *,
    path: Path,
    examples: int,
    seed: int,
    content_ids: Sequence[str],
) -> dict[str, str]:
    """Create a fixed pairing once, or validate an existing candidate before finalization."""

    path = path.expanduser().resolve()
    if path.exists():
        expected = sha256_file(path)
        load_fixed_pairing(
            path,
            expected_sha256=expected,
            dataset_size=len(dataset),
            examples=examples,
            seed=seed,
            content_ids=content_ids,
        )
        return {"path": str(path), "sha256": expected}
    payload = build_matched_wrong_image_pairing(
        dataset,
        recipient_count=examples,
        seed=seed,
        content_ids=content_ids,
        epoch=0,
    )
    raw = serialize_matched_wrong_image_pairing(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.building")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to replace fixed pairing {path}") from error
    finally:
        if temporary.exists():
            temporary.unlink()
    expected = hashlib.sha256(raw).hexdigest()
    load_fixed_pairing(
        path,
        expected_sha256=expected,
        dataset_size=len(dataset),
        examples=examples,
        seed=seed,
        content_ids=content_ids,
    )
    return {"path": str(path), "sha256": expected}


__all__ = [
    "build_validation_datasets",
    "content_ids_sha256",
    "create_or_validate_pairing",
    "load_fixed_pairing",
    "load_validation_manifest",
    "source_spec",
    "validate_live_validation_dataset",
]
