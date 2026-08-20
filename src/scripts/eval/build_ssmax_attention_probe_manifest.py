"""Freeze bounded SSMax attention queries from the audited Vision Alignment validation split.

This command reads a completed Vision Alignment ``config.json`` solely as an artifact contract. It
verifies the pinned validation-manifest bytes and live Arrow split, selects rows by deterministic
SHA-256 priority, tokenizes them through the pinned recipe, and writes a canonical probe manifest.
No model checkpoint is opened.

Example::

    python src/scripts/eval/build_ssmax_attention_probe_manifest.py \
        --config /path/to/vision-alignment/step0/config.json \
        --output /path/to/ssmax-attention-probe-v1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from olmo_core.data.multimodal.ssmax_single_response import (
    SSMAX_SINGLE_RESPONSE_PROJECTION_SEED,
    SSMaxSingleResponseDataset,
)
from olmo_core.data.multimodal.vision_alignment_perception_provenance import (
    build_selected_perception_dataset,
    load_perception_provenance_manifest,
)
from olmo_core.data.multimodal.vision_alignment_sources import (
    VisionAlignmentSourceSpec,
    build_vision_alignment_dataset_config,
    load_pinned_vision_alignment_tokenizer,
    pixmo_row_path_inventory,
    runtime_dataset_fingerprint,
)
from olmo_core.eval.ssmax_attention_diagnostics import (
    ProbeSequence,
    SSMaxProbeManifest,
    build_probe_manifest,
    iter_ssmax_probe_batches,
    serialize_probe_manifest,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Vision Alignment checkpoint config JSON.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source",
        choices=("pixmo_caption", "pixmo_transcript"),
        default="pixmo_caption",
    )
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--max-queries-per-category-per-row", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--perception-provenance",
        help=(
            "Build over the exact provenance-selected perception validation population instead "
            "of the bridge validation population. Only pixmo_caption is supported."
        ),
    )
    parser.add_argument(
        "--expected-perception-provenance-sha256",
        help="Required independent raw-byte pin for --perception-provenance.",
    )
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not decode JSON artifact {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON artifact {path} must contain a mapping")
    return payload


def _validation_population(
    raw_config: Mapping[str, Any],
) -> tuple[Path, str, Mapping[str, Any], tuple[str, ...]]:
    evaluation = raw_config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("Vision Alignment config lacks evaluation metadata")
    path_value = evaluation.get("validation_manifest_path")
    expected_sha = evaluation.get("validation_manifest_sha256")
    if not isinstance(path_value, str) or not isinstance(expected_sha, str):
        raise ValueError("Config does not pin a validation-manifest path and SHA-256")
    path = Path(path_value).expanduser().resolve()
    actual_sha = _sha256_file(path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"Validation-manifest SHA mismatch: expected {expected_sha}, got {actual_sha}"
        )
    manifest = _load_json(path)
    if (
        manifest.get("format") != "vision_alignment_validation_manifest"
        or manifest.get("version") != 3
    ):
        raise ValueError("Validation manifest has an incompatible format")
    try:
        validation = manifest["output"]["splits"]["validation"]
        content_relative = validation["row_image_content_path"]
        content_sha = validation["row_image_content_sha256"]
    except (KeyError, TypeError) as error:
        raise ValueError("Validation manifest lacks row-content identities") from error
    content_path = (path.parent / content_relative).resolve()
    if not content_path.is_relative_to(path.parent):
        raise ValueError("Validation row-content identity path escapes its artifact directory")
    content_bytes = content_path.read_bytes()
    if hashlib.sha256(content_bytes).hexdigest() != content_sha:
        raise ValueError("Validation row-content identity file differs from its SHA-256 pin")
    content_ids = tuple(content_bytes.decode().splitlines())
    if len(content_ids) != validation["examples"]:
        raise ValueError("Validation row-content count differs from its manifest")
    return path, actual_sha, manifest, content_ids


def _source_spec(raw_config: Mapping[str, Any]) -> VisionAlignmentSourceSpec:
    try:
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
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Vision Alignment config lacks a complete source specification") from error


def _validate_live_dataset(dataset: Any, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    validation = manifest["output"]["splits"]["validation"]
    inventory = pixmo_row_path_inventory(dataset)
    actual = {
        "dataset_fingerprint": runtime_dataset_fingerprint(dataset),
        "examples": len(dataset),
        "row_image_paths_algorithm": inventory["algorithm"],
        "row_image_paths_sha256": inventory["sha256"],
        "unique_image_paths": inventory["unique_paths"],
    }
    expected = {
        "dataset_fingerprint": validation["dataset_fingerprint"],
        "examples": validation["examples"],
        "row_image_paths_algorithm": manifest["builder"]["row_image_paths_algorithm"],
        "row_image_paths_sha256": validation["row_image_paths_sha256"],
        "unique_image_paths": validation["unique_image_paths"],
    }
    if actual != expected:
        differing = sorted(name for name in expected if actual[name] != expected[name])
        raise ValueError(f"Live validation dataset differs in fields {differing}")
    return actual


def _selected_indices(
    content_ids: Sequence[str], *, source: str, rows: int, seed: int
) -> list[int]:
    if rows <= 0 or rows > len(content_ids):
        raise ValueError(f"--rows must be within [1, {len(content_ids)}]")
    if seed < 0:
        raise ValueError("--seed must be non-negative")
    ranked = []
    for index, content_id in enumerate(content_ids):
        priority = hashlib.sha256(
            f"ssmax-probe-row-v1\0{seed}\0{source}\0{index}\0{content_id}".encode()
        ).digest()
        ranked.append((priority, index))
    ranked.sort()
    return sorted(index for _, index in ranked[:rows])


def _get_example(dataset: Any, index: int) -> Mapping[str, Any]:
    get = getattr(dataset, "get", None)
    example = get(index, 0) if callable(get) else dataset[index]
    if not isinstance(example, Mapping):
        raise ValueError(f"Validation row {index} did not produce a mapping")
    return example


def _content_ids_sha256(content_ids: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in content_ids).encode()).hexdigest()


def _perception_population(
    raw_config: Mapping[str, Any],
    *,
    provenance_path: Path,
    expected_sha256: str,
    source: str,
    tokenizer: Any,
    token_ids: Any,
) -> tuple[Path, str, Any, tuple[str, ...], dict[str, Any]]:
    if source != "pixmo_caption":
        raise ValueError("Perception attention probes are fixed to pixmo_caption")
    provenance = load_perception_provenance_manifest(
        provenance_path,
        expected_sha256=expected_sha256,
        verify_finevision_materialization=False,
        load_image_path_signatures=False,
    )
    artifacts = raw_config.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Vision Alignment config lacks artifact metadata")
    for field_name, expected in (
        ("tokenizer_id", provenance.source_spec.tokenizer_id),
        ("tokenizer_revision", provenance.source_spec.tokenizer_revision),
        ("tokenizer_fingerprint", provenance.source_spec.tokenizer_fingerprint),
    ):
        if artifacts.get(field_name) != expected:
            raise ValueError(
                f"Config {field_name} differs from the perception provenance source spec"
            )
    model = raw_config.get("model")
    if not isinstance(model, Mapping) or model.get("image_patch_token_id") != token_ids.im_patch_id:
        raise ValueError("Config and pinned tokenizer image-patch token IDs differ")
    base_dataset = build_selected_perception_dataset(
        provenance,
        tokenizer,
        token_ids,
        source,
        logical_split="validation",
        validate_required_annotations=True,
        verify_finevision_materialization=False,
    )
    projection = raw_config.get("data", {}).get("ssmax_single_response_projection")
    if not isinstance(projection, Mapping):
        raise ValueError("Perception config lacks its SSMax single-response projection")
    projection_seed = projection.get("seed")
    loss_token_weighting = raw_config["data"].get("loss_token_weighting")
    if projection_seed != SSMAX_SINGLE_RESPONSE_PROJECTION_SEED:
        raise ValueError("Perception config has a non-canonical SSMax projection seed")
    dataset = SSMaxSingleResponseDataset(
        base_dataset,
        source_name=source,
        logical_split="validation",
        seed=projection_seed,
        loss_token_weighting=loss_token_weighting,
    )
    selection = provenance.selection(source, "validation")
    content_ids = tuple(selection.row_image_content_sha256)
    if len(content_ids) != len(dataset):
        raise ValueError("Perception provenance content identities differ from selected rows")
    identity = {
        "contract": "perception-provenance-selected-validation-v1",
        "dataset_fingerprint": dataset.content_fingerprint,
        "base_dataset_fingerprint": base_dataset.content_fingerprint,
        "examples": len(dataset),
        "logical_split": "validation",
        "physical_split": selection.physical_split,
        "selection_indices_sha256": selection.selection_indices_sha256,
        "row_image_content_sha256": _content_ids_sha256(content_ids),
        "provenance_content_sha256": provenance.content_sha256,
        "source_spec_sha256": provenance.source_spec_sha256,
        "single_response_projection": dataset.contract,
    }
    return provenance.path, provenance.raw_sha256, dataset, content_ids, identity


def _preflight_probe_rows(
    dataset: Any,
    manifest: SSMaxProbeManifest,
    *,
    content_ids: Sequence[str],
) -> dict[str, Any]:
    """Reconstruct and verify every row without loading a model or allocating GPU state."""

    dataset_indices = [
        int(row["dataset_index"])
        for row in sorted(manifest.rows_by_sample_id.values(), key=lambda row: row["dataset_index"])
    ]
    validate_images = getattr(dataset, "validate_image_content", None)
    image_bytes_sha256 = validate_images(dataset_indices) if callable(validate_images) else None
    batches = iter_ssmax_probe_batches(
        dataset,
        manifest,
        content_ids=content_ids,
        collate=lambda rows: {"examples": len(rows)},
        rank=0,
        world_size=1,
        batch_size=8,
    )
    sample_ids = [sample_id for batch in batches for sample_id in batch.sample_ids]
    expected_ids = [
        str(row["sample_id"])
        for row in sorted(manifest.rows_by_sample_id.values(), key=lambda row: row["dataset_index"])
    ]
    if sample_ids != expected_ids:
        raise ValueError("Data-only probe preflight reconstructed a different sample order")
    result = {
        "protocol": "ssmax-probe-data-only-preflight-v1",
        "rows": len(sample_ids),
        "sample_ids_sha256": hashlib.sha256(
            "".join(f"{value}\n" for value in sample_ids).encode()
        ).hexdigest(),
        "live_selected_image_bytes_sha256": image_bytes_sha256,
    }
    result["sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace existing probe manifest {output_path}")
    raw_config = _load_json(config_path)
    if bool(args.perception_provenance) != bool(args.expected_perception_provenance_sha256):
        raise ValueError(
            "--perception-provenance and --expected-perception-provenance-sha256 are required together"
        )
    artifacts = raw_config["artifacts"]
    tokenizer, token_ids = load_pinned_vision_alignment_tokenizer(
        identifier=artifacts["tokenizer_id"],
        revision=artifacts["tokenizer_revision"],
        expected_fingerprint=artifacts["tokenizer_fingerprint"],
        cache_dir=artifacts["hf_cache_dir"],
    )
    if args.perception_provenance:
        (
            validation_path,
            validation_sha,
            dataset,
            content_ids,
            live_identity,
        ) = _perception_population(
            raw_config,
            provenance_path=Path(args.perception_provenance).expanduser().resolve(),
            expected_sha256=args.expected_perception_provenance_sha256,
            source=args.source,
            tokenizer=tokenizer,
            token_ids=token_ids,
        )
    else:
        validation_path, validation_sha, validation_manifest, content_ids = _validation_population(
            raw_config
        )
        spec = _source_spec(raw_config)
        dataset = build_vision_alignment_dataset_config(
            spec, token_ids, args.source, split="validation"
        ).build(tokenizer)
        live_identity = dict(_validate_live_dataset(dataset, validation_manifest))
    indices = _selected_indices(content_ids, source=args.source, rows=args.rows, seed=args.seed)
    sequences = []
    for index in indices:
        example = _get_example(dataset, index)
        input_ids = torch.as_tensor(example["input_ids"])
        sequences.append(
            ProbeSequence(
                sample_id=f"{args.source}:{index}:{content_ids[index]}",
                dataset_index=index,
                input_ids=input_ids,
                token_type_ids=torch.as_tensor(example["token_type_ids"]),
                loss_masks=torch.as_tensor(example["loss_masks"]),
                valid_tokens=torch.ones_like(input_ids, dtype=torch.bool),
            )
        )
    manifest = build_probe_manifest(
        sequences,
        validation_manifest_path=validation_path,
        validation_manifest_sha256=validation_sha,
        seed=args.seed,
        max_queries_per_category_per_row=args.max_queries_per_category_per_row,
    )
    payload = manifest.as_dict()
    payload["population"] = {
        "source": args.source,
        "split": "validation",
        "epoch": 0,
        "row_selection_algorithm": "sha256-priority-over-content-id-v1",
        "row_selection_seed": args.seed,
        "selected_dataset_indices": indices,
        "selected_content_ids": [content_ids[index] for index in indices],
        "live_dataset": live_identity,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "tokenizer": {
            "id": artifacts["tokenizer_id"],
            "revision": artifacts["tokenizer_revision"],
            "fingerprint": artifacts["tokenizer_fingerprint"],
            "token_ids": token_ids.as_config_dict(),
        },
    }
    manifest = SSMaxProbeManifest.from_dict(payload)
    preflight = _preflight_probe_rows(dataset, manifest, content_ids=content_ids)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_bytes(serialize_probe_manifest(manifest))
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": manifest.sha256,
                "rows": len(indices),
                "source": args.source,
                "preflight": preflight,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
