#!/usr/bin/env python3
"""Convert a legacy OLMoE v2 checkpoint to a model-only OLMoDDP checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

from olmo_core.distributed.checkpoint import RemoteFileSystemReader, save_state_dict
from olmo_core.nn.moe.v2.checkpoint_conversion import (
    convert_legacy_config,
    convert_legacy_model_state,
    expected_olmo_ddp_main_tensors,
    get_legacy_dense_layer_specs,
    validate_main_tensor_schema,
)
from olmo_core.train.checkpoint import CheckpointMetadata


log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        help="Legacy checkpoint root (stepN) or its model_and_optim directory.",
    )
    parser.add_argument("output", help="New model-only checkpoint root.")
    parser.add_argument(
        "--config",
        default=None,
        help="Source config.json; defaults to SOURCE/config.json.",
    )
    parser.add_argument("--load-thread-count", type=int, default=4)
    parser.add_argument("--save-thread-count", type=int, default=4)
    parser.add_argument("--save-overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _resolve_source(source: Path) -> tuple[Path, Path]:
    if (source / "model_and_optim" / ".metadata").is_file():
        return source, source / "model_and_optim"
    if (source / ".metadata").is_file():
        return source.parent, source
    raise FileNotFoundError(f"Could not find distributed checkpoint metadata under {source}")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)
        file.write("\n")


def _load_model_main_tensors(checkpoint_dir: Path, *, thread_count: int) -> Dict[str, torch.Tensor]:
    reader = RemoteFileSystemReader(checkpoint_dir, thread_count=thread_count)
    metadata = reader.read_metadata()
    if not isinstance(metadata, Metadata):
        raise TypeError(f"Unexpected checkpoint metadata type: {type(metadata)}")

    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in metadata.state_dict_metadata.items():
        if not key.endswith(".main"):
            continue
        if not isinstance(value, TensorStorageMetadata):
            raise TypeError(f"Expected tensor metadata for {key}, got {type(value)}")
        state_dict[key] = torch.empty(value.size, dtype=value.properties.dtype, device="cpu")
    if not state_dict:
        raise ValueError(f"No optimizer-main model tensors found in {checkpoint_dir}")

    log.info("Loading %d model tensors", len(state_dict))
    dist_cp.state_dict_loader.load(
        state_dict,
        checkpoint_id=checkpoint_dir,
        storage_reader=reader,
        process_group=None,
    )
    return state_dict


def _validate_saved_checkpoint(
    checkpoint_dir: Path,
    expected: Dict[str, tuple[torch.Size, torch.dtype]],
    *,
    thread_count: int,
) -> None:
    reader = RemoteFileSystemReader(checkpoint_dir, thread_count=thread_count)
    metadata = reader.read_metadata()
    actual = metadata.state_dict_metadata
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        raise ValueError(
            f"Saved checkpoint key mismatch: missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    for key, (shape, dtype) in expected.items():
        value = actual[key]
        if not isinstance(value, TensorStorageMetadata):
            raise TypeError(f"Saved entry is not a tensor: {key}")
        if value.size.numel() != shape.numel() or value.properties.dtype != dtype:
            raise ValueError(
                f"Saved metadata mismatch for {key}: expected {tuple(shape)} {dtype}, "
                f"got {tuple(value.size)} {value.properties.dtype}"
            )


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    source_arg = Path(args.source).expanduser().resolve()
    source_root, source_model_dir = _resolve_source(source_arg)
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config is not None
        else source_root / "config.json"
    )
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No source config found at {config_path}; pass --config explicitly"
        )

    output_root = Path(args.output).expanduser().resolve()
    if output_root.exists() and not args.save_overwrite:
        raise FileExistsError(
            f"Output already exists: {output_root}; pass --save-overwrite to replace it"
        )

    source_config = _load_json(config_path)
    dense_layers = get_legacy_dense_layer_specs(source_config)
    if dense_layers:
        log.info("Legacy dense layers: %s", [spec.layer_idx for spec in dense_layers])
    else:
        log.info("No legacy dense layers; preserving the all-MoE model tensor layout")

    converted_config = convert_legacy_config(source_config)
    expected = expected_olmo_ddp_main_tensors(converted_config)
    source_state = _load_model_main_tensors(source_model_dir, thread_count=args.load_thread_count)
    converted_state = convert_legacy_model_state(source_state, dense_layers)
    del source_state
    validate_main_tensor_schema(converted_state, expected)

    temp_root = output_root.with_name(f".{output_root.name}.incomplete-{uuid.uuid4().hex}")
    temp_root.mkdir(parents=True)
    try:
        model_dir = temp_root / "model_and_optim"
        log.info(
            "Saving %d model tensors (%s elements) to %s",
            len(converted_state),
            f"{sum(tensor.numel() for tensor in converted_state.values()):,}",
            model_dir,
        )
        save_state_dict(
            model_dir,
            converted_state,
            thread_count=args.save_thread_count,
        )
        _validate_saved_checkpoint(model_dir, expected, thread_count=args.load_thread_count)

        _write_json(temp_root / "source_config.json", source_config)
        _write_json(temp_root / "config.json", converted_config)
        _write_json(
            temp_root / "conversion_manifest.json",
            {
                "format": "legacy_olmoe_v2_to_olmo_ddp_model_only_v1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_checkpoint": str(source_root),
                "source_model_checkpoint": str(source_model_dir),
                "source_config": str(config_path),
                "dense_layers": [spec.layer_idx for spec in dense_layers],
                "model_tensor_count": len(converted_state),
                "model_numel": sum(tensor.numel() for tensor in converted_state.values()),
                "optimizer_state_included": False,
                "trainer_state_included": False,
            },
        )
        _write_json(
            temp_root / ".metadata.json",
            CheckpointMetadata(ephemeral=False).as_config_dict(),
        )
        data_paths = source_root / "data_paths.txt"
        if data_paths.is_file():
            shutil.copy2(data_paths, temp_root / "data_paths.txt")

        output_root.parent.mkdir(parents=True, exist_ok=True)
        if output_root.exists():
            _remove_path(output_root)
        temp_root.rename(output_root)
    except BaseException:
        _remove_path(temp_root)
        raise

    log.info("Conversion complete: %s", output_root)


if __name__ == "__main__":
    main()
