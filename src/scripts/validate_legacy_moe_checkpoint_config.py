#!/usr/bin/env python3
"""Require a legacy config's model schema to match a source DCP checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

from olmo_core.nn.transformer.config import MoEFusedV2TransformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Legacy step root or model_and_optim directory")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _resolve_model_dir(path: Path) -> Path:
    if (path / "model_and_optim" / ".metadata").is_file():
        return path / "model_and_optim"
    if (path / ".metadata").is_file():
        return path
    raise FileNotFoundError(f"No distributed checkpoint metadata under {path}")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    checkpoint_arg = Path(args.checkpoint).expanduser().resolve()
    checkpoint_dir = _resolve_model_dir(checkpoint_arg)
    config_path = Path(args.config).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    config = _load_json(config_path)
    model_config = MoEFusedV2TransformerConfig.from_dict(config["model"])
    model_config.validate()
    model = model_config.build(init_device="meta")
    expected = {
        f"module.{name}.main": (parameter.shape, parameter.numel(), torch.float32)
        for name, parameter in model.named_parameters()
    }

    metadata = FileSystemReader(checkpoint_dir).read_metadata()
    if not isinstance(metadata, Metadata):
        raise TypeError(f"Unexpected checkpoint metadata type: {type(metadata)}")
    actual = {
        key: value for key, value in metadata.state_dict_metadata.items() if key.endswith(".main")
    }
    if set(actual) != set(expected):
        raise ValueError(
            "Legacy config/checkpoint key mismatch: "
            f"missing={sorted(set(expected) - set(actual))[:20]}, "
            f"unexpected={sorted(set(actual) - set(expected))[:20]}"
        )

    errors: list[str] = []
    for key, (parameter_shape, expected_numel, expected_dtype) in expected.items():
        tensor_metadata = actual[key]
        if not isinstance(tensor_metadata, TensorStorageMetadata):
            errors.append(f"{key}: checkpoint entry is not a tensor")
            continue
        if tensor_metadata.size.numel() != expected_numel:
            errors.append(
                f"{key}: config parameter {tuple(parameter_shape)} has "
                f"{expected_numel:,} elements; checkpoint shape "
                f"{tuple(tensor_metadata.size)} has {tensor_metadata.size.numel():,}"
            )
        if tensor_metadata.properties.dtype != expected_dtype:
            errors.append(
                f"{key}: expected checkpoint dtype {expected_dtype}, "
                f"got {tensor_metadata.properties.dtype}"
            )
    if errors:
        raise ValueError("Legacy config/checkpoint schema mismatch:\n" + "\n".join(errors[:20]))

    report = {
        "protocol": "legacy_config_checkpoint_schema_v1",
        "status": "LEGACY_CONFIG_SCHEMA_MATCH",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_arg),
        "checkpoint_model_dir": str(checkpoint_dir),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "model_tensor_count": len(expected),
        "model_numel": sum(item[1] for item in expected.values()),
        "checkpoint_main_tensor_count": len(actual),
        "checkpoint_main_numel": sum(
            value.size.numel()
            for value in actual.values()
            if isinstance(value, TensorStorageMetadata)
        ),
        "keys_exact": True,
        "numel_exact": True,
        "dtypes_exact": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)


if __name__ == "__main__":
    main()
