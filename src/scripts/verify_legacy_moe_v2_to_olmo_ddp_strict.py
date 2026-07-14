#!/usr/bin/env python3
"""Exhaustively verify a saved legacy-to-OLMoDDP checkpoint conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

from olmo_core.distributed.checkpoint import RemoteFileSystemReader
from olmo_core.nn.moe.v2.checkpoint_conversion import get_legacy_dense_layer_specs
from olmo_core.nn.moe.v2.checkpoint_verification import verify_converted_state_exact


log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Legacy step root or model_and_optim directory")
    parser.add_argument("target", help="Converted step root or model_and_optim directory")
    parser.add_argument(
        "--config", default=None, help="Legacy config; defaults to SOURCE/config.json"
    )
    parser.add_argument("--output", default=None, help="Verification JSON output path")
    parser.add_argument("--load-thread-count", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _resolve_checkpoint(path: Path) -> tuple[Path, Path]:
    if (path / "model_and_optim" / ".metadata").is_file():
        return path, path / "model_and_optim"
    if (path / ".metadata").is_file():
        return path.parent, path
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


def _load_main_tensors(
    checkpoint_dir: Path, *, thread_count: int, require_model_only: bool = False
) -> dict[str, torch.Tensor]:
    reader = RemoteFileSystemReader(checkpoint_dir, thread_count=thread_count)
    metadata = reader.read_metadata()
    if not isinstance(metadata, Metadata):
        raise TypeError(f"Unexpected checkpoint metadata type: {type(metadata)}")
    non_main_keys = sorted(key for key in metadata.state_dict_metadata if not key.endswith(".main"))
    if require_model_only and non_main_keys:
        raise ValueError(
            f"Converted checkpoint contains non-model-main state: {non_main_keys[:20]}"
        )
    state: dict[str, torch.Tensor] = {}
    for key, value in metadata.state_dict_metadata.items():
        if not key.endswith(".main"):
            continue
        if not isinstance(value, TensorStorageMetadata):
            raise TypeError(f"Expected tensor metadata for {key}, got {type(value)}")
        state[key] = torch.empty(value.size, dtype=value.properties.dtype, device="cpu")
    if not state:
        raise ValueError(f"No model main tensors in {checkpoint_dir}")
    log.info("Loading %d tensors from %s", len(state), checkpoint_dir)
    dist_cp.state_dict_loader.load(
        state,
        checkpoint_id=checkpoint_dir,
        storage_reader=reader,
        process_group=None,
    )
    return state


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    source_root, source_dir = _resolve_checkpoint(Path(args.source).expanduser().resolve())
    target_root, target_dir = _resolve_checkpoint(Path(args.target).expanduser().resolve())
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config is not None
        else source_root / "config.json"
    )
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else target_root / "strict_tensor_verification.json"
    )
    config = _load_json(config_path)
    dense_layers = get_legacy_dense_layer_specs(config)
    if dense_layers:
        log.info("Legacy dense layers: %s", [spec.layer_idx for spec in dense_layers])
    else:
        log.info("No legacy dense layers; verifying the all-MoE state bitwise unchanged")

    # Reload both persisted DCPs; never reuse the converter's in-memory output.
    source_state = _load_main_tensors(source_dir, thread_count=args.load_thread_count)
    target_state = _load_main_tensors(
        target_dir,
        thread_count=args.load_thread_count,
        require_model_only=True,
    )
    result = verify_converted_state_exact(source_state, target_state, dense_layers)
    report = {
        "protocol": "strict_tensor_v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(source_root),
        "source_model_checkpoint": str(source_dir),
        "target_checkpoint": str(target_root),
        "target_model_checkpoint": str(target_dir),
        "source_config": str(config_path),
        "source_config_sha256": _sha256(config_path),
        "dense_layers": [spec.layer_idx for spec in dense_layers],
        "target_model_only": True,
        "optimizer_state_included": False,
        "trainer_state_included": False,
        **result,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    log.info("Strict tensor verification passed: %s", output_path)


if __name__ == "__main__":
    main()
