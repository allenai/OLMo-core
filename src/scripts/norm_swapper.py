#!/usr/bin/env python3
"""
Swap q_norm and k_norm weights from one checkpoint into another.

Usage:
    python -m scripts.norm_swapper \
        --model /path/to/base_checkpoint \
        --norm_model /path/to/norm_checkpoint \
        --output_dir /path/to/output
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch.distributed as dist
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata

from olmo_core.distributed.checkpoint import get_checkpoint_metadata, save_state_dict
from olmo_core.distributed.checkpoint.filesystem import RemoteFileSystemReader
from olmo_core.io import normalize_path
from olmo_core.utils import prepare_cli_environment
from olmo_core.version import VERSION

log = logging.getLogger(__name__)


def init_single_process_distributed():
    """Initialize a single-process distributed group for checkpoint saving."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


def resolve_checkpoint_dir(checkpoint_dir: str) -> str:
    """
    Resolve the checkpoint directory to the model_and_optim subdirectory.
    Handles both direct model_and_optim paths and parent checkpoint paths.
    """
    checkpoint_dir = normalize_path(checkpoint_dir)
    model_and_optim = f"{checkpoint_dir.rstrip('/')}/model_and_optim"

    # Check if model_and_optim subdirectory exists
    if Path(model_and_optim).exists() or (
        not checkpoint_dir.endswith("model_and_optim")
        and Path(f"{model_and_optim}/.metadata").exists()
    ):
        return model_and_optim

    return checkpoint_dir


def get_checkpoint_parent(checkpoint_dir: str) -> str:
    """Get the parent checkpoint directory from a model_and_optim path."""
    checkpoint_dir = normalize_path(checkpoint_dir)
    if checkpoint_dir.endswith("/model_and_optim"):
        return checkpoint_dir[:-len("/model_and_optim")]
    if checkpoint_dir.endswith("model_and_optim"):
        return checkpoint_dir[:-len("model_and_optim")].rstrip("/")
    return checkpoint_dir


def save_checkpoint_metadata(output_dir: Path):
    """Save .metadata.json with version info."""
    metadata = {"version": VERSION}
    metadata_path = output_dir / ".metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    log.info(f"Saved metadata to {metadata_path}")


def copy_config(source_checkpoint: str, output_dir: Path, config_path: Optional[str] = None):
    """Copy config.json to output directory."""
    if config_path:
        # Use provided config
        src = Path(config_path)
        if src.exists():
            dst = output_dir / "config.json"
            shutil.copy(src, dst)
            log.info(f"Copied config from {src} to {dst}")
            return
        else:
            log.warning(f"Provided config path does not exist: {config_path}")

    # Try to copy from source checkpoint
    parent = get_checkpoint_parent(source_checkpoint)
    src = Path(parent) / "config.json"
    if src.exists():
        dst = output_dir / "config.json"
        shutil.copy(src, dst)
        log.info(f"Copied config from {src} to {dst}")
    else:
        log.warning(f"No config.json found at {src}, skipping")


def load_model_weights(checkpoint_dir: str, keys: List[str] | None = None) -> Dict[str, Any]:
    """
    Load model weights from a distributed checkpoint.

    If keys is provided, only load those specific keys.
    Otherwise, load all model keys.
    """
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    checkpoint_dir = normalize_path(checkpoint_dir)
    metadata = get_checkpoint_metadata(checkpoint_dir)

    if keys is None:
        # Get all model keys from metadata
        keys = [
            key for key in metadata.state_dict_metadata.keys() if key.startswith("model.")
        ]

    if not keys:
        raise ValueError(f"No keys to load from checkpoint: {checkpoint_dir}")

    log.info(f"Loading {len(keys)} tensors from {checkpoint_dir}")

    state_dict: Dict[str, Any] = {}
    _load_state_dict(
        state_dict,
        storage_reader=RemoteFileSystemReader(checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(keys=keys),
        no_dist=True,
    )

    return state_dict


def find_norm_keys(metadata: Metadata) -> List[str]:
    """Find all q_norm and k_norm model tensor keys in the checkpoint."""
    norm_keys = []
    for key, value in metadata.state_dict_metadata.items():
        # Only include model tensors, not optimizer state
        if not key.startswith("model."):
            continue
        # Only include actual tensors, not nested dict structures
        if isinstance(value, TensorStorageMetadata):
            if ".q_norm." in key or ".k_norm." in key:
                norm_keys.append(key)
    return sorted(norm_keys)


def get_nested_value(d: Dict[str, Any], key: str) -> Any:
    """Get a value from a dict using a dotted key path. Handles flat or nested structures."""
    # First check if flat key exists
    if key in d:
        return d[key]

    # Otherwise navigate nested structure
    if "." not in key:
        raise KeyError(key)

    root, rest = key.split(".", 1)
    if root not in d:
        raise KeyError(root)

    return get_nested_value(d[root], rest)


def set_nested_value(d: Dict[str, Any], key: str, value: Any):
    """Set a value in a dict using a dotted key path. Handles flat or nested structures."""
    # First check if flat key exists
    if key in d:
        d[key] = value
        return

    # Otherwise navigate nested structure
    if "." not in key:
        d[key] = value
        return

    root, rest = key.split(".", 1)
    if root not in d:
        raise KeyError(root)

    set_nested_value(d[root], rest, value)


def swap_norms(
    base_state_dict: Dict[str, Any],
    norm_state_dict: Dict[str, Any],
    norm_keys: List[str],
) -> Dict[str, Any]:
    """Replace norm weights in base_state_dict with values from norm_state_dict."""
    for key in norm_keys:
        try:
            norm_value = get_nested_value(norm_state_dict, key)
        except KeyError:
            raise ValueError(f"Key '{key}' not found in norm_model checkpoint")

        set_nested_value(base_state_dict, key, norm_value)
        log.info(f"Swapped: {key}")

    return base_state_dict


def main():
    parser = argparse.ArgumentParser(
        description="Swap q_norm and k_norm weights from one checkpoint into another"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to base model checkpoint (model_and_optim directory)",
    )
    parser.add_argument(
        "--norm_model",
        type=str,
        required=True,
        help="Path to checkpoint to pull q_norm/k_norm from (model_and_optim directory)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for modified checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.json to include in output. If not provided, copies from base model.",
    )
    args = parser.parse_args()

    # Initialize distributed for saving
    init_single_process_distributed()

    try:
        # Get metadata to find norm keys
        log.info("Reading checkpoint metadata...")
        base_metadata = get_checkpoint_metadata(normalize_path(args.model))
        norm_metadata = get_checkpoint_metadata(normalize_path(args.norm_model))

        # Find norm keys in both checkpoints (only actual tensors)
        base_norm_keys = find_norm_keys(base_metadata)
        norm_model_keys = find_norm_keys(norm_metadata)

        if not base_norm_keys:
            raise ValueError("No q_norm or k_norm keys found in base model")
        if not norm_model_keys:
            raise ValueError("No q_norm or k_norm keys found in norm_model")

        # Verify norm keys match
        if set(base_norm_keys) != set(norm_model_keys):
            raise ValueError(
                f"Norm keys don't match between checkpoints.\n"
                f"Base model: {base_norm_keys}\n"
                f"Norm model: {norm_model_keys}"
            )

        log.info(f"Found {len(base_norm_keys)} norm keys to swap")

        # Load base model (all weights)
        log.info("Loading base model...")
        base_state_dict = load_model_weights(args.model)

        # Load only norm weights from norm_model
        log.info("Loading norm weights from norm_model...")
        norm_state_dict = load_model_weights(args.norm_model, keys=norm_model_keys)

        # Swap norm weights
        log.info("Swapping norm weights...")
        result_state_dict = swap_norms(base_state_dict, norm_state_dict, norm_model_keys)

        # Free memory
        del base_state_dict
        del norm_state_dict

        # Prepare output directory structure
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_and_optim_dir = output_dir / "model_and_optim"

        # Save result to model_and_optim subdirectory
        log.info(f"Saving modified checkpoint to {model_and_optim_dir}")
        save_state_dict(
            str(model_and_optim_dir),
            result_state_dict,
            save_overwrite=True,
        )

        # Save checkpoint metadata
        save_checkpoint_metadata(output_dir)

        # Copy config.json
        copy_config(args.model, output_dir, args.config)

        log.info(f"Done! Checkpoint saved to {output_dir}")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    prepare_cli_environment()
    main()
