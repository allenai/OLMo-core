#!/usr/bin/env python3
"""
Swap q_norm and k_norm weights from one checkpoint into another.

Usage:
    python -m scripts.norm_swapper \
        --model /path/to/base_checkpoint/model_and_optim \
        --norm_model /path/to/norm_checkpoint/model_and_optim \
        --output_dir /path/to/output
"""

import argparse
import logging
import os
from typing import Any, Dict, List, Set

import torch.distributed as dist

from olmo_core.distributed.checkpoint import get_checkpoint_metadata, save_state_dict
from olmo_core.distributed.checkpoint.filesystem import RemoteFileSystemReader
from olmo_core.io import normalize_path
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def init_single_process_distributed():
    """Initialize a single-process distributed group for checkpoint saving."""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    dist.init_process_group(backend="gloo", rank=0, world_size=1)


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


def find_norm_keys(metadata_keys: Set[str]) -> List[str]:
    """Find all q_norm and k_norm keys in the checkpoint."""
    norm_keys = [
        key for key in metadata_keys
        if "q_norm" in key or "k_norm" in key
    ]
    return sorted(norm_keys)


def swap_norms(
    base_state_dict: Dict[str, Any],
    norm_state_dict: Dict[str, Any],
    norm_keys: List[str],
) -> Dict[str, Any]:
    """Replace norm weights in base_state_dict with values from norm_state_dict."""
    result = base_state_dict.copy()

    for key in norm_keys:
        if key not in norm_state_dict:
            raise ValueError(f"Key '{key}' not found in norm_model checkpoint")

        # Navigate nested dict structure
        parts = key.split(".")

        # Set value in result
        current = result
        for part in parts[:-1]:
            current = current[part]
        current[parts[-1]] = norm_state_dict[key]

        log.info(f"Swapped: {key}")

    return result


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
    args = parser.parse_args()

    # Initialize distributed for saving
    init_single_process_distributed()

    try:
        # Get metadata to find norm keys
        log.info("Reading checkpoint metadata...")
        base_metadata = get_checkpoint_metadata(normalize_path(args.model))
        norm_metadata = get_checkpoint_metadata(normalize_path(args.norm_model))

        # Find norm keys in both checkpoints
        base_norm_keys = find_norm_keys(set(base_metadata.state_dict_metadata.keys()))
        norm_model_keys = find_norm_keys(set(norm_metadata.state_dict_metadata.keys()))

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

        # Save result
        log.info(f"Saving modified checkpoint to {args.output_dir}")
        save_state_dict(
            args.output_dir,
            result_state_dict,
            save_overwrite=True,
        )

        log.info("Done!")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    prepare_cli_environment()
    main()
