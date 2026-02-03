#!/usr/bin/env python3
"""
Merge two OLMo-core distributed checkpoints by computing a weighted average of model weights.

Usage:
    python -m olmo_core.scripts.model_merger \
        --model1 /path/to/checkpoint1/model_and_optim \
        --model2 /path/to/checkpoint2/model_and_optim \
        --output_dir /path/to/output \
        --percent_model1 0.5
"""

import argparse
import logging
import os
from typing import Any, Dict, List

import torch
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


def load_model_weights(checkpoint_dir: str) -> Dict[str, Any]:
    """Load model weights from a distributed checkpoint."""
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    checkpoint_dir = normalize_path(checkpoint_dir)
    metadata = get_checkpoint_metadata(checkpoint_dir)

    # Get all model keys from metadata
    model_keys: List[str] = [
        key for key in metadata.state_dict_metadata.keys() if key.startswith("model.")
    ]

    if not model_keys:
        raise ValueError(f"No model keys found in checkpoint: {checkpoint_dir}")

    log.info(f"Loading {len(model_keys)} model tensors from {checkpoint_dir}")

    state_dict: Dict[str, Any] = {}
    _load_state_dict(
        state_dict,
        storage_reader=RemoteFileSystemReader(checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(keys=model_keys),
        no_dist=True,
    )

    return state_dict


def merge_weights(
    state_dict1: Dict[str, Any],
    state_dict2: Dict[str, Any],
    percent_model1: float,
) -> Dict[str, Any]:
    """Merge two state dicts with weighted averaging."""
    percent_model2 = 1.0 - percent_model1

    # Verify both have the same keys
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())

    if keys1 != keys2:
        missing_in_1 = keys2 - keys1
        missing_in_2 = keys1 - keys2
        raise ValueError(
            f"Checkpoints have different keys.\n"
            f"Missing in model1: {missing_in_1}\n"
            f"Missing in model2: {missing_in_2}"
        )

    merged: Dict[str, Any] = {}

    for key in state_dict1.keys():
        val1 = state_dict1[key]
        val2 = state_dict2[key]

        # Handle nested dicts (like "model" key containing the actual weights)
        if isinstance(val1, dict) and isinstance(val2, dict):
            merged[key] = merge_weights(val1, val2, percent_model1)
        elif isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            # Verify shapes match
            if val1.shape != val2.shape:
                raise ValueError(
                    f"Shape mismatch for key '{key}': {val1.shape} vs {val2.shape}"
                )
            # Compute weighted average
            merged[key] = percent_model1 * val1.float() + percent_model2 * val2.float()
            # Convert back to original dtype
            merged[key] = merged[key].to(val1.dtype)
        else:
            raise ValueError(
                f"Unexpected types for key '{key}': {type(val1)} vs {type(val2)}"
            )

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Merge two OLMo-core distributed checkpoints"
    )
    parser.add_argument(
        "--model1",
        type=str,
        required=True,
        help="Path to first model checkpoint (model_and_optim directory)",
    )
    parser.add_argument(
        "--model2",
        type=str,
        required=True,
        help="Path to second model checkpoint (model_and_optim directory)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for merged checkpoint",
    )
    parser.add_argument(
        "--percent_model1",
        type=float,
        default=0.5,
        help="Weight for model1 (default: 0.5). Model2 weight = 1 - percent_model1",
    )
    args = parser.parse_args()

    # Validate percent_model1
    if not 0.0 <= args.percent_model1 <= 1.0:
        raise ValueError(f"percent_model1 must be between 0 and 1, got {args.percent_model1}")

    log.info(f"Merging checkpoints with weights: model1={args.percent_model1}, model2={1 - args.percent_model1}")

    # Initialize distributed for saving
    init_single_process_distributed()

    try:
        # Load both checkpoints
        log.info("Loading model1...")
        state_dict1 = load_model_weights(args.model1)

        log.info("Loading model2...")
        state_dict2 = load_model_weights(args.model2)

        # Merge weights
        log.info("Merging weights...")
        merged_state_dict = merge_weights(state_dict1, state_dict2, args.percent_model1)

        # Free memory from original state dicts
        del state_dict1
        del state_dict2

        # Save merged checkpoint
        log.info(f"Saving merged checkpoint to {args.output_dir}")
        save_state_dict(
            args.output_dir,
            merged_state_dict,
            save_overwrite=True,
        )

        log.info("Done!")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    prepare_cli_environment()
    main()
