"""
Count the total number of parameters in an OLMo model.

This script loads a model from a checkpoint and prints its total parameters,
excluding any teacher model stored alongside it.

Usage:
    python count_parameters.py --checkpoint-dir /path/to/olmo/checkpoint
"""

import argparse
import json
import logging
from typing import Tuple

import torch
from cached_path import cached_path

from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.distributed.parallel import build_world_mesh
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.utils import (
    get_rank,
    is_distributed,
    scatter_object,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import join_path, normalize_path
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import seed_all


log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count total parameters in an OLMo model"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to the OLMo checkpoint directory",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cpu"),
        help="Device to use for loading (cpu/cuda)",
    )
    return parser.parse_args()


def _load_config(checkpoint_dir) -> Tuple[TransformerConfig, TokenizerConfig, dict]:
    """Load transformer and tokenizer config from checkpoint."""
    config_path = join_path(checkpoint_dir, "config.json")
    with cached_path(config_path).open() as f:
        config_dict = json.load(f)
    try:
        transformer_config = TransformerConfig.from_dict(config_dict["model"])
        tokenizer_config = TokenizerConfig.from_dict(config_dict["dataset"]["tokenizer"])
    except KeyError as e:
        raise OLMoConfigurationError(
            f"Failed to load config from checkpoint at {config_path}: missing required field {e}"
        ) from e

    return transformer_config, tokenizer_config, config_dict


def _load_state_dict(checkpoint_dir, model):
    """Load model state dict from checkpoint."""
    train_module_dir = join_path(checkpoint_dir, "model_and_optim")
    incompatible_keys = load_model_and_optim_state(
        train_module_dir, model, strict=False
    )
    if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
        log.warning(f"Incompatible keys when loading checkpoint: {incompatible_keys}")
    return incompatible_keys


def _load_model(checkpoint_dir, device):
    """Load model from checkpoint."""
    if get_rank() == 0:
        transformer_config, tokenizer_config, config_dict = _load_config(checkpoint_dir)
    else:
        transformer_config = None
        tokenizer_config = None
        config_dict = None

    # Broadcast config to all ranks
    transformer_config, tokenizer_config, config_dict = scatter_object(
        (transformer_config, tokenizer_config, config_dict)
    )

    if is_distributed():
        world_mesh = build_world_mesh()
    else:
        world_mesh = None

    # Remove teacher config to exclude it from parameter count
    transformer_config = transformer_config.replace(teacher_config=None)  # type: ignore

    model = transformer_config.build(init_device="meta")  # type: ignore
    model = parallelize_model(
        model,
        world_mesh=world_mesh,
        device=device,
        rank_microbatch_size=1,
        compile_model=False,
    )

    _load_state_dict(checkpoint_dir, model)

    return model, transformer_config


def count_parameters(model):
    """Count total parameters in model."""
    total = sum(p.numel() for p in model.parameters())
    return total


def main():
    seed_all(0)
    args = parse_args()

    checkpoint_dir = normalize_path(args.checkpoint_dir)

    log.info(f"Loading model from {checkpoint_dir}")
    model, transformer_config = _load_model(checkpoint_dir, args.device)

    total_params = count_parameters(model)

    log.info(f"Total parameters: {total_params:,}")
    print(checkpoint_dir, total_params)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
