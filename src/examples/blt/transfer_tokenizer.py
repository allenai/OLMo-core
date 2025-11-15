"""
Transfer an OLMo model to a new tokenizer using FOCUS-inspired embedding initialization.

This script implements embedding transfer similar to the FOCUS method
(https://github.com/konstantinjdobler/focus) for OLMo models.

Usage:
    python transfer_tokenizer.py \
        --checkpoint-dir /path/to/olmo/checkpoint \
        --target-tokenizer scb10x/typhoon-7b \
        --output /path/to/output \
        --training-data /path/to/target/language/data.jsonl
"""

import os
import argparse
import json
import logging
import tempfile
from typing import Optional, Dict, List
from collections import defaultdict

import torch
from cached_path import cached_path
from transformers import AutoTokenizer

from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_model_and_optim_state
from olmo_core.distributed.parallel import build_world_mesh
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.utils import (
    get_rank,
    is_distributed,
    scatter_object,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import copy_file, join_path, normalize_path
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import seed_all
from deepfocus import FOCUS


log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transfer OLMo model to a new tokenizer using FOCUS-inspired initialization"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to the source OLMo checkpoint directory",
    )
    parser.add_argument(
        "--target-tokenizer",
        type=str,
        required=True,
        help="HuggingFace identifier for target tokenizer (e.g., 'scb10x/typhoon-7b')",
    )
    parser.add_argument(
        "--target_language_code",
        type=str,
        required=True,
        help="Language code for the target language (e.g., 'th' for Thai)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output directory for the transferred model",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        default=torch.device("cpu"),
        help="Device to use for transfer (cpu/cuda)",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=4096,
        help="Max sequence length",
    )
    return parser.parse_args()


def _load_config(checkpoint_dir):
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


def _load_model(checkpoint_dir, device, **kwargs):
    if get_rank() == 0:
        transformer_config, tokenizer_config, config_dict = _load_config(checkpoint_dir)
    else:
        transformer_config = None
        tokenizer_config = None
        config_dict = None

    # Broadcast config and work_dir to all ranks
    transformer_config, tokenizer_config, config_dict = scatter_object((transformer_config, tokenizer_config, config_dict))

    if is_distributed():
        world_mesh = build_world_mesh()
    else:
        world_mesh = None

    transformer_config = transformer_config.replace(teacher_config=None)  # type: ignore

    model = transformer_config.build(init_device="meta")  # type: ignore
    model = parallelize_model(
        model,
        world_mesh=world_mesh,
        device=device,
        rank_microbatch_size=1,
        compile_model=True,
        **kwargs,
    )

    _load_state_dict(checkpoint_dir, model)

    return transformer_config, tokenizer_config, config_dict, model

def _load_target_model(source_transformer_config, target_tokenizer_config, device, **kwargs):
    """Build target model with new vocabulary size."""
    target_transformer_config = source_transformer_config.replace(
        vocab_size=target_tokenizer_config.padded_vocab_size()
    )

    if is_distributed():
        world_mesh = build_world_mesh()
    else:
        world_mesh = None

    target_model = target_transformer_config.build(init_device="meta")
    target_model = parallelize_model(
        target_model,
        world_mesh=world_mesh,
        device=device,
        rank_microbatch_size=1,
        compile_model=False,
        **kwargs,
    )

    return target_transformer_config, target_model


def main():
    seed_all(0)
    args = parse_args()

    checkpoint_dir = normalize_path(args.checkpoint_dir)
    output = normalize_path(args.output)

    log.info(f"Loading source model from {checkpoint_dir}")

    transformer_config, tokenizer_config, config_dict, source_model = _load_model(checkpoint_dir, args.device, max_sequence_length=args.max_sequence_length)

    # Load source and target tokenizers
    log.info(f"Loading source tokenizer: {tokenizer_config.identifier}")
    source_tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.identifier)

    log.info(f"Loading target tokenizer: {args.target_tokenizer}")
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer)

    # Create target tokenizer config
    target_tokenizer_config = TokenizerConfig.from_hf(args.target_tokenizer)

    log.info(f"Source vocab size: {tokenizer_config.vocab_size}")
    log.info(f"Target vocab size: {target_tokenizer_config.vocab_size}")

    # Get source embeddings
    source_input_embeddings = source_model.embeddings.weight.data
    source_output_embeddings = source_model.lm_head.w_out.weight.data

    target_input_embeddings = FOCUS(
        source_embeddings=source_input_embeddings,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        auxiliary_embedding_mode="fasttext-wordlevel",
        language_identifier=args.target_language_code,
        device=args.device,
    )
    target_output_embeddings = FOCUS(
        source_embeddings=source_output_embeddings,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        auxiliary_embedding_mode="fasttext-wordlevel",
        language_identifier=args.target_language_code,
        device=source_output_embeddings.device,
    )

    # Create new model with target vocab size
    log.info("Building target model with new vocabulary...")
    target_transformer_config, target_model = _load_target_model(
        transformer_config,
        target_tokenizer_config,
        args.device,
        max_sequence_length=args.max_sequence_length,
    )

    # Copy all weights from source model
    log.info("Copying model weights...")
    source_state = source_model.state_dict()

    for key, value in target_model.named_parameters():
        if "embeddings" in key or "lm_head" in key:
            # Skip embedding layers, we'll handle these separately
            log.info(f"Skipping parameter {key} for separate transfer")
            continue
        if key in source_state:
            if value.shape == source_state[key].shape:
                value.data.copy_(source_state[key])
            else:
                log.warning(f"Shape mismatch for {key}: {value.shape} vs {source_state[key].shape}")

    # Set transferred embeddings
    log.info("Setting transferred embeddings...")
    padded_vocab_size = target_tokenizer_config.padded_vocab_size()

    target_model.embeddings.weight.data[:target_input_embeddings.shape[0], :] = target_input_embeddings
    target_model.lm_head.w_out.weight.data[:target_output_embeddings.shape[0], :] = target_output_embeddings

    # Save the transferred model
    model_and_optim_dir = join_path(output, "model_and_optim")
    log.info(f"Saving transferred model to '{model_and_optim_dir}'")
    save_model_and_optim_state(model_and_optim_dir, target_model, save_overwrite=True)
    log.info(f"Successfully saved model to '{output}'")

    # Save updated config
    config_path = join_path(output, "config.json")
    log.info(f"Writing config to '{config_path}'")

    # Create new config dict without model and dataset
    new_config_dict = {k: v for k, v in config_dict.items() if k not in ["model", "dataset"]}

    experiment_config_dict = {
        "model": target_transformer_config.as_config_dict(),
        "dataset": {
            "tokenizer": target_tokenizer_config.as_config_dict(),
        },
        **new_config_dict,
    }

    with tempfile.NamedTemporaryFile(mode="w") as temp_file:
        json.dump(experiment_config_dict, temp_file, indent=2)
        temp_file.flush()
        copy_file(temp_file.name, config_path, save_overwrite=True)
        log.info(f"Successfully wrote config to '{config_path}'")

    log.info("Transfer complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
