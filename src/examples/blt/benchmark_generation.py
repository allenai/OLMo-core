#!/usr/bin/env python3
"""
Generation script for loading OLMo-core models from checkpoints and responding to prompts.
This script is designed to sanity check the implementation with real data.

Usage:
    python generate_from_checkpoint.py CHECKPOINT_PATH [--device cuda:0] [--max-length 100]
    python generate_from_checkpoint.py CHECKPOINT_PATH --interactive
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from dataclasses import dataclass
import os
import math

import torch
from transformers import AutoTokenizer

from olmo_core.config import Config, DType
from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig, ByteTokenizerConfig
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.blt.config import BLTConfig, LocalEncoderConfig, LocalDecoderConfig
from olmo_core.generate.generation_module import TransformerGenerationModule, BLTTransformerGenerationModule
from olmo_core.generate.generation_module.config import GenerationConfig
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerBlockType, TransformerConfig, TransformerType
from olmo_core.nn.xlstm import XLSTMConfig
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.utils import get_default_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

MODEL_STYLE = os.environ.get("MODEL_STYLE", "hnet") # or baseline
DTYPE = os.environ.get("DTYPE", "bfloat16")
OLMO_ARCH = os.environ.get("OLMO_ARCH", "olmo2_1B_v2") # or gpt
GENERATE_LENGTH = int(os.environ.get("GENERATE_LENGTH", 2048))
PREFILL_LENGTH = int(os.environ.get("PREFILL_LENGTH", 1024))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1))
N_BATCHES = int(os.environ.get("N_BATCHES", 1))
FORCE_BOUNDARY_EVERY_K = os.environ.get("FORCE_BOUNDARY_EVERY_K", 4)

def main(run_name: str, overrides: list[str]):
    # Setup generation config
    generation_config = GenerationConfig(
        max_new_tokens=GENERATE_LENGTH,
        temperature=0.0,
        do_sample=False,
        use_cache=True,
        pad_token_id=1_000_000, # outside vocab size, don't treat anything as padding
        eos_token_id=1_000_001,  # outside vocab size, don't treat anything as eos
    )

    log.info(f"Generation config: {generation_config}")

    if MODEL_STYLE == "hnet":
        tokenizer_config = ByteTokenizerConfig.blt()
        tokenizer = tokenizer_config.build()

        model_config = getattr(TransformerConfig, OLMO_ARCH)(
            vocab_size=tokenizer_config.padded_vocab_size(),
            dtype=getattr(DType, DTYPE),
        )
        local_d_model = model_config.d_model
        local_encoder_n_layers = 0
        local_decoder_n_layers = 0

        local_block = TransformerBlockConfig(
            name=TransformerBlockType.xlstm,
            attention=AttentionConfig(), # not used
            xlstm=XLSTMConfig(
                num_heads=16,
                dtype=model_config.dtype,
            ),
            feed_forward=model_config.block.feed_forward.replace(
                hidden_size=int(local_d_model * 1.5),
                bias=False,
            ),
            layer_norm=model_config.block.layer_norm,
        )
        model_config = model_config.replace(
            name=TransformerType.blt_distill,
            local_encoder=LocalEncoderConfig(
                add_hash_embeddings=False,
                add_expanded_embeddings=False,
                d_model=local_d_model,
                n_layers=local_encoder_n_layers,
                sliding_window_size=0,
                cross_attn_n_heads=0,
                block_config=local_block,
                add_norm_after_last_block=True,
                boundary_predictor="hnet",
                add_out_projection=True,
                pooling="hnet",
                dtype=model_config.dtype,
            ),
            local_decoder=LocalDecoderConfig(
                d_model=local_d_model,
                n_layers=local_decoder_n_layers,
                sliding_window_size=0,
                cross_attn_n_heads=0,
                block_config=local_block,
                add_norm_before_first_block=True,
                add_norm_onto_residual=False,
                add_in_projection=True,
                hnet_smooth=False,
                hnet_modulate=False,
                depooling="hnet",
                dtype=model_config.dtype,
            )
        )
    else:
        tokenizer_config = TokenizerConfig.dolma2()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.identifier)
        model_config = getattr(TransformerConfig, OLMO_ARCH)(
            vocab_size=tokenizer_config.padded_vocab_size(),
            dtype=getattr(DType, DTYPE),
        )

    # make sure flash attn
    model_config = model_config.replace(
        block=model_config.block.replace(
            attention=model_config.block.attention.replace(
                use_flash=True,
            )
        )
    )

    model_config = model_config.merge(overrides)
    model = model_config.build(init_device="meta")
    print(model)
    model = parallelize_model(
        model,
        world_mesh=None,
        device=get_default_device(),
        max_sequence_length=PREFILL_LENGTH + GENERATE_LENGTH,
        rank_microbatch_size=BATCH_SIZE,
        compile_model=False,
    )

    if MODEL_STYLE == "hnet":
        generation_module = BLTTransformerGenerationModule(
            model,
            tokenizer,
            BLTConfig(teacher_force_boundaries=True),
            generation_config,
        )
        generate_kwargs = {
            "force_boundary_every": FORCE_BOUNDARY_EVERY_K, 
        }
    else:
        generation_module = TransformerGenerationModule(
            model,
            generation_config,
        )
        generate_kwargs = {}

    for _ in range(N_BATCHES):
        generation_module.generate_batch(
            input_ids=torch.randint(0, 100_000, (BATCH_SIZE, PREFILL_LENGTH), device=get_default_device()),
            return_logits=False,
            return_logprobs=False,
            **generate_kwargs,  # type: ignore
        )

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]
    main(run_name, overrides=overrides)