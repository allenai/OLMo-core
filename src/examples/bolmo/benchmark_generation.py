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
import tempfile
import json

import torch
from transformers import AutoTokenizer

from olmo_core.config import Config, DType
from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig, ByteTokenizerConfig
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.blt.config import BolmoConfig, LocalEncoderConfig, LocalDecoderConfig
from olmo_core.generate.generation_module import TransformerGenerationModule, BolmoTransformerGenerationModule
from olmo_core.generate.generation_module.config import GenerationConfig
from olmo_core.nn.mamba import MambaConfig
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerBlockType, TransformerConfig, TransformerType
from olmo_core.nn.xlstm import XLSTMConfig
from olmo_core.nn.fla import FLAConfig
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.utils import get_default_device
from olmo_core.io import join_path, copy_file, file_exists

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

SAVE_FOLDER = os.environ.get("SAVE_FOLDER", "/tmp/benchmark_outputs")
MODEL_STYLE = os.environ.get("MODEL_STYLE", "hnet") # or baseline
LOCAL_MODEL_BLOCKS = os.environ.get("LOCAL_MODEL_BLOCKS", "xlstm") # or mamba2 or xlstm_no_ffn
DTYPE = os.environ.get("DTYPE", "bfloat16")
OLMO_ARCH = os.environ.get("OLMO_ARCH", "olmo2_1B_v2") # or gpt
GENERATE_LENGTH = int(os.environ.get("GENERATE_LENGTH", 256))
PREFILL_LENGTH = int(os.environ.get("PREFILL_LENGTH", 1024))
PROFILE = os.environ.get("PROFILE", "0") == "1"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 1))
N_BATCHES = int(os.environ.get("N_BATCHES", 1))
AVG_BYTES_PER_TOKEN = float(os.environ.get("AVG_BYTES_PER_TOKEN", 4.3))

def main(run_name: str, overrides: list[str]):
    save_path = join_path(SAVE_FOLDER, f"{run_name}_generation_benchmark.json")

    if file_exists(save_path):
        log.info(f"Benchmark output file '{save_path}' already exists, skipping benchmark.")
        return

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

        if LOCAL_MODEL_BLOCKS == "xlstm":
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
        elif LOCAL_MODEL_BLOCKS == "xlstm_no_ffn":
            local_block = TransformerBlockConfig(
                name=TransformerBlockType.xlstm,
                attention=AttentionConfig(), # not used
                xlstm=XLSTMConfig(
                    num_heads=16,
                    dtype=model_config.dtype,
                ),
                feed_forward=None,
                layer_norm=model_config.block.layer_norm,
            )
        elif LOCAL_MODEL_BLOCKS == "gdn":
            local_block = TransformerBlockConfig(
                name=TransformerBlockType.fla,
                attention=AttentionConfig(), # not used
                fla=FLAConfig(
                    name="GatedDeltaNet",
                    dtype=model_config.dtype,
                    fla_layer_kwargs={"num_heads": 16},
                ),
                feed_forward=model_config.block.feed_forward.replace(
                    hidden_size=int(local_d_model * 1.5),
                    bias=False,
                ),
                layer_norm=model_config.block.layer_norm,
            )
        elif LOCAL_MODEL_BLOCKS == "gdn_no_ffn":
            local_block = TransformerBlockConfig(
                name=TransformerBlockType.fla,
                attention=AttentionConfig(), # not used
                fla=FLAConfig(
                    name="GatedDeltaNet",
                    dtype=model_config.dtype,
                ),
                feed_forward=None,
                layer_norm=model_config.block.layer_norm,
            )
        elif LOCAL_MODEL_BLOCKS == "mamba2":
            local_block = TransformerBlockConfig(
                name=TransformerBlockType.mamba,
                attention=AttentionConfig(), # not used
                mamba=MambaConfig(
                    chunk_size=256,
                    d_conv=4,
                    d_state=128,
                    expand=2,
                    dtype=model_config.dtype,
                ),
                feed_forward=None,
                layer_norm=model_config.block.layer_norm,
            )
        else:
            raise ValueError(f"Unknown LOCAL_MODEL_BLOCKS: {LOCAL_MODEL_BLOCKS}")
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
        generation_config = GenerationConfig(
            max_new_tokens=GENERATE_LENGTH,
            temperature=0.0,
            do_sample=False,
            use_cache=True,
            pad_token_id=1_000_000, # outside vocab size, don't treat anything as padding
            eos_token_id=1_000_001,  # outside vocab size, don't treat anything as eos
        )
    else:
        tokenizer_config = TokenizerConfig.dolma2()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.identifier)
        model_config = getattr(TransformerConfig, OLMO_ARCH)(
            vocab_size=tokenizer_config.padded_vocab_size(),
            dtype=getattr(DType, DTYPE),
        )
        generation_config = GenerationConfig(
            max_new_tokens=GENERATE_LENGTH,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            use_cache=True,
            pad_token_id=1_000_000, # outside vocab size, don't treat anything as padding
            eos_token_id=1_000_001,  # outside vocab size, don't treat anything as eos
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
 
    log.info(f"Generation config: {generation_config}")
    log.info(f"Model: {model}")

    if MODEL_STYLE == "hnet":
        generation_module = BolmoTransformerGenerationModule(
            model,
            tokenizer,
            BolmoConfig(teacher_force_boundaries=True),
            generation_config,
            compile_model=True,
            device=torch.device("cuda"),
        )
        generate_kwargs = {
            "avg_bytes_per_token": AVG_BYTES_PER_TOKEN,
        }
    else:
        generation_module = TransformerGenerationModule(
            model,
            generation_config,
            compile_model=True,
            device=torch.device("cuda"),
        )
        generate_kwargs = {}

    all_timings = []

    print(f"Benchmarking {N_BATCHES} batches of size {BATCH_SIZE} with prefill length {PREFILL_LENGTH} and generate length {GENERATE_LENGTH}")

    for batch_idx in range(N_BATCHES):
        print(f"Running batch {batch_idx+1}/{N_BATCHES}")
        try:
            timings = generation_module.benchmark(  # type: ignore
                batch_size=BATCH_SIZE,
                n_prefill=PREFILL_LENGTH,
                n_generate=GENERATE_LENGTH,
                profile=PROFILE,
                **generate_kwargs,
            )
        except Exception as e:
            print(f"Error during benchmarking: {type(e).__name__}")
            sys.exit(1)
        all_timings.append(timings)

    with tempfile.NamedTemporaryFile(mode="w") as temp_file:
        json.dump(all_timings, temp_file, indent=4)
        temp_file.flush()  # make sure data is written to disk, json.dump doesn't flush.
        copy_file(temp_file.name, save_path, save_overwrite=True)
        print(f"Successfully wrote timings to '{save_path}'")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} run_name [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]
    main(run_name, overrides=overrides)