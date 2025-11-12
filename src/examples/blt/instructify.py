import os
import argparse
import json
import logging
import tempfile
from typing import List, Optional
from pathlib import Path

from cached_path import cached_path
import torch
from torch.distributed.checkpoint.metadata import Metadata

from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_model_and_optim_state
from olmo_core.distributed.parallel import build_world_mesh
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_state_dict
from olmo_core.distributed.utils import (
    get_fs_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    scatter_object,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import copy_file, join_path, normalize_path
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import seed_all

log = logging.getLogger(__name__)

BYTE_EXPANSION_FACTOR = int(os.environ.get("BYTE_EXPANSION_FACTOR", "6"))

def parse_args():
    parser = argparse.ArgumentParser(description="Instructify OLMo model merging script")
    parser.add_argument("--checkpoint-dir", type=str, default="/weka/oe-training-default/benjaminm/runs/stage2n_hnet_v8-e1d4-w-ee-no-sm-150k-half-local-lr_wd01gc05/step150000",
                        help="Path to the main checkpoint directory")
    parser.add_argument("--output", type=str, default="/weka/oe-training-default/benjaminm/merges/stage2n_hnet_v8-e1d4-w-ee-no-sm-150k-half-local-lr_wd01gc05_zeroshot_instruct_alpha1",
                        help="Path to output directory")
    parser.add_argument("--base-checkpoint-dir", type=str, default="/weka/oe-training-default/benjaminm/checkpoints/olmo2_1b",
                        help="Path to base checkpoint directory")
    parser.add_argument("--instruct-checkpoint-dir", type=str, default="/weka/oe-training-default/benjaminm/checkpoints/olmo2_1b_instruct",
                        help="Path to instruct checkpoint directory")
    parser.add_argument("--include-instruct-teacher", action="store_true",
                        default=True,
                        help="Include instruct teacher in the merged model")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Alpha value for diff merge")
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"),
                        help="Device to use for instructifying.")
    parser.add_argument("--max-sequence-length", type=int, default=4096,
                        help="Max sequence length")
    return parser.parse_args()

def _load_state_dict(checkpoint_dir, model):
    train_module_dir = join_path(checkpoint_dir, "model_and_optim")
    
    if hasattr(model, "teacher") and model.teacher is not None:
        key_mapping = {
            f"teacher.{key}": None for key in model.teacher.state_dict().keys()  # type: ignore
        }
    else:
        key_mapping = {}

    incompatible_keys = load_model_and_optim_state(train_module_dir, model, key_mapping=key_mapping, strict=False)
    print(incompatible_keys)

def _load_config(checkpoint_dir):
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

    return transformer_config, tokenizer_config

def _load_our_model(checkpoint_dir, device, max_sequence_length: int, teacher_config=None):
    if get_rank() == 0:
        transformer_config, tokenizer_config = _load_config(checkpoint_dir)
    else:
        transformer_config = None
        tokenizer_config = None

    transformer_config, tokenizer_config = scatter_object((transformer_config, tokenizer_config))

    if is_distributed():
        world_mesh = build_world_mesh()
    else:
        world_mesh = None

    transformer_config = transformer_config.replace(teacher_config=teacher_config)  # type: ignore

    model = transformer_config.build(init_device="meta")  # type: ignore
    model = parallelize_model(
        model,
        world_mesh=world_mesh,
        device=device,
        max_sequence_length=max_sequence_length * BYTE_EXPANSION_FACTOR,
        rank_microbatch_size=1,
        compile_model=True,
    )

    _load_state_dict(checkpoint_dir, model)

    return transformer_config, tokenizer_config, model

def _load_model(checkpoint_dir, device, max_sequence_length):
    if get_rank() == 0:
        transformer_config, tokenizer_config = _load_config(checkpoint_dir)
    else:
        transformer_config = None
        tokenizer_config = None

    transformer_config, tokenizer_config = scatter_object((transformer_config, tokenizer_config))

    if is_distributed():
        world_mesh = build_world_mesh()
    else:
        world_mesh = None

    model = transformer_config.build(init_device="meta")  # type: ignore
    model = parallelize_model(
        model,
        world_mesh=world_mesh,
        device=device,
        max_sequence_length=max_sequence_length,
        rank_microbatch_size=1,
        compile_model=True,
    )

    _load_state_dict(checkpoint_dir, model)

    return transformer_config, tokenizer_config, model

def main():
    seed_all(0)
    args = parse_args()

    checkpoint_dir = normalize_path(args.checkpoint_dir)
    base_checkpoint_dir = normalize_path(args.base_checkpoint_dir)
    instruct_checkpoint_dir = normalize_path(args.instruct_checkpoint_dir)
    output = normalize_path(args.output)

    _, _, base_model = _load_model(base_checkpoint_dir, args.device, args.max_sequence_length)
    instruct_config, instruct_tokenizer_config, instruct_model = _load_model(instruct_checkpoint_dir, args.device, args.max_sequence_length)
    transformer_config, tokenizer_config, model = _load_our_model(
        checkpoint_dir,
        device=args.device,
        max_sequence_length=args.max_sequence_length,
        teacher_config=instruct_config if args.include_instruct_teacher else None,
    )

    for key, value in model.state_dict().items():
        if key.startswith("teacher."):
            value[:] = instruct_model.state_dict()[key[len("teacher."):]]
        elif key in instruct_model.state_dict():
            diff = instruct_model.state_dict()[key] - base_model.state_dict()[key]
            if diff.shape == value.shape:
                value[:] += args.alpha * diff
            else:
                log.warning(f"Key {key} has different shape across models: {diff.shape} vs {value.shape} (this is okay for embeddings).")
        else:
            log.warning(f"Key {key} not found in instruct model state dict, skipping (this is okay for local enc/dec and boundary predictor).")

    model_and_optim_dir = join_path(output, "model_and_optim")
    log.info(f"Saving OLMo core checkpoint to '{model_and_optim_dir}'")
    save_model_and_optim_state(model_and_optim_dir, model, save_overwrite=True)
    log.info(f"Successfully saved converted model to '{output}'")

    config_path = join_path(output, "config.json")
    log.info(f"Writing partial experiment config to '{config_path}'")

    experiment_config_dict = {
        "model": transformer_config.as_config_dict(),
        "dataset": {
            "tokenizer": tokenizer_config.as_config_dict(),  # type: ignore
        }
    }

    with tempfile.NamedTemporaryFile(mode="w") as temp_file:
        json.dump(experiment_config_dict, temp_file)
        temp_file.flush()
        copy_file(temp_file.name, config_path, save_overwrite=True)
        log.info(f"Successfully wrote partial experiment config to '{config_path}'")

if __name__ == "__main__":
    main()