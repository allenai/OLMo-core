import json
import logging
import tempfile
from typing import List, Optional
from pathlib import Path

from cached_path import cached_path
from torch.distributed.checkpoint.metadata import Metadata

from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_model_and_optim_state
from olmo_core.distributed.parallel import build_world_mesh
from olmo_core.utils import get_default_device
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

CHECKPOINT_DIR = "/weka/oe-training-default/benjaminm/runs/stage2_hnet_v4_global_2dot6e-5_fixed_bsx4_local_5e-4_zero_bos-fula-100k-no-tf/step100000"
OUTPUT = "/weka/oe-training-default/benjaminm/merges/bolmo_v0_instruct"
BASE_CHECKPOINT_DIR = "/weka/oe-training-default/benjaminm/checkpoints/olmo2_1b"
INSTRUCT_CHECKPOINT_DIR = "/weka/oe-training-default/benjaminm/checkpoints/olmo2_1b_instruct"
BYTE_EXPANSION_FACTOR = 6
MAX_SEQUENCE_LENGTH = 1024


def _load_state_dict(checkpoint_dir, model):
    train_module_dir = join_path(checkpoint_dir, "model_and_optim")
    load_model_and_optim_state(train_module_dir, model)


def _load_config(checkpoint_dir):
    config_path = join_path(checkpoint_dir, "config.json")
    with cached_path(config_path).open() as f:
        config_dict = json.load(f)
    try:
        # Avoid loading the entire experiment config b/c we don't care about validation outside
        # of the transformer config and the tokenizer config
        transformer_config = TransformerConfig.from_dict(config_dict["model"])
        tokenizer_config = TokenizerConfig.from_dict(config_dict["dataset"]["tokenizer"])
    except KeyError as e:
        raise OLMoConfigurationError(
            f"Failed to load config from checkpoint at {config_path}: missing required field {e}"
        ) from e


    return transformer_config, tokenizer_config


def _load_model(checkpoint_dir):
    if get_rank() == 0:
        transformer_config, tokenizer_config = _load_config(checkpoint_dir)
    else:
        transformer_config = None
        tokenizer_config = None

    # Broadcast config and work_dir to all ranks
    transformer_config, tokenizer_config = scatter_object((transformer_config, tokenizer_config))

    if is_distributed():
        world_mesh = build_world_mesh()
    else:
        world_mesh = None

    model = transformer_config.build(init_device="meta")  # type: ignore
    model = parallelize_model(
        model,
        world_mesh=world_mesh,
        device=get_default_device(),
        max_sequence_length=MAX_SEQUENCE_LENGTH * BYTE_EXPANSION_FACTOR,
        rank_microbatch_size=1,
        compile_model=True,
    )

    _load_state_dict(checkpoint_dir, model)

    return transformer_config, tokenizer_config, model


def main():
    seed_all(0)

    checkpoint_dir = normalize_path(CHECKPOINT_DIR)
    base_checkpoint_dir = normalize_path(BASE_CHECKPOINT_DIR)
    instruct_checkpoint_dir = normalize_path(INSTRUCT_CHECKPOINT_DIR)

    transformer_config, tokenizer_config, model = _load_model(checkpoint_dir)
    _, _, base_model = _load_model(base_checkpoint_dir)
    _, instruct_tokenizer_config, instruct_model = _load_model(instruct_checkpoint_dir)

    for key, value in model.state_dict().items():
        if key in instruct_model.state_dict():
            diff = instruct_model.state_dict()[key] - base_model.state_dict()[key]
            if diff.shape == value.shape:
                value[:] += diff
            else:
                log.warning(f"Key {key} has different shape across models: {diff.shape} vs {value.shape} (this is okay for embeddings).")
        else:
            log.warning(f"Key {key} not found in instruct model state dict, skipping (this is okay for local enc/dec and boundary predictor).")

    # instruction tuned model uses bos token
    transformer_config = transformer_config.replace(prepend_embedding_to_global=True)  # type: ignore
    model.state_dict()["prepend_embedding.weight"] = instruct_model.state_dict()["embeddings.weight"][[instruct_tokenizer_config.eos_token_id]]  # type: ignore
 
    model_and_optim_dir = join_path(OUTPUT, "model_and_optim")
    log.info(f"Saving OLMo core checkpoint to '{model_and_optim_dir}'")
    save_model_and_optim_state(model_and_optim_dir, model, save_overwrite=True)
    log.info(f"Successfully saved converted model to '{OUTPUT}'")

    config_path = join_path(OUTPUT, "config.json")
    log.info(f"Writing partial experiment config to '{config_path}'")

    experiment_config_dict = {
        "model": transformer_config.as_config_dict(),
        "dataset": {
            "tokenizer": tokenizer_config.as_config_dict(),  # type: ignore
        }
    }

    with tempfile.NamedTemporaryFile(mode="w") as temp_file:
        json.dump(experiment_config_dict, temp_file)
        copy_file(temp_file.name, config_path, save_overwrite=True)
        log.info(f"Successfully wrote partial experiment config to '{config_path}'")

if __name__ == "__main__":
    main()
