import gc
import json
import logging
import tempfile
from tempfile import TemporaryDirectory
from typing import List, Optional, Dict, Tuple, Any

import click
import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from cached_path import cached_path

from olmo_core.aliases import PathOrStr
from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_model_and_optim_state
from olmo_core.io import file_exists, join_path, copy_file
from olmo_core.nn.transformer import TransformerConfig, TransformerBlockConfig
from olmo_core.optim import OptimConfig
from olmo_core.utils import prepare_cli_environment


log = logging.getLogger(__name__)


def load_config(checkpoint_input_dir: PathOrStr) -> Dict:
    if not file_exists(f"{checkpoint_input_dir}/config.json"):
        raise RuntimeError(f"Config file not found at {checkpoint_input_dir}")

    with cached_path(f"{checkpoint_input_dir}/config.json").open("r", encoding="utf-8") as f:
        config_dict = json.load(f)

    if "model" not in config_dict:
        raise RuntimeError(
            f"Config file at {checkpoint_input_dir} is not an OLMo core experiment config, ignoring"
        )

    return config_dict


def config_dicts_from_path(model_path: str) -> Tuple[Dict, Dict, Dict]:
    # Load and preprocess configs
    experiment_config = load_config(model_path)

    transformer_config_dict = experiment_config["model"]
    # Remove deprecated transformer config options
    if "compile" in transformer_config_dict:
        del transformer_config_dict["compile"]
    if "dp_config" in transformer_config_dict:
        del transformer_config_dict["dp_config"]
    if "tp_config" in transformer_config_dict:
        del transformer_config_dict["tp_config"]
    if "float8_config" in transformer_config_dict:
        del transformer_config_dict["float8_config"]

    tokenizer_config_dict = experiment_config.get("dataset", {}).get("tokenizer")
    optim_config_dict = experiment_config["train_module"]["optim"]

    return transformer_config_dict, tokenizer_config_dict, optim_config_dict


def model_and_optim_config_from_path(model_path: str) -> Tuple[TransformerConfig, OptimConfig]:
    transformer_config_dict, _, optim_config_dict = config_dicts_from_path(model_path)
    model_config = TransformerConfig.from_dict(transformer_config_dict)
    optim_config = OptimConfig.from_dict(optim_config_dict)
    return model_config, optim_config


def state_dict_from_path(model_path: str) -> Dict[str, Any]:
    model_config, optim_config = model_and_optim_config_from_path(model_path)
    model = model_config.build(init_device="meta")
    model.to_empty(device=torch.device("cpu"))

    with TemporaryDirectory(prefix="merge_core_checkpoints-") as work_dir:
        model_and_optim_dir = join_path(model_path, "model_and_optim")
        log.info(f"Loading checkpoint from '{model_and_optim_dir}'")
        load_model_and_optim_state(
            model_and_optim_dir,
            model,
            work_dir=work_dir,
        )
        state_dict_options = dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        return dist_cp_sd.get_model_state_dict(model, options=state_dict_options)


def merge_checkpoints(
    model_paths: List[str],
    output_path: str,
) -> None:
    # Load the first model, in float32 for accuracy
    accumulator_sd = state_dict_from_path(model_paths[0])
    dtypes = {}
    for key in accumulator_sd.keys():
        t = accumulator_sd[key]
        if isinstance(t, torch.Tensor):
            dtypes[key] = t.dtype
            accumulator_sd[key] = t.to(torch.float32)
    gc.collect()

    # Load the rest of the models and accumulate
    for model_path in model_paths[1:]:
        sd = state_dict_from_path(model_path)
        if sd.keys() != accumulator_sd.keys():
            raise RuntimeError(
                f"Checkpoint at {model_path} has different keys than the first checkpoint. "
                f"Cannot merge checkpoints with different architectures."
            )
        for key in sd.keys():
            t = sd[key]
            if isinstance(t, torch.Tensor):
                accumulator_sd[key] += t
        del sd
        gc.collect()

    # Average and cast back down
    for key in accumulator_sd.keys():
        t = accumulator_sd[key]
        if isinstance(t, torch.Tensor):
            t /= len(model_paths)
            accumulator_sd[key] = t.to(dtypes[key])
    gc.collect()

    # Save the model
    transformer_config_dict, tokenizer_config_dict, optim_config_dict = config_dicts_from_path(model_paths[0])
    model_config, optim_config = model_and_optim_config_from_path(model_paths[0])
    model = model_config.build(init_device="meta")
    model.to_empty(device=torch.device("cpu"))
    optim = optim_config.build(model, strict=True)
    first_model_path = model_paths[0]
    if not first_model_path.endswith("/model_and_optim"):
        first_model_path = first_model_path.rstrip("/") + "/model_and_optim"
    load_model_and_optim_state(first_model_path, model, optim, flatten_optimizer_state=True)
    model.load_state_dict(accumulator_sd)

    model_and_optim_dir = join_path(output_path, "model_and_optim")
    log.info(f"Saving OLMo core checkpoint to '{model_and_optim_dir}'")
    save_model_and_optim_state(model_and_optim_dir, model, optim, save_overwrite=True)
    log.info(f"Saved merged model to '{output_path}'")

    copy_file(
        join_path(model_paths[0], ".metadata.json"),
        join_path(output_path, ".metadata.json"),
        save_overwrite=True)
    copy_file(
        join_path(model_paths[0], "config.json"),
        join_path(output_path, "config.json"),
        save_overwrite=True)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--model",
    "-m",
    "model_paths",
    multiple=True,
    required=True,
    help="Model checkpoint path. Should be specified multiple times for different checkpoints"
)
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    help="Output directory for the merged checkpoint",
)
def main(model_paths: tuple, output_path: str):
    """
    Merge OLMo-core model checkpoints by averaging their weights.

    Weights are accumulated in float32 for numerical stability, then converted
    back to the original dtype of the checkpoints.

    Optimizer state is taken from the first model in the list.

    Examples:

    \b
    # Merge different local model checkpoints
    python merge_core_checkpoints.py \\
        --model gs://ai2-llm/checkpoints/stego32-highlr-filter3/step656000 \\
        --model gs://ai2-llm/checkpoints/stego32-highlr-filter3/step655000 \\
        --output ./merged_checkpoint
    """
    merge_checkpoints(model_paths=list(model_paths), output_path=output_path)


if __name__ == "__main__":
    prepare_cli_environment()
    main()
