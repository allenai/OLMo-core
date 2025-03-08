"""
Example script to convert a OLMo Core model checkpoint to a HuggingFace model checkpoint.
"""

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from cached_path import cached_path
from safetensors.torch import load_file
from transformers import AutoTokenizer, GPT2Tokenizer, Olmo2Config

from olmo_core.aliases import PathOrStr
from olmo_core.io import file_exists
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.optim.adamw import AdamWConfig
from olmo_core.train.checkpoint import CheckpointerConfig, CheckpointFormat
from olmo_core.train.train_module.transformer import TransformerTrainModuleConfig
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def load_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """
    Load a state dict from either a PyTorch checkpoint or safetensors file.

    Args:
        checkpoint_path: Path to checkpoint file (.pt, .pth, .bin, or .safetensors)

    Returns:
        Dictionary containing model state dict
    """
    checkpoint_path = Path(checkpoint_path)

    if checkpoint_path.suffix == ".safetensors":
        # Load safetensors format
        state_dict = load_file(checkpoint_path)
    else:
        # Load PyTorch format (.pt, .pth, .bin)
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # Handle both cases:
        # 1. Direct state dict
        # 2. Nested state dict under 'model' or 'state_dict' key
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

    return state_dict


def convert_to_hf_checkpoint(
    olmo_checkpoint_path: str | Path,
    output_path: str | Path,
    olmo_core_config: dict,
    tokenizer: GPT2Tokenizer,
    max_sequence_length: int = -1,
) -> None:
    """
    Convert an OLMo core checkpoint to Hugging Face format.

    Args:
        olmo_checkpoint_path: Path to the OLMo core checkpoint
        output_path: Where to save the converted HF model
        olmo_core_config: OLMo core configuration dictionary
        tokenizer: HuggingFace tokenizer instance
    """
    if max_sequence_length <= 0:
        dataset_config = olmo_core_config["dataset"]
        if "max_sequence_length" in dataset_config:
            max_sequence_length = int(dataset_config["max_sequence_length"])
        elif "sequence_length" in dataset_config:
            max_sequence_length = int(dataset_config["sequence_length"])
        else:
            max_sequence_length = tokenizer.model_max_length

    if max_sequence_length <= 0:
        raise ValueError(f"Missing or invalid sequence length: {max_sequence_length}")

    with TemporaryDirectory() as work_dir:
        checkpointer_config = CheckpointerConfig(
            work_dir=work_dir, save_overwrite=True, checkpoint_save_format=CheckpointFormat.hf
        )
        checkpointer = checkpointer_config.build()

        model_config = olmo_core_config["model"]

        # Remove deprecated options
        if "compile" in model_config:
            del model_config["compile"]
        if "dp_config" in model_config:
            del model_config["dp_config"]
        if "tp_config" in model_config:
            del model_config["tp_config"]
        if "float8_config" in model_config:
            del model_config["float8_config"]

        model = TransformerConfig.from_dict(olmo_core_config["model"]).build()
        train_module = TransformerTrainModuleConfig(
            rank_microbatch_size=max_sequence_length,
            max_sequence_length=max_sequence_length,
            optim=AdamWConfig(),
        ).build(model)

        log.info(f"Loading OLMo core checkpoint from '{olmo_checkpoint_path}'")
        checkpointer.load(olmo_checkpoint_path, train_module, load_trainer_state=False)
        log.info(f"Saving HF checkpoint to '{output_path}'")
        checkpointer.save(output_path, train_module, train_state={})
        log.info(f"Successfully saved HF model to '{output_path}'")

    log.info("Fixing config using tokenizer data and script arguments")
    if not hasattr(tokenizer, "vocab") or not isinstance(tokenizer.vocab, dict):
        raise ValueError("Tokenizer must have a vocab dictionary")

    huggingface_config = Olmo2Config.from_pretrained(output_path)
    huggingface_config.max_position_embeddings = max_sequence_length
    huggingface_config.pad_token_id = tokenizer.vocab.get(tokenizer.pad_token, None)
    huggingface_config.bos_token_id = tokenizer.vocab.get(tokenizer.bos_token, None)
    huggingface_config.eos_token_id = tokenizer.vocab.get(tokenizer.eos_token, None)
    huggingface_config.save_pretrained(output_path)
    log.info("Successfully fixed config using tokenizer data and script arguments")

    # Save the tokenizer in HF format
    tokenizer.save_pretrained(output_path)
    log.info(f"Successfully saved HF tokenizer to '{output_path}'")


def load_config(checkpoint_input_dir: PathOrStr) -> dict:
    assert file_exists(
        f"{checkpoint_input_dir}/config.json"
    ), f"Config file not found at {checkpoint_input_dir}"

    with cached_path(f"{checkpoint_input_dir}/config.json").open("r", encoding="utf-8") as f:
        config_dict = json.load(f)

    return config_dict


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--checkpoint-input-dir", type=str, required=True)
    parser.add_argument("-o", "--huggingface-output-dir", type=Path, required=True)
    parser.add_argument("-t", "--tokenizer-name-or-path", type=str, default=None)
    parser.add_argument("-s", "--max-sequence-length", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_config = load_config(args.checkpoint_input_dir)

    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = (
            experiment_config.get("dataset", {}).get("tokenizer", {}).get("identifier", None)
        )

        if args.tokenizer_name_or_path is None:
            raise ValueError(
                "Tokenizer identifier not found in the config; please provide it manually"
            )

    tokenizer_config = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)

    convert_to_hf_checkpoint(
        olmo_checkpoint_path=args.checkpoint_input_dir,
        output_path=args.huggingface_output_dir,
        olmo_core_config=experiment_config,
        max_sequence_length=args.max_sequence_length,
        tokenizer=tokenizer_config,  # type: ignore
    )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
