"""
Example script to convert a OLMo Core model checkpoint to a HuggingFace model checkpoint.

Note that this script is architecture-dependent, meaning it may only work for OLMo Core models that
have support in the `transformers` library.
"""

import json
import logging
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional

import torch
from cached_path import cached_path
from transformers import AutoConfig, AutoModelForCausalLM

from olmo_core.aliases import PathOrStr
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.io import file_exists
from olmo_core.nn.hf.checkpoint import save_hf_model
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.nn.transformer.model import Transformer
from olmo_core.optim.adamw import AdamWConfig
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.train_module.transformer import TransformerTrainModuleConfig
from olmo_core.utils import get_default_device, prepare_cli_environment

log = logging.getLogger(__name__)


def convert_checkpoint(
    original_checkpoint_path: str | Path,
    output_path: str | Path,
    transformer_config_dict: Dict[str, Any],
    tokenizer_config_dict: Dict[str, Any],
    *,
    max_sequence_length: int = -1,
    validate: bool = True,
) -> None:
    """
    Convert a checkpoint to a different OLMo core compatible format.

    Args:
        original_checkpoint_path: Path to the original checkpoint
        output_format: Format of converted checkpoint
        output_path: Where to save the converted model
        transformer_config_dict: Dictionary form of OLMo core model config
        tokenizer_config_dict: Dictionary form of OLMo core tokenizer config
    """
    if max_sequence_length <= 0:
        raise ValueError(f"Missing or invalid sequence length: {max_sequence_length}")

    # Remove deprecated transformer config options
    if "compile" in transformer_config_dict:
        del transformer_config_dict["compile"]
    if "dp_config" in transformer_config_dict:
        del transformer_config_dict["dp_config"]
    if "tp_config" in transformer_config_dict:
        del transformer_config_dict["tp_config"]
    if "float8_config" in transformer_config_dict:
        del transformer_config_dict["float8_config"]

    model = TransformerConfig.from_dict(transformer_config_dict).build()
    train_module = TransformerTrainModuleConfig(
        rank_microbatch_size=max_sequence_length,
        max_sequence_length=max_sequence_length,
        optim=AdamWConfig(),
    ).build(model)

    tokenizer_config = TokenizerConfig.from_dict(tokenizer_config_dict)

    with TemporaryDirectory() as work_dir:
        checkpointer_config = CheckpointerConfig(work_dir=work_dir, save_overwrite=True)
        checkpointer = checkpointer_config.build()

        log.info(f"Loading checkpoint from '{original_checkpoint_path}'")
        checkpointer.load(original_checkpoint_path, train_module, load_trainer_state=False)
        log.info(f"Saving checkpoint to '{output_path}'")
        save_hf_model(
            output_path,
            train_module.state_dict_to_save(optim=False)["model"],
            train_module.model,
            process_group=checkpointer.process_group,
            work_dir=checkpointer.work_dir,
            save_overwrite=checkpointer.save_overwrite,
        )
        # checkpointer.save(output_path, train_module, train_state={}, format=output_format)
        log.info(f"Successfully saved converted model to '{output_path}'")

    log.info("Fixing HF config using tokenizer config data and script arguments")
    huggingface_config = AutoConfig.from_pretrained(output_path)
    huggingface_config.max_position_embeddings = max_sequence_length
    huggingface_config.pad_token_id = tokenizer_config.pad_token_id
    huggingface_config.bos_token_id = tokenizer_config.bos_token_id
    huggingface_config.eos_token_id = tokenizer_config.eos_token_id
    huggingface_config.save_pretrained(output_path)
    log.info("Successfully fixed config using tokenizer config data and script arguments")

    if validate:
        log.info("Validating converted model")
        validate_conversion(output_path, model, tokenizer_config.vocab_size)
        log.info("Validation completed successful")


def validate_conversion(
    hf_path: str | Path,
    model: Transformer,
    vocab_size: int,
):
    device = get_default_device()

    B, T = 1, 120
    input_ids = torch.randint(0, vocab_size, (B, T)).to(device)

    log.info("Loading converted checkpoint for validation...")
    hf_model = AutoModelForCausalLM.from_pretrained(hf_path).to(device).eval()

    log.info("Running OLMo core and HF models for validation...")
    with torch.no_grad():
        hf_logits, *_ = hf_model(input_ids=input_ids, return_dict=False)

    del hf_model

    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids)

    torch.testing.assert_close(hf_logits, logits)


def load_config(checkpoint_input_dir: PathOrStr) -> Optional[dict]:
    if not file_exists(f"{checkpoint_input_dir}/config.json"):
        log.warning(f"Config file not found at {checkpoint_input_dir}")
        return None

    with cached_path(f"{checkpoint_input_dir}/config.json").open("r", encoding="utf-8") as f:
        config_dict = json.load(f)

    if "model" not in config_dict:
        log.warning(
            f"Config file at {checkpoint_input_dir} is not an OLMo core experiment config, ignoring"
        )
        return None

    return config_dict


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--checkpoint-input-path", type=str, required=True)
    parser.add_argument("-c", "--config-path", type=str, required=True)

    parser.add_argument("-o", "--huggingface-output-dir", type=Path, required=True)
    parser.add_argument("-s", "--max-sequence-length", type=int, required=True)
    parser.add_argument("--skip-validation", dest="validate", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()

    experiment_config = load_config(args.config_path)
    if experiment_config is None:
        raise RuntimeError("Experiment config not found, cannot convert to HF checkpoint")

    transformer_config_dict = experiment_config["model"]
    tokenizer_config_dict = experiment_config.get("dataset", {}).get("tokenizer")

    assert transformer_config_dict is not None
    assert tokenizer_config_dict is not None

    convert_checkpoint(
        original_checkpoint_path=args.checkpoint_input_path,
        output_path=args.huggingface_output_dir,
        transformer_config_dict=transformer_config_dict,
        tokenizer_config_dict=tokenizer_config_dict,
        max_sequence_length=args.max_sequence_length,
        validate=args.validate,
    )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
