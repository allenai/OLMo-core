"""
Convert an olmo-core Qwen3 checkpoint to HuggingFace format.

The standard convert_checkpoint_to_hf.py only supports OLMo block types.
This script handles Qwen3's default TransformerBlock by providing the
HF config directly rather than trying to auto-detect it.

Usage:
    python src/scripts/train/sft/convert_qwen3_to_hf.py \
        -i /path/to/olmo-core-checkpoint \
        -o /path/to/hf-output \
        -c src/scripts/train/sft/qwen3-1.7b-config.json \
        --max-sequence-length 32768
"""

import json
import logging
import tempfile
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from olmo_core.aliases import PathOrStr
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.io import file_exists, join_path
from olmo_core.nn.hf.convert import convert_state_to_hf
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--checkpoint-input-path", type=str, required=True,
                        help="Path to the olmo-core checkpoint directory (containing model_and_optim/)")
    parser.add_argument("-o", "--output-dir", type=Path, required=True,
                        help="Directory to save the HuggingFace model")
    parser.add_argument("-c", "--config-path", type=str, default=None,
                        help="Path to the olmo-core experiment config JSON (defaults to <checkpoint>/config.json)")
    parser.add_argument("--hf-model-id", type=str, default="Qwen/Qwen3-1.7B-Base",
                        help="HuggingFace model ID to use for the config (default: Qwen/Qwen3-1.7B-Base)")
    parser.add_argument("--max-sequence-length", type=int, default=32768,
                        help="Max sequence length for the HF config")
    parser.add_argument("-t", "--tokenizer-path", type=str, default=None,
                        help="Path to tokenizer to include in output (if not specified, uses --hf-model-id)")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"],
                        help="Dtype to save weights as")
    args = parser.parse_args()

    # Load experiment config
    config_path = args.config_path or str(Path(args.checkpoint_input_path) / "config.json")
    log.info(f"Loading experiment config from '{config_path}'")
    with open(config_path) as f:
        experiment_config = json.load(f)

    transformer_config_dict = experiment_config["model"]
    tokenizer_config_dict = experiment_config.get("dataset", {}).get("tokenizer")

    # Remove deprecated fields
    for field in ("compile", "dp_config", "tp_config", "float8_config"):
        transformer_config_dict.pop(field, None)

    model_config = TransformerConfig.from_dict(transformer_config_dict)
    log.info(f"Built model config: {model_config.d_model=}, {model_config.n_layers=}, {model_config.vocab_size=}")

    # Build olmo-core model and load checkpoint
    model = model_config.build(init_device="meta")
    model.to_empty(device=torch.device("cpu"))

    state_dict_options = dist_cp_sd.StateDictOptions(flatten_optimizer_state_dict=True, cpu_offload=True)
    model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)

    checkpoint_path = args.checkpoint_input_path
    model_and_optim_dir = join_path(checkpoint_path, "model_and_optim")
    if not file_exists(model_and_optim_dir):
        model_and_optim_dir = checkpoint_path
    log.info(f"Loading checkpoint from '{model_and_optim_dir}'")

    with tempfile.TemporaryDirectory() as work_dir:
        load_model_and_optim_state(model_and_optim_dir, model, work_dir=work_dir)

    model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)

    # Get HF config from the base model
    log.info(f"Loading HF config from '{args.hf_model_id}'")
    hf_config = AutoConfig.from_pretrained(args.hf_model_id)
    hf_config.max_position_embeddings = args.max_sequence_length

    # Set tokenizer IDs in HF config
    if tokenizer_config_dict:
        tc = TokenizerConfig.from_dict(tokenizer_config_dict)
        hf_config.pad_token_id = tc.pad_token_id
        hf_config.eos_token_id = tc.eos_token_id
        if tc.bos_token_id is not None:
            hf_config.bos_token_id = tc.bos_token_id

    # Convert weights from olmo-core to HF format
    log.info("Converting state dict to HF format")
    hf_state_dict = convert_state_to_hf(hf_config, model_state_dict)
    hf_state_dict = {key: state.contiguous() for key, state in hf_state_dict.items()}

    # Cast to target dtype
    save_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    hf_state_dict = {key: state.to(dtype=save_dtype) for key, state in hf_state_dict.items()}

    # Create HF model and load weights
    log.info("Creating HF model and loading converted weights")
    hf_model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=save_dtype)
    hf_model.load_state_dict(hf_state_dict, strict=True)

    # Save model
    output_path = str(args.output_dir)
    log.info(f"Saving HF model to '{output_path}'")
    hf_model.save_pretrained(output_path)

    # Save tokenizer
    tokenizer_path = args.tokenizer_path or args.hf_model_id
    log.info(f"Saving tokenizer from '{tokenizer_path}'")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(output_path)

    log.info(f"Successfully saved HF model to '{output_path}'")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
