"""
Example script to convert a OLMo Core model checkpoint to a HuggingFace model checkpoint.
"""

import json
import logging
import re
import shutil
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, Olmo2Config

from olmo_core.distributed.checkpoint import unshard_checkpoint
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


try:
    from accelerate import init_empty_weights
except ImportError:
    pass

    @contextmanager
    def init_empty_weights(include_buffers: bool = False) -> Generator[None, None, None]:
        log.warning("accelerate not installed, will initialize weights.")
        yield None


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
    log.info(f"Loading OLMo core checkpoint from '{olmo_checkpoint_path}'")
    olmo_state_dict = load_state_dict(olmo_checkpoint_path)

    # Initialize new state dict for HF format
    hf_state_dict = {}

    # Map OLMo-core keys to HF keys
    hf_state_dict["model.embed_tokens.weight"] = olmo_state_dict.pop("embeddings.weight")  # ok

    if "norm.weight" in olmo_state_dict:
        log.info("Using norm.weight as model.norm.weight")
        hf_state_dict["model.norm.weight"] = olmo_state_dict.pop("norm.weight")
    elif "lm_head.norm.weight" in olmo_state_dict:
        log.info("Using lm_head.norm.weight as model.norm.weight")
        hf_state_dict["model.norm.weight"] = olmo_state_dict.pop("lm_head.norm.weight")
    else:
        raise ValueError("No norm.weight or lm_head.norm.weight found in the state dict")

    if "w_out.weight" in olmo_state_dict:
        log.info("Using w_out.weight as lm_head.weight")
        hf_state_dict["lm_head.weight"] = olmo_state_dict.pop("w_out.weight")
    elif "lm_head.w_out.weight" in olmo_state_dict:
        log.info("Using lm_head.w_out.weight as lm_head.weight")
        hf_state_dict["lm_head.weight"] = olmo_state_dict.pop("lm_head.w_out.weight")
    else:
        raise ValueError("No w_out.weight or lm_head.w_out.weight found in the state dict")

    # Count number of layers from the state dict keys
    layer_ids = [
        match.group(1)
        for key in olmo_state_dict.keys()
        if (match := re.match(r"blocks\.(\d+)\.", key))
    ]
    assert layer_ids, "No layer IDs found in the state dict keys"
    n_layers = max(map(int, layer_ids)) + 1

    for block in range(n_layers):
        # Attention
        hf_state_dict[f"model.layers.{block}.self_attn.q_proj.weight"] = olmo_state_dict.pop(
            f"blocks.{block}.attention.w_q.weight"
        )
        hf_state_dict[f"model.layers.{block}.self_attn.k_proj.weight"] = olmo_state_dict.pop(
            f"blocks.{block}.attention.w_k.weight"
        )
        hf_state_dict[f"model.layers.{block}.self_attn.v_proj.weight"] = olmo_state_dict.pop(
            f"blocks.{block}.attention.w_v.weight"
        )
        hf_state_dict[f"model.layers.{block}.self_attn.o_proj.weight"] = olmo_state_dict.pop(
            f"blocks.{block}.attention.w_out.weight"
        )

        # MLP
        hf_state_dict[f"model.layers.{block}.mlp.gate_proj.weight"] = olmo_state_dict.pop(
            f"blocks.{block}.feed_forward.w1.weight"
        )
        hf_state_dict[f"model.layers.{block}.mlp.down_proj.weight"] = olmo_state_dict.pop(
            f"blocks.{block}.feed_forward.w2.weight"
        )
        hf_state_dict[f"model.layers.{block}.mlp.up_proj.weight"] = olmo_state_dict.pop(
            f"blocks.{block}.feed_forward.w3.weight"
        )

        # Check if we have q_norm and k_norm (non-Llama model)
        has_qk_norm = f"blocks.{block}.attention.q_norm.weight" in olmo_state_dict

        if has_qk_norm:
            # Non-Llama model
            hf_state_dict[
                f"model.layers.{block}.post_attention_layernorm.weight"
            ] = olmo_state_dict.pop(f"blocks.{block}.attention_norm.weight")
            hf_state_dict[
                f"model.layers.{block}.post_feedforward_layernorm.weight"
            ] = olmo_state_dict.pop(f"blocks.{block}.feed_forward_norm.weight")
            hf_state_dict[f"model.layers.{block}.self_attn.q_norm.weight"] = olmo_state_dict.pop(
                f"blocks.{block}.attention.q_norm.weight"
            )
            hf_state_dict[f"model.layers.{block}.self_attn.k_norm.weight"] = olmo_state_dict.pop(
                f"blocks.{block}.attention.k_norm.weight"
            )
        else:
            # Llama model
            hf_state_dict[
                f"model.layers.{block}.post_attention_layernorm.weight"
            ] = olmo_state_dict.pop(f"blocks.{block}.feed_forward_norm.weight")
            hf_state_dict[f"model.layers.{block}.input_layernorm.weight"] = olmo_state_dict.pop(
                f"blocks.{block}.attention_norm.weight"
            )

    # Verify we used all keys
    if len(olmo_state_dict) > 0:
        _unused_keys = "\n".join(f"\t-{k}" for k in olmo_state_dict.keys())
        raise ValueError(f"Unused keys in the state dict:\n{_unused_keys}")

    if not hasattr(tokenizer, "vocab") or not isinstance(tokenizer.vocab, dict):
        raise ValueError("Tokenizer must have a vocab dictionary")

    if (
        k := olmo_core_config.get("model", {})
        .get("block", {})
        .get("layer_norm", {})
        .get("name", None)
    ) != "rms":
        raise ValueError(f"Only RMSNorm is supported, found {k}")

    max_sequence_length = min(
        int(olmo_core_config.get("dataset", {}).get("sequence_length", max_sequence_length)),
        tokenizer.model_max_length,
    )

    if max_sequence_length <= 0:
        raise ValueError(f"Missing or invalid sequence length: {max_sequence_length}")

    # Create HF model instance and load state dict
    huggingface_config = Olmo2Config(
        vocab_size=olmo_core_config["model"]["vocab_size"],
        hidden_size=olmo_core_config["model"]["d_model"],
        intermediate_size=olmo_core_config["model"]["block"]["feed_forward"]["hidden_size"],
        num_hidden_layers=olmo_core_config["model"]["n_layers"],
        num_attention_heads=(n_heads := olmo_core_config["model"]["block"]["attention"]["n_heads"]),
        num_key_value_heads=(
            olmo_core_config["model"]["block"]["attention"].get("n_kv_heads") or n_heads
        ),
        hidden_act="silu",
        max_position_embeddings=max_sequence_length,
        rope_theta=olmo_core_config["model"]["block"]["attention"]["rope"]["theta"],
        attention_bias=olmo_core_config["model"]["block"]["attention"].get("bias") or False,
        pad_token_id=tokenizer.vocab.get(tokenizer.pad_token, None),
        bos_token_id=tokenizer.vocab.get(tokenizer.bos_token, None),
        eos_token_id=tokenizer.vocab.get(tokenizer.eos_token, None),
        rms_norm_eps=olmo_core_config["model"]["block"]["layer_norm"]["eps"],
        tie_word_embeddings=False,
    )

    with init_empty_weights():
        log.info("Initializing HF model with empty weights...")
        model = AutoModelForCausalLM.from_config(huggingface_config)

    log.info("Loading state dict into HF model...")
    model.load_state_dict(hf_state_dict, assign=True)

    # Save the model in HF format
    log.info(f"Saving HF model checkpoint to {output_path}...")
    model.save_pretrained(output_path)
    log.info(f"Successfully saved HF model to '{output_path}'")

    # Save the tokenizer in HF format
    tokenizer.save_pretrained(output_path)
    log.info(f"Successfully saved HF tokenizer to '{output_path}'")


def load_config(checkpoint_input_dir: Path) -> dict:
    assert (
        checkpoint_input_dir / "config.json"
    ).exists(), f"Config file not found at {checkpoint_input_dir}"

    with open(checkpoint_input_dir / "config.json", "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    return config_dict


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--checkpoint-input-dir", type=Path, required=True)
    parser.add_argument("-u", "--unsharded-output-dir", type=Path, default=None)
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

    with TemporaryDirectory() as _unsharded_dir:
        if args.unsharded_output_dir:
            log.info(f"Using provided unsharded output directory: {args.unsharded_output_dir}")
            _unsharded_dir = args.unsharded_output_dir

        shards_dir = args.checkpoint_input_dir / "model_and_optim"
        if shards_dir.exists() and shards_dir.is_dir():
            logging.info(f"Unsharding checkpoint from {shards_dir} to {_unsharded_dir}")
            (unsharded_dir := Path(_unsharded_dir)).mkdir(parents=True, exist_ok=True)
            unshard_checkpoint(dir=shards_dir, target_dir=unsharded_dir, optim=False)

            logging.info("Copying config.json to unsharded directory")
            shutil.copy(args.checkpoint_input_dir / "config.json", unsharded_dir / "config.json")
        else:
            logging.info("No sharded checkpoint found, using input directory as unsharded")
            unsharded_dir = args.checkpoint_input_dir

        convert_to_hf_checkpoint(
            olmo_checkpoint_path=unsharded_dir / "model.pt",
            output_path=args.huggingface_output_dir,
            olmo_core_config=experiment_config,
            max_sequence_length=args.max_sequence_length,
            tokenizer=tokenizer_config,  # type: ignore
        )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
