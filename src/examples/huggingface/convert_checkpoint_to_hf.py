"""
Example script to convert a OLMo Core model checkpoint to a HuggingFace model checkpoint.
"""

from argparse import ArgumentParser
import json
import logging
from pathlib import Path
from contextlib import contextmanager
import re
from tempfile import TemporaryDirectory

import torch
from transformers import Olmo2Config, AutoModelForCausalLM
from safetensors.torch import load_file


from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state, save_state_dict
from olmo_core.internal.experiment import ExperimentConfig
from olmo_core.io import clear_directory, dir_is_empty
from olmo_core.nn.rope import RoPEScalingConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.utils import get_default_device, prepare_cli_environment
from olmo_core.distributed.checkpoint import unshard_checkpoint

try:
    from accelerate import init_empty_weights
except ImportError:
    pass

    @contextmanager
    def init_empty_weights(include_buffers: bool = False):
        yield None



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

    if checkpoint_path.suffix == '.safetensors':
        # Load safetensors format
        state_dict = load_file(checkpoint_path)
    else:
        # Load PyTorch format (.pt, .pth, .bin)
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        # Handle both cases:
        # 1. Direct state dict
        # 2. Nested state dict under 'model' or 'state_dict' key
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

    return state_dict


def convert_to_hf_checkpoint(
    olmo_checkpoint_path: str | Path,
    output_path: str | Path,
    config: dict
) -> None:
    """
    Convert an OLMo core checkpoint to Hugging Face format.

    Args:
        olmo_checkpoint_path: Path to the OLMo core checkpoint
        hf_model_name: Name/path of the HF model to use for configuration
        output_path: Where to save the converted HF model
    """
    log.info(f"Loading OLMo core checkpoint from '{olmo_checkpoint_path}'")
    olmo_state_dict = load_state_dict(olmo_checkpoint_path)

    # Initialize new state dict for HF format
    hf_state_dict = {}

    # Map OLMo-core keys to HF keys
    hf_state_dict["model.embed_tokens.weight"] = olmo_state_dict.pop("embeddings.weight")   # ok
    hf_state_dict["model.norm.weight"] = olmo_state_dict.pop("norm.weight")  # ok
    hf_state_dict["lm_head.weight"] = olmo_state_dict.pop("w_out.weight")  # ok

    # Count number of layers from the state dict keys
    layer_ids = [match.group(1) for key in olmo_state_dict.keys() if (match := re.match(r"blocks\.(\d+)\.", key))]
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
            hf_state_dict[f"model.layers.{block}.post_attention_layernorm.weight"] = olmo_state_dict.pop(
                f"blocks.{block}.attention_norm.weight"
            )
            hf_state_dict[f"model.layers.{block}.post_feedforward_layernorm.weight"] = olmo_state_dict.pop(
                f"blocks.{block}.feed_forward_norm.weight"
            )
            hf_state_dict[f"model.layers.{block}.self_attn.q_norm.weight"] = olmo_state_dict.pop(
                f"blocks.{block}.attention.q_norm.weight"
            )
            hf_state_dict[f"model.layers.{block}.self_attn.k_norm.weight"] = olmo_state_dict.pop(
                f"blocks.{block}.attention.k_norm.weight"
            )
        else:
            # Llama model
            hf_state_dict[f"model.layers.{block}.post_attention_layernorm.weight"] = olmo_state_dict.pop(
                f"blocks.{block}.feed_forward_norm.weight"
            )
            hf_state_dict[f"model.layers.{block}.input_layernorm.weight"] = olmo_state_dict.pop(
                f"blocks.{block}.attention_norm.weight"
            )

    # Verify we used all keys
    assert len(olmo_state_dict) == 0

    log.info(f"Saving HF model checkpoint to {output_path}...")

    breakpoint()

    # Create HF model instance and load state dict
    config = Olmo2Config(
        vocab_size=hf_state_dict["model.embed_tokens.weight"].size(0),
        hidden_size=hf_state_dict["model.norm.weight"].size(0),
        intermediate_size=hf_state_dict["model.layers.0.mlp.down_proj.weight"].size(0),
        num_hidden_layers=n_layers,
        num_attention_heads=hf_state_dict["model.layers.0.self_attn.q_proj.weight"].size(0),
    )

    with init_empty_weights():
        # this will not reinit weights in case
        model = AutoModelForCausalLM.from_config(config)

    model.load_state_dict(hf_state_dict)


    # Save the model in HF format
    model.save_pretrained(output_path)
    log.info(f"Successfully saved HF model to '{output_path}'")



def load_config(checkpoint_input_dir: Path) -> dict:
    assert (checkpoint_input_dir / "config.json").exists(), f"Config file not found at {checkpoint_input_dir}"

    with open(checkpoint_input_dir / "config.json", "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    return config_dict


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--checkpoint-input-dir', type=Path, required=True)
    parser.add_argument('-o', '--huggingface-output-dir', type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_config = load_config(args.checkpoint_input_dir)

    with TemporaryDirectory() as _unsharded_dir:
        if (shards_dir := (args.checkpoint_input_dir / "model_and_optim")).exists() and shards_dir.is_dir():
            unsharded_dir = Path(_unsharded_dir)
            unshard_checkpoint(dir=shards_dir, target_dir=unsharded_dir, optim=False)
        else:
            unsharded_dir = args.checkpoint_input_dir

        convert_to_hf_checkpoint(
            olmo_checkpoint_path=unsharded_dir / "model.pt",
            output_path=args.huggingface_output_dir,
            config=experiment_config
        )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
