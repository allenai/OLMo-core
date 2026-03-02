"""
Convert an OLMo Core hybrid model checkpoint (with GatedDeltaNet layers) to a HuggingFace
``olmo_hybrid`` model checkpoint.

Hybrid models use a mix of standard attention layers and GatedDeltaNet (linear attention) layers.
In OLMo-core, both layer types share the same block-level key prefix (``blocks.{i}.attention.*``),
but in HF they map to different prefixes (``self_attn.*`` vs ``linear_attn.*``). This script uses
explicit per-layer key mapping based on runtime block type inspection to handle the difference.

Usage::

    python convert_checkpoint_to_hf_hybrid.py \\
        -i /path/to/olmo-core-checkpoint \\
        -o /path/to/output-hf-model \\
        --skip-validation
"""

import json
import logging
import re
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

import rich
import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from cached_path import cached_path
from safetensors.torch import save_file
from transformers import AutoTokenizer

from olmo_core.aliases import PathOrStr
from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.io import file_exists, join_path
from olmo_core.nn.attention import Attention, AttentionBackendName, AttentionConfig
from olmo_core.nn.attention.recurrent import GatedDeltaNet
from olmo_core.nn.transformer.block import ReorderedNormTransformerBlock, TransformerBlock
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig
from olmo_core.nn.transformer.model import Transformer
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Key mapping tables
# ---------------------------------------------------------------------------

SHARED_KEY_MAP: Dict[str, str] = {
    "embeddings.weight": "model.embed_tokens.weight",
    "lm_head.norm.weight": "model.norm.weight",
    "lm_head.w_out.weight": "lm_head.weight",
}

# GDN layers: OLMo-core ``blocks.{i}.attention.*`` -> HF ``model.layers.{i}.linear_attn.*``
# These layers use pre-norm in HF (input_layernorm before the sequence mixer).
GDN_KEY_MAP: Dict[str, str] = {
    "attention.w_q.weight": "linear_attn.q_proj.weight",
    "attention.w_k.weight": "linear_attn.k_proj.weight",
    "attention.w_v.weight": "linear_attn.v_proj.weight",
    "attention.w_a.weight": "linear_attn.a_proj.weight",
    "attention.w_b.weight": "linear_attn.b_proj.weight",
    "attention.w_g.weight": "linear_attn.g_proj.weight",
    "attention.w_out.weight": "linear_attn.o_proj.weight",
    "attention.q_conv1d.weight": "linear_attn.q_conv1d.weight",
    "attention.k_conv1d.weight": "linear_attn.k_conv1d.weight",
    "attention.v_conv1d.weight": "linear_attn.v_conv1d.weight",
    "attention.o_norm.weight": "linear_attn.o_norm.weight",
    "attention.A_log": "linear_attn.A_log",
    "attention.dt_bias": "linear_attn.dt_bias",
    "attention_norm.weight": "input_layernorm.weight",
    "feed_forward_norm.weight": "post_attention_layernorm.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
}

# Attention layers: OLMo-core ``blocks.{i}.attention.*`` -> HF ``model.layers.{i}.self_attn.*``
# These layers use post-norm in HF (layernorm after the sequence mixer and after the MLP).
ATTN_KEY_MAP: Dict[str, str] = {
    "attention.w_q.weight": "self_attn.q_proj.weight",
    "attention.w_k.weight": "self_attn.k_proj.weight",
    "attention.w_v.weight": "self_attn.v_proj.weight",
    "attention.w_out.weight": "self_attn.o_proj.weight",
    "attention.q_norm.weight": "self_attn.q_norm.weight",
    "attention.k_norm.weight": "self_attn.k_norm.weight",
    "attention_norm.weight": "post_attention_layernorm.weight",
    "feed_forward_norm.weight": "post_feedforward_layernorm.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
}

# Regex to match per-layer block keys: ``blocks.<layer_idx>.<rest>``
_BLOCK_KEY_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")


# ---------------------------------------------------------------------------
# Block type helpers
# ---------------------------------------------------------------------------


def is_gdn_layer(block: TransformerBlock) -> bool:
    """Return ``True`` if the block's sequence mixer is a :class:`GatedDeltaNet`."""
    return isinstance(block.attention, GatedDeltaNet)


def get_layer_types(model: Transformer) -> List[str]:
    """
    Iterate over blocks and return a list of ``"linear_attention"`` or ``"full_attention"``
    strings matching the HF ``olmo_hybrid`` config format.
    """
    layer_types: List[str] = []
    for idx, block in model.blocks.items():
        if is_gdn_layer(block):
            layer_types.append("linear_attention")
        elif isinstance(block.attention, Attention):
            layer_types.append("full_attention")
        else:
            raise ValueError(
                f"Unknown sequence mixer type at layer {idx}: {type(block.attention)}"
            )
    return layer_types


# ---------------------------------------------------------------------------
# State dict conversion
# ---------------------------------------------------------------------------


def convert_state_dict(
    state_dict: Dict[str, Any],
    layer_types: List[str],
) -> Dict[str, Any]:
    """
    Convert an OLMo-core hybrid state dict to HF ``olmo_hybrid`` format.

    Uses :data:`SHARED_KEY_MAP` for non-block keys, and per-layer
    :data:`GDN_KEY_MAP` / :data:`ATTN_KEY_MAP` based on ``layer_types``.
    """
    hf_state: Dict[str, Any] = {}

    for olmo_key, value in state_dict.items():
        # Try shared (non-block) keys first.
        if olmo_key in SHARED_KEY_MAP:
            hf_state[SHARED_KEY_MAP[olmo_key]] = value
            continue

        m = _BLOCK_KEY_RE.match(olmo_key)
        if m is None:
            log.warning(f"Unmapped key (skipped): {olmo_key}")
            continue

        layer_idx = int(m.group(1))
        suffix = m.group(2)

        key_map = GDN_KEY_MAP if layer_types[layer_idx] == "linear_attention" else ATTN_KEY_MAP
        if suffix not in key_map:
            log.warning(f"Unmapped block suffix (skipped): {olmo_key}")
            continue

        hf_key = f"model.layers.{layer_idx}.{key_map[suffix]}"
        hf_state[hf_key] = value

    return hf_state


# ---------------------------------------------------------------------------
# HF config builder
# ---------------------------------------------------------------------------


def _get_rope_scaling(model: Transformer, layer_types: List[str]) -> Optional[dict]:
    """
    Extract the RoPE scaling config from attention blocks.  GDN layers are skipped
    because they don't use RoPE.
    """
    attn_blocks = [
        (int(idx), block)
        for idx, block in model.blocks.items()
        if layer_types[int(idx)] == "full_attention"
    ]

    layers_with_scaling = [
        (idx, block)
        for idx, block in attn_blocks
        if block.attention.rope is not None and block.attention.rope.scaling is not None
    ]
    if not layers_with_scaling:
        return None

    first_config = layers_with_scaling[0][1].attention.rope.scaling.to_hf_config()
    for idx, block in layers_with_scaling[1:]:
        cfg = block.attention.rope.scaling.to_hf_config()
        if cfg != first_config:
            raise NotImplementedError(
                f"Inconsistent RoPE scaling configs. First: {first_config}, Layer {idx}: {cfg}"
            )
    return first_config


def build_hf_config(
    model: Transformer,
    layer_types: List[str],
    max_seq_len: int,
) -> Dict[str, Any]:
    """
    Build the ``config.json`` dict for a HF ``olmo_hybrid`` model.

    Extracts standard fields from the first attention block and GDN-specific fields
    from the first GDN block.
    """
    blocks = list(model.blocks.values())

    attn_block: Optional[TransformerBlock] = None
    gdn_block: Optional[TransformerBlock] = None
    for lt, block in zip(layer_types, blocks):
        if lt == "full_attention" and attn_block is None:
            attn_block = block
        elif lt == "linear_attention" and gdn_block is None:
            gdn_block = block

    if attn_block is None:
        raise ValueError("Hybrid model must have at least one attention layer")
    if gdn_block is None:
        raise ValueError("Hybrid model must have at least one GDN layer")

    attn: Attention = attn_block.attention
    gdn: GatedDeltaNet = gdn_block.attention

    # RoPE (from attention blocks only)
    rope_parameters: Optional[dict] = None
    if attn.rope is not None:
        rope_theta = float(attn.rope.theta)
        rope_scaling = _get_rope_scaling(model, layer_types)
        rope_parameters = {"rope_theta": rope_theta}
        if rope_scaling:
            rope_parameters.update(rope_scaling)
        else:
            rope_parameters["rope_type"] = "default"
        log.info(f"RoPE: {rope_parameters}")
    else:
        log.info("No RoPE configured")

    # Warn if GDN blocks are post-norm (ReorderedNormTransformerBlock) but HF expects pre-norm.
    if isinstance(gdn_block, ReorderedNormTransformerBlock):
        log.warning(
            "GDN block uses post-norm (ReorderedNormTransformerBlock) but HF olmo_hybrid "
            "expects pre-norm for linear_attention layers. The conversion will proceed, but "
            "outputs may not match exactly."
        )

    config: Dict[str, Any] = {
        "model_type": "olmo_hybrid",
        "architectures": ["OlmoHybridForCausalLM"],
        # Standard transformer fields
        "vocab_size": model.vocab_size,
        "hidden_size": model.d_model,
        "intermediate_size": attn_block.feed_forward.hidden_size,
        "num_hidden_layers": len(blocks),
        "num_attention_heads": attn.n_heads,
        "num_key_value_heads": attn.n_kv_heads,
        "hidden_act": "silu",
        "max_position_embeddings": max_seq_len,
        "initializer_range": 0.02,
        "use_cache": True,
        "attention_bias": attn.w_out.bias is not None,
        "attention_dropout": 0.0,
        "rms_norm_eps": attn_block.feed_forward_norm.eps,
        "tie_word_embeddings": False,
        # Hybrid layer configuration
        "layer_types": layer_types,
        # GDN (linear attention) parameters
        "linear_num_key_heads": gdn.n_heads,
        "linear_num_value_heads": gdn.n_v_heads,
        "linear_key_head_dim": gdn.head_k_dim,
        "linear_value_head_dim": gdn.head_v_dim,
        "linear_conv_kernel_dim": gdn.conv_size,
        "linear_allow_neg_eigval": gdn.allow_neg_eigval,
        # Token IDs (updated later after tokenizer is saved)
        "pad_token_id": None,
        "bos_token_id": None,
        "eos_token_id": None,
    }

    if rope_parameters is not None:
        config["rope_parameters"] = rope_parameters
    else:
        config["rope_theta"] = None

    return config


# ---------------------------------------------------------------------------
# Checkpoint saving
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    state_dict: Dict[str, torch.Tensor],
    config_dict: Dict[str, Any],
    dtype: Optional[DType],
    vocab_size: Optional[int],
) -> None:
    """Write ``config.json`` and ``model.safetensors`` to *path*."""
    path.mkdir(parents=True, exist_ok=True)

    if dtype is not None:
        state_dict = {
            k: v.to(dtype.as_pt()) if torch.is_tensor(v) else v for k, v in state_dict.items()
        }

    if vocab_size is not None:
        config_dict["vocab_size"] = vocab_size
        if "model.embed_tokens.weight" in state_dict:
            state_dict["model.embed_tokens.weight"] = state_dict["model.embed_tokens.weight"][
                :vocab_size
            ]
        if "lm_head.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["lm_head.weight"][:vocab_size]

    log.info(f"Converted state dict has {len(state_dict)} keys")
    log.info("Sample keys for layer 0:")
    for k in sorted(state_dict):
        if "layers.0." in k:
            log.info(f"  {k}")

    config_path = path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    log.info(f"Saved config to {config_path}")

    save_file(state_dict, path / "model.safetensors")
    log.info(f"Saved weights to {path / 'model.safetensors'}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def convert_checkpoint_to_hf(
    original_checkpoint_path: str | Path,
    output_path: str | Path,
    transformer_config_dict: Dict[str, Any],
    tokenizer_config_dict: Dict[str, Any],
    *,
    dtype: Optional[DType] = None,
    tokenizer_id: str | None = None,
    max_sequence_length: int | None = None,
    validate: bool = True,
    device: torch.device | None = None,
) -> None:
    """
    Convert an OLMo-core hybrid checkpoint to HuggingFace ``olmo_hybrid`` format.
    """
    if max_sequence_length is not None and max_sequence_length <= 0:
        raise ValueError(f"Invalid sequence length: {max_sequence_length}")

    # Remove deprecated config options that may be present in old configs.
    for key in ("compile", "dp_config", "tp_config", "float8_config"):
        transformer_config_dict.pop(key, None)

    model_config = TransformerConfig.from_dict(transformer_config_dict)
    rich.print(model_config)

    # Override attention backends to torch for CPU conversion.
    if isinstance(model_config.block, dict):
        block_configs = model_config.block.values()
    else:
        block_configs = [model_config.block]
    for block_config in block_configs:
        if isinstance(block_config.sequence_mixer, AttentionConfig):
            block_config.sequence_mixer.backend = AttentionBackendName.torch
            block_config.sequence_mixer.use_flash = False

    # Build the model on meta device, then materialise to CPU.
    model = model_config.build(init_device="meta")
    model.to_empty(device=device or torch.device("cpu"))

    tokenizer_config = TokenizerConfig.from_dict(tokenizer_config_dict)
    vocab_size = tokenizer_config.vocab_size

    with TemporaryDirectory() as work_dir:
        model_and_optim_dir = join_path(original_checkpoint_path, "model_and_optim")
        log.info(f"Loading checkpoint from '{model_and_optim_dir}'")
        load_model_and_optim_state(model_and_optim_dir, model, work_dir=work_dir)

        state_dict_options = dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)

    # Detect layer types and convert state dict.
    layer_types = get_layer_types(model)
    log.info(f"Layer types ({len(layer_types)} layers): {layer_types}")

    hf_state_dict = convert_state_dict(model_state_dict, layer_types)

    max_sequence_length = max_sequence_length or 65536
    hf_config_dict = build_hf_config(model, layer_types, max_sequence_length)

    output_path = Path(output_path)
    save_checkpoint(output_path, hf_state_dict, hf_config_dict, dtype, vocab_size)

    # Save tokenizer.
    tokenizer_id = tokenizer_id or tokenizer_config.identifier
    if tokenizer_id is not None:
        log.info(f"Saving tokenizer {tokenizer_id}")
        huggingface_tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        max_sequence_length = max_sequence_length or getattr(
            huggingface_tokenizer, "model_max_length", None
        )
        huggingface_tokenizer.model_max_length = max_sequence_length
        huggingface_tokenizer.pad_token_id = tokenizer_config.pad_token_id
        huggingface_tokenizer.bos_token_id = tokenizer_config.bos_token_id
        huggingface_tokenizer.eos_token_id = tokenizer_config.eos_token_id
        huggingface_tokenizer.save_pretrained(output_path)
        log.info(f"Saved tokenizer to {output_path}")

    # Update config.json with tokenizer info and max_position_embeddings.
    config_path = output_path / "config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config_dict["max_position_embeddings"] = max_sequence_length or config_dict.get(
        "max_position_embeddings", 65536
    )
    config_dict["pad_token_id"] = tokenizer_config.pad_token_id
    config_dict["bos_token_id"] = tokenizer_config.bos_token_id
    config_dict["eos_token_id"] = tokenizer_config.eos_token_id

    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Validation
    if validate:
        log.warning(
            "Validation for hybrid models requires 'olmo_hybrid' to be registered in "
            "transformers. Skipping validation."
        )

    log.info(f"Successfully saved converted model to '{output_path}'")


# ---------------------------------------------------------------------------
# Config loading & CLI
# ---------------------------------------------------------------------------


def load_config(checkpoint_input_dir: PathOrStr) -> Optional[dict]:
    """Load the experiment config from the checkpoint directory."""
    config_path = f"{checkpoint_input_dir}/config.json"
    if not file_exists(config_path):
        raise RuntimeError(f"Config file not found at {checkpoint_input_dir}")

    with cached_path(config_path).open("r", encoding="utf-8") as f:
        config_dict = json.load(f)

    if "model" not in config_dict:
        raise RuntimeError(
            f"Config at {checkpoint_input_dir} is not an OLMo core experiment config"
        )

    return config_dict


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--checkpoint-input-path",
        type=str,
        required=True,
        help="Path to OLMo Core checkpoint directory.",
    )
    parser.add_argument(
        "-o",
        "--huggingface-output-dir",
        type=str,
        required=True,
        help="Output path for HuggingFace model.",
    )
    parser.add_argument(
        "-s",
        "--max-sequence-length",
        type=int,
        help="Max sequence length. Defaults to tokenizer's model_max_length or 65536.",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        help="HuggingFace tokenizer identifier.",
    )
    parser.add_argument(
        "--skip-validation",
        dest="validate",
        action="store_false",
        help="Skip validation.",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        help="Device for conversion. Defaults to CPU.",
    )
    parser.add_argument(
        "--dtype",
        type=DType,
        default=DType.bfloat16,
        help="Output dtype. Defaults to bfloat16.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    experiment_config = load_config(args.checkpoint_input_path)
    if experiment_config is None:
        raise RuntimeError("Experiment config not found")

    transformer_config_dict = experiment_config["model"]
    tokenizer_config_dict = experiment_config.get("dataset", {}).get("tokenizer")

    assert transformer_config_dict is not None
    assert tokenizer_config_dict is not None

    max_sequence_length = args.max_sequence_length
    if max_sequence_length is None:
        max_sequence_length = experiment_config.get("train_module", {}).get("max_sequence_length")
    if max_sequence_length is None:
        max_sequence_length = experiment_config.get("dataset", {}).get("sequence_length")

    convert_checkpoint_to_hf(
        original_checkpoint_path=args.checkpoint_input_path,
        output_path=args.huggingface_output_dir,
        transformer_config_dict=transformer_config_dict,
        tokenizer_config_dict=tokenizer_config_dict,
        dtype=args.dtype,
        max_sequence_length=max_sequence_length,
        tokenizer_id=args.tokenizer,
        validate=args.validate,
        device=args.device,
    )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
