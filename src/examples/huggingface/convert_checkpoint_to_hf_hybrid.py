"""
Convert an OLMo Core hybrid model checkpoint (with FLA/GatedDeltaNet layers) 
to a HuggingFace model checkpoint.

This script extends the standard conversion to support OLMo 3.5 Hybrid models that mix
attention layers with linear attention (GatedDeltaNet) layers.

UPDATED: Now uses simplified conversion without weight fusion, since the HF architecture
matches FLA directly (separate Q/K/V projections and convolutions).

Usage:
    python convert_checkpoint_to_hf_hybrid.py \
        -i /path/to/olmo-core-checkpoint \
        -o /path/to/output-hf-model \
        --skip-validation
"""

import json
import logging
import re
import sys
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple

import rich
import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn.functional as F
from cached_path import cached_path
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from olmo_core.aliases import PathOrStr
from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.io import file_exists, join_path
from olmo_core.nn.attention import AttentionBackendName, AttentionType
from olmo_core.nn.conversion.state_converter import StateConverter
from olmo_core.nn.conversion.state_mapping import (
    StateMapping,
    StateMappingTemplate,
    StateType,
    TemplatePlaceholder,
)
from olmo_core.nn.hf.convert import (
    OLMO_CORE_TO_HF_MODULE_MAPPINGS,
    OLMO_CORE_TO_HF_WEIGHT_MAPPINGS,
    OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS,
    get_converter_to_hf,
)
from olmo_core.nn.moe.moe import MoEType
from olmo_core.nn.transformer.block import FLABlock, ReorderedNormTransformerBlock
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig
from olmo_core.nn.transformer.model import Transformer
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)

LAYER = TemplatePlaceholder.LAYER

FLA_OLMO_CORE_TO_HF_WEIGHT_MAPPINGS: Dict[str, str] = {
    f"blocks.{LAYER}.fla.inner.q_proj.weight": f"model.layers.{LAYER}.linear_attn.q_proj.weight",
    f"blocks.{LAYER}.fla.inner.k_proj.weight": f"model.layers.{LAYER}.linear_attn.k_proj.weight",
    f"blocks.{LAYER}.fla.inner.v_proj.weight": f"model.layers.{LAYER}.linear_attn.v_proj.weight",
    f"blocks.{LAYER}.fla.inner.g_proj.weight": f"model.layers.{LAYER}.linear_attn.g_proj.weight",
    f"blocks.{LAYER}.fla.inner.a_proj.weight": f"model.layers.{LAYER}.linear_attn.a_proj.weight",
    f"blocks.{LAYER}.fla.inner.b_proj.weight": f"model.layers.{LAYER}.linear_attn.b_proj.weight",
    f"blocks.{LAYER}.fla.inner.o_proj.weight": f"model.layers.{LAYER}.linear_attn.o_proj.weight",
    f"blocks.{LAYER}.fla.inner.q_conv1d.weight": f"model.layers.{LAYER}.linear_attn.q_conv1d.weight",
    f"blocks.{LAYER}.fla.inner.k_conv1d.weight": f"model.layers.{LAYER}.linear_attn.k_conv1d.weight",
    f"blocks.{LAYER}.fla.inner.v_conv1d.weight": f"model.layers.{LAYER}.linear_attn.v_conv1d.weight",
    f"blocks.{LAYER}.fla.inner.o_norm.weight": f"model.layers.{LAYER}.linear_attn.o_norm.weight",
    f"blocks.{LAYER}.fla.inner.A_log": f"model.layers.{LAYER}.linear_attn.A_log",
    f"blocks.{LAYER}.fla.inner.dt_bias": f"model.layers.{LAYER}.linear_attn.dt_bias",
    f"blocks.{LAYER}.fla_norm.weight": f"model.layers.{LAYER}.post_attention_layernorm.weight",
}

FLA_OLMO_CORE_TO_HF_MODULE_MAPPINGS: Dict[str, str] = {
    f"blocks.{LAYER}.fla": f"model.layers.{LAYER}.linear_attn",
    f"blocks.{LAYER}.fla.inner": f"model.layers.{LAYER}.linear_attn",
    f"blocks.{LAYER}.fla_norm": f"model.layers.{LAYER}.post_attention_layernorm",
}


def is_hybrid_model(model: Transformer) -> bool:
    """Check if the model has FLA (linear attention) blocks."""
    blocks = list(model.blocks.values())
    return any(isinstance(block, FLABlock) for block in blocks)


def get_hybrid_hf_config_dict(model: Transformer, transformer_config: TransformerConfig) -> Dict[str, Any]:
    """
    Build a config dict for Olmo3_5Hybrid model.
    
    Returns a dict that can be saved as config.json directly, without needing
    to import the Olmo3_5HybridConfig class from transformers.
    """
    blocks = list(model.blocks.values())
    n_layers = len(blocks)

    # Determine layer types and find reference blocks
    layer_types = []
    attention_block = None
    fla_block = None

    for idx, block in enumerate(blocks):
        if isinstance(block, FLABlock):
            layer_types.append("linear_attention")
            if fla_block is None:
                fla_block = block
        elif isinstance(block, ReorderedNormTransformerBlock):
            if hasattr(block.attention, 'backend') and block.attention.backend is not None:
                if block.attention.backend.window_size != (-1, -1):
                    layer_types.append("sliding_attention")
                else:
                    layer_types.append("full_attention")
            else:
                layer_types.append("full_attention")
            if attention_block is None:
                attention_block = block
        else:
            raise ValueError(f"Unknown block type at layer {idx}: {type(block)}")

    if attention_block is None:
        raise ValueError("Hybrid model must have at least one attention layer")
    if fla_block is None:
        raise ValueError("Hybrid model must have at least one FLA layer")

    # Extract attention config
    attn = attention_block.attention
    if attn.rope is None:
        raise ValueError("Attention does not use RoPE")

    # Extract FLA config
    fla_inner = fla_block.fla.inner

    # Get sliding window size if any
    sliding_window = 4096  # Default
    for block in blocks:
        if isinstance(block, ReorderedNormTransformerBlock):
            if hasattr(block.attention, 'backend') and block.attention.backend is not None:
                if block.attention.backend.window_size != (-1, -1):
                    sliding_window = block.attention.backend.window_size[0] + 1
                    break

    # Get FLA layer kwargs from original config
    block_config = transformer_config.block
    fla_config = block_config.fla
    fla_kwargs = fla_config.fla_layer_kwargs if fla_config else {}

    # Extract dimensions from the actual FLA layer
    linear_num_heads = getattr(fla_inner, 'num_heads', attn.n_heads)
    linear_key_head_dim = fla_kwargs.get('head_dim', getattr(fla_inner, 'head_k_dim', 96))
    linear_value_head_dim = getattr(fla_inner, 'head_v_dim', linear_key_head_dim * 2)

    # Build the config dict - this matches what Olmo3_5HybridConfig expects
    config_dict = {
        # Required field for HF to know which model class to use
        "model_type": "olmo3_5_hybrid",
        "architectures": ["Olmo3_5HybridForCausalLM"],
        
        # Standard transformer config
        "vocab_size": model.vocab_size,
        "hidden_size": model.d_model,
        "intermediate_size": attention_block.feed_forward.hidden_size,
        "num_hidden_layers": n_layers,
        "num_attention_heads": attn.n_heads,
        "num_key_value_heads": attn.n_kv_heads or attn.n_heads,
        "hidden_act": "silu",
        "max_position_embeddings": 65536,
        "initializer_range": 0.02,
        "use_cache": True,
        "attention_bias": attn.w_out.bias is not None,
        "attention_dropout": 0.0,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": float(attn.rope.theta),
        },
        "rms_norm_eps": attention_block.feed_forward_norm.eps,
        "tie_word_embeddings": False,
        
        # Sliding window
        "sliding_window": sliding_window,
        
        # Hybrid layer configuration
        "layer_types": layer_types,
        "fla_hybrid_attention_indices": [
            i for i, t in enumerate(layer_types) if t in {"full_attention", "sliding_attention"}
        ],
        
        # Linear attention (GatedDeltaNet) parameters
        "linear_num_key_heads": linear_num_heads,
        "linear_num_value_heads": linear_num_heads,
        "linear_key_head_dim": linear_key_head_dim,
        "linear_value_head_dim": linear_value_head_dim,
        "linear_conv_kernel_dim": 4,
        "linear_use_gate": fla_kwargs.get('use_gate', True),
        "linear_allow_neg_eigval": fla_kwargs.get('allow_neg_eigval', True),
        
        # Token IDs (will be updated later)
        "pad_token_id": None,
        "bos_token_id": None,
        "eos_token_id": None,
        
        # Transformers version info
        "transformers_version": "4.52.0",
    }

    return config_dict


def get_hybrid_converter_to_hf() -> StateConverter:
    """
    Get a state converter that handles both attention and FLA layers.
    """
    # Start with standard mappings
    mapping_templates = {
        olmo_core_key: StateMappingTemplate(olmo_core_key, hf_key, state_type=StateType.module)
        for olmo_core_key, hf_key in OLMO_CORE_TO_HF_MODULE_MAPPINGS.items()
    }
    mapping_templates.update(
        {
            olmo_core_key: StateMappingTemplate(olmo_core_key, hf_key, state_type=StateType.weight)
            for olmo_core_key, hf_key in OLMO_CORE_TO_HF_WEIGHT_MAPPINGS.items()
        }
    )
    mapping_templates.update(OLMO_CORE_TO_HF_TEMPLATE_MAPPINGS)

    # Add FLA-specific mappings
    mapping_templates.update(
        {
            olmo_core_key: StateMappingTemplate(olmo_core_key, hf_key, state_type=StateType.weight)
            for olmo_core_key, hf_key in FLA_OLMO_CORE_TO_HF_WEIGHT_MAPPINGS.items()
        }
    )
    mapping_templates.update(
        {
            olmo_core_key: StateMappingTemplate(olmo_core_key, hf_key, state_type=StateType.module)
            for olmo_core_key, hf_key in FLA_OLMO_CORE_TO_HF_MODULE_MAPPINGS.items()
        }
    )

    return StateConverter(list(mapping_templates.values()))


def convert_hybrid_state_to_hf(
    model_state_dict: Dict[str, Any],
    n_layers: int,
) -> Dict[str, Any]:
    """
    Convert OLMo Core hybrid model state dict to HuggingFace format.
    """
    converter = get_hybrid_converter_to_hf()

    placeholder_bounds = {
        TemplatePlaceholder.LAYER: n_layers,
    }

    return converter.convert(model_state_dict, placeholder_bounds)


def save_hybrid_hf_model(
    output_path: str | Path,
    model_state_dict: Dict[str, Any],
    model: Transformer,
    transformer_config: TransformerConfig,
    *,
    dtype: Optional[DType] = None,
    vocab_size: Optional[int] = None,
) -> None:
    """
    Save a hybrid model in HuggingFace format.
    
    UPDATED: Simplified version without weight fusion. The HF architecture now
    matches FLA directly, so we only need key renaming.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    hf_config_dict = get_hybrid_hf_config_dict(model, transformer_config)
    
    n_layers = len(list(model.blocks.values()))
    hf_state_dict = convert_hybrid_state_to_hf(model_state_dict, n_layers)
    

    if dtype is not None:
        hf_state_dict = {
            k: v.to(dtype.as_pt()) if torch.is_tensor(v) else v 
            for k, v in hf_state_dict.items()
        }

    # Truncate embeddings if vocab_size specified
    if vocab_size is not None:
        hf_config_dict["vocab_size"] = vocab_size
        if "model.embed_tokens.weight" in hf_state_dict:
            hf_state_dict["model.embed_tokens.weight"] = hf_state_dict["model.embed_tokens.weight"][:vocab_size]
        if "lm_head.weight" in hf_state_dict:
            hf_state_dict["lm_head.weight"] = hf_state_dict["lm_head.weight"][:vocab_size]

    # Log info
    original_keys = set(model_state_dict.keys())
    converted_keys = set(hf_state_dict.keys())
    log.info(f"Original state dict has {len(original_keys)} keys")
    log.info(f"Converted state dict has {len(converted_keys)} keys")
    
    # Debug: print some converted keys to verify mapping
    log.info("Sample converted keys for layer 0:")
    for k in sorted(converted_keys):
        if 'layers.0.' in k:
            log.info(f"  {k}")

    # Save config as JSON directly (no need for HF config class)
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(hf_config_dict, f, indent=2)
    log.info(f"Saved config to {config_path}")

    save_file(hf_state_dict, output_path / "model.safetensors")
    log.info(f"Saved weights to {output_path / 'model.safetensors'}")


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
    debug: bool = False,
    device: torch.device | None = None,
    moe_capacity_factor: float | None = None,
    validation_device: torch.device | None = None,
    validation_sliding_window: int | None = None,
) -> None:
    """
    Convert a checkpoint to HuggingFace format.
    Supports both standard OLMo models and hybrid models with FLA layers.
    """
    if max_sequence_length is not None and max_sequence_length <= 0:
        raise ValueError(f"Invalid sequence length: {max_sequence_length}")

    # Remove deprecated config options
    for key in ["compile", "dp_config", "tp_config", "float8_config"]:
        transformer_config_dict.pop(key, None)

    model_config = TransformerConfig.from_dict(transformer_config_dict)
    rich.print(model_config)

    validation_device = validation_device or torch.device("cpu")

    # Detect hybrid model
    is_hybrid = (
        model_config.block.name == "fla_hybrid"
        or model_config.block.fla is not None
        or model_config.block.fla_hybrid_attention_indices is not None
    )

    if is_hybrid:
        log.info("Detected hybrid model with FLA layers")

    # Prepare blocks for conversion
    block_entries: list[tuple[str, TransformerBlockConfig]] = [("base block", model_config.block)]
    if model_config.block_overrides:
        block_entries.extend(
            (f"block override {idx}", block_config)
            for idx, block_config in sorted(model_config.block_overrides.items())
        )

    for block_label, block_config in block_entries:
        # Skip attention prep for FLA-only blocks
        if block_config.attention is None:
            continue

        attention_config = block_config.attention
        if attention_config.name == AttentionType.fused:
            backend = attention_config.backend
            if backend is None:
                assert attention_config.use_flash
                backend = AttentionBackendName.flash_2

            try:
                backend.assert_supported()
                log.info(f"Using GPU and {backend} for {block_label}")
                device = torch.device("cuda")
                validation_device = torch.device("cuda")
                attention_config.backend = backend
            except RuntimeError as e:
                raise RuntimeError(f"Flash attention required but not supported: {e}")

        elif validate and attention_config.backend != AttentionBackendName.torch:
            log.info(f"Overriding attention backend to torch for {block_label}")
            attention_config.backend = AttentionBackendName.torch
            attention_config.use_flash = False

        if moe_capacity_factor is not None and block_config.feed_forward_moe is not None:
            block_config.feed_forward_moe.capacity_factor = moe_capacity_factor

    # Build model
    model = model_config.build(init_device="meta")
    model.to_empty(device=device or torch.device("cpu"))

    tokenizer_config = TokenizerConfig.from_dict(tokenizer_config_dict)
    vocab_size = tokenizer_config.vocab_size

    with TemporaryDirectory() as work_dir:
        model_and_optim_dir = join_path(original_checkpoint_path, "model_and_optim")
        log.info(f"Loading checkpoint from '{model_and_optim_dir}'")
        load_model_and_optim_state(model_and_optim_dir, model, work_dir=work_dir)
        
        log.info(f"Saving checkpoint to '{output_path}'")
        state_dict_options = dist_cp_sd.StateDictOptions(
            flatten_optimizer_state_dict=True, cpu_offload=True
        )
        model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)

        # Handle MoE reshaping if needed
        if (moe_config := model_config.block.feed_forward_moe) is not None:
            if moe_config.name == MoEType.dropless:
                for k, v in model_state_dict.items():
                    if k.endswith(".feed_forward_moe.experts.mlp.w1") or k.endswith(
                        ".feed_forward_moe.experts.mlp.w3"
                    ):
                        assert isinstance(v, torch.Tensor)
                        model_state_dict[k] = (
                            v.reshape(moe_config.num_experts, moe_config.hidden_size, -1)
                            .permute(0, 2, 1)
                            .reshape(-1, moe_config.hidden_size)
                        )
                        log.info(f"Reshaped {k} for dropless MoE")

        if is_hybrid:
            save_hybrid_hf_model(
                output_path,
                model_state_dict,
                model,
                model_config,
                dtype=dtype,
                vocab_size=None,
            )
        else:
            # Use standard conversion
            from olmo_core.nn.hf.checkpoint import save_hf_model
            save_hf_model(
                output_path,
                model_state_dict,
                model,
                dtype=dtype,
                vocab_size=vocab_size,
                work_dir=work_dir,
                save_overwrite=True,
            )
        
        log.info(f"Successfully saved converted model to '{output_path}'")

    # Save tokenizer
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

    # Update config with tokenizer info
    log.info("Updating config with tokenizer settings")
    hf_config_path = Path(output_path) / "config.json"
    with open(hf_config_path, "r") as f:
        config_dict = json.load(f)
    
    config_dict["max_position_embeddings"] = max_sequence_length or config_dict.get("max_position_embeddings", 65536)
    config_dict["pad_token_id"] = tokenizer_config.pad_token_id
    config_dict["bos_token_id"] = tokenizer_config.bos_token_id
    config_dict["eos_token_id"] = tokenizer_config.eos_token_id
    
    with open(hf_config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    log.info("Updated config.json")

    # Validation
    if validate:
        if is_hybrid:
            log.warning("Validation for hybrid models not yet implemented. Skipping.")
        else:
            log.info("Validating converted model")
            validate_conversion(
                output_path,
                model,
                tokenizer_config.vocab_size,
                debug=debug,
                dtype=dtype,
                device=validation_device,
                sliding_window=validation_sliding_window,
            )
            log.info("Validation completed")


def validate_conversion(
    hf_path: str | Path,
    model: Transformer,
    vocab_size: int,
    debug: bool = False,
    dtype: DType | None = None,
    device: torch.device | None = None,
    sliding_window: int | None = None,
):
    """Validate the conversion by comparing outputs."""
    device = device or torch.device("cpu")
    log.info(f"Running validation on {device}")

    B, T = 1, 60
    input_ids = torch.randint(0, vocab_size, (B, T)).to(device)

    is_sliding = any(
        hasattr(block, "attention") and 
        hasattr(block.attention, "window_size") and 
        block.attention.window_size != (-1, -1)
        for block in model.blocks.values()
    )

    log.info("Loading converted checkpoint for validation...")
    kwargs = {}
    if is_sliding and sliding_window is not None:
        kwargs["sliding_window"] = sliding_window
    
    hf_config = AutoConfig.from_pretrained(hf_path, **kwargs)
    hf_model = (
        AutoModelForCausalLM.from_pretrained(
            hf_path,
            torch_dtype="auto",
            config=hf_config,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )

    log.info("Running models for validation...")
    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids).logits

    del hf_model

    if is_sliding and sliding_window is not None:
        for block in model.blocks.values():
            if hasattr(block, "attention") and block.attention.window_size != (-1, -1):
                block.attention.window_size = (sliding_window - 1, 0)
    
    if dtype:
        model = model.to(dtype.as_pt())
    model = model.to(device=device).eval()
    
    with torch.no_grad():
        logits = model(input_ids=input_ids)

    torch.testing.assert_close(
        hf_logits[..., :vocab_size].float(), 
        logits[..., :vocab_size].float(), 
        rtol=1e-4, 
        atol=1e-4
    )


def load_config(checkpoint_input_dir: PathOrStr) -> Optional[dict]:
    config_path = f"{checkpoint_input_dir}/config.json"
    if not file_exists(config_path):
        raise RuntimeError(f"Config file not found at {checkpoint_input_dir}")

    with cached_path(config_path).open("r", encoding="utf-8") as f:
        config_dict = json.load(f)

    if "model" not in config_dict:
        raise RuntimeError(f"Config at {checkpoint_input_dir} is not an OLMo core experiment config")

    return config_dict


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--checkpoint-input-path",
        type=str, required=True,
        help="Path to OLMo Core checkpoint directory.",
    )
    parser.add_argument(
        "-o", "--huggingface-output-dir",
        type=str, required=True,
        help="Output path for HuggingFace model.",
    )
    parser.add_argument(
        "-s", "--max-sequence-length",
        type=int,
        help="Max sequence length. Defaults to tokenizer's model_max_length.",
    )
    parser.add_argument(
        "-t", "--tokenizer",
        help="HuggingFace tokenizer identifier.",
    )
    parser.add_argument(
        "--skip-validation",
        dest="validate", action="store_false",
        help="Skip validation.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output.",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        help="Device for conversion. Defaults to CPU.",
    )
    parser.add_argument(
        "--dtype",
        type=DType, default=DType.bfloat16,
        help="Output dtype. Defaults to bfloat16.",
    )
    parser.add_argument(
        "--validation-device",
        type=torch.device,
        help="Device for validation.",
    )
    parser.add_argument(
        "--validation-sliding-window",
        type=int,
        help="Override sliding window size for validation.",
    )
    parser.add_argument(
        "--moe-capacity-factor",
        type=float,
        help="MoE capacity factor.",
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

    convert_checkpoint_to_hf(
        original_checkpoint_path=args.checkpoint_input_path,
        output_path=args.huggingface_output_dir,
        transformer_config_dict=transformer_config_dict,
        tokenizer_config_dict=tokenizer_config_dict,
        dtype=args.dtype,
        max_sequence_length=args.max_sequence_length,
        tokenizer_id=args.tokenizer,
        validate=args.validate,
        debug=args.debug,
        device=args.device,
        moe_capacity_factor=args.moe_capacity_factor,
        validation_device=args.validation_device or args.device,
        validation_sliding_window=args.validation_sliding_window,
    )


if __name__ == "__main__":
    prepare_cli_environment()
    main()