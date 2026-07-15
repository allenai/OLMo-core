"""Config-construction tests for the Qwen3-MoE variant builders."""

from olmo_core.nn.moe.v2.qwen import (
    QWEN3_MOE_LAYER_PATTERN,
    build_debug_qwen3_moe_config,
    build_qwen3_moe_config,
    build_qwen3_moe_config_from_hf_config,
)
from olmo_core.nn.transformer.config import OLMoDDPModelConfig


def test_build_debug_qwen3_moe_config_uses_interleaved_layer_pattern():
    config = build_debug_qwen3_moe_config(vocab_size=128, n_layers=4)
    assert isinstance(config, OLMoDDPModelConfig)
    assert config.n_layers == 4
    # 4 distinct layer_types (3x linear + 1 full) -> dict of blocks with an explicit pattern.
    assert config.block_pattern == list(QWEN3_MOE_LAYER_PATTERN)
    assert isinstance(config.block, dict)
    assert set(config.block) == {"linear_attention", "full_attention"}


def test_build_qwen3_moe_config_dense_only_collapses_to_single_block():
    config = build_qwen3_moe_config(
        n_layers=2,
        layer_types=("full_attention",),
        vocab_size=128,
        d_model=256,
        num_attention_heads=4,
        num_key_value_heads=1,
        attention_head_dim=64,
        attention_rotary_dim=32,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=None,
    )
    # A single layer type -> one block config, no block_pattern.
    assert config.block_pattern is None
    assert not isinstance(config.block, dict)
    # rotary_dim / head_dim = 32 / 64 -> partial_rotary_factor 0.5.
    assert config.block.sequence_mixer.rope.partial_rotary_factor == 0.5  # type: ignore[union-attr]


def test_build_qwen3_moe_config_from_hf_config_maps_partial_rotary():
    hf_config = {
        "vocab_size": 128,
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "partial_rotary_factor": 0.25,
        "rope_theta": 5_000_000,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 128,
        "rms_norm_eps": 1e-6,
    }
    config = build_qwen3_moe_config_from_hf_config(hf_config)
    assert config.n_layers == 2
    # head_dim=64, partial_rotary_factor=0.25 -> attention_rotary_dim=16 -> factor 16/64 = 0.25.
    assert config.block.sequence_mixer.rope.partial_rotary_factor == 0.25  # type: ignore[union-attr]
    assert config.block.sequence_mixer.rope.theta == 5_000_000  # type: ignore[union-attr]
