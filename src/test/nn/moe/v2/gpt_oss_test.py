"""Config-construction tests for the GPT-OSS variant builders."""

import pytest

from olmo_core.nn.moe.v2.gpt_oss import (
    GPT_OSS_FULL_ATTENTION,
    GPT_OSS_LAYER_PATTERN,
    GPT_OSS_SLIDING_ATTENTION,
    build_debug_gpt_oss_20b_config,
    build_gpt_oss_20b_config,
    build_gpt_oss_20b_config_from_hf_config,
)
from olmo_core.nn.transformer.config import OLMoDDPModelConfig


def test_build_debug_gpt_oss_config_uses_interleaved_layer_pattern():
    config = build_debug_gpt_oss_20b_config(vocab_size=128, n_layers=4)
    assert isinstance(config, OLMoDDPModelConfig)
    assert config.n_layers == 4
    # Two interleaved layer types -> dict of blocks with an explicit repeated pattern.
    assert isinstance(config.block, dict)
    assert set(config.block) == {GPT_OSS_SLIDING_ATTENTION, GPT_OSS_FULL_ATTENTION}
    assert config.block_pattern == list(GPT_OSS_LAYER_PATTERN) * 2

    sliding = config.block[GPT_OSS_SLIDING_ATTENTION].sequence_mixer
    full = config.block[GPT_OSS_FULL_ATTENTION].sequence_mixer
    # Only the sliding layer uses a window; both carry attention sinks.
    assert sliding.sliding_window is not None
    assert full.sliding_window is None
    assert sliding.attention_sinks is True
    assert full.attention_sinks is True


def test_build_gpt_oss_config_single_layer_type_collapses_to_single_block():
    config = build_gpt_oss_20b_config(
        n_layers=2,
        layer_types=(GPT_OSS_FULL_ATTENTION,),
        vocab_size=128,
        d_model=256,
        num_attention_heads=4,
        num_key_value_heads=1,
        attention_head_dim=64,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
    )
    # A single layer type -> one block config, no block_pattern.
    assert config.block_pattern is None
    assert not isinstance(config.block, dict)
    assert config.block.sequence_mixer.attention_sinks is True  # type: ignore[union-attr]
    # GPT-OSS rotates the full head dim.
    assert config.block.sequence_mixer.rope.partial_rotary_factor == 1.0  # type: ignore[union-attr]


def test_build_gpt_oss_config_rejects_unknown_layer_type():
    with pytest.raises(ValueError, match="Unsupported gpt-oss layer_types"):
        build_gpt_oss_20b_config(n_layers=2, layer_types=("mystery_attention",), vocab_size=128)


def test_build_gpt_oss_config_from_hf_config_maps_hyperparameters():
    hf_config = {
        "vocab_size": 128,
        "hidden_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 64,
        "sliding_window": 64,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "intermediate_size": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 150_000,
        "layer_types": (GPT_OSS_SLIDING_ATTENTION, GPT_OSS_FULL_ATTENTION),
    }
    config = build_gpt_oss_20b_config_from_hf_config(hf_config)
    assert config.n_layers == 2
    assert isinstance(config.block, dict)
    assert config.block[GPT_OSS_SLIDING_ATTENTION].sequence_mixer.rope.theta == 150_000
