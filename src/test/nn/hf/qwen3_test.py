import os

import pytest
import torch
import transformers

from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.transformer import TransformerConfig


def test_qwen3_has_head_qk_norm():
    config = TransformerConfig.qwen3_8B(vocab_size=1024, n_layers=2)
    assert config.block.attention.qk_norm is not None
    assert config.block.attention.use_head_qk_norm is True
    assert config.block.attention.head_dim == 128

    model = config.build(init_device="cpu")

    head_dim = config.block.attention.head_dim
    assert model.blocks["0"].attention.q_norm.weight.shape == (head_dim,)
    assert model.blocks["0"].attention.k_norm.weight.shape == (head_dim,)


def test_qwen3_conversion_mappings():
    hf_config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        rope_theta=1_000_000,
        rms_norm_eps=1e-6,
    )

    hf_state = {}
    for i in range(hf_config.num_hidden_layers):
        hf_state[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.randn(
            hf_config.hidden_size
        )

    converted_state = convert_state_from_hf(hf_config, hf_state, model_type="qwen3")

    for i in range(hf_config.num_hidden_layers):
        torch.testing.assert_close(
            converted_state[f"blocks.{i}.feed_forward_norm.weight"],
            hf_state[f"model.layers.{i}.post_attention_layernorm.weight"],
        )
