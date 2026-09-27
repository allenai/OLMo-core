"""Qwen2 bias and normalization mappings round-trip real HF parameter names."""

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from olmo_core.nn.hf.convert import convert_state_from_hf, convert_state_to_hf


def test_qwen2_state_roundtrip_preserves_qkv_bias_and_pre_norms():
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=False,
    )
    state = Qwen2ForCausalLM(config).state_dict()
    # Biases initialize to zero; distinct values expose wrong projection mappings.
    for index, (name, value) in enumerate(state.items()):
        if name.endswith(".bias"):
            value.copy_(torch.arange(value.numel()).reshape(value.shape) + index)
    native = convert_state_from_hf(config, state, model_type="qwen2")
    for layer in range(config.num_hidden_layers):
        for projection in ("q", "k", "v"):
            torch.testing.assert_close(
                native[f"blocks.{layer}.attention.w_{projection}.bias"],
                state[f"model.layers.{layer}.self_attn.{projection}_proj.bias"],
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            native[f"blocks.{layer}.feed_forward_norm.weight"],
            state[f"model.layers.{layer}.post_attention_layernorm.weight"],
        )
        assert f"blocks.{layer}.attention.w_out.bias" not in native
    restored = convert_state_to_hf(config, native)
    assert restored.keys() == state.keys()
    for name, value in state.items():
        torch.testing.assert_close(restored[name], value, rtol=0, atol=0)
