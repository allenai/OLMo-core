import os

import pytest
import torch
import transformers

from olmo_core.nn.hf.convert import convert_state_from_hf
from olmo_core.nn.transformer import TransformerConfig


def test_qwen3_forward_pass():
    config = TransformerConfig.qwen3_0_6B(vocab_size=1024, n_layers=2)
    model = config.build(init_device="cpu")
    model.eval()

    input_ids = torch.randint(0, 1024, (2, 16))
    with torch.no_grad():
        logits = model(input_ids)

    assert logits.shape == (2, 16, 1024)


def test_qwen3_has_head_qk_norm():
    config = TransformerConfig.qwen3_8B(vocab_size=1024, n_layers=2)
    assert config.block.attention.qk_norm is not None
    assert config.block.attention.use_head_qk_norm is True
    assert config.block.attention.head_dim == 128

    model = config.build(init_device="cpu")

    head_dim = config.block.attention.head_dim
    assert model.blocks["0"].attention.q_norm.weight.shape == (head_dim,)
    assert model.blocks["0"].attention.k_norm.weight.shape == (head_dim,)


@pytest.mark.skipif(
    not hasattr(transformers, "Qwen3Config"), reason="Qwen3Config not available in transformers"
)
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


@pytest.mark.skipif(
    not hasattr(transformers, "Qwen3Config"), reason="Qwen3Config not available in transformers"
)
@pytest.mark.skipif(
    not os.environ.get("HF_TOKEN"),
    reason="HF_TOKEN not set",
)
def test_qwen3_matches_huggingface():
    model_name = "Qwen/Qwen3-0.6B"

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        token=os.environ.get("HF_TOKEN"),
    )
    hf_model.eval()
    hf_config = hf_model.config

    olmo_config = TransformerConfig.qwen3_0_6B(
        vocab_size=hf_config.vocab_size,
    )
    olmo_model = olmo_config.build(init_device="cpu")

    converted_state = convert_state_from_hf(
        hf_config,
        hf_model.state_dict(),
        model_type="qwen3",
    )
    olmo_model.load_state_dict(converted_state)
    olmo_model.eval()

    input_ids = torch.randint(0, 1000, (2, 16))

    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits
        olmo_logits = olmo_model(input_ids)

    torch.testing.assert_close(hf_logits, olmo_logits, rtol=1e-4, atol=1e-4)
