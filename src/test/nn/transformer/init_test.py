"""
Tests for the fan_in InitMethod.
"""

import math

import pytest
import torch

from olmo_core.nn.attention import AttentionConfig, GateConfig, GateGranularity
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe import MoEConfig
from olmo_core.nn.transformer import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
    TransformerType,
)
from olmo_core.nn.transformer.init import InitMethod


@pytest.mark.parametrize("d_model", [256, 512])
@pytest.mark.parametrize(
    "init_device, device",
    [
        pytest.param("cpu", "cpu", id="cpu->cpu"),
        pytest.param("meta", "cpu", id="meta->cpu"),
    ],
)
def test_fan_in_init_dense(d_model, init_device, device):
    """Test that fan_in init uses correct std for embeddings, attention, feed-forward, and LM head."""
    hidden_size = d_model * 4
    config = TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=1000,
        n_layers=4,
        n_heads=d_model // 64,
        feed_forward=FeedForwardConfig(hidden_size=hidden_size, bias=False),
        init_method=InitMethod.fan_in,
    )
    model = config.build(init_device=init_device)
    model.init_weights(device=torch.device(device))

    # Embeddings should have std ~1.0
    embedding_std = model.embeddings.weight.std().item()
    assert abs(embedding_std - 1.0) < 0.1, f"Expected embedding std ~1.0, got {embedding_std}"

    expected_d = 1.0 / math.sqrt(d_model)
    expected_h = 1.0 / math.sqrt(hidden_size)
    tol_d = expected_d * 0.2
    tol_h = expected_h * 0.2

    block = model.blocks["0"]

    # Attention weights (all fan-in = d_model)
    assert hasattr(block.attention, "w_q"), "Expected non-fused attention"
    for name in ("w_q", "w_k", "w_v", "w_out"):
        actual = getattr(block.attention, name).weight.std().item()
        assert (
            abs(actual - expected_d) < tol_d
        ), f"attention.{name} std: expected ~{expected_d:.5f}, got {actual:.5f}"

    # Feed-forward weights (w1/w3 fan-in = d_model, w2 fan-in = hidden_size)
    for name, expected, tol in [
        ("w1", expected_d, tol_d),
        ("w2", expected_h, tol_h),
        ("w3", expected_d, tol_d),
    ]:
        actual = getattr(block.feed_forward, name).weight.std().item()
        assert (
            abs(actual - expected) < tol
        ), f"feed_forward.{name} std: expected ~{expected:.5f}, got {actual:.5f}"

    # LM head (fan-in = d_model)
    lm_head_std = model.lm_head.w_out.weight.std().item()
    assert (
        abs(lm_head_std - expected_d) < tol_d
    ), f"LM head std: expected ~{expected_d:.5f}, got {lm_head_std:.5f}"


@pytest.mark.parametrize("d_model", [256, 512])
@pytest.mark.parametrize(
    "init_device, device",
    [
        pytest.param("cpu", "cpu", id="cpu->cpu"),
        pytest.param("meta", "cpu", id="meta->cpu"),
    ],
)
def test_fan_in_init_moe(d_model, init_device, device):
    """Test that fan_in init works with MoE layers."""
    hidden_size = d_model * 2
    num_experts = 8

    config = TransformerConfig(
        name=TransformerType.moe,
        d_model=d_model,
        n_layers=2,
        vocab_size=1000,
        init_method=InitMethod.fan_in,
        block=TransformerBlockConfig(
            name=TransformerBlockType.moe,
            sequence_mixer=AttentionConfig(n_heads=d_model // 64),
            feed_forward_moe=MoEConfig(num_experts=num_experts, hidden_size=hidden_size),
            layer_norm=LayerNormConfig(name=LayerNormType.rms, bias=False),
        ),
        lm_head=LMHeadConfig(bias=False),
    )

    model = config.build(init_device=init_device)
    model.init_weights(device=torch.device(device))

    block = model.blocks["0"]
    assert hasattr(block, "feed_forward_moe"), "Expected MoE block to have feed_forward_moe"
    moe = block.feed_forward_moe
    mlp = moe.experts.mlp

    expected_d = 1.0 / math.sqrt(d_model)
    expected_h = 1.0 / math.sqrt(hidden_size)

    router_std = moe.router.weight.std().item()
    w1_std = mlp.w1.std().item()
    w2_std = mlp.w2.std().item()
    w3_std = mlp.w3.std().item()

    # 30% tolerance (truncated normal + many experts = more variance)
    for name, actual, expected in [
        ("router", router_std, expected_d),
        ("w1", w1_std, expected_d),
        ("w2", w2_std, expected_h),
        ("w3", w3_std, expected_d),
    ]:
        tolerance = expected * 0.3
        assert (
            abs(actual - expected) < tolerance
        ), f"MoE {name} std: expected ~{expected:.5f}, got {actual:.5f}"


@pytest.mark.parametrize("d_model", [256, 512])
@pytest.mark.parametrize(
    "init_device, device",
    [
        pytest.param("cpu", "cpu", id="cpu->cpu"),
        pytest.param("meta", "cpu", id="meta->cpu"),
    ],
)
@pytest.mark.parametrize(
    "granularity",
    [
        pytest.param(GateGranularity.headwise, id="headwise"),
        pytest.param(GateGranularity.elementwise, id="elementwise"),
    ],
)
def test_fan_in_init_attention_gate(d_model, init_device, device, granularity):
    """Test that fan_in init initializes the attention gate projection."""
    config = TransformerConfig.llama_like(
        d_model=d_model,
        vocab_size=1000,
        n_layers=2,
        n_heads=d_model // 64,
        init_method=InitMethod.fan_in,
        gate=GateConfig(granularity=granularity),
    )
    model = config.build(init_device=init_device)
    model.init_weights(device=torch.device(device))

    expected_std = 1.0 / math.sqrt(d_model)

    block = model.blocks["0"]
    assert block.attention.w_g is not None, "Expected attention gate w_g to be present"

    g_std = block.attention.w_g.weight.std().item()
    tolerance = expected_std * 0.2
    assert abs(g_std - expected_std) < tolerance, f"w_g std: expected ~{expected_std}, got {g_std}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
