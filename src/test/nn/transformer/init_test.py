"""
Tests for the fan_in InitMethod.
"""

import math

import pytest
import torch
import torch.nn as nn

from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.moe import MoEConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.init import InitMethod


@pytest.mark.parametrize(
    "init_device, device",
    [
        pytest.param("cpu", "cpu", id="cpu->cpu"),
    ],
)
def test_fan_in_init_embeddings(init_device, device):
    """Test that fan_in init uses std=1.0 for embeddings."""
    config = TransformerConfig.llama2_271M(vocab_size=50257, init_method=InitMethod.fan_in)
    model = config.build(init_device=init_device)
    model.init_weights(device=torch.device(device))

    # Check embedding initialization - should have std close to 1.0
    embedding_std = model.embeddings.weight.std().item()
    # Allow some tolerance since it's a random initialization
    assert abs(embedding_std - 1.0) < 0.1, f"Expected embedding std ~1.0, got {embedding_std}"


@pytest.mark.parametrize(
    "init_device, device",
    [
        pytest.param("cpu", "cpu", id="cpu->cpu"),
    ],
)
def test_fan_in_init_attention(init_device, device):
    """Test that fan_in init uses 1/√d_model for attention weights."""
    d_model = 512
    config = TransformerConfig.llama2_271M(
        vocab_size=50257, d_model=d_model, init_method=InitMethod.fan_in
    )
    model = config.build(init_device=init_device)
    model.init_weights(device=torch.device(device))

    expected_std = 1.0 / math.sqrt(d_model)

    # Check first block's attention weights
    block = model.blocks["0"]
    if hasattr(block.attention, "w_q"):
        # Non-fused attention
        q_std = block.attention.w_q.weight.std().item()
        k_std = block.attention.w_k.weight.std().item()
        v_std = block.attention.w_v.weight.std().item()
        out_std = block.attention.w_out.weight.std().item()

        # Allow 20% tolerance
        tolerance = expected_std * 0.2
        assert abs(q_std - expected_std) < tolerance, f"Q std: expected ~{expected_std}, got {q_std}"
        assert abs(k_std - expected_std) < tolerance, f"K std: expected ~{expected_std}, got {k_std}"
        assert abs(v_std - expected_std) < tolerance, f"V std: expected ~{expected_std}, got {v_std}"
        assert (
            abs(out_std - expected_std) < tolerance
        ), f"Out std: expected ~{expected_std}, got {out_std}"


@pytest.mark.parametrize(
    "init_device, device",
    [
        pytest.param("cpu", "cpu", id="cpu->cpu"),
    ],
)
def test_fan_in_init_feed_forward(init_device, device):
    """Test that fan_in init uses correct std for feed-forward weights."""
    d_model = 512
    hidden_size = 2048
    config = TransformerConfig.llama2_271M(
        vocab_size=50257,
        d_model=d_model,
        mlp_hidden_size=hidden_size,
        init_method=InitMethod.fan_in,
    )
    model = config.build(init_device=init_device)
    model.init_weights(device=torch.device(device))

    # w1 and w3 should use 1/√d_model (d_in = d_model)
    expected_std_w1 = 1.0 / math.sqrt(d_model)
    # w2 should use 1/√hidden_size (d_in = hidden_size)
    expected_std_w2 = 1.0 / math.sqrt(hidden_size)

    # Check first block's feed-forward weights
    block = model.blocks["0"]
    w1_std = block.feed_forward.w1.weight.std().item()
    w2_std = block.feed_forward.w2.weight.std().item()
    w3_std = block.feed_forward.w3.weight.std().item()

    # Allow 20% tolerance
    tolerance_w1 = expected_std_w1 * 0.2
    tolerance_w2 = expected_std_w2 * 0.2

    assert (
        abs(w1_std - expected_std_w1) < tolerance_w1
    ), f"w1 std: expected ~{expected_std_w1}, got {w1_std}"
    assert (
        abs(w2_std - expected_std_w2) < tolerance_w2
    ), f"w2 std: expected ~{expected_std_w2}, got {w2_std}"
    assert (
        abs(w3_std - expected_std_w1) < tolerance_w1
    ), f"w3 std: expected ~{expected_std_w1}, got {w3_std}"


@pytest.mark.parametrize(
    "init_device, device",
    [
        pytest.param("cpu", "cpu", id="cpu->cpu"),
    ],
)
def test_fan_in_init_final_w_out(init_device, device):
    """Test that fan_in init uses 1/√d_model for final LM head."""
    d_model = 512
    config = TransformerConfig.llama2_271M(
        vocab_size=50257, d_model=d_model, init_method=InitMethod.fan_in
    )
    model = config.build(init_device=init_device)
    model.init_weights(device=torch.device(device))

    expected_std = 1.0 / math.sqrt(d_model)

    # Check LM head weight
    lm_head_std = model.lm_head.weight.std().item()

    # Allow 20% tolerance
    tolerance = expected_std * 0.2
    assert (
        abs(lm_head_std - expected_std) < tolerance
    ), f"LM head std: expected ~{expected_std}, got {lm_head_std}"


@pytest.mark.parametrize(
    "init_device, device",
    [
        pytest.param("cpu", "cpu", id="cpu->cpu"),
    ],
)
def test_fan_in_init_different_from_normal(init_device, device):
    """Test that fan_in init produces different results from normal init."""
    # Build two identical models with different init methods
    config_fan_in = TransformerConfig.llama2_271M(vocab_size=50257, init_method=InitMethod.fan_in)
    config_normal = TransformerConfig.llama2_271M(vocab_size=50257, init_method=InitMethod.normal)

    model_fan_in = config_fan_in.build(init_device=init_device)
    model_normal = config_normal.build(init_device=init_device)

    # Use different seeds to ensure they're initialized differently
    torch.manual_seed(42)
    model_fan_in.init_weights(device=torch.device(device))

    torch.manual_seed(42)
    model_normal.init_weights(device=torch.device(device))

    # Check that feed-forward w2 has different std (most obvious difference)
    fan_in_w2_std = model_fan_in.blocks["0"].feed_forward.w2.weight.std().item()
    normal_w2_std = model_normal.blocks["0"].feed_forward.w2.weight.std().item()

    # These should be significantly different
    assert abs(fan_in_w2_std - normal_w2_std) > 0.01, (
        f"fan_in and normal init should produce different stds for w2, "
        f"but got fan_in={fan_in_w2_std}, normal={normal_w2_std}"
    )


def test_fan_in_init_moe():
    """Test that fan_in init works with MoE layers."""
    d_model = 256
    hidden_size = 512
    num_experts = 8

    # Create a simple MoE config
    config = TransformerConfig(
        d_model=d_model,
        n_heads=4,
        n_layers=2,
        vocab_size=1000,
        mlp_hidden_size=hidden_size,
        init_method=InitMethod.fan_in,
        moe=MoEConfig(num_experts=num_experts, hidden_size=hidden_size),
    )

    model = config.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))

    # Check that the model was initialized without errors
    # and that MoE weights have reasonable stds
    block = model.blocks["0"]
    if hasattr(block, "moe"):
        # Router should use 1/√d_model
        expected_router_std = 1.0 / math.sqrt(d_model)
        router_weight = block.moe.router.weight
        router_std = router_weight.std().item()

        # Allow 30% tolerance for MoE (more parameters, more variance)
        tolerance = expected_router_std * 0.3
        assert (
            abs(router_std - expected_router_std) < tolerance
        ), f"Router std: expected ~{expected_router_std}, got {router_std}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
