"""
Tests for the fan_in InitMethod.
"""

import math

import pytest
import torch

from olmo_core.nn.attention import AttentionConfig, GateConfig, GateGranularity
from olmo_core.nn.attention.flash_linear_attn_api import has_fla as _has_fla
from olmo_core.nn.attention.recurrent import GatedDeltaNetConfig
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


@pytest.mark.skipif(not _has_fla(), reason="fla not installed")
def test_fan_in_init_raises_for_gdn():
    """Test that fan_in init raises NotImplementedError for GatedDeltaNet."""
    d_model = 256
    config = TransformerConfig(
        d_model=d_model,
        vocab_size=1000,
        n_layers=2,
        init_method=InitMethod.fan_in,
        block=TransformerBlockConfig(
            sequence_mixer=GatedDeltaNetConfig(n_heads=d_model // 64),
            feed_forward=FeedForwardConfig(hidden_size=d_model * 4, bias=False),
            layer_norm=LayerNormConfig(name=LayerNormType.rms, bias=False),
        ),
        lm_head=LMHeadConfig(bias=False),
    )
    model = config.build(init_device="cpu")
    with pytest.raises(NotImplementedError, match="fan_in.*not supported.*GatedDeltaNet"):
        model.init_weights(device=torch.device("cpu"))


class _FakePPMesh:
    def __init__(self, local_rank: int, size: int):
        self._local_rank = local_rank
        self._size = size

    def get_local_rank(self) -> int:
        return self._local_rank

    def size(self) -> int:
        return self._size


def _build_tiny_model(init_seed: int = 42):
    return TransformerConfig.llama_like(
        d_model=64,
        vocab_size=128,
        n_layers=2,
        n_heads=2,
        feed_forward=FeedForwardConfig(hidden_size=128, bias=False),
        init_seed=init_seed,
    ).build(init_device="meta")


def test_init_weights_diversifies_seed_across_model_parts(monkeypatch):
    """
    Under interleaved pipeline parallelism a single rank can own multiple model
    chunks. They must each get a distinct init seed, or their parameters would
    collide.
    """
    fake_pp_mesh = _FakePPMesh(local_rank=0, size=2)
    monkeypatch.setattr(
        "olmo_core.nn.transformer.model.get_pp_mesh",
        lambda _world_mesh: fake_pp_mesh,
    )

    part_a = _build_tiny_model()
    part_a._pp_enabled = True
    part_a.init_weights(device=torch.device("cpu"), world_mesh=object(), model_part_idx=0)

    part_b = _build_tiny_model()
    part_b._pp_enabled = True
    part_b.init_weights(device=torch.device("cpu"), world_mesh=object(), model_part_idx=1)

    assert not torch.equal(part_a.embeddings.weight, part_b.embeddings.weight)
    assert not torch.equal(
        part_a.blocks["0"].attention.w_q.weight,
        part_b.blocks["0"].attention.w_q.weight,
    )


def test_init_weights_same_model_part_idx_is_deterministic(monkeypatch):
    """Two chunks with the same (pp_rank, model_part_idx) must init identically."""
    fake_pp_mesh = _FakePPMesh(local_rank=0, size=2)
    monkeypatch.setattr(
        "olmo_core.nn.transformer.model.get_pp_mesh",
        lambda _world_mesh: fake_pp_mesh,
    )

    part_a = _build_tiny_model()
    part_a._pp_enabled = True
    part_a.init_weights(device=torch.device("cpu"), world_mesh=object(), model_part_idx=1)

    part_b = _build_tiny_model()
    part_b._pp_enabled = True
    part_b.init_weights(device=torch.device("cpu"), world_mesh=object(), model_part_idx=1)

    assert torch.equal(part_a.embeddings.weight, part_b.embeddings.weight)
    assert torch.equal(
        part_a.blocks["0"].attention.w_q.weight,
        part_b.blocks["0"].attention.w_q.weight,
    )


def test_init_weights_model_part_idx_ignored_without_pp():
    """
    Without PP enabled, ``model_part_idx`` must not affect initialization —
    existing non-PP training runs must produce identical RNG streams to before.
    """
    part_a = _build_tiny_model()
    part_a.init_weights(device=torch.device("cpu"), model_part_idx=0)

    part_b = _build_tiny_model()
    part_b.init_weights(device=torch.device("cpu"), model_part_idx=7)

    assert torch.equal(part_a.embeddings.weight, part_b.embeddings.weight)
    assert torch.equal(
        part_a.blocks["0"].attention.w_q.weight,
        part_b.blocks["0"].attention.w_q.weight,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
