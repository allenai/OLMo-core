"""
Tests that the landmark attention variants support the optional output gate inherited from
:class:`Attention` (``att * sigmoid(w_g(x))`` applied just before ``w_out``), which is what lets
landmark attention drop into gated models like Qwen3.5 while preserving the gated-attention
functionality. Also guards that the *non-gated* path (``gate=None``) is completely unaffected.
"""

import pytest
import torch
import torch.nn as nn

from olmo_core.nn.attention import (
    AttentionConfig,
    AttentionType,
    FastLandmarkAttention,
    GateConfig,
    GateGranularity,
    LandmarkAttention,
    SparseLandmarkAttention,
)
from olmo_core.nn.attention.landmark_kernel import has_landmark_kernel
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.testing import requires_gpu

D_MODEL = 64
N_HEADS = 8
N_KV_HEADS = 2
HEAD_DIM = 8


def _build(
    name: AttentionType,
    *,
    gate: bool,
    granularity: GateGranularity = GateGranularity.elementwise,
    mem_freq: int = 3,
    **extra,
):
    config = AttentionConfig(
        name=name,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        bias=False,
        mem_freq=mem_freq,
        gate=GateConfig(granularity=granularity) if gate else None,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
        rope=RoPEConfig(name=RoPEType.default, theta=10_000),
        **extra,
    )
    return config.build(D_MODEL, layer_idx=0, n_layers=2)


def _gate_reference(attn, x: torch.Tensor) -> torch.Tensor:
    """Independent reference: extract the *ungated* attention output (the tensor fed to ``w_out``) by
    temporarily swapping ``w_out`` for an identity and disabling the gate, then re-apply the gate and
    the real ``w_out`` by hand. This pins down both that the gate is applied in the right place
    (immediately before the output projection) and that the landmark core itself is unchanged.
    """
    w_out, w_g, gate = attn.w_out, attn.w_g, attn.gate
    assert gate is not None and w_g is not None
    attn.w_out = nn.Identity()
    attn.gate = None
    try:
        with torch.no_grad():
            att_flat = attn(x)  # (B, T, n_heads * head_dim)
    finally:
        attn.w_out = w_out
        attn.gate = gate

    B, T, _ = x.shape
    g = torch.sigmoid(w_g(x).float()).to(att_flat.dtype)
    if gate.granularity == GateGranularity.headwise:
        att_flat = (att_flat.view(B, T, -1, attn.head_dim) * g.unsqueeze(-1)).view(B, T, -1)
    else:
        att_flat = att_flat * g
    return w_out(att_flat)


# --------------------------------------------------------------------------------------------------
# Gate is wired up at config/build time.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", [AttentionType.landmark, AttentionType.fast_landmark, AttentionType.sparse_landmark]
)
def test_gated_landmark_builds_w_g(name: AttentionType):
    # fast_landmark requires mem_freq >= 15.
    attn = _build(name, gate=True, mem_freq=15)
    assert attn.gate is not None
    assert isinstance(attn.w_g, nn.Linear)
    # elementwise gate -> one value per output element (n_heads * head_dim).
    assert attn.w_g.out_features == N_HEADS * HEAD_DIM


@pytest.mark.parametrize(
    "name", [AttentionType.landmark, AttentionType.fast_landmark, AttentionType.sparse_landmark]
)
def test_ungated_landmark_has_no_w_g(name: AttentionType):
    attn = _build(name, gate=False, mem_freq=15)
    assert attn.gate is None
    assert attn.w_g is None


def test_gated_headwise_landmark_w_g_shape():
    attn = _build(AttentionType.landmark, gate=True, granularity=GateGranularity.headwise)
    assert attn.w_g is not None
    assert attn.w_g.out_features == N_HEADS  # one gate per head


# --------------------------------------------------------------------------------------------------
# Forward correctness on CPU (eager paths): the gate is applied immediately before w_out.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("granularity", [GateGranularity.elementwise, GateGranularity.headwise])
def test_gated_landmark_eager_matches_reference(granularity: GateGranularity):
    torch.manual_seed(0)
    attn = _build(AttentionType.landmark, gate=True, granularity=granularity, mem_freq=3)
    assert isinstance(attn, LandmarkAttention) and attn.use_kernel is False
    attn.eval()
    B, T = 2, 12  # multiple of block_size (4)
    x = torch.randn(B, T, D_MODEL)
    with torch.no_grad():
        out = attn(x)
    ref = _gate_reference(attn, x)
    assert out.shape == (B, T, D_MODEL)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_gated_sparse_landmark_eager_matches_reference(monkeypatch):
    monkeypatch.setenv("LM_SPARSE_KERNEL", "0")  # force the eager torch fallback
    torch.manual_seed(0)
    attn = _build(AttentionType.sparse_landmark, gate=True, mem_freq=3, num_landmarks=1)
    assert isinstance(attn, SparseLandmarkAttention)
    attn.eval()
    B, T = 2, 12  # multiple of block_size (mem_freq + num_landmarks = 4)
    x = torch.randn(B, T, D_MODEL)
    with torch.no_grad():
        out = attn(x)
    ref = _gate_reference(attn, x)
    assert out.shape == (B, T, D_MODEL)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_gate_actually_changes_output():
    # A gated module with shared q/k/v/out weights must differ from the ungated one (the gate is not
    # a no-op when configured).
    torch.manual_seed(0)
    gated = _build(AttentionType.landmark, gate=True, mem_freq=3)
    ungated = _build(AttentionType.landmark, gate=False, mem_freq=3)
    ungated.load_state_dict(
        {k: v for k, v in gated.state_dict().items() if not k.startswith("w_g.")}
    )
    gated.eval()
    ungated.eval()
    x = torch.randn(2, 12, D_MODEL)
    with torch.no_grad():
        assert not torch.allclose(gated(x), ungated(x))


def test_gated_landmark_backward():
    # Gated landmark attention must be fully differentiable on CPU, including the gate projection.
    torch.manual_seed(0)
    attn = _build(AttentionType.landmark, gate=True, mem_freq=3)
    attn.train()
    x = torch.randn(2, 12, D_MODEL, requires_grad=True)
    (attn(x) ** 2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert attn.w_g is not None and attn.w_g.weight.grad is not None
    assert torch.isfinite(attn.w_g.weight.grad).all()


# --------------------------------------------------------------------------------------------------
# No-regression: the gate=None path is an exact no-op.
# --------------------------------------------------------------------------------------------------


def test_apply_gate_is_identity_without_gate():
    attn = _build(AttentionType.landmark, gate=False, mem_freq=3)
    att = torch.randn(2, 12, N_HEADS * HEAD_DIM)
    x = torch.randn(2, 12, D_MODEL)
    # Returns the very same tensor object: a true no-op.
    assert attn._apply_gate(att, x) is att


def test_ungated_landmark_forward_unchanged():
    # With no gate, the landmark forward must be bit-identical to manually running the core and the
    # output projection (i.e. _apply_gate added nothing).
    torch.manual_seed(0)
    attn = _build(AttentionType.landmark, gate=False, mem_freq=3)
    attn.eval()
    x = torch.randn(2, 12, D_MODEL)
    with torch.no_grad():
        out = attn(x)
        w_out = attn.w_out
        attn.w_out = nn.Identity()
        att_flat = attn(x)
        attn.w_out = w_out
        expected = w_out(att_flat)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


# --------------------------------------------------------------------------------------------------
# FastLandmarkAttention forward needs CUDA + the triton kernel.
# --------------------------------------------------------------------------------------------------


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize("granularity", [GateGranularity.elementwise, GateGranularity.headwise])
def test_gated_fast_landmark_matches_reference(granularity: GateGranularity):
    torch.manual_seed(0)
    attn = _build(
        AttentionType.fast_landmark, gate=True, granularity=granularity, mem_freq=15
    ).cuda()
    assert isinstance(attn, FastLandmarkAttention)
    attn.eval()
    B, T = 1, (15 + 1) * 4  # multiple of block_size (16)
    x = torch.randn(B, T, D_MODEL, device="cuda")
    with torch.no_grad():
        out = attn(x)
    ref = _gate_reference(attn, x)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
