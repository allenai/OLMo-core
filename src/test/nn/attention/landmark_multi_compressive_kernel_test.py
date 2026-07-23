"""Tests for the multi-landmark compressive kernel (:mod:`landmark_multi_compressive`).

CPU tests validate the eager reference (:func:`multi_compressive_landmark_reference`) -- most
importantly that ``num_landmarks == 1`` reduces *exactly* to the single-landmark compressive
reference. GPU tests validate the fused Triton forward/backward against the eager reference for
several ``num_landmarks`` / pool combinations, including a ``num_landmarks == 1`` case that must match
the single-landmark kernel and module bit-for-bit up to tolerance.
"""

import math

import pytest
import torch

from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.attention.landmark_compressive import fused_compressive_landmark_attention
from olmo_core.nn.attention.landmark_kernel import has_landmark_kernel
from olmo_core.nn.attention.landmark_multi_compressive import (
    fused_multi_compressive_landmark_attention,
    multi_compressive_landmark_reference,
)
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.testing import requires_gpu

# Import the single-landmark eager reference so we can pin the num_landmarks == 1 reduction.
from .landmark_compressive_kernel_test import (
    _eager_compressive_landmark_reference,
    _eager_compressive_landmark_reference_docmask,
)


def _multi_is_mem(T, block_size, num_landmarks, device):
    return (torch.arange(T, device=device) % block_size) >= (block_size - num_landmarks)


# --------------------------------------------------------------------------------------------------
# CPU tests: the eager reference.
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("agg", ["mean", "max"])
def test_reference_reduces_to_single_landmark_cpu(agg):
    """num_landmarks == 1 must reproduce the single-landmark compressive reference exactly."""
    torch.manual_seed(0)
    B, H, T, d = 2, 3, 48, 16
    block_size = 16
    q = torch.randn(B, H, T, d, dtype=torch.float64)
    k = torch.randn(B, H, T, d, dtype=torch.float64)
    v = torch.randn(B, H, T, d, dtype=torch.float64)
    ref = _eager_compressive_landmark_reference(q, k, v, block_size)
    out = multi_compressive_landmark_reference(q, k, v, block_size, 1, agg)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("agg", ["mean", "max"])
def test_reference_docmask_reduces_to_single_landmark_cpu(agg):
    torch.manual_seed(1)
    B, H, d = 2, 2, 16
    block_size = 16
    n_blocks = 4
    T = n_blocks * block_size
    q = torch.randn(B, H, T, d, dtype=torch.float64)
    k = torch.randn(B, H, T, d, dtype=torch.float64)
    v = torch.randn(B, H, T, d, dtype=torch.float64)
    # Two documents, each spanning two whole blocks.
    doc_id = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=torch.int32)
    ref = _eager_compressive_landmark_reference_docmask(q, k, v, block_size, doc_id)
    out = multi_compressive_landmark_reference(q, k, v, block_size, 1, agg, doc_id=doc_id)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_reference_probabilities_are_valid_cpu():
    """The implied per-key attention weights are non-negative and sum to 1 (causal rows)."""
    torch.manual_seed(2)
    B, H, T, d = 1, 1, 64, 8
    block_size, num_landmarks = 32, 4
    q = torch.randn(B, H, T, d, dtype=torch.float64)
    k = torch.randn(B, H, T, d, dtype=torch.float64)
    v = torch.eye(T, dtype=torch.float64).view(1, 1, T, T).expand(B, H, T, T).contiguous()
    for agg in ("mean", "max"):
        # With value == identity, output row i is exactly the attention weight vector for query i.
        probs = multi_compressive_landmark_reference(q, k, v, block_size, num_landmarks, agg)[0, 0]
        assert torch.all(probs >= -1e-9)
        torch.testing.assert_close(
            probs.sum(-1), torch.ones(T, dtype=torch.float64), rtol=0, atol=1e-6
        )


def test_reference_mean_vs_max_differ_cpu():
    """mean and max pooling give genuinely different gates when a block's landmarks disagree."""
    torch.manual_seed(3)
    B, H, T, d = 1, 2, 64, 8
    block_size, num_landmarks = 32, 4
    q = torch.randn(B, H, T, d, dtype=torch.float64)
    k = torch.randn(B, H, T, d, dtype=torch.float64)
    v = torch.randn(B, H, T, d, dtype=torch.float64)
    out_mean = multi_compressive_landmark_reference(q, k, v, block_size, num_landmarks, "mean")
    out_max = multi_compressive_landmark_reference(q, k, v, block_size, num_landmarks, "max")
    assert (out_mean - out_max).abs().max() > 1e-3


# --------------------------------------------------------------------------------------------------
# GPU tests: the fused kernel vs. the eager reference.
# --------------------------------------------------------------------------------------------------


def _rand_qkv(B, H, T, d, dtype, device, requires_grad=False):
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(
        B, H, T, d, dtype=dtype, device=device, generator=g, requires_grad=requires_grad
    )
    k = torch.randn(
        B, H, T, d, dtype=dtype, device=device, generator=g, requires_grad=requires_grad
    )
    v = torch.randn(
        B, H, T, d, dtype=dtype, device=device, generator=g, requires_grad=requires_grad
    )
    return q, k, v


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires the fused landmark Triton kernel")
@pytest.mark.parametrize("agg", ["mean", "max"])
@pytest.mark.parametrize("block_size,num_landmarks", [(16, 1), (32, 1), (32, 2), (32, 4), (64, 8)])
def test_kernel_forward_matches_eager(block_size, num_landmarks, agg):
    device = "cuda"
    B, H, T, d = 2, 3, block_size * 5, 64
    q, k, v = _rand_qkv(B, H, T, d, torch.bfloat16, device)
    is_mem = _multi_is_mem(T, block_size, num_landmarks, device)
    out = fused_multi_compressive_landmark_attention(
        q,
        k,
        v,
        is_mem,
        sm_scale=1.0 / math.sqrt(d),
        block_size=block_size,
        num_landmarks=num_landmarks,
        agg=agg,
    )
    ref = multi_compressive_landmark_reference(
        q.float(), k.float(), v.float(), block_size, num_landmarks, agg
    ).to(out.dtype)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires the fused landmark Triton kernel")
@pytest.mark.parametrize("agg", ["mean", "max"])
@pytest.mark.parametrize("block_size,num_landmarks", [(32, 1), (32, 2), (32, 4)])
def test_kernel_backward_matches_eager(block_size, num_landmarks, agg):
    device = "cuda"
    B, H, T, d = 2, 2, block_size * 4, 64
    q, k, v = _rand_qkv(B, H, T, d, torch.float32, device, requires_grad=True)
    is_mem = _multi_is_mem(T, block_size, num_landmarks, device)
    do = torch.randn_like(q)

    out = fused_multi_compressive_landmark_attention(
        q,
        k,
        v,
        is_mem,
        sm_scale=1.0 / math.sqrt(d),
        block_size=block_size,
        num_landmarks=num_landmarks,
        agg=agg,
    )
    (dq, dk, dv) = torch.autograd.grad(out, (q, k, v), do)

    qr, kr, vr = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    ref = multi_compressive_landmark_reference(qr, kr, vr, block_size, num_landmarks, agg)
    (rdq, rdk, rdv) = torch.autograd.grad(ref, (qr, kr, vr), do)

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dq, rdq, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dk, rdk, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dv, rdv, rtol=1e-3, atol=1e-3)


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires the fused landmark Triton kernel")
@pytest.mark.parametrize("agg", ["mean", "max"])
def test_kernel_docmask_forward_matches_eager(agg):
    device = "cuda"
    from olmo_core.nn.attention.landmark import build_block_doc_id

    block_size, num_landmarks = 32, 4
    B, H, d = 2, 2, 64
    n_blocks = 6
    T = n_blocks * block_size
    q, k, v = _rand_qkv(B, H, T, d, torch.bfloat16, device)
    is_mem = _multi_is_mem(T, block_size, num_landmarks, device)
    # cu_doc_lens per batch element, block-aligned document boundaries.
    cu = [
        torch.tensor([0, 2 * block_size, T], device=device),
        torch.tensor([0, 3 * block_size, T], device=device),
    ]
    doc_id = torch.stack([build_block_doc_id(c, T, block_size) for c in cu], dim=0)
    out = fused_multi_compressive_landmark_attention(
        q,
        k,
        v,
        is_mem,
        sm_scale=1.0 / math.sqrt(d),
        block_size=block_size,
        num_landmarks=num_landmarks,
        agg=agg,
        doc_id=doc_id,
    )
    ref = multi_compressive_landmark_reference(
        q.float(), k.float(), v.float(), block_size, num_landmarks, agg, doc_id=doc_id
    ).to(out.dtype)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires the fused landmark Triton kernel")
@pytest.mark.parametrize("agg", ["mean", "max"])
def test_kernel_num_landmarks_1_matches_single_landmark_kernel(agg):
    """num_landmarks == 1 (either pool) must match the single-landmark compressive kernel."""
    device = "cuda"
    block_size = 32
    B, H, T, d = 2, 3, block_size * 5, 64
    q, k, v = _rand_qkv(B, H, T, d, torch.float32, device, requires_grad=True)
    is_mem = _multi_is_mem(T, block_size, 1, device)
    do = torch.randn_like(q)

    out_multi = fused_multi_compressive_landmark_attention(
        q,
        k,
        v,
        is_mem,
        sm_scale=1.0 / math.sqrt(d),
        block_size=block_size,
        num_landmarks=1,
        agg=agg,
    )
    gm = torch.autograd.grad(out_multi, (q, k, v), do, retain_graph=True)

    out_single = fused_compressive_landmark_attention(
        q, k, v, is_mem, sm_scale=1.0 / math.sqrt(d), block_size=block_size
    )
    gs = torch.autograd.grad(out_single, (q, k, v), do)

    torch.testing.assert_close(out_multi, out_single, rtol=1e-4, atol=1e-5)
    for a, b in zip(gm, gs):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)


def _build_multi(*, mem_freq, num_landmarks, landmark_gate_pool, head_dim, device):
    attn = AttentionConfig(
        name=AttentionType.multi_compressive_landmark,
        n_heads=2,
        n_kv_heads=2,
        head_dim=head_dim,
        bias=False,
        mem_freq=mem_freq,
        num_landmarks=num_landmarks,
        landmark_gate_pool=landmark_gate_pool,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
    ).build(head_dim, layer_idx=0, n_layers=1, init_device=device)
    return attn


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires the fused landmark Triton kernel")
@pytest.mark.parametrize("landmark_gate_pool", ["mean", "max"])
def test_module_num_landmarks_1_matches_fast_compressive(landmark_gate_pool):
    """Full module forward + backward: MultiCompressive(num_landmarks=1) == FastCompressive.

    This is the primary class-level sanity check from the plan: with one landmark per block the new
    class must reproduce ``FastCompressiveLandmarkAttention`` given identical weights and input.
    """
    device = "cuda"
    mem_freq = 31  # block_size = 32
    head_dim = 64
    d_model = 128

    multi = _build_multi(
        mem_freq=mem_freq,
        num_landmarks=1,
        landmark_gate_pool=landmark_gate_pool,
        head_dim=head_dim,
        device=device,
    )
    single = AttentionConfig(
        name=AttentionType.fast_compressive_landmark,
        n_heads=2,
        n_kv_heads=2,
        head_dim=head_dim,
        bias=False,
        mem_freq=mem_freq,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
    ).build(head_dim, layer_idx=0, n_layers=1, init_device=device)

    single.load_state_dict(multi.state_dict())

    B, T = 2, 32 * 5
    x = torch.randn(B, T, d_model, device=device, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    grad_out = torch.randn(B, T, d_model, device=device)

    out_multi = multi(x)
    out_single = single(x2)
    torch.testing.assert_close(out_multi, out_single, rtol=1e-4, atol=1e-5)

    out_multi.backward(grad_out)
    out_single.backward(grad_out)
    torch.testing.assert_close(x.grad, x2.grad, rtol=1e-4, atol=1e-5)
    for (n1, p1), (n2, p2) in zip(multi.named_parameters(), single.named_parameters()):
        assert n1 == n2
        if p1.grad is not None or p2.grad is not None:
            torch.testing.assert_close(p1.grad, p2.grad, rtol=1e-4, atol=1e-5, msg=n1)
