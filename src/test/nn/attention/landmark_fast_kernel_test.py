"""
GPU tests for the fast landmark kernel (``landmark_fast``) across head dims, including
``head_dim=256`` (Qwen3.5). The ``<= 128`` cases double as a regression check that extending the
kernel to larger head dims did not change its behavior: the fast kernel must stay bit-identical to
the original ``FusedLandmarkAttention``.
"""

import math

import pytest
import torch

from olmo_core.nn.attention.landmark import landmark_grouped_softmax
from olmo_core.nn.attention.landmark_fast import fused_landmark_attention_fast
from olmo_core.nn.attention.landmark_kernel import (
    fused_landmark_attention,
    has_landmark_kernel,
)
from olmo_core.testing import requires_gpu


def _eager_landmark_reference(q, k, v, block_size):
    """Dense eager landmark attention over ``(B, H, T, d)`` tensors (full-context, causal)."""
    B, H, T, d = q.shape
    att = (q @ k.transpose(-1, -2)) / math.sqrt(d)
    att_mask = torch.tril(torch.ones((1, 1, T, T), device=q.device), diagonal=0) == 1.0
    sec = torch.arange(T, device=q.device) // block_size
    last_section_mask = (sec[None, :] == sec[:, None]).unsqueeze(0).unsqueeze(1)
    is_mem = ((torch.arange(T, device=q.device) % block_size) == (block_size - 1)).view(1, 1, 1, T)
    mask = att_mask & ~(last_section_mask & is_mem)
    last_section_mask = (last_section_mask & mask).expand(B, H, T, T)
    is_mem_ = (is_mem & mask).expand(B, H, T, T)
    att = att.masked_fill(~mask, float("-inf"))
    att = landmark_grouped_softmax(att, -1, is_mem_, last_section_mask).to(q.dtype)
    att = att.masked_fill(~mask, 0.0)
    return att @ v


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize(
    "head_dim, mem_freq",
    [(64, 15), (128, 15), (256, 15), (256, 63)],  # (256, 63) is the Qwen3.5 training config
)
def test_fast_kernel_forward_matches_eager(head_dim: int, mem_freq: int):
    torch.manual_seed(0)
    block_size = mem_freq + 1
    B, n_heads = 2, 4
    T = block_size * 4
    q = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.bfloat16)
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)

    out_kernel = fused_landmark_attention_fast(q, k, v, is_mem, block_size=block_size)
    out_eager = _eager_landmark_reference(q, k, v, block_size)

    torch.testing.assert_close(out_kernel, out_eager, rtol=1e-2, atol=1e-2)


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize(
    "head_dim, mem_freq, dtype",
    [
        # fp32 makes the eager comparison exact up to accumulation noise (mirrors the original
        # kernel test).
        (64, 15, torch.float32),
        (128, 15, torch.float32),
        (256, 15, torch.float32),
        # The Qwen3.5 training config. fp32 is not runnable here -- at head_dim 256 and BLOCK=64
        # the backward's fp32 dot/trans operand tiles exceed H100 shared memory (~278KB vs 232KB)
        # at any num_stages -- so validate in bf16, the dtype training actually uses, with
        # accumulation-noise tolerances.
        (256, 63, torch.bfloat16),
    ],
)
def test_fast_kernel_backward_matches_eager(head_dim: int, mem_freq: int, dtype: torch.dtype):
    torch.manual_seed(0)
    block_size = mem_freq + 1
    B, n_heads = 2, 4
    T = block_size * 4
    scale = head_dim**-0.5
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)
    base = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=dtype)
    grad_out = torch.rand_like(base)

    def grads(use_kernel):
        q, k, v = (base.clone().requires_grad_(True) for _ in range(3))
        if use_kernel:
            out = fused_landmark_attention_fast(
                q, k, v, is_mem, sm_scale=scale, block_size=block_size
            )
        else:
            out = _eager_landmark_reference(q, k, v, block_size)
        out.backward(grad_out)
        return out, q.grad, k.grad, v.grad

    out_k, dq_k, dk_k, dv_k = grads(True)
    out_e, dq_e, dk_e, dv_e = grads(False)

    out_tol = dict(rtol=1e-4, atol=1e-4) if dtype == torch.float32 else dict(rtol=1e-2, atol=1e-2)
    grad_tol = dict(rtol=1e-3, atol=1e-3) if dtype == torch.float32 else dict(rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out_k, out_e, **out_tol)
    torch.testing.assert_close(dq_k, dq_e, **grad_tol)
    torch.testing.assert_close(dk_k, dk_e, **grad_tol)
    torch.testing.assert_close(dv_k, dv_e, **grad_tol)


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fast_kernel_unchanged_vs_original_for_small_head_dims(head_dim: int, dtype: torch.dtype):
    # Regression check for the head_dim <= 128 path after adding head_dim 256 support: the fast
    # kernel reuses the original forward kernel, so the forward output must be exactly equal.
    # Gradients match only up to last-ulp reassociation (1-2 ulps on a handful of elements): the
    # two backwards run with different num_warps, which retiles the tl.dot reductions. torch's
    # dtype-aware default tolerances comfortably cover that ulp noise while staying far tighter
    # than the eager-reference comparisons above.
    torch.manual_seed(0)
    mem_freq = 15
    block_size = mem_freq + 1
    B, n_heads = 2, 4
    T = block_size * 4
    scale = head_dim**-0.5
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)
    base = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=dtype)
    grad_out = torch.rand_like(base)

    def grads(fn):
        q, k, v = (base.clone().requires_grad_(True) for _ in range(3))
        out = fn(q, k, v, is_mem, sm_scale=scale, block_size=block_size)
        out.backward(grad_out)
        return out, q.grad, k.grad, v.grad

    out_f, dq_f, dk_f, dv_f = grads(fused_landmark_attention_fast)
    out_o, dq_o, dk_o, dv_o = grads(fused_landmark_attention)

    torch.testing.assert_close(out_f, out_o, rtol=0, atol=0)
    torch.testing.assert_close(dq_f, dq_o)
    torch.testing.assert_close(dk_f, dk_o)
    torch.testing.assert_close(dv_f, dv_o)
