"""GPU tests for the compressive landmark kernel (:mod:`landmark_compressive`).

The compressive kernel keeps the cross-block (gate) softmax of normal landmark attention but folds
each past block's landmark token into that block's within-block softmax, so the landmark contributes
its value to the output. Forward and backward are validated against an independent eager reference
that builds the full (B, H, T, T) compressive attention matrix.
"""

import math

import pytest
import torch

from olmo_core.nn.attention.landmark_compressive import (
    fused_compressive_landmark_attention,
)
from olmo_core.nn.attention.landmark_kernel import has_landmark_kernel
from olmo_core.testing import requires_gpu


def _eager_compressive_landmark_reference(q, k, v, block_size):
    """Dense eager *compressive* landmark attention over ``(B, H, T, d)`` (full-context, causal).

    Each fully-past block's gate weight (from its landmark score, exactly as in normal landmark
    attention) is spread over the block's content tokens AND its landmark via a within-block softmax
    over all ``block_size`` tokens; the local section is plain causal attention and never attends its
    own block's landmark.
    """
    B, H, T, d = q.shape
    device = q.device
    scale = 1.0 / math.sqrt(d)
    scores = (q @ k.transpose(-1, -2)).float() * scale  # (B, H, T, T)
    neg_inf = torch.finfo(scores.dtype).min

    pos = torch.arange(T, device=device)
    sec = pos // block_size
    is_mem = (pos % block_size) == (block_size - 1)  # (T,)
    causal = pos[None, :] <= pos[:, None]
    same_block = sec[None, :] == sec[:, None]
    past_block = sec[None, :] < sec[:, None]
    kmem = is_mem[None, :]

    local_content = same_block & (~kmem) & causal  # (T, T)
    past_landmark = past_block & kmem
    gate_set = (local_content | past_landmark).view(1, 1, T, T)
    gate_w = torch.softmax(scores.masked_fill(~gate_set, neg_inf), dim=-1)  # (B, H, T, T)

    # Full within-block softmax over every block of keys (only the past-block entries are used).
    within = torch.softmax(scores.reshape(B, H, T, T // block_size, block_size), dim=-1)
    within = within.reshape(B, H, T, T)

    block_gate = gate_w[..., is_mem]  # (B, H, T, n_blocks): gate weight at each block's landmark
    block_gate_full = block_gate.repeat_interleave(block_size, dim=-1)  # (B, H, T, T)

    past_mask = past_block.view(1, 1, T, T)
    local_mask = local_content.view(1, 1, T, T)
    final = torch.where(past_mask, block_gate_full * within, torch.zeros_like(within))
    final = torch.where(local_mask, gate_w, final)
    return final.to(v.dtype) @ v


def _eager_compressive_landmark_reference_docmask(q, k, v, block_size, doc_id):
    """Eager *compressive* landmark attention with intra-document (packing) masking.

    ``doc_id`` is an int32 ``(B, n_blocks)`` per-landmark-block document id. A query may only attend
    to keys in the same document, so cross-document landmarks are dropped from the gate set (and
    their blocks therefore receive zero gate weight, contributing nothing). Within-block softmaxes
    are unaffected (every block is single-document, and cross-document blocks are gated to zero).
    """
    B, H, T, d = q.shape
    device = q.device
    scale = 1.0 / math.sqrt(d)
    scores = (q @ k.transpose(-1, -2)).float() * scale  # (B, H, T, T)
    neg_inf = torch.finfo(scores.dtype).min

    pos = torch.arange(T, device=device)
    sec = pos // block_size
    is_mem = (pos % block_size) == (block_size - 1)  # (T,)
    causal = pos[None, :] <= pos[:, None]
    same_block = sec[None, :] == sec[:, None]
    past_block = sec[None, :] < sec[:, None]
    kmem = is_mem[None, :]

    # Per-token document id (B, T) and the symmetric same-document mask (B, 1, T, T).
    tok_doc = doc_id[:, sec.to(torch.long)]  # (B, T)
    same_doc = (tok_doc[:, :, None] == tok_doc[:, None, :]).view(B, 1, T, T)

    local_content = (same_block & (~kmem) & causal).view(1, 1, T, T)  # (1, 1, T, T)
    past_landmark = (past_block & kmem).view(1, 1, T, T)
    gate_set = (local_content | past_landmark) & same_doc  # (B, 1, T, T)
    gate_w = torch.softmax(scores.masked_fill(~gate_set, neg_inf), dim=-1)  # (B, H, T, T)

    # Within-block softmax on the (unmasked) scores: valid for every block; cross-document blocks
    # are multiplied by a zero gate weight below, so their (finite) within distribution is unused.
    within = torch.softmax(scores.reshape(B, H, T, T // block_size, block_size), dim=-1)
    within = within.reshape(B, H, T, T)

    block_gate = gate_w[..., is_mem]  # (B, H, T, n_blocks)
    block_gate_full = block_gate.repeat_interleave(block_size, dim=-1)  # (B, H, T, T)

    past_mask = past_block.view(1, 1, T, T)
    final = torch.where(past_mask, block_gate_full * within, torch.zeros_like(within))
    final = torch.where(local_content, gate_w, final)
    return final.to(v.dtype) @ v


def test_docmask_eager_reference_equals_per_document_cpu():
    """CPU regression for the doc-mask eager reference (the GPU oracle): packing two documents into
    one window and masking across the boundary must reproduce running each document on its own."""
    torch.manual_seed(0)
    block_size, head_dim = 16, 32
    n_blocks_a, n_blocks_b = 2, 3
    Ta, Tb = n_blocks_a * block_size, n_blocks_b * block_size
    T = Ta + Tb

    q = torch.randn(1, 2, T, head_dim)
    k = torch.randn(1, 2, T, head_dim)
    v = torch.randn(1, 2, T, head_dim)
    doc_id = torch.tensor([[0] * n_blocks_a + [1] * n_blocks_b], dtype=torch.int32)  # (1, n_blocks)

    out_packed = _eager_compressive_landmark_reference_docmask(q, k, v, block_size, doc_id)

    out_a = _eager_compressive_landmark_reference(
        q[:, :, :Ta], k[:, :, :Ta], v[:, :, :Ta], block_size
    )
    out_b = _eager_compressive_landmark_reference(
        q[:, :, Ta:], k[:, :, Ta:], v[:, :, Ta:], block_size
    )
    torch.testing.assert_close(out_packed[:, :, :Ta], out_a, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_packed[:, :, Ta:], out_b, rtol=1e-5, atol=1e-5)


def test_eager_reference_matches_decode_cpu():
    """CPU cross-check: tie the eager reference (the GPU oracle) to the brute-force-validated decode
    path. ``_decode_one`` for a non-landmark, non-eval query computes exactly the eager reference's
    row, so this transitively validates the reference math (and hence the GPU comparison) on CPU."""
    from olmo_core.nn.attention import AttentionConfig, AttentionType

    block_size, head_dim = 16, 32
    T = block_size * 3
    attn = AttentionConfig(
        name=AttentionType.fast_compressive_landmark,
        n_heads=1,
        n_kv_heads=1,
        head_dim=head_dim,
        bias=False,
        mem_freq=block_size - 1,
    ).build(head_dim, layer_idx=0, n_layers=1, init_device="cpu")
    attn.eval()

    torch.manual_seed(0)
    q = torch.randn(1, 1, T, head_dim)
    k = torch.randn(1, 1, T, head_dim)
    v = torch.randn(1, 1, T, head_dim)

    out_ref = _eager_compressive_landmark_reference(q, k, v, block_size)  # (1, 1, T, d)
    qpos = T - 2  # 46 -> not a landmark (46 % 16 == 14), no self-key drop
    with torch.no_grad():
        out_dec = attn._decode_one(
            q[:, :, qpos : qpos + 1], k[:, :, : qpos + 1], v[:, :, : qpos + 1], qpos
        )
    torch.testing.assert_close(out_ref[:, :, qpos], out_dec[:, :, 0], rtol=1e-5, atol=1e-5)


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize(
    "head_dim, mem_freq",
    [(64, 15), (128, 15), (256, 15), (256, 63)],  # (256, 63) is the Qwen3.5 training config
)
def test_compressive_kernel_forward_matches_eager(head_dim: int, mem_freq: int):
    torch.manual_seed(0)
    block_size = mem_freq + 1
    B, n_heads = 2, 4
    T = block_size * 4
    q = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.bfloat16)
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)

    out_kernel = fused_compressive_landmark_attention(q, k, v, is_mem, block_size=block_size)
    out_eager = _eager_compressive_landmark_reference(q, k, v, block_size)

    torch.testing.assert_close(out_kernel, out_eager, rtol=1e-2, atol=1e-2)


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize(
    "head_dim, mem_freq, dtype",
    [
        (64, 15, torch.float32),
        (128, 15, torch.float32),
        (256, 15, torch.float32),
        # Qwen3.5 config: head_dim 256 / block 64 fp32 overflows H100 shared memory in the backward,
        # so validate in bf16 (the training dtype) with accumulation-noise tolerances.
        (256, 63, torch.bfloat16),
    ],
)
def test_compressive_kernel_backward_matches_eager(
    head_dim: int, mem_freq: int, dtype: torch.dtype
):
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
            out = fused_compressive_landmark_attention(
                q, k, v, is_mem, sm_scale=scale, block_size=block_size
            )
        else:
            out = _eager_compressive_landmark_reference(q, k, v, block_size)
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
def test_compressive_differs_from_normal_landmark():
    # Sanity: the compressive kernel must NOT equal the normal fast-landmark kernel (the landmark
    # tokens now contribute their values to the output).
    from olmo_core.nn.attention.landmark_fast import fused_landmark_attention_fast

    torch.manual_seed(0)
    block_size = 16
    B, n_heads, head_dim = 2, 4, 64
    T = block_size * 4
    q = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.float32)
    k = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.float32)
    v = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.float32)
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)

    out_c = fused_compressive_landmark_attention(q, k, v, is_mem, block_size=block_size)
    out_n = fused_landmark_attention_fast(q, k, v, is_mem, block_size=block_size)
    assert not torch.allclose(out_c, out_n, atol=1e-3)


def _two_doc_doc_id(B, n_blocks, split, device):
    """Per-block doc id splitting each row's blocks into doc 0 (first ``split``) and doc 1."""
    blk = torch.arange(n_blocks, device=device)
    return (blk >= split).to(torch.int32).view(1, n_blocks).expand(B, n_blocks).contiguous()


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize("head_dim, mem_freq", [(64, 15), (128, 15), (256, 15)])
def test_compressive_kernel_docmask_forward_matches_eager(head_dim: int, mem_freq: int):
    torch.manual_seed(0)
    block_size = mem_freq + 1
    B, n_heads = 2, 4
    n_blocks = 5
    T = block_size * n_blocks
    q = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.bfloat16)
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)
    doc_id = _two_doc_doc_id(B, n_blocks, split=2, device="cuda")

    out_kernel = fused_compressive_landmark_attention(
        q, k, v, is_mem, block_size=block_size, doc_id=doc_id
    )
    out_eager = _eager_compressive_landmark_reference_docmask(q, k, v, block_size, doc_id)
    torch.testing.assert_close(out_kernel, out_eager, rtol=1e-2, atol=1e-2)


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
def test_compressive_kernel_no_docmask_byte_identical():
    """Passing doc_id=None must be byte-identical to before (DOC_MASK compiles out)."""
    torch.manual_seed(0)
    block_size, head_dim = 16, 64
    B, n_heads = 2, 4
    T = block_size * 5
    q = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.float32)
    k = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.float32)
    v = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=torch.float32)
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)

    out_none = fused_compressive_landmark_attention(q, k, v, is_mem, block_size=block_size)
    # A single-document doc_id should give the same result as no masking.
    doc_id = torch.zeros(B, T // block_size, dtype=torch.int32, device="cuda")
    out_onedoc = fused_compressive_landmark_attention(
        q, k, v, is_mem, block_size=block_size, doc_id=doc_id
    )
    torch.testing.assert_close(out_none, out_onedoc, rtol=0, atol=0)


@requires_gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="requires triton landmark kernel")
@pytest.mark.parametrize(
    "head_dim, mem_freq, dtype",
    [
        (64, 15, torch.float32),
        (128, 15, torch.float32),
        (256, 15, torch.float32),
    ],
)
def test_compressive_kernel_docmask_backward_matches_eager(
    head_dim: int, mem_freq: int, dtype: torch.dtype
):
    torch.manual_seed(0)
    block_size = mem_freq + 1
    B, n_heads = 2, 4
    n_blocks = 5
    T = block_size * n_blocks
    scale = head_dim**-0.5
    is_mem = (torch.arange(T, device="cuda") % block_size) == (block_size - 1)
    doc_id = _two_doc_doc_id(B, n_blocks, split=2, device="cuda")
    base = torch.rand(B, n_heads, T, head_dim, device="cuda", dtype=dtype)
    grad_out = torch.rand_like(base)

    def grads(use_kernel):
        q, k, v = (base.clone().requires_grad_(True) for _ in range(3))
        if use_kernel:
            out = fused_compressive_landmark_attention(
                q, k, v, is_mem, sm_scale=scale, block_size=block_size, doc_id=doc_id
            )
        else:
            out = _eager_compressive_landmark_reference_docmask(q, k, v, block_size, doc_id)
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
