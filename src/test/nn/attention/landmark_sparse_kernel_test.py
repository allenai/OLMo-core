"""
Tests for ``SparseLandmarkAttention`` sequence packing (intra-document masking):

  * CPU: the doc-aware eager forms (``sparse_landmark_attention`` / ``..._ref``) agree, and a packed
    forward/backward equals gradient accumulation over each (batch-element, document) sub-sequence.
  * GPU: the fused Triton kernel's document masking (forward + backward) matches the eager reference.
"""

import pytest
import torch

from olmo_core.nn.attention.landmark import build_block_doc_id
from olmo_core.nn.attention.landmark_sparse import (
    sparse_landmark_attention,
    sparse_landmark_attention_ref,
)
from olmo_core.nn.attention.landmark_sparse_kernel import (
    has_sparse_kernel,
    sparse_landmark_attention_triton_train,
)
from olmo_core.testing import requires_gpu


def _layout(block_size, device="cpu"):
    """Two batch rows with distinct chunk-aligned document layouts (T = 4 chunks each)."""
    L = block_size
    T = L * 4
    # Flattened-over-batch cu_doc_lens: row 0 docs [2 chunks, 2 chunks]; row 1 docs [1 chunk, 3].
    cu_doc_lens = torch.tensor([0, 2 * L, T, T + L, 2 * T], dtype=torch.int32, device=device)
    doc_id = build_block_doc_id(cu_doc_lens, 2, T, L)
    return T, doc_id


def test_sparse_packing_ref_matches_efficient():
    torch.manual_seed(0)
    B, H, D, L, G = 2, 4, 16, 4, 1
    T, doc_id = _layout(L)
    q, k, v = (torch.randn(B, H, T, D, dtype=torch.float64) for _ in range(3))
    o_eff = sparse_landmark_attention(q, k, v, L, num_landmarks=G, doc_id=doc_id)
    o_ref = sparse_landmark_attention_ref(q, k, v, L, num_landmarks=G, doc_id=doc_id)
    torch.testing.assert_close(o_eff, o_ref, rtol=1e-12, atol=1e-12)


def test_sparse_packing_matches_grad_accumulation():
    """Packed forward/backward == gradient accumulation over each (row, document) sub-sequence."""
    torch.manual_seed(0)
    B, H, D, L, G = 2, 4, 16, 4, 1
    T, doc_id = _layout(L)
    # Per-row document layouts (token spans), matching ``_layout``.
    rows = [[(0, 2 * L), (2 * L, 4 * L)], [(0, L), (L, 4 * L)]]
    base = torch.randn(B, H, T, D, dtype=torch.float64)
    grad_out = torch.randn_like(base)

    q = base.clone().requires_grad_(True)
    k = base.clone().requires_grad_(True)
    v = base.clone().requires_grad_(True)
    out = sparse_landmark_attention(q, k, v, L, num_landmarks=G, doc_id=doc_id)
    out.backward(grad_out)

    qr = base.clone().requires_grad_(True)
    kr = base.clone().requires_grad_(True)
    vr = base.clone().requires_grad_(True)
    out_ref = torch.zeros_like(base)
    for b, docs in enumerate(rows):
        for s, e in docs:
            out_ref[b : b + 1, :, s:e] = sparse_landmark_attention(
                qr[b : b + 1, :, s:e],
                kr[b : b + 1, :, s:e],
                vr[b : b + 1, :, s:e],
                L,
                num_landmarks=G,
            )
    out_ref.backward(grad_out)

    torch.testing.assert_close(out, out_ref, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(q.grad, qr.grad, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(k.grad, kr.grad, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(v.grad, vr.grad, rtol=1e-10, atol=1e-10)


@requires_gpu
@pytest.mark.skipif(not has_sparse_kernel(), reason="requires triton sparse landmark kernel")
@pytest.mark.parametrize("block_size, num_landmarks", [(16, 1), (16, 4), (64, 1)])
def test_sparse_kernel_packing_matches_eager(block_size: int, num_landmarks: int):
    # The fused kernel's document masking (fwd + bwd) must match the eager reference, in fp32 so the
    # comparison is exact up to accumulation noise.
    torch.manual_seed(0)
    B, H, D, L, G = 2, 4, 64, block_size, num_landmarks
    T, doc_id = _layout(L, device="cuda")
    scale = D**-0.5
    base = torch.rand(B, H, T, D, device="cuda", dtype=torch.float32)
    grad_out = torch.rand_like(base)

    def run(use_kernel):
        q, k, v = (base.clone().requires_grad_(True) for _ in range(3))
        if use_kernel:
            out = sparse_landmark_attention_triton_train(
                q, k, v, L, num_landmarks=G, scale=scale, doc_id=doc_id
            )
        else:
            out = sparse_landmark_attention(q, k, v, L, num_landmarks=G, scale=scale, doc_id=doc_id)
        out.backward(grad_out)
        return out, q.grad, k.grad, v.grad

    out_k, dq_k, dk_k, dv_k = run(True)
    out_e, dq_e, dk_e, dv_e = run(False)

    torch.testing.assert_close(out_k, out_e, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dq_k, dq_e, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dk_k, dk_e, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(dv_k, dv_e, rtol=1e-3, atol=1e-3)
