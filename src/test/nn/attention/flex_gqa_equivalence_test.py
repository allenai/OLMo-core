"""Pin that FlexAttention's native GQA equals the materialised-KV path.

``FlexAttentionBackend.forward`` expands k/v from ``n_kv_heads`` to ``n_heads`` with
``_repeat_kv`` before calling ``flex_attention``. That expansion is
``.expand(...).reshape(...)``, and a reshape of an expanded tensor cannot be a view, so it
copies: at Stage-2 geometry k and v are materialised at 4x size and the kernel then reads
32 kv heads where 8 would do.

``flex_attention(..., enable_gqa=True)`` broadcasts in-register instead, which is the
cheaper path. Switching to it is a change of code path, not of arithmetic -- but
``pad_isolation_test.test_flex_and_dense_rules_agree`` only pins the *mask predicate* and
never runs attention, so nothing currently pins the outputs. This does.

GPU-only: the flex kernels live on CUDA, and the eager CPU fallback does not exercise the
path under test.
"""

import pytest
import torch

from olmo_core.nn.attention.backend import FlexAttentionBackend, _repeat_kv

B, S, N_HEADS, N_KV_HEADS, HEAD_DIM = 1, 256, 8, 2, 32


def _block_mask(device):
    # Two packed examples plus a pad tail, so the mask exercises the example-isolation and
    # pad rules rather than a plain causal triangle.
    example_id = torch.full((B, S), -1, dtype=torch.int32, device=device)
    example_id[:, 0:100] = 0
    example_id[:, 100:220] = 1
    is_image = torch.zeros((B, S), dtype=torch.bool, device=device)
    is_image[:, 20:80] = True
    return FlexAttentionBackend.build_block_mask_from_vectors(
        B=B, S=S, device=device, is_image=is_image, example_id=example_id
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_enable_gqa_matches_materialized_kv():
    from torch.nn.attention.flex_attention import flex_attention

    device = torch.device("cuda")
    torch.manual_seed(0)
    block_mask = _block_mask(device)
    scale = HEAD_DIM**-0.5

    # (B, S, H, D) is the layout `_repeat_kv` expects; the backend transposes afterwards.
    q = torch.randn(B, S, N_HEADS, HEAD_DIM, device=device, dtype=torch.float32)
    k = torch.randn(B, S, N_KV_HEADS, HEAD_DIM, device=device, dtype=torch.float32)
    v = torch.randn(B, S, N_KV_HEADS, HEAD_DIM, device=device, dtype=torch.float32)

    expanded = flex_attention(
        q.transpose(1, 2),
        _repeat_kv(k, N_HEADS // N_KV_HEADS).transpose(1, 2),
        _repeat_kv(v, N_HEADS // N_KV_HEADS).transpose(1, 2),
        block_mask=block_mask,
        scale=scale,
    )
    native = flex_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        block_mask=block_mask,
        scale=scale,
        enable_gqa=True,
    )

    torch.testing.assert_close(expanded, native, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_enable_gqa_matches_materialized_kv_in_backward():
    """Forward parity alone would not catch a wrong gradient reduction over kv groups.

    The materialised path accumulates dk/dv across the ``n_rep`` copies through the
    reshape's backward; the native path sums over the group inside the kernel. Those are
    the same sum by different routes, which is exactly the kind of thing that can diverge.
    """
    from torch.nn.attention.flex_attention import flex_attention

    device = torch.device("cuda")
    torch.manual_seed(0)
    block_mask = _block_mask(device)
    scale = HEAD_DIM**-0.5
    n_rep = N_HEADS // N_KV_HEADS

    base = [
        torch.randn(B, S, n, HEAD_DIM, device=device, dtype=torch.float32)
        for n in (N_HEADS, N_KV_HEADS, N_KV_HEADS)
    ]
    grads = []
    for native in (False, True):
        q, k, v = (t.clone().requires_grad_(True) for t in base)
        if native:
            out = flex_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                block_mask=block_mask, scale=scale, enable_gqa=True,
            )
        else:
            out = flex_attention(
                q.transpose(1, 2),
                _repeat_kv(k, n_rep).transpose(1, 2),
                _repeat_kv(v, n_rep).transpose(1, 2),
                block_mask=block_mask, scale=scale,
            )
        out.sum().backward()
        grads.append((q.grad, k.grad, v.grad))

    for name, a, b in zip(("dq", "dk", "dv"), grads[0], grads[1]):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4, msg=lambda m, n=name: f"{n}: {m}")
