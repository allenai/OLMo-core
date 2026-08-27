"""Parity of the ragged (right-padded, cross-length) ``SparseLandmarkAttention`` decode against the
legacy bs=1 per-row decode.

Chunk boundaries are tied to ABSOLUTE position, so left-padding is illegal and the legacy path can
only batch prompts of *exactly* equal length -- effective batch size ~= 1 on any variable-length
eval. Right-padding is legal instead (each row's content still starts at position 0; only the pad
TAIL differs), which is what ``_decode_ragged`` decodes: every row at its OWN absolute position,
prompt length and top-k. This file is the parity gate for that: for every row the batched output
must equal the scalar ``_decode_one`` run on that row alone.

The eager decode math is pure torch, so this runs on CPU with no Triton kernel and no GPU.
"""

import os

import pytest
import torch

from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.attention.landmark import repeat_kv
from olmo_core.nn.layer_norm import LayerNormConfig

os.environ.setdefault("LM_SPARSE_KERNEL", "0")

MEM_FREQ = 7
NUM_LANDMARKS = 1
BLOCK = MEM_FREQ + NUM_LANDMARKS  # 8


def _build(num_landmarks=NUM_LANDMARKS, mem_freq=MEM_FREQ, n_heads=2, n_kv_heads=2, head_dim=24):
    attn = AttentionConfig(
        name=AttentionType.sparse_landmark,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        bias=False,
        mem_freq=mem_freq,
        num_landmarks=num_landmarks,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
    ).build(n_heads * head_dim, layer_idx=0, n_layers=1, init_device="cpu")
    attn.eval()
    return attn


@pytest.mark.parametrize("mode", ["extend_last_block", "generation_only"])
@pytest.mark.parametrize("top_k", [None, 1, 2, 50])
def test_ragged_decode_matches_scalar_per_row(mode, top_k):
    torch.manual_seed(0)
    attn = _build()
    H, D = attn.n_heads, attn.head_dim

    # Rows covering: a final-prompt-token query (non-eval, per-chunk rule), generated tokens with
    # short and long prompts (eval / one-long-local-block), a landmark-position prompt query, and a
    # row whose prompt is a single chunk (no past chunks at all).
    #        prompt_len, qpos
    rows = [
        (24, 23),  # final prompt token of a 3-chunk prompt (non-eval)
        (16, 20),  # generated token, short prompt (eval)
        (40, 47),  # generated token, long prompt (eval)
        (32, 31),  # landmark-position prompt token (31 % 8 == 7), non-eval
        (8, 12),  # generated token, single-chunk prompt
    ]
    B = len(rows)
    plen = torch.tensor([r[0] for r in rows])
    qpos = torch.tensor([r[1] for r in rows])
    total = int(qpos.max().item()) + 1

    # Shared right-padded cache: keys beyond a row's own qpos are "pad" (random) and must be ignored.
    k = torch.randn(B, H, total, D)
    v = torch.randn(B, H, total, D)
    q = torch.randn(B, H, 1, D)

    top_k_t = None if top_k is None else torch.full((B,), top_k, dtype=torch.long)
    attn.set_landmark_ragged_decode(plen, mode=mode, top_k=top_k_t)
    attn.set_ragged_qpos(qpos)
    with torch.no_grad():
        out = attn._decode_ragged(q, k, v)  # (B,H,1,D)

    attn.clear_ragged_decode()
    for b in range(B):
        attn.set_landmark_eval_decode(int(plen[b].item()), mode, top_k=top_k)
        qb = int(qpos[b].item())
        with torch.no_grad():
            ref = attn._decode_one(
                q[b : b + 1], k[b : b + 1, :, : qb + 1], v[b : b + 1, :, : qb + 1], qb
            )
        torch.testing.assert_close(out[b : b + 1], ref, rtol=1e-5, atol=1e-5, msg=f"row {b}")


def test_ragged_decode_per_row_top_k():
    """Each row gets its OWN chunk budget -- including a row whose budget exceeds its chunk count
    (which the scalar path leaves entirely unmasked)."""
    torch.manual_seed(1)
    attn = _build()
    H, D = attn.n_heads, attn.head_dim
    plen = torch.tensor([40, 40, 40])
    qpos = torch.tensor([44, 44, 44])
    top_k = torch.tensor([1, 3, 99])  # third row: more budget than it has chunks
    total = int(qpos.max().item()) + 1
    k, v, q = torch.randn(3, H, total, D), torch.randn(3, H, total, D), torch.randn(3, H, 1, D)

    attn.set_landmark_ragged_decode(plen, mode="extend_last_block", top_k=top_k)
    attn.set_ragged_qpos(qpos)
    with torch.no_grad():
        out = attn._decode_ragged(q, k, v)

    attn.clear_ragged_decode()
    for b in range(3):
        attn.set_landmark_eval_decode(40, "extend_last_block", top_k=int(top_k[b].item()))
        with torch.no_grad():
            ref = attn._decode_one(q[b : b + 1], k[b : b + 1, :, :45], v[b : b + 1, :, :45], 44)
        torch.testing.assert_close(out[b : b + 1], ref, rtol=1e-5, atol=1e-5, msg=f"row {b}")


@pytest.mark.parametrize("num_landmarks", [1, 3])
def test_ragged_decode_gqa_and_multi_landmark(num_landmarks):
    torch.manual_seed(2)
    attn = _build(num_landmarks=num_landmarks, mem_freq=5, n_heads=4, n_kv_heads=2, head_dim=16)
    L, H, D = attn.block_size, attn.n_heads, attn.head_dim
    plen = torch.tensor([4 * L, 2 * L, 5 * L])
    qpos = torch.tensor([4 * L + 2, 2 * L, 5 * L - 1])
    total = int(qpos.max().item()) + 1
    # ``_decode_ragged`` takes already GQA-expanded k/v, like the shipped ``_forward_generate``.
    k, v, q = torch.randn(3, H, total, D), torch.randn(3, H, total, D), torch.randn(3, H, 1, D)

    attn.set_landmark_ragged_decode(plen, mode="extend_last_block", top_k=torch.tensor([2, 1, 3]))
    attn.set_ragged_qpos(qpos)
    with torch.no_grad():
        out = attn._decode_ragged(q, k, v)

    attn.clear_ragged_decode()
    for b in range(3):
        attn.set_landmark_eval_decode(int(plen[b].item()), "extend_last_block", top_k=[2, 1, 3][b])
        qb = int(qpos[b].item())
        with torch.no_grad():
            ref = attn._decode_one(
                q[b : b + 1], k[b : b + 1, :, : qb + 1], v[b : b + 1, :, : qb + 1], qb
            )
        torch.testing.assert_close(out[b : b + 1], ref, rtol=1e-5, atol=1e-5, msg=f"row {b}")


def test_sparse_landmark_advertises_ragged_support():
    assert _build()._supports_ragged_decode is True


# --- the sparse ragged decode: batching and sparsity composed --------------------------------


@pytest.mark.parametrize("mode", ["extend_last_block", "generation_only"])
@pytest.mark.parametrize("top_k", [None, 1, 2, 50])
def test_sparse_ragged_decode_matches_scalar_per_row(mode, top_k):
    """``sparse_chunk_decode_ragged`` must match the scalar bs=1 decode row for row -- the same gate
    as the dense ragged decode above, for the version that also skips the dense scan."""
    from olmo_core.nn.attention.landmark_sparse_decode import sparse_chunk_decode_ragged

    torch.manual_seed(4)
    attn = _build(n_heads=4, n_kv_heads=2, head_dim=16)
    L, H, Hkv, D = attn.block_size, attn.n_heads, attn.n_kv_heads, attn.head_dim
    rows = [(3 * L, 3 * L - 1), (2 * L, 2 * L + 3), (5 * L, 5 * L + L), (L, L + 1)]
    B = len(rows)
    plen = torch.tensor([r[0] for r in rows])
    qpos = torch.tensor([r[1] for r in rows])
    total = int(qpos.max().item()) + 1
    k_kv, v_kv = torch.randn(B, Hkv, total, D), torch.randn(B, Hkv, total, D)
    q = torch.randn(B, H, 1, D)

    top_k_t = None if top_k is None else torch.full((B,), top_k, dtype=torch.long)
    attn.set_landmark_ragged_decode(plen, mode=mode, top_k=top_k_t)
    attn.set_ragged_qpos(qpos)
    out = sparse_chunk_decode_ragged(
        q,
        k_kv,
        v_kv,
        block_size=L,
        num_landmarks=attn.num_landmarks,
        softmax_scale=attn.softmax_scale,
        section_start=attn._ragged_section_start(),
        qpos=qpos.view(B, 1),
        top_k=top_k_t,
    )

    attn.clear_ragged_decode()
    n_rep = H // Hkv
    for b in range(B):
        attn.set_landmark_eval_decode(int(plen[b].item()), mode, top_k=top_k)
        qb = int(qpos[b].item())
        kb = repeat_kv(k_kv[b : b + 1, :, : qb + 1], n_rep)
        vb = repeat_kv(v_kv[b : b + 1, :, : qb + 1], n_rep)
        with torch.no_grad():
            ref = attn._decode_one(q[b : b + 1], kb, vb, qb)
        torch.testing.assert_close(out[b : b + 1], ref, rtol=1e-5, atol=1e-5, msg=f"row {b}")


def test_sparse_ragged_decode_per_row_top_k_and_multi_landmark():
    from olmo_core.nn.attention.landmark_sparse_decode import sparse_chunk_decode_ragged

    torch.manual_seed(5)
    attn = _build(num_landmarks=2, mem_freq=6, n_heads=4, n_kv_heads=4, head_dim=16)
    L, H, D = attn.block_size, attn.n_heads, attn.head_dim
    plen = torch.tensor([6 * L, 4 * L, 6 * L])
    qpos = torch.tensor([6 * L + 1, 4 * L + 5, 6 * L - 1])
    top_k = torch.tensor([1, 2, 99])
    total = int(qpos.max().item()) + 1
    k_kv, v_kv, q = (
        torch.randn(3, H, total, D),
        torch.randn(3, H, total, D),
        torch.randn(3, H, 1, D),
    )

    attn.set_landmark_ragged_decode(plen, mode="extend_last_block", top_k=top_k)
    attn.set_ragged_qpos(qpos)
    out = sparse_chunk_decode_ragged(
        q,
        k_kv,
        v_kv,
        block_size=L,
        num_landmarks=attn.num_landmarks,
        softmax_scale=attn.softmax_scale,
        section_start=attn._ragged_section_start(),
        qpos=qpos.view(3, 1),
        top_k=top_k,
    )
    attn.clear_ragged_decode()
    for b in range(3):
        attn.set_landmark_eval_decode(int(plen[b].item()), "extend_last_block", top_k=int(top_k[b]))
        qb = int(qpos[b].item())
        with torch.no_grad():
            ref = attn._decode_one(
                q[b : b + 1], k_kv[b : b + 1, :, : qb + 1], v_kv[b : b + 1, :, : qb + 1], qb
            )
        torch.testing.assert_close(out[b : b + 1], ref, rtol=1e-5, atol=1e-5, msg=f"row {b}")
