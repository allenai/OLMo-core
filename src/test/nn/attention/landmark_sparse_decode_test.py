"""Parity of the genuinely sparse ``SparseLandmarkAttention`` decode against the shipped
dense-masked decode.

``SparseLandmarkAttention._decode_one`` (the shipped path) ``repeat_kv``-expands the whole KV cache,
scores every cached key, and masks all but the local section and the past chunks' landmarks to
``-inf``. :func:`sparse_chunk_decode` scores only the keys that survive that mask. The two must
agree, so this file pins:

* per-step equality of ``sparse_chunk_decode`` vs ``_decode_one`` across decode modes, top-k values,
  GQA vs MHA, ``num_landmarks`` > 1 and both query regimes (prompt-position / generated);
* identical *selection* under top-k (a hard, discrete decision -- not a tolerance question);
* end-to-end equality of a patched vs unpatched layer's ``_forward_generate`` over a prefill +
  multi-step decode, which is what :func:`enable_sparse_decode` actually installs.

The sparse mixer's eager decode math is pure torch, so all of this runs on CPU with no Triton kernel
and no GPU. Everything runs in float32; the sparse path splits the softmax denominator and the
``A @ V`` reduction into a landmark term and a local term, so agreement is to floating-point
reassociation, not bitwise -- except for the selection tests, which are exact.
"""

import os

import pytest
import torch

from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.attention.landmark import repeat_kv
from olmo_core.nn.attention.landmark_sparse_decode import (
    disable_sparse_decode,
    enable_sparse_decode,
    landmark_positions,
    sparse_chunk_decode,
)
from olmo_core.nn.layer_norm import LayerNormConfig

os.environ.setdefault("LM_SPARSE_KERNEL", "0")  # CPU: eager sparse-landmark core


def _build(*, mem_freq=3, num_landmarks=1, n_heads=4, n_kv_heads=4, head_dim=16):
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


def _reference(attn, q, k_kv, v_kv, qpos):
    """The shipped decode: GQA-expand the cache, dense scan, mask."""
    n_rep = attn.n_heads // attn.n_kv_heads
    total = qpos + 1
    kh = repeat_kv(k_kv[:, :, :total], n_rep)
    vh = repeat_kv(v_kv[:, :, :total], n_rep)
    with torch.no_grad():
        return attn._decode_one(q, kh, vh, qpos)


def _sparse(attn, q, k_kv, v_kv, qpos, section_start):
    return sparse_chunk_decode(
        q,
        k_kv,
        v_kv,
        block_size=attn.block_size,
        num_landmarks=attn.num_landmarks,
        softmax_scale=attn.softmax_scale,
        section_start=section_start,
        total=qpos + 1,
        top_k=attn._eval_top_k,
    )


def _section_start(attn, qpos, prompt_len, mode):
    L = attn.block_size
    if prompt_len is not None and qpos >= prompt_len:
        return (prompt_len // L) * L if mode == "extend_last_block" else prompt_len
    return (qpos // L) * L


def test_landmark_positions_matches_mask():
    # The O(n_lm) index construction must reproduce the boolean mask the shipped decode builds,
    # including a section boundary that falls *inside* a chunk (generation_only mode).
    for L, G in ((4, 1), (8, 3), (16, 2)):
        for section_start in range(0, 3 * L + 1):
            idx = landmark_positions(section_start, L, G, torch.device("cpu"))
            j = torch.arange(section_start)
            expect = j[(j % L) >= (L - G)]
            assert torch.equal(idx, expect.to(idx.dtype)), (L, G, section_start)


@pytest.mark.parametrize("mode", ["extend_last_block", "generation_only"])
@pytest.mark.parametrize("top_k", [None, 1, 2, 100])
@pytest.mark.parametrize("n_kv_heads", [4, 2, 1])
@pytest.mark.parametrize("num_landmarks", [1, 2])
def test_sparse_chunk_decode_matches_shipped(mode, top_k, n_kv_heads, num_landmarks):
    torch.manual_seed(0)
    mem_freq = 4
    attn = _build(mem_freq=mem_freq, num_landmarks=num_landmarks, n_kv_heads=n_kv_heads)
    L = attn.block_size  # 4 + num_landmarks
    H, Hkv, D = attn.n_heads, attn.n_kv_heads, attn.head_dim

    prompt_len = 5 * L  # 5 full chunks
    max_len = prompt_len + 2 * L
    k_kv = torch.randn(2, Hkv, max_len, D)
    v_kv = torch.randn(2, Hkv, max_len, D)

    # Query positions covering: an early prompt position (per-chunk decode), the final prompt token
    # (per-chunk, the first decode step of the generation loop), a landmark-position prompt token,
    # and several generated positions (eval / one-long-local-block).
    positions = [
        L + 1,
        L - 1,
        2 * L - 1,
        prompt_len - 1,
        prompt_len,
        prompt_len + 3,
        prompt_len + L,
    ]
    for qpos in positions:
        q = torch.randn(2, H, 1, D)
        attn.set_landmark_eval_decode(prompt_len, mode, top_k=top_k)
        ref = _reference(attn, q, k_kv, v_kv, qpos)
        got = _sparse(attn, q, k_kv, v_kv, qpos, _section_start(attn, qpos, prompt_len, mode))
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5, msg=f"qpos={qpos}")


def test_sparse_chunk_decode_non_eval_mode():
    # With eval decode cleared (``_eval_prompt_len is None``) every query uses the per-chunk rule.
    torch.manual_seed(1)
    attn = _build(mem_freq=7, num_landmarks=1, n_heads=4, n_kv_heads=2)
    L, H, Hkv, D = attn.block_size, attn.n_heads, attn.n_kv_heads, attn.head_dim
    k_kv, v_kv = torch.randn(1, Hkv, 5 * L, D), torch.randn(1, Hkv, 5 * L, D)
    attn.clear_landmark_eval_decode()
    for qpos in (0, 3, L, 2 * L + 5, 4 * L - 1):
        q = torch.randn(1, H, 1, D)
        ref = _reference(attn, q, k_kv, v_kv, qpos)
        got = _sparse(attn, q, k_kv, v_kv, qpos, (qpos // L) * L)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5, msg=f"qpos={qpos}")


@pytest.mark.parametrize("top_k", [1, 2, 3])
def test_sparse_decode_selects_the_same_chunks(top_k):
    """Top-k is a hard discrete decision: the *support* of the output weights must match exactly."""
    torch.manual_seed(2)
    attn = _build(mem_freq=3, num_landmarks=1, n_heads=1, n_kv_heads=1, head_dim=15)
    L = attn.block_size  # 4
    prompt_len, qpos = 5 * L, 5 * L + 2
    total = qpos + 1
    k_kv = torch.randn(1, 1, total, 15)
    q = torch.randn(1, 1, 1, 15)
    # One-hot values read the attention weights straight off the output.
    v_kv = torch.eye(total).view(1, 1, total, total)
    attn.set_landmark_eval_decode(prompt_len, "extend_last_block", top_k=top_k)

    with torch.no_grad():
        ref = attn._decode_one(q, k_kv, v_kv, qpos).view(-1)
    got = sparse_chunk_decode(
        q,
        k_kv,
        v_kv,
        block_size=L,
        num_landmarks=1,
        softmax_scale=attn.softmax_scale,
        section_start=prompt_len,
        total=total,
        top_k=top_k,
    ).view(-1)
    support_ref = {i for i in range(total) if abs(float(ref[i])) > 1e-6}
    support_got = {i for i in range(total) if abs(float(got[i])) > 1e-6}
    assert support_got == support_ref
    # Exactly ``top_k`` past chunks retrieved, plus the local section.
    landmarks = {i for i in support_ref if i < prompt_len}
    assert len(landmarks) == top_k
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def test_enable_sparse_decode_end_to_end_matches():
    """A prefill + multi-step decode through the patched ``_forward_generate`` must match the
    unpatched layer step for step -- this is what the generation module installs."""
    torch.manual_seed(3)
    attn = _build(mem_freq=7, num_landmarks=1, n_heads=4, n_kv_heads=2, head_dim=16)
    L, d_model = attn.block_size, attn.n_heads * attn.head_dim
    prompt_len, n_steps = 6 * L, 5
    max_len = prompt_len + n_steps + 1

    prompt = torch.randn(1, prompt_len, d_model)
    steps = [torch.randn(1, 1, d_model) for _ in range(n_steps)]

    model = torch.nn.Module()  # a bare holder so enable_sparse_decode can walk .modules()
    model.attn = attn

    def run():
        attn.init_kv_cache_manager(1, max_len)
        attn.set_landmark_eval_decode(prompt_len, "extend_last_block", top_k=2)
        outs = []
        with torch.no_grad():
            attn(prompt)  # prefill 0..prompt_len-1
            for s in steps:
                outs.append(attn(s))
        attn.clear_landmark_eval_decode()
        attn.kv_cache_manager = None
        return torch.cat(outs, dim=1)

    ref = run()
    assert enable_sparse_decode(model) == 1
    got = run()
    assert disable_sparse_decode(model) == 1
    again = run()

    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(again, ref, rtol=0, atol=0)  # unpatching is exact


def test_enable_sparse_decode_non_strict_is_a_noop():
    model = torch.nn.Module()
    model.lin = torch.nn.Linear(4, 4)
    with pytest.raises(RuntimeError):
        enable_sparse_decode(model)
    assert enable_sparse_decode(model, strict=False) == 0
