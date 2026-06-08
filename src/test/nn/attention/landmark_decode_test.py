"""Correctness of KV-cache generation (prefill + incremental decode) for the landmark mixers.

The gold-standard check: a single no-cache full-sequence forward must equal a cached single-shot
prefill of a prompt prefix followed by one-token-at-a-time decode for the rest. This exercises the
attention math, the cache read/write indexing, and RoPE position handling together.
"""
import os

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.attention.kv_cache import KVCacheManager
from olmo_core.nn.attention.landmark_kernel import has_landmark_kernel
from olmo_core.nn.attention.landmark_sparse_kernel import has_sparse_kernel
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.testing import requires_gpu


def _build(name, *, mem_freq, num_landmarks=None, dtype, device, d_model=128):
    kwargs = dict(
        name=name,
        n_heads=8,
        n_kv_heads=2,  # exercise GQA repeat_kv on the cache read path
        head_dim=64,
        bias=False,
        mem_freq=mem_freq,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
        rope=RoPEConfig(name=RoPEType.default, theta=10_000),
        dtype=dtype,
    )
    if num_landmarks is not None:
        kwargs["num_landmarks"] = num_landmarks
    attn = AttentionConfig(**kwargs).build(d_model, layer_idx=0, n_layers=1, init_device=device)
    attn.eval()
    return attn


def _check_decode_matches_full(attn, *, N, P, d_model, dtype, device, atol, rtol):
    torch.manual_seed(0)
    x = torch.randn(1, N, d_model, device=device, dtype=dtype)

    with torch.no_grad():
        # Reference: one no-cache full-sequence forward.
        out_ref = attn(x)

        # Cached: single-shot prefill of the first P tokens, then decode the rest one at a time.
        # Build the cache with the test dtype (init_kv_cache_manager hardcodes bf16; in real
        # generation the whole model is bf16, but here we want fp32 caches for the fp32 case).
        attn.kv_cache_manager = KVCacheManager(
            batch_size=1,
            max_seq_len=N,
            num_kv_heads=attn.n_kv_heads,
            head_dim=attn.head_dim,
            device=x.device,
            dtype=x.dtype,
        )
        chunks = [attn(x[:, :P])]
        for t in range(P, N):
            chunks.append(attn(x[:, t : t + 1]))
        out_gen = torch.cat(chunks, dim=1)

    attn.kv_cache_manager = None  # restore no-cache mode

    assert out_gen.shape == out_ref.shape
    assert torch.isfinite(out_gen).all()
    max_err = (out_gen.float() - out_ref.float()).abs().max().item()
    torch.testing.assert_close(out_gen, out_ref, atol=atol, rtol=rtol)
    return max_err


@requires_gpu
@pytest.mark.gpu
def test_sparse_landmark_decode_matches_full_eager():
    # Eager torch path (LM_SPARSE_KERNEL=0); fp32 -> tight tolerance, validates the cache/RoPE
    # plumbing and the decode mask precisely.
    os.environ["LM_SPARSE_KERNEL"] = "0"
    attn = _build(
        AttentionType.sparse_landmark,
        mem_freq=15,
        num_landmarks=1,
        dtype=DType.float32,
        device="cuda",
    )
    err = _check_decode_matches_full(
        attn, N=64, P=20, d_model=128, dtype=torch.float32, device="cuda", atol=1e-4, rtol=1e-4
    )
    print(f"sparse eager max_err={err:.2e}")


@requires_gpu
@pytest.mark.gpu
@pytest.mark.skipif(not has_sparse_kernel(), reason="sparse landmark triton kernel unavailable")
def test_sparse_landmark_decode_matches_full_kernel():
    os.environ["LM_SPARSE_KERNEL"] = "1"
    attn = _build(
        AttentionType.sparse_landmark, mem_freq=15, num_landmarks=1, dtype=DType.bfloat16, device="cuda"
    )
    err = _check_decode_matches_full(
        attn, N=64, P=20, d_model=128, dtype=torch.bfloat16, device="cuda", atol=4e-2, rtol=4e-2
    )
    print(f"sparse kernel max_err={err:.2e}")


@requires_gpu
@pytest.mark.gpu
@pytest.mark.skipif(not has_landmark_kernel(), reason="landmark triton kernel unavailable")
def test_fast_landmark_decode_matches_full_kernel():
    attn = _build(AttentionType.fast_landmark, mem_freq=15, dtype=DType.bfloat16, device="cuda")
    err = _check_decode_matches_full(
        attn, N=64, P=20, d_model=128, dtype=torch.bfloat16, device="cuda", atol=4e-2, rtol=4e-2
    )
    print(f"fast kernel max_err={err:.2e}")
