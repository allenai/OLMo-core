"""Sparse cached prefill must reproduce the full sparse mask, including GQA and ragged tails."""
import pytest
import torch

from olmo_core.nn.attention import AttentionConfig, AttentionType
from olmo_core.nn.attention.landmark import repeat_kv
from olmo_core.nn.attention.landmark_sparse import sparse_landmark_attention_ref
from olmo_core.nn.attention.landmark_sparse_kernel import (
    sparse_landmark_attention_triton,
)
from olmo_core.nn.rope import RoPEConfig


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("landmarks", [1, 2])
def test_sparse_suffix_matches_reference(device, kv_heads, landmarks):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(17)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    q = torch.randn(2, 4, 80, 16, device=device, dtype=dtype)
    k = torch.randn(2, kv_heads, 80, 16, device=device, dtype=dtype)
    v = torch.randn_like(k)
    expected = sparse_landmark_attention_ref(
        q.float(),
        repeat_kv(k, 4 // kv_heads).float(),
        repeat_kv(v, 4 // kv_heads).float(),
        16,
        landmarks,
    )
    attn = AttentionConfig(
        name=AttentionType.sparse_landmark,
        n_heads=4,
        n_kv_heads=kv_heads,
        head_dim=16,
        mem_freq=16 - landmarks,
        num_landmarks=landmarks,
    ).build(64, layer_idx=0, n_layers=1, init_device=device)
    for start, stop in [(0, 32), (32, 64), (64, 77)]:
        got = attn._prefill(q[:, :, start:stop], k[:, :, :stop], v[:, :, :stop])
        torch.testing.assert_close(
            got.float(),
            expected[:, :, start:stop],
            atol=0.025 if device == "cuda" else 1e-5,
            rtol=0.025 if device == "cuda" else 1e-5,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("chunk", [16, 32, 48])
def test_sparse_cached_chunks_and_decode_match_one_shot(device, chunk):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(23)
    a = AttentionConfig(
        name=AttentionType.sparse_landmark,
        n_heads=4,
        n_kv_heads=1,
        head_dim=16,
        mem_freq=15,
        rope=RoPEConfig(),
    ).build(64, layer_idx=0, n_layers=1, init_device=device)
    a.eval()
    x = torch.randn(1, 79, 64, device=device)
    with torch.no_grad():
        a.init_kv_cache_manager(1, 100)
        expected = torch.cat([a(x[:, :77]), a(x[:, 77:78]), a(x[:, 78:])], dim=1)
        a.init_kv_cache_manager(1, 100)
        actual = torch.cat(
            [a(x[:, s : min(s + chunk, 77)]) for s in range(0, 77, chunk)]
            + [a(x[:, 77:78]), a(x[:, 78:])],
            dim=1,
        )
    assert int(a.kv_cache_manager.current_position()) == 79
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_sparse_misaligned_prefill_rejected():
    from olmo_core.exceptions import OLMoConfigurationError

    a = AttentionConfig(
        name=AttentionType.sparse_landmark, n_heads=2, head_dim=16, mem_freq=15
    ).build(32, layer_idx=0, n_layers=1)
    a.eval()
    a.init_kv_cache_manager(1, 64)
    with torch.no_grad():
        a(torch.randn(1, 17, 32))
        with pytest.raises(OLMoConfigurationError, match="block boundary"):
            a(torch.randn(1, 16, 32))
