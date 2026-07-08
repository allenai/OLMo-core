"""Tests for DilatedSlidingWindowAttention -- a dilated causal sliding-window mask whose dilation
stride rotates with transformer depth ("Hierarchical K")."""

import pytest
import torch

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    AttentionBackendName,
    AttentionConfig,
    AttentionType,
    DilatedSlidingWindowAttention,
)
from olmo_core.nn.attention.dilated_window import (
    build_dilated_window_allowed_mask,
    layer_dilation,
)
from olmo_core.nn.attention.kv_cache import KVCacheManager
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType
from olmo_core.nn.transformer import TransformerConfig


def _attention(name: AttentionType, *, layer_idx: int = 0, n_layers: int = 12, **kw):
    config = AttentionConfig(
        name=name,
        n_heads=8,
        n_kv_heads=2,
        head_dim=8,
        bias=False,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
        rope=RoPEConfig(name=RoPEType.default, theta=10_000),
        **kw,
    )
    return config.build(64, layer_idx=layer_idx, n_layers=n_layers)


def _dilated_attention(*, layer_idx: int = 0, n_layers: int = 12, **kw):
    attn = _attention(
        AttentionType.dilated_sliding_window, layer_idx=layer_idx, n_layers=n_layers, **kw
    )
    assert isinstance(attn, DilatedSlidingWindowAttention)
    return attn


# ---------------------------------------------------------------------------
# Pure mask logic: exact attend/no-attend pattern + rotation.
# ---------------------------------------------------------------------------


def _allowed_offsets(dilation: int, window: int, *, t: int = 20) -> set:
    """The set of key *offsets* (t - k) a query at position ``t`` may attend, from the mask."""
    positions = torch.arange(t + 1)
    allowed = build_dilated_window_allowed_mask(
        positions, positions, dilation=dilation, window=window
    )
    row = allowed[t]  # (t+1,) over keys 0..t
    keys = row.nonzero(as_tuple=True)[0].tolist()
    return {t - k for k in keys}


def test_layer_dilation_rotation():
    # base=2, L=3 -> strides {1, 2, 4} repeating, resetting at layer L.
    strides = [layer_dilation(ell, num_configs=3, base=2) for ell in range(7)]
    assert strides == [1, 2, 4, 1, 2, 4, 1]


@pytest.mark.parametrize(
    "cycle_pos, dilation, expected_offsets",
    [
        (0, 1, {0, 1, 2}),  # ...OOOXXX
        (1, 2, {0, 2, 4}),  # ...OXOXOX
        (2, 4, {0, 4, 8}),  # ...OXOOOX
    ],
)
def test_dilated_window_offsets_per_cycle_position(cycle_pos, dilation, expected_offsets):
    assert layer_dilation(cycle_pos, num_configs=3, base=2) == dilation
    assert _allowed_offsets(dilation, window=3) == expected_offsets


def test_dilated_window_is_causal():
    positions = torch.arange(8)
    allowed = build_dilated_window_allowed_mask(positions, positions, dilation=2, window=3)
    # No query may attend a strictly-future key: upper triangle (k > q) is all False.
    upper = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
    assert not allowed[upper].any()
    # Every query attends its own position (offset 0 is always in-window).
    assert allowed.diagonal().all()


def test_dilated_window_truncates_at_sequence_start():
    # Query at position 3 with dilation 4, K=3 wants offsets {0,4,8} -> only offset 0 (pos 3) is valid.
    assert _allowed_offsets(dilation=4, window=3, t=3) == {0}
    # Query at position 5 with dilation 4 -> offsets {0,4} (pos 5 and 1); offset 8 is out of range.
    assert _allowed_offsets(dilation=4, window=3, t=5) == {0, 4}


def test_dilated_window_reset_at_cycle_length():
    # Layer L (== num_configs) reuses layer 0's stride: identical masks.
    positions = torch.arange(12)
    d0 = layer_dilation(0, num_configs=3, base=2)
    d3 = layer_dilation(3, num_configs=3, base=2)
    assert d0 == d3 == 1
    m0 = build_dilated_window_allowed_mask(positions, positions, dilation=d0, window=3)
    m3 = build_dilated_window_allowed_mask(positions, positions, dilation=d3, window=3)
    assert torch.equal(m0, m3)


# ---------------------------------------------------------------------------
# Module wiring / config.
# ---------------------------------------------------------------------------


def test_dilated_window_config_builds_defaults():
    attn = _dilated_attention(layer_idx=0)
    assert attn.window == 3
    assert attn.num_configs == 3
    assert attn.base == 2
    assert attn.dilation == 1


def test_dilated_window_layer_idx_sets_stride():
    assert _dilated_attention(layer_idx=0).dilation == 1
    assert _dilated_attention(layer_idx=1).dilation == 2
    assert _dilated_attention(layer_idx=2).dilation == 4
    assert _dilated_attention(layer_idx=3).dilation == 1  # reset


def test_dilated_window_custom_fields():
    attn = _dilated_attention(
        layer_idx=2, dilated_window_k=4, dilated_window_num_configs=2, dilated_window_base=3
    )
    assert attn.window == 4
    assert attn.num_configs == 2
    assert attn.base == 3
    # layer_idx=2, L=2 -> cycle_pos 0 -> stride 3**0 = 1.
    assert attn.dilation == 1


def test_dilated_window_fields_rejected_on_other_type():
    with pytest.raises(OLMoConfigurationError):
        _attention(AttentionType.default, dilated_window_k=3)


def test_dilated_window_window_size_rejected():
    from olmo_core.nn.attention import SlidingWindowAttentionConfig

    with pytest.raises(OLMoConfigurationError):
        _attention(
            AttentionType.dilated_sliding_window,
            sliding_window=SlidingWindowAttentionConfig(
                pattern=[4], force_full_attention_on_first_layer=False
            ),
        )


def test_dilated_window_bad_k_rejected():
    with pytest.raises(OLMoConfigurationError):
        _dilated_attention(dilated_window_k=0)


# ---------------------------------------------------------------------------
# Forward / backward behavior.
# ---------------------------------------------------------------------------


def test_dilated_window_forward_runs():
    attn = _dilated_attention(layer_idx=1)
    attn.eval()
    B, T, d_model = 2, 16, 64
    x = torch.randn(B, T, d_model)
    with torch.no_grad():
        out = attn(x)
    assert out.shape == (B, T, d_model)
    assert torch.isfinite(out).all()


def test_dilated_window_backward():
    attn = _dilated_attention(layer_idx=2)
    x = torch.randn(1, 16, 64, requires_grad=True)
    attn(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_dilated_window_respects_mask_via_perturbation():
    # Layer with dilation d=2, K=3: query at position 8 attends only positions {8, 6, 4}. Perturbing
    # an *allowed* key (pos 6) must change its output; perturbing a *disallowed* key (pos 7 or pos 2)
    # must not.
    attn = _dilated_attention(layer_idx=1)  # dilation 2
    assert attn.dilation == 2
    attn.eval()
    x = torch.randn(1, 16, 64)

    def out_at(perturb_pos):
        x2 = x.clone()
        x2[0, perturb_pos] += 5.0
        with torch.no_grad():
            return attn(x2)[0, 8]

    with torch.no_grad():
        base = attn(x)[0, 8]
    # Allowed key (offset 2 -> pos 6): output changes.
    assert not torch.allclose(base, out_at(6), atol=1e-4)
    # Disallowed key on wrong stride (offset 1 -> pos 7): output unchanged.
    assert torch.allclose(base, out_at(7), atol=1e-6)
    # Disallowed key out of window (offset 6 -> pos 2): output unchanged.
    assert torch.allclose(base, out_at(2), atol=1e-6)


# ---------------------------------------------------------------------------
# KV-cache decoding: cached prefill + incremental decode must equal the eager full recompute.
# ---------------------------------------------------------------------------


def test_dilated_window_rectangular_decode_mask_matches_square():
    """The single-query decode geometry (query at absolute position ``P`` over cached keys
    ``0..P``) must reproduce exactly row ``P`` of the square prefill mask -- this is why the
    offset-based mask generalizes from prefill to decode unchanged."""
    T = 24
    for dilation, window in [(1, 3), (2, 3), (4, 3), (2, 4), (3, 2)]:
        square = build_dilated_window_allowed_mask(
            torch.arange(T), torch.arange(T), dilation=dilation, window=window
        )
        for P in range(T):
            rect = build_dilated_window_allowed_mask(
                torch.arange(P, P + 1), torch.arange(P + 1), dilation=dilation, window=window
            )
            assert torch.equal(rect[0], square[P, : P + 1])


def _check_decode_matches_full(attn, *, N, P, d_model, dtype, device, atol, rtol):
    """Gold standard: one no-cache full-sequence forward must equal a single-shot cached prefill of
    the first ``P`` tokens followed by one-token-at-a-time decode of the rest. Exercises the cache
    read/write indexing, the absolute-position dilated mask, and RoPE position handling together."""
    torch.manual_seed(0)
    x = torch.randn(1, N, d_model, device=device, dtype=dtype)

    with torch.no_grad():
        # Reference: one no-cache full-sequence forward.
        out_ref = attn(x)

        # Cached: prefill the first P tokens, then decode the rest one token at a time. Build an
        # fp32 cache directly (init_kv_cache_manager hardcodes bf16; here we want a matching dtype).
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


@pytest.mark.parametrize(
    "layer_idx, dilation",
    [(0, 1), (1, 2), (2, 4), (3, 1)],  # spans strides 1, 2, 4 and the cycle reset.
)
def test_dilated_window_decode_matches_full_eager(layer_idx, dilation):
    attn = _dilated_attention(layer_idx=layer_idx)
    attn.eval()
    assert attn.dilation == dilation
    # Prefill length P must clip the window (P > window*dilation, worst case 3*4=12) so decode
    # actually drops keys that fall out of the dilated window rather than trivially attending all.
    err = _check_decode_matches_full(
        attn, N=48, P=24, d_model=64, dtype=torch.float32, device="cpu", atol=1e-4, rtol=1e-4
    )
    assert err < 1e-4


def test_dilated_window_multilayer_decode_matches_full():
    """End-to-end: a multi-layer transformer whose dilated layers span strides {1, 2, 4} must
    produce identical (teacher-forced) logits whether decoded with the KV cache (prefill +
    incremental decode) or recomputed over the full sequence each step."""
    torch.manual_seed(0)
    N, P, vocab = 40, 20, 256
    config = TransformerConfig.llama_like(
        d_model=64,
        vocab_size=vocab,
        n_layers=4,
        n_heads=8,
        n_kv_heads=2,  # GQA -> exercises _repeat_kv on the cache read path
        qk_norm=True,
        use_head_qk_norm=True,
        rope_type=RoPEType.default,
        rope_theta=10_000,
        dilated_sliding_window=True,
        dilated_window_k=3,
        dilated_window_num_configs=3,
        dilated_window_base=2,
        attn_backend=AttentionBackendName.torch,  # CPU: no flash / KV-cache backend support
        dtype=DType.float32,
    )
    model = config.build(init_device="cpu")
    model.eval()

    blocks = list(model.blocks.values())
    attns = [b.attention for b in blocks]
    assert all(isinstance(a, DilatedSlidingWindowAttention) for a in attns)
    # Strides genuinely vary across depth (the whole point of the "Hierarchical K" rotation).
    assert [a.dilation for a in attns] == [1, 2, 4, 1]

    x = torch.randint(0, vocab, (1, N))
    with torch.no_grad():
        # Reference: full-sequence forward, next-token logits at every position.
        logits_ref = model(x)  # (1, N, vocab)

        # Cached: attach fp32 KV caches, prefill the first P tokens, decode the rest one at a time.
        for a in attns:
            a.kv_cache_manager = KVCacheManager(
                batch_size=1,
                max_seq_len=N,
                num_kv_heads=a.n_kv_heads,
                head_dim=a.head_dim,
                device=x.device,
                dtype=torch.float32,
            )
        chunks = [model(x[:, :P])]  # (1, P, vocab)
        for t in range(P, N):
            chunks.append(model(x[:, t : t + 1]))  # (1, 1, vocab)
        logits_gen = torch.cat(chunks, dim=1)  # (1, N, vocab)
        for a in attns:
            a.kv_cache_manager = None

    assert logits_gen.shape == logits_ref.shape
    torch.testing.assert_close(logits_gen, logits_ref, atol=1e-4, rtol=1e-4)


def test_dilated_window_init_kv_cache_ignores_backend_support():
    """DilatedSlidingWindowAttention self-manages its cache and never calls the backend during
    decode, so it must allocate a KV cache even with a backend (torch) that reports no KV-cache
    support -- this is what enables fast cached eval regardless of backend."""
    attn = _dilated_attention(layer_idx=1, backend=AttentionBackendName.torch)
    # The torch backend refuses KV caching...
    with pytest.raises(RuntimeError):
        attn.backend.assert_supports_kv_cache()
    # ...but the dilated layer allocates its own cache anyway.
    assert attn.kv_cache_manager is None
    attn.init_kv_cache_manager(batch_size=1, max_seq_len=16)
    assert attn.kv_cache_manager is not None
    assert attn.kv_cache_manager.k_cache.shape == (1, 16, attn.n_kv_heads, attn.head_dim)
    attn.kv_cache_manager = None


def test_dilated_window_decode_left_padding_masks_pad_columns():
    """With left-padding the offset-based pattern is unchanged (a per-row shift cancels in
    ``q_pos - k_pos``), but the padded key columns must be forbidden so no query attends padding."""
    attn = _dilated_attention(layer_idx=0)  # dilation 1, window 3 -> attends offsets {0, 1, 2}
    q_pos = torch.tensor([2])
    kv_pos = torch.arange(3)
    m_no = attn._additive_mask_from_positions(q_pos, kv_pos, dtype=torch.float32)
    m_lp = attn._additive_mask_from_positions(
        q_pos,
        kv_pos,
        dtype=torch.float32,
        cache_leftpad=torch.tensor([2], dtype=torch.int32),  # cols 0,1 are padding
        batch_size=1,
    )
    assert m_no.shape == (1, 1, 1, 3)
    assert m_lp.shape == (1, 1, 1, 3)
    # No left-pad: all three in-window keys (cols 0,1,2) are allowed.
    assert (m_no[0, 0, 0] == 0.0).tolist() == [True, True, True]
    # Left-pad 2: cols 0,1 are padding -> masked; only the real key (col 2) survives.
    assert (m_lp[0, 0, 0] == 0.0).tolist() == [False, False, True]
