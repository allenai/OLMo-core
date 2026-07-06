"""
Tests for the multiple-landmarks-per-block variants (:mod:`olmo_core.nn.attention.landmark_multi`).

Covers, in particular:
  * **regression / non-interference** -- with ``num_landmarks == 1`` the new grouped softmax and the
    new attention modules reproduce the original single-landmark
    :func:`~olmo_core.nn.attention.landmark.landmark_grouped_softmax` /
    :class:`~olmo_core.nn.attention.LandmarkAttention` /
    :class:`~olmo_core.nn.attention.DocumentLandmarkAttention` exactly;
  * **correctness for num_landmarks > 1** -- the ``"sum"`` and ``"max"`` pools match an independent
    pure-Python two-level-softmax reference, and every query row normalizes to 1;
  * a coarse **speed** check that the new ``num_landmarks == 1`` path is not dramatically slower than
    the legacy grouped softmax.
"""

import time

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    AttentionConfig,
    AttentionType,
    DocumentLandmarkAttention,
    DocumentMultiLandmarkAttention,
    LandmarkAttention,
    MultiLandmarkAttention,
)
from olmo_core.nn.attention.landmark import landmark_grouped_softmax
from olmo_core.nn.attention.landmark_multi import (
    _single_doc_masks,
    multi_landmark_grouped_softmax,
)
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType


def _multi_landmark(*, mem_freq=3, num_landmarks=2, landmark_pool="sum", qk_norm=True, **kw):
    config = AttentionConfig(
        name=AttentionType.multi_landmark,
        n_heads=8,
        n_kv_heads=2,
        head_dim=8,
        bias=False,
        mem_freq=mem_freq,
        num_landmarks=num_landmarks,
        landmark_pool=landmark_pool,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False) if qk_norm else None,
        use_head_qk_norm=qk_norm,
        rope=RoPEConfig(name=RoPEType.default, theta=10_000),
        **kw,
    )
    attn = config.build(64, layer_idx=0, n_layers=1)
    assert isinstance(attn, MultiLandmarkAttention)
    return attn


def _reference_two_level_softmax(raw, attn_mask, is_mem, last_section, block_size, pool):
    """
    Independent pure-Python reference for the dense multi-landmark grouped softmax, operating on a
    single ``(T, T)`` logit matrix and the additive/boolean masks. Mirrors the documented semantics:
    a top ("memory") softmax over the query's own section plus every active landmark; each block's
    gate pools its landmarks (``"sum"`` = add; ``"max"`` = keep only the block's argmax landmark,
    others demoted to ``-inf`` before the sum -- exactly what the implementation does); each earlier
    block's regular tokens get a within-block softmax scaled by that gate.
    """
    T = raw.shape[0]
    finfo_min = float(torch.finfo(raw.dtype).min)
    eff = (raw + attn_mask).double()
    block = (torch.arange(T) // block_size).tolist()
    n_blocks = max(block) + 1
    out = torch.zeros(T, T, dtype=torch.float64)

    for i in range(T):
        row = eff[i].clone()
        if pool == "max":
            for b in range(n_blocks):
                idxs = [j for j in range(T) if bool(is_mem[i, j]) and block[j] == b]
                if idxs:
                    bmax = max(row[j].item() for j in idxs)
                    for j in idxs:
                        if row[j].item() < bmax:
                            row[j] = finfo_min

        topm = [bool(is_mem[i, j] or last_section[i, j]) for j in range(T)]
        tl = row.clone()
        for j in range(T):
            if not topm[j]:
                tl[j] = float("-inf")
        p_top = torch.softmax(tl, dim=0)

        gate = [0.0] * n_blocks
        for j in range(T):
            if bool(is_mem[i, j]):
                gate[block[j]] += p_top[j].item()

        for b in range(n_blocks):
            regm = [block[j] == b and not topm[j] for j in range(T)]
            if any(regm):
                wl = row.clone()
                for j in range(T):
                    if not regm[j]:
                        wl[j] = float("-inf")
                w = torch.softmax(wl, dim=0)
                out[i] += w * gate[b]

        for j in range(T):
            if bool(last_section[i, j]):
                out[i, j] += p_top[j]

    return out


# ---------------------------------------------------------------------------
# Config / construction
# ---------------------------------------------------------------------------


def test_multi_landmark_config_builds():
    attn = _multi_landmark(mem_freq=3, num_landmarks=2, landmark_pool="max")
    assert attn.mem_freq == 3
    assert attn.num_landmarks == 2
    assert attn.block_size == 5
    assert attn.landmark_pool == "max"


def test_multi_landmark_rejects_bad_args():
    with pytest.raises(OLMoConfigurationError):
        _multi_landmark(num_landmarks=0)
    with pytest.raises(OLMoConfigurationError):
        _multi_landmark(landmark_pool="mean")
    with pytest.raises(OLMoConfigurationError):
        # The fused kernel is not supported for multi-landmark.
        MultiLandmarkAttention(mem_freq=15, num_landmarks=2, use_kernel=True, n_heads=8, d_model=64)


def test_landmark_pool_rejected_on_non_multi():
    with pytest.raises(OLMoConfigurationError):
        AttentionConfig(
            name=AttentionType.landmark, n_heads=8, mem_freq=3, landmark_pool="sum"
        ).build(64, layer_idx=0, n_layers=1)


# ---------------------------------------------------------------------------
# Regression: num_landmarks == 1 reproduces the legacy single-landmark path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool", ["sum", "max"])
def test_grouped_softmax_matches_legacy_when_single(pool):
    torch.manual_seed(0)
    block_size = 4  # mem_freq=3, num_landmarks=1
    B, H, T = 2, 3, 12
    attn_mask, is_mem, last_section = _single_doc_masks(
        T,
        torch.device("cpu"),
        torch.float32,
        block_size=block_size,
        num_landmarks=1,
        cu_doc_lens=None,
        batch_size=B,
    )
    logits = torch.randn(B, H, T, T)
    logits = torch.maximum(logits + attn_mask, torch.tensor(torch.finfo(logits.dtype).min))
    is_mem_e = is_mem.expand(B, H, T, T)
    lsm_e = last_section.expand(B, 1, T, T)

    legacy = landmark_grouped_softmax(logits, -1, is_mem_e, lsm_e)
    new = multi_landmark_grouped_softmax(
        logits, -1, is_mem_e, lsm_e, block_size=block_size, pool=pool
    )
    # With one landmark per block, sum and max coincide and both equal the legacy grouped softmax.
    torch.testing.assert_close(new, legacy, rtol=1e-5, atol=1e-6)


def test_masks_match_legacy_when_single():
    legacy_attn = LandmarkAttention(mem_freq=3, n_heads=8, d_model=64)
    T = 12
    a0, m0, l0 = legacy_attn._landmark_masks(T, torch.device("cpu"), torch.float32)
    a1, m1, l1 = _single_doc_masks(
        T,
        torch.device("cpu"),
        torch.float32,
        block_size=4,
        num_landmarks=1,
        cu_doc_lens=None,
        batch_size=1,
    )
    torch.testing.assert_close(a0, a1)
    assert torch.equal(m0, m1) and torch.equal(l0, l1)


def test_multi_landmark_attention_matches_single_landmark():
    # MultiLandmarkAttention(num_landmarks=1) must equal LandmarkAttention with the same weights.
    torch.manual_seed(0)
    rope = RoPEConfig(name=RoPEType.default, theta=10_000)
    single = LandmarkAttention(
        mem_freq=3,
        n_heads=8,
        n_kv_heads=2,
        head_dim=8,
        bias=False,
        d_model=64,
        rope=rope,
    )
    multi = MultiLandmarkAttention(
        mem_freq=3,
        num_landmarks=1,
        landmark_pool="sum",
        n_heads=8,
        n_kv_heads=2,
        head_dim=8,
        bias=False,
        d_model=64,
        rope=rope,
    )
    multi.load_state_dict(single.state_dict())
    single.eval()
    multi.eval()
    x = torch.randn(2, 12, 64)
    with torch.no_grad():
        torch.testing.assert_close(multi(x), single(x), rtol=1e-5, atol=1e-6)


def test_document_multi_landmark_matches_document_single():
    torch.manual_seed(0)
    rope = RoPEConfig(name=RoPEType.default, theta=10_000)
    common = dict(n_heads=8, n_kv_heads=2, head_dim=8, bias=False, d_model=64)
    single = DocumentLandmarkAttention(mem_freq=3, cross_doc_mode="chunked", rope=rope, **common)
    multi = DocumentMultiLandmarkAttention(
        mem_freq=3,
        num_landmarks=1,
        landmark_pool="sum",
        cross_doc_mode="chunked",
        rope=rope,
        **common,
    )
    multi.load_state_dict(single.state_dict())
    single.eval()
    multi.eval()
    x = torch.randn(1, 12, 64)
    # chunk0 = 0..3, chunk1 = 4..7, FREE = 8..11
    chunk_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1]])
    with torch.no_grad():
        torch.testing.assert_close(
            multi(x, chunk_ids=chunk_ids), single(x, chunk_ids=chunk_ids), rtol=1e-5, atol=1e-6
        )


# ---------------------------------------------------------------------------
# Correctness for num_landmarks > 1 against an independent reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pool", ["sum", "max"])
@pytest.mark.parametrize("num_landmarks", [2, 3])
def test_grouped_softmax_matches_reference(pool, num_landmarks):
    torch.manual_seed(num_landmarks)
    mem_freq = 3
    block_size = mem_freq + num_landmarks
    T = block_size * 4
    attn_mask, is_mem, last_section = _single_doc_masks(
        T,
        torch.device("cpu"),
        torch.float32,
        block_size=block_size,
        num_landmarks=num_landmarks,
        cu_doc_lens=None,
        batch_size=1,
    )
    raw = torch.randn(1, 1, T, T)
    logits = torch.maximum(raw + attn_mask, torch.tensor(torch.finfo(raw.dtype).min))
    probs = multi_landmark_grouped_softmax(
        logits,
        -1,
        is_mem.expand(1, 1, T, T),
        last_section.expand(1, 1, T, T),
        block_size=block_size,
        pool=pool,
    )[0, 0]
    ref = _reference_two_level_softmax(
        raw[0, 0], attn_mask[0, 0], is_mem[0, 0], last_section[0, 0], block_size, pool
    )
    torch.testing.assert_close(probs.double(), ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("pool", ["sum", "max"])
@pytest.mark.parametrize("num_landmarks", [1, 2, 4])
def test_grouped_softmax_rows_sum_to_one(pool, num_landmarks):
    torch.manual_seed(num_landmarks)
    mem_freq = 5
    block_size = mem_freq + num_landmarks
    B, H, T = 2, 4, block_size * 3
    attn_mask, is_mem, last_section = _single_doc_masks(
        T,
        torch.device("cpu"),
        torch.float32,
        block_size=block_size,
        num_landmarks=num_landmarks,
        cu_doc_lens=None,
        batch_size=B,
    )
    logits = torch.randn(B, H, T, T)
    logits = torch.maximum(logits + attn_mask, torch.tensor(torch.finfo(logits.dtype).min))
    probs = multi_landmark_grouped_softmax(
        logits,
        -1,
        is_mem.expand(B, H, T, T),
        last_section.expand(B, 1, T, T),
        block_size=block_size,
        pool=pool,
    )
    # Every query attends to at least itself, so all rows normalize to 1.
    assert torch.allclose(probs.sum(-1), torch.ones(B, H, T), atol=1e-5)


@pytest.mark.parametrize("pool", ["sum", "max"])
def test_multi_landmark_eager_backward(pool):
    attn = _multi_landmark(mem_freq=3, num_landmarks=2, landmark_pool=pool)
    attn.train()
    x = torch.randn(2, 15, 64, requires_grad=True)  # T multiple of block_size=5
    out = attn(x)
    assert out.shape == (2, 15, 64) and torch.isfinite(out).all()
    (out**2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in attn.parameters())


def test_max_pool_gradient_flows_only_to_winning_landmark():
    # With "max" pooling, only the highest-logit landmark of a past block should receive gradient
    # from a downstream query; the demoted landmarks get none.
    torch.manual_seed(0)
    block_size, num_landmarks = 5, 2  # mem_freq=3
    T = block_size * 2
    attn_mask, is_mem, last_section = _single_doc_masks(
        T,
        torch.device("cpu"),
        torch.float32,
        block_size=block_size,
        num_landmarks=num_landmarks,
        cu_doc_lens=None,
        batch_size=1,
    )
    raw = torch.randn(1, 1, T, T, requires_grad=True)
    logits = raw + attn_mask
    probs = multi_landmark_grouped_softmax(
        logits,
        -1,
        is_mem.expand(1, 1, T, T),
        last_section.expand(1, 1, T, T),
        block_size=block_size,
        pool="max",
    )
    # Loss = mass query T-1 puts on block 0's regular tokens (0..mem_freq-1), which is gated by block
    # 0's gate = its argmax landmark. (Plain ``probs.sum()`` is constant -- rows normalize to 1 -- so
    # it has no gradient; we need a loss that actually depends on block 0's gate.)
    mem_freq = block_size - num_landmarks
    probs[0, 0, T - 1, :mem_freq].sum().backward()
    # Block 0's landmark positions are 3 and 4; the last query row (T-1) sees both. Exactly the
    # higher-logit one should carry gradient from that row.
    g = raw.grad[0, 0, T - 1]
    lm0, lm1 = 3, 4
    winner = lm0 if logits[0, 0, T - 1, lm0] >= logits[0, 0, T - 1, lm1] else lm1
    loser = lm1 if winner == lm0 else lm0
    assert g[winner].abs() > 0
    assert g[loser].abs() == 0


# ---------------------------------------------------------------------------
# Speed sanity: the new single-landmark path is not dramatically slower than legacy
# ---------------------------------------------------------------------------


def test_single_landmark_speed_comparable_to_legacy():
    torch.manual_seed(0)
    block_size = 16  # mem_freq=15, num_landmarks=1
    B, H, T = 2, 8, block_size * 16  # 256
    attn_mask, is_mem, last_section = _single_doc_masks(
        T,
        torch.device("cpu"),
        torch.float32,
        block_size=block_size,
        num_landmarks=1,
        cu_doc_lens=None,
        batch_size=B,
    )
    logits = torch.maximum(
        torch.randn(B, H, T, T) + attn_mask, torch.tensor(torch.finfo(torch.float32).min)
    )
    is_mem_e, lsm_e = is_mem.expand(B, H, T, T), last_section.expand(B, 1, T, T)

    def bench(fn, n=20):
        fn()  # warmup
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n

    legacy_t = bench(lambda: landmark_grouped_softmax(logits, -1, is_mem_e, lsm_e))
    new_t = bench(
        lambda: multi_landmark_grouped_softmax(
            logits, -1, is_mem_e, lsm_e, block_size=block_size, pool="sum"
        )
    )
    # Generous bound: catch only catastrophic (orders-of-magnitude) regressions, not noise.
    assert new_t < 6 * legacy_t + 5e-3, f"new={new_t:.4e}s legacy={legacy_t:.4e}s"
