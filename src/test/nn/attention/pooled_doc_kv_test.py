"""Tests for PooledDocKVAttention (train-time per-document KV pooling with full-attention
inference transfer). The load-bearing test is the exact equivalence against plain causal attention
over the "perturbed corpus" (pooled documents' KV entries replaced by their mean)."""

import pytest
import torch
import torch.nn.functional as F

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import (
    Attention,
    AttentionConfig,
    AttentionType,
    PooledDocKVAttention,
)
from olmo_core.nn.attention.backend import _repeat_kv
from olmo_core.nn.attention.pooled_doc_kv import (
    PooledDocKeepHolder,
    make_fingerprint_keep_docs_fn,
)
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType

D_MODEL = 64


def _attention(name: AttentionType, *, layer_idx=0, n_layers=1, **kw):
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
    return config.build(D_MODEL, layer_idx=layer_idx, n_layers=n_layers)


def _pooled_attention(**kw):
    attn = _attention(AttentionType.pooled_doc_kv, **kw)
    assert isinstance(attn, PooledDocKVAttention)
    return attn


def _set_keep(attn: PooledDocKVAttention, keep_rows) -> None:
    attn._pooled_keep_holder = PooledDocKeepHolder(
        keep_docs=torch.tensor(keep_rows, dtype=torch.bool)
    )


# doc0 = 0..3, doc1 = 4..7, doc2 = 8..11, FREE (query/answer) = 12..15.
CHUNK_IDS = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, -1, -1, -1, -1]])
T = CHUNK_IDS.shape[1]


def _reference_perturbed_causal(attn, q, k, v, chunk_ids, keep, len_bias=True):
    """Plain causal SDPA over the perturbed corpus: pooled docs' K/V rows replaced by their mean.

    With ``len_bias=True`` the pooled path must match this EXACTLY (for queries outside pooled
    docs): one slot with ``+log(L)`` logit bias is ``L`` copies of the mean KV entry.
    """
    assert len_bias, "the exact-copies equivalence only holds with the log-length bias"
    k2, v2 = k.clone(), v.clone()
    B = k.shape[0]
    for b in range(B):
        for d in range(int(chunk_ids[b].max().item()) + 1):
            if keep[b][d]:
                continue
            idx = (chunk_ids[b] == d).nonzero(as_tuple=True)[0]
            k2[b, idx] = k[b, idx].mean(dim=0, keepdim=True)
            v2[b, idx] = v[b, idx].mean(dim=0, keepdim=True)
    n_rep = q.shape[2] // k.shape[2]
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        _repeat_kv(k2, n_rep).transpose(1, 2),
        _repeat_kv(v2, n_rep).transpose(1, 2),
        is_causal=True,
        scale=attn.softmax_scale,
    )
    return out.transpose(1, 2)


def _run_sdpa(attn, q, k, v, chunk_ids):
    attn._chunk_ids = chunk_ids
    try:
        return attn._sdpa_masked(q, k, v)
    finally:
        attn._chunk_ids = None


def test_config_builds_and_knob_gate():
    attn = _pooled_attention(pooled_keep_prob=0.25, pooled_keep_seed=7, pooled_len_bias=True)
    assert attn.keep_prob == 0.25 and attn.keep_seed == 7 and attn.len_bias
    # The pooled knobs are rejected on other attention types.
    with pytest.raises(OLMoConfigurationError):
        _attention(AttentionType.document_chunked, pooled_keep_prob=0.5)


def test_no_chunk_ids_matches_default_causal():
    pooled = _pooled_attention()
    base = _attention(AttentionType.default)
    assert isinstance(base, Attention) and not isinstance(base, PooledDocKVAttention)
    base.load_state_dict(pooled.state_dict())
    pooled.eval()
    base.eval()
    x = torch.randn(2, T, D_MODEL)
    with torch.no_grad():
        assert torch.allclose(pooled(x), base(x), atol=1e-5)


def test_keep_all_matches_default_causal():
    # Every doc kept -> no slot is visible and every real edge is causal-allowed: plain causal.
    pooled = _pooled_attention()
    _set_keep(pooled, [[True, True, True]])
    base = _attention(AttentionType.default)
    base.load_state_dict(pooled.state_dict())
    pooled.eval()
    base.eval()
    x = torch.randn(1, T, D_MODEL)
    with torch.no_grad():
        assert torch.allclose(pooled(x, chunk_ids=CHUNK_IDS), base(x), atol=1e-5)


def test_all_free_roles_match_default_causal():
    # Mask-mix compatibility: roles collapsed to all-FREE -> plain causal (n_docs == 0 branch).
    pooled = _pooled_attention()
    base = _attention(AttentionType.default)
    base.load_state_dict(pooled.state_dict())
    pooled.eval()
    base.eval()
    x = torch.randn(1, T, D_MODEL)
    with torch.no_grad():
        out = pooled(x, chunk_ids=torch.full_like(CHUNK_IDS, -1))
        assert torch.allclose(out, base(x), atol=1e-5)


@pytest.mark.parametrize("keep", [[[True, False, True]], [[False, False, False]]])
def test_exact_equivalence_to_perturbed_full_attention(keep):
    # THE load-bearing test: for every query OUTSIDE the pooled documents, the pooled path equals
    # plain full causal attention over the perturbed corpus (pooled docs' KV replaced by the mean).
    torch.manual_seed(0)
    attn = _pooled_attention()
    _set_keep(attn, keep)
    B, H, HKV, HD = 1, 8, 2, 8
    q = torch.randn(B, T, H, HD)
    k = torch.randn(B, T, HKV, HD)
    v = torch.randn(B, T, HKV, HD)
    out = _run_sdpa(attn, q, k, v, CHUNK_IDS)
    ref = _reference_perturbed_causal(attn, q, k, v, CHUNK_IDS, keep)
    pooled_docs = {d for d in range(3) if not keep[0][d]}
    outside = torch.tensor([int(CHUNK_IDS[0, t].item()) not in pooled_docs for t in range(T)])
    assert torch.allclose(out[:, outside], ref[:, outside], atol=1e-5), (
        (out[:, outside] - ref[:, outside]).abs().max()
    )


def test_within_pooled_doc_attention_is_real_and_causal():
    # Queries inside a pooled doc attend their own doc's REAL tokens causally (no self-pool, no
    # future leak): perturbing a LATER token of the same pooled doc leaves earlier outputs unchanged.
    torch.manual_seed(1)
    attn = _pooled_attention()
    _set_keep(attn, [[True, False, True]])
    B, H, HKV, HD = 1, 8, 2, 8
    q = torch.randn(B, T, H, HD)
    k = torch.randn(B, T, HKV, HD)
    v = torch.randn(B, T, HKV, HD)
    out = _run_sdpa(attn, q, k, v, CHUNK_IDS)
    k2, v2 = k.clone(), v.clone()
    k2[0, 7] += 1.0  # last token of pooled doc1
    v2[0, 7] += 1.0
    out2 = _run_sdpa(attn, q, k2, v2, CHUNK_IDS)
    # Positions before/at 6 never see position 7 (causal, and the slot only opens after the doc).
    assert torch.allclose(out[:, :7], out2[:, :7], atol=1e-6)
    # A FREE query after the doc sees the perturbation through the pooled mean.
    assert not torch.allclose(out[:, 12:], out2[:, 12:], atol=1e-4)


def test_pooled_doc_only_exposes_its_mean():
    # A mean-preserving perturbation WITHIN a pooled doc (+delta on one token's V, -delta on
    # another's) is invisible to queries outside the doc -- only the mean is exposed. The same
    # perturbation on a KEPT doc changes them (per-token values weighted by per-token scores).
    # NB a (k, v)-pair permutation would be a no-op in BOTH arms (softmax attention is
    # permutation-invariant over key-value pairs), so it cannot distinguish pooled from kept.
    torch.manual_seed(2)
    B, H, HKV, HD = 1, 8, 2, 8
    q = torch.randn(B, T, H, HD)
    k = torch.randn(B, T, HKV, HD)
    v = torch.randn(B, T, HKV, HD)
    delta = torch.randn(HKV, HD)
    for keep_doc1, expect_same in [(False, True), (True, False)]:
        attn = _pooled_attention()
        _set_keep(attn, [[True, keep_doc1, True]])
        out = _run_sdpa(attn, q, k, v, CHUNK_IDS)
        v2 = v.clone()
        v2[0, 4] += delta  # doc1 spans positions 4..7; the doc mean V is unchanged
        v2[0, 5] -= delta
        out2 = _run_sdpa(attn, q, k, v2, CHUNK_IDS)
        same = torch.allclose(out[:, 12:], out2[:, 12:], atol=1e-5)
        assert same == expect_same


def test_pad_never_attended():
    # Rows with trailing PAD: perturbing a PAD position's KV changes nothing else, outputs finite.
    torch.manual_seed(3)
    cids = torch.tensor([[0, 0, 0, 1, 1, 1, -1, -1, -2, -2]])
    Tp = cids.shape[1]
    attn = _pooled_attention()
    _set_keep(attn, [[True, False]])
    B, H, HKV, HD = 1, 8, 2, 8
    q = torch.randn(B, Tp, H, HD)
    k = torch.randn(B, Tp, HKV, HD)
    v = torch.randn(B, Tp, HKV, HD)
    out = _run_sdpa(attn, q, k, v, cids)
    assert torch.isfinite(out).all()
    k2, v2 = k.clone(), v.clone()
    k2[0, 8:] += 5.0
    v2[0, 8:] += 5.0
    out2 = _run_sdpa(attn, q, k2, v2, cids)
    assert torch.allclose(out[:, :8], out2[:, :8], atol=1e-6)


def test_len_bias_changes_slot_mass():
    torch.manual_seed(4)
    a_bias = _pooled_attention(pooled_len_bias=True)
    a_nobias = _pooled_attention(pooled_len_bias=False)
    a_nobias.load_state_dict(a_bias.state_dict())
    for a in (a_bias, a_nobias):
        _set_keep(a, [[True, False, True]])
        a.eval()
    x = torch.randn(1, T, D_MODEL)
    with torch.no_grad():
        out_b = a_bias(x, chunk_ids=CHUNK_IDS)
        out_n = a_nobias(x, chunk_ids=CHUNK_IDS)
    # FREE queries weigh the pooled slot differently; within-doc rows see no slot at all.
    assert not torch.allclose(out_b[:, 12:], out_n[:, 12:], atol=1e-5)


def test_fallback_keep_is_deterministic_and_seed_dependent():
    attn_a = _pooled_attention(pooled_keep_prob=0.5, pooled_keep_seed=1)
    attn_b = _pooled_attention(pooled_keep_prob=0.5, pooled_keep_seed=2)
    cids = torch.arange(24).repeat_interleave(4).unsqueeze(0) // 4  # 24 docs of 4 tokens
    cids = cids.to(torch.int32)
    keep1 = attn_a._resolve_keep_docs(cids, 24)
    keep2 = attn_a._resolve_keep_docs(cids, 24)
    keep3 = attn_b._resolve_keep_docs(cids, 24)
    assert torch.equal(keep1, keep2)
    assert not torch.equal(keep1, keep3)
    # ~half kept at keep_prob=0.5 (loose bound; the draw is a fixed hash, not statistical noise).
    assert 4 <= int(keep1.sum()) <= 20


def test_training_backward_flows_through_pool():
    attn = _pooled_attention()
    _set_keep(attn, [[True, False, False]])
    x = torch.randn(1, T, D_MODEL, requires_grad=True)
    attn(x, chunk_ids=CHUNK_IDS).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # Pooled docs' tokens still receive gradient (through the mean and their within-doc attention).
    assert x.grad[0, 4:12].abs().sum() > 0


def test_fingerprint_keep_docs_fn_gold_plus_random():
    # 3 docs wrapped in markers, gold = doc 1, n_random=0 -> keep exactly {1}; unknown rows keep all.
    from olmo_core.nn.attention.gold_grad_mask import content_fingerprint_from_row

    START, END, EOS = 900, 901, 999
    row = [START, 1, 2, END, START, 3, 4, END, START, 5, 6, END, 7, 8, EOS]
    fp = content_fingerprint_from_row(row, EOS)
    fn = make_fingerprint_keep_docs_fn(
        {fp: [1]},
        doc_start_id=START,
        doc_end_id=END,
        eos_id=EOS,
        n_random=0,
        mode="gold_plus_random",
        seed=0,
    )
    ids = torch.tensor([row, [START, 9, 9, END, START, 9, 9, END, START, 9, 9, END, 9, 9, EOS]])
    keep = fn(ids)
    assert keep.shape == (2, 3)
    assert keep[0].tolist() == [False, True, False]
    assert keep[1].tolist() == [True, True, True]  # fingerprint miss -> all real


def test_fingerprint_keep_docs_fn_n_random_range():
    # (lo, hi) breadth: gold always kept, negative count varies per call within [lo, hi].
    from olmo_core.nn.attention.gold_grad_mask import content_fingerprint_from_row

    START, END, EOS = 900, 901, 999
    row = []
    for d in range(12):
        row += [START, 10 + 2 * d, 11 + 2 * d, END]
    row += [7, 8, EOS]
    fp = content_fingerprint_from_row(row, EOS)
    fn = make_fingerprint_keep_docs_fn(
        {fp: [3, 7]},
        doc_start_id=START,
        doc_end_id=END,
        eos_id=EOS,
        n_random=2,
        n_random_range=(1, 8),
        mode="gold_plus_random",
        seed=0,
    )
    counts = []
    for _ in range(8):
        keep = fn(torch.tensor([row]))
        assert keep.shape == (1, 12)
        assert keep[0, 3] and keep[0, 7]
        counts.append(int(keep.sum().item()))
    assert len(set(counts)) > 1  # breadth varies across calls (per-epoch variation)
    assert all(2 + 1 <= c <= 2 + 8 for c in counts)


@pytest.mark.parametrize("n_docs,frac", [(12, 0.5), (24, 0.25), (6, 1.0 / 6)])
def test_fingerprint_keep_docs_fn_fixed_fraction(n_docs, frac):
    """n_random_frac keeps a context-length-INVARIANT share of the non-gold docs real (gold always
    kept): the FLOP-scaling study's KV arms. Same fraction at every doc count."""
    from olmo_core.nn.attention.gold_grad_mask import content_fingerprint_from_row

    START, END, EOS = 900, 901, 999
    row = []
    for d in range(n_docs):
        row += [START, 10 + 2 * d, 11 + 2 * d, END]
    row += [7, 8, EOS]
    fp = content_fingerprint_from_row(row, EOS)
    gold = [1, 3]
    fn = make_fingerprint_keep_docs_fn(
        {fp: gold},
        doc_start_id=START,
        doc_end_id=END,
        eos_id=EOS,
        n_random=999,  # would keep everything -- must be overridden by the fraction
        n_random_frac=frac,
        mode="gold_plus_random",
        seed=0,
    )
    keep = fn(torch.tensor([row]))
    assert keep.shape == (1, n_docs)
    assert keep[0, 1] and keep[0, 3]
    expected = len(gold) + max(1, int(round(frac * (n_docs - len(gold)))))
    assert int(keep.sum().item()) == expected
    # deterministic per (seed, fingerprint)
    assert torch.equal(keep, fn(torch.tensor([row])))
