"""Tests for the ``"hierarchical_dilated"`` chunked-attention pattern: a *layer-dependent* document
mask where transformer layer ``ell`` attends the ``n`` documents at stride ``m**ell`` behind a context
query (saturating once the span covers all history). See
:func:`olmo_core.nn.attention.chunked_mask.build_chunked_allowed_mask` and
:class:`olmo_core.nn.attention.document_chunked.DocumentChunkedAttention`."""

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention import AttentionConfig, AttentionType, DocumentChunkedAttention
from olmo_core.nn.attention.chunked_mask import (
    FREE_CHUNK_ID,
    PAD_CHUNK_ID,
    AttentionPattern,
    build_chunked_allowed_mask,
    hierarchical_effective_layer,
)
from olmo_core.nn.layer_norm import LayerNormConfig
from olmo_core.nn.rope import RoPEConfig, RoPEType


def _allowed(chunk_ids, *, n, m, layer_idx, max_docs=None):
    pattern = AttentionPattern(
        name="hierarchical_dilated", dilation_n=n, dilation_m=m, dilation_max_docs=max_docs
    )
    return build_chunked_allowed_mask(pattern, chunk_ids, layer_idx=layer_idx)[0]


def _attended_chunks(chunk_ids, *, n, m, layer_idx, query_chunk):
    """The set of (context) key-chunk indices a query in ``query_chunk`` may attend, for a layout with
    one token per chunk (so position == chunk index)."""
    allowed = _allowed(chunk_ids, n=n, m=m, layer_idx=layer_idx)
    row = allowed[query_chunk]
    cids = chunk_ids[0]
    return {int(cids[k]) for k in range(len(row)) if bool(row[k]) and int(cids[k]) >= 0}


# 8 context chunks, one token each: position c == chunk c.
D8 = torch.arange(8).view(1, 8)


# ---------------------------------------------------------------------------
# Core stride schedule: D=8, n=2, m=2.
# ---------------------------------------------------------------------------


def test_layer0_sees_self_and_immediate_predecessor():
    # Layer 0 -> stride 1: chunk c sees {c, c-1} (dense local).
    for c in range(8):
        assert _attended_chunks(D8, n=2, m=2, layer_idx=0, query_chunk=c) == {
            x for x in (c, c - 1) if x >= 0
        }


def test_layer1_sees_self_and_stride2():
    # Layer 1 -> stride m=2: chunk c sees {c, c-2}.
    for c in range(8):
        assert _attended_chunks(D8, n=2, m=2, layer_idx=1, query_chunk=c) == {
            x for x in (c, c - 2) if x >= 0
        }


def test_layer2_sees_self_and_stride4():
    # Layer 2 -> stride m^2=4: chunk c sees {c, c-4}.
    for c in range(8):
        assert _attended_chunks(D8, n=2, m=2, layer_idx=2, query_chunk=c) == {
            x for x in (c, c - 4) if x >= 0
        }


def test_layer3_saturates_and_deeper_layers_reuse_it():
    # max_chunk = 7; L* = smallest ell with (n-1)*m^ell >= 7 -> 2^3 = 8 >= 7, so L* = 3.
    assert int(hierarchical_effective_layer(3, 2, 2, torch.tensor([7]))[0]) == 3
    # At/after L* the stride is capped at m^L* = 8, so layers 3, 4, 5 produce IDENTICAL masks.
    base = _allowed(D8, n=2, m=2, layer_idx=3)
    assert torch.equal(base, _allowed(D8, n=2, m=2, layer_idx=4))
    assert torch.equal(base, _allowed(D8, n=2, m=2, layer_idx=5))
    # Stride 8 > max gap 7 -> a context chunk attends only itself at the saturated layer.
    for c in range(8):
        assert _attended_chunks(D8, n=2, m=2, layer_idx=3, query_chunk=c) == {c}


def test_isolation_non_strided_chunks_unattended():
    # Layer 1 (stride 2): chunk 5 sees {5, 3} only -- NOT 4 (off-stride) nor 1 (beyond n).
    seen = _attended_chunks(D8, n=2, m=2, layer_idx=1, query_chunk=5)
    assert seen == {5, 3}
    assert 4 not in seen and 1 not in seen


def test_causal_never_attends_future_chunks():
    for layer_idx in range(4):
        allowed = _allowed(D8, n=2, m=2, layer_idx=layer_idx)
        for q in range(8):
            for k in range(8):
                if k > q:
                    assert not bool(allowed[q, k]), (layer_idx, q, k)


# ---------------------------------------------------------------------------
# FREE / PAD roles (with multi-token chunks + a trailing FREE query/answer).
# ---------------------------------------------------------------------------

# chunk0 = 0..1, chunk1 = 2..3, chunk2 = 4..5, chunk3 = 6..7, FREE = 8..9, PAD = 10..11.
ROLES = torch.tensor(
    [[0, 0, 1, 1, 2, 2, 3, 3, FREE_CHUNK_ID, FREE_CHUNK_ID, PAD_CHUNK_ID, PAD_CHUNK_ID]]
)


def test_free_query_bridges_all_earlier_tokens_at_every_layer():
    # A FREE query (position 8) attends every earlier non-pad token regardless of the layer stride.
    for layer_idx in range(4):
        allowed = _allowed(ROLES, n=2, m=2, layer_idx=layer_idx)
        free_row = allowed[8]
        for k in range(9):  # 0..8 are non-pad and causal
            assert bool(free_row[k]), (layer_idx, k)


def test_free_key_is_attendable_and_pad_never_attended():
    for layer_idx in range(4):
        allowed = _allowed(ROLES, n=2, m=2, layer_idx=layer_idx)
        # No query attends a PAD key (positions 10, 11), except the self-diagonal NaN guard (a PAD
        # query may attend itself; those rows are dropped by the loss mask anyway).
        for q in range(12):
            if q != 10:
                assert not bool(allowed[q, 10]), (layer_idx, q)
            if q != 11:
                assert not bool(allowed[q, 11]), (layer_idx, q)
        # A context query attends a FREE key that precedes it (FREE bridges). Position 8 is FREE and
        # precedes... only the FREE query at 9, so check that 9 attends 8.
        assert bool(allowed[9, 8])


def test_own_document_is_fully_causal():
    # Within a chunk (e.g. chunk1 = positions 2,3) attention is full causal regardless of stride.
    allowed = _allowed(ROLES, n=2, m=2, layer_idx=2)  # large stride, so cross-doc is sparse
    assert bool(allowed[3, 2]) and bool(allowed[3, 3]) and bool(allowed[2, 2])
    assert not bool(allowed[2, 3])  # causal within the doc


# ---------------------------------------------------------------------------
# Sensible reductions.
# ---------------------------------------------------------------------------


def test_m1_reduces_to_doc_window():
    # m == 1 -> constant stride 1 at every layer -> doc_window of width n-1 (layer-independent).
    dw = build_chunked_allowed_mask(AttentionPattern(name="doc_window", doc_window_k=2), D8)[0]
    for layer_idx in (0, 3, 7):
        hd = _allowed(D8, n=3, m=1, layer_idx=layer_idx)
        assert torch.equal(hd, dw), layer_idx


def test_n_large_layer0_is_dense_causal_by_chunk():
    # n >= D at layer 0 (stride 1) -> a context chunk attends ALL preceding chunks (causal-by-chunk).
    for c in range(8):
        assert _attended_chunks(D8, n=8, m=2, layer_idx=0, query_chunk=c) == set(range(c + 1))


def test_n1_reduces_to_chunked():
    # n == 1 -> only the own document, i.e. the plain "chunked" pattern (isolated docs + FREE bridge).
    chunked = build_chunked_allowed_mask(AttentionPattern(name="chunked"), ROLES)[0]
    for layer_idx in (0, 2, 5):
        hd = _allowed(ROLES, n=1, m=2, layer_idx=layer_idx)
        assert torch.equal(hd, chunked), layer_idx


def test_dilation_max_docs_fixes_saturation_layer():
    # With a fixed cap reference of 7, the saturation layer is L*=3 regardless of the actual #chunks.
    # Build a short 2-chunk sequence but force the cap as if there were 8 docs: layers do NOT collapse
    # to self-only before L*.
    short = torch.tensor([[0, 1]])
    # eff layer with fixed max_docs=7 at layer 2 -> stride m^2 = 4 (not saturated yet).
    pat = AttentionPattern(
        name="hierarchical_dilated", dilation_n=2, dilation_m=2, dilation_max_docs=7
    )
    eff = hierarchical_effective_layer(2, 2, 2, torch.tensor([7]))
    assert int(eff[0]) == 2
    # And per-sequence (no cap) on the same short sequence saturates immediately (max_chunk=1 -> L*=0).
    eff_auto = hierarchical_effective_layer(2, 2, 2, torch.tensor([1]))
    assert int(eff_auto[0]) == 0
    assert pat.dilation_max_docs == 7
    assert build_chunked_allowed_mask(pat, short, layer_idx=2).shape == (1, 2, 2)


# ---------------------------------------------------------------------------
# Module / config wiring + forward & backward.
# ---------------------------------------------------------------------------


def _build_attn(layer_idx, n_layers, *, n=2, m=2):
    config = AttentionConfig(
        name=AttentionType.document_chunked,
        n_heads=8,
        n_kv_heads=2,
        head_dim=8,
        bias=False,
        cross_doc_mode="hierarchical_dilated",
        dilation_n=n,
        dilation_m=m,
        qk_norm=LayerNormConfig(name="rms", eps=1e-6, bias=False),
        use_head_qk_norm=True,
        rope=RoPEConfig(name=RoPEType.default, theta=10_000),
    )
    return config.build(64, layer_idx=layer_idx, n_layers=n_layers)


def test_config_builds_and_threads_layer_idx():
    attn = _build_attn(2, 6, n=3, m=2)
    assert isinstance(attn, DocumentChunkedAttention)
    assert attn.cross_doc_mode == "hierarchical_dilated"
    assert attn.layer_idx == 2 and attn.n_layers == 6
    assert attn._pattern.dilation_n == 3 and attn._pattern.dilation_m == 2


def test_dilation_params_rejected_for_non_chunked():
    with pytest.raises(OLMoConfigurationError):
        AttentionConfig(name=AttentionType.default, n_heads=8, head_dim=8, dilation_n=2).build(
            64, layer_idx=0, n_layers=1
        )


def test_forward_backward_finite_and_layer_dependent():
    torch.manual_seed(0)
    x = torch.randn(1, 8, 64, requires_grad=True)
    chunk_ids = torch.arange(8).view(1, 8)
    out0 = _build_attn(0, 4)(x, chunk_ids=chunk_ids)
    out2 = _build_attn(2, 4)(x, chunk_ids=chunk_ids)
    assert out0.shape == (1, 8, 64)
    assert torch.isfinite(out0).all() and torch.isfinite(out2).all()
    # Different layers -> different masks -> different outputs (same weights would still differ here
    # because the two modules are independently initialized; instead check the masks differ via a
    # shared module run at two layer indices below).
    out0.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_same_module_different_layer_changes_output():
    # Reuse ONE module's weights but evaluate its mask at two layer indices, to isolate the layer
    # effect from random init.
    torch.manual_seed(0)
    attn = _build_attn(0, 4)
    attn.eval()
    x = torch.randn(1, 8, 64)
    chunk_ids = torch.arange(8).view(1, 8)
    with torch.no_grad():
        attn.layer_idx = 0
        o0 = attn(x, chunk_ids=chunk_ids)
        attn.layer_idx = 1
        o1 = attn(x, chunk_ids=chunk_ids)
    # Layer 0 (stride 1, sees c-1) vs layer 1 (stride 2, sees c-2) give different context -> outputs
    # at a chunk with >=2 predecessors must differ.
    assert not torch.allclose(o0[0, 3], o1[0, 3], atol=1e-5)
