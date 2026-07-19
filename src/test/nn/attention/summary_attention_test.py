"""Tests for the ``"summary_attention"`` chunked-attention pattern: documents are grouped into
**cells** of ``summary_every_k`` docs, each followed by a **summary span** (its own chunk). Within a
cell attention is full; the span reads its whole cell and is attendable by later cells, so any two
documents in different cells sit exactly **2 hops** apart with no gold-aware term anywhere.

The ``summary_bandwidth`` knob is the experiment's dose axis: it throttles how many of each span's
leading tokens later chunks may attend, **without changing the data** -- so ``bandwidth=0`` is the
ladder's floor control and must reproduce a pure cell-blocks mask exactly.

See :func:`olmo_core.nn.attention.chunked_mask.build_chunked_allowed_mask` and
``records/multihop-gold-routing-experiment.md`` §"Approach C -- summary attention".
"""

import pytest
import torch

from olmo_core.nn.attention.chunked_mask import (
    FREE_CHUNK_ID,
    PAD_CHUNK_ID,
    AttentionPattern,
    build_chunked_allowed_mask,
    build_chunked_mask_mod,
    chunk_token_offset,
)

INF = 10**6


def _pattern(*, k: int, bandwidth: int, relay: bool = True) -> AttentionPattern:
    return AttentionPattern(
        name="summary_attention",
        summary_every_k=k,
        summary_bandwidth=bandwidth,
        summary_relay=relay,
    )


def _allowed(chunk_ids, *, k, bandwidth, relay=True):
    return build_chunked_allowed_mask(_pattern(k=k, bandwidth=bandwidth, relay=relay), chunk_ids)[0]


def _one_token_per_chunk(n_docs: int, k: int) -> torch.Tensor:
    """Chunk ids for a layout with one token per chunk, so the ``(S, S)`` mask IS the chunk graph."""
    n_cells = n_docs // k
    return torch.arange(n_cells * (k + 1)).unsqueeze(0)


def _realistic_chunk_ids(n_docs: int, k: int, b_max: int, seed: int = 3):
    """Variable-length docs, a FREE instruction prefix, a FREE query/answer tail, and a PAD tail --
    i.e. what the tokenized shard actually looks like."""
    g = torch.Generator().manual_seed(seed)
    ids: list = [FREE_CHUNK_ID] * 6
    cid = 0
    for _ in range(n_docs // k):
        for _ in range(k):
            ids += [cid] * int(torch.randint(3, 9, (1,), generator=g).item())
            cid += 1
        ids += [cid] * b_max  # the summary span
        cid += 1
    ids += [FREE_CHUNK_ID] * 10 + [PAD_CHUNK_ID] * 5
    return torch.tensor(ids, dtype=torch.int32).unsqueeze(0), cid


def _chunk_graph(allowed: torch.Tensor, chunk_ids: torch.Tensor, n_chunks: int) -> torch.Tensor:
    """Chunk-level graph induced by a token-level mask: chunk ``b`` attends chunk ``a`` iff ANY token
    of ``b`` attends ANY token of ``a``."""
    c = chunk_ids[0]
    g = torch.zeros((n_chunks, n_chunks), dtype=torch.bool)
    for qb in range(n_chunks):
        rq = torch.nonzero(c == qb, as_tuple=True)[0]
        for ka in range(n_chunks):
            rk = torch.nonzero(c == ka, as_tuple=True)[0]
            g[qb, ka] = bool(allowed[rq][:, rk].any())
    return g


def _apsp(g: torch.Tensor) -> torch.Tensor:
    """All-pairs shortest path lengths over the boolean adjacency (Floyd-Warshall)."""
    n = g.shape[0]
    d = torch.where(g, torch.ones_like(g, dtype=torch.long), torch.full(g.shape, INF))
    d.fill_diagonal_(0)
    for m in range(n):
        d = torch.minimum(d, d[:, m].unsqueeze(1) + d[m, :].unsqueeze(0))
    return d


def _cell_blocks_reference(n_chunks: int, k: int) -> torch.Tensor:
    """A pure cell-blocks chunk graph: every chunk attends its own cell's earlier chunks, nothing else.
    Built independently of the pattern under test."""
    p = k + 1
    idx = torch.arange(n_chunks)
    return (idx.unsqueeze(1) >= idx.unsqueeze(0)) & (idx.unsqueeze(1) // p == idx.unsqueeze(0) // p)


# ---------------------------------------------------------------------------
# The floor control: bandwidth=0 must be EXACTLY cell blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [5, 10, 25])
def test_bandwidth_zero_is_exactly_cell_blocks(k):
    """``bandwidth=0`` is the ladder's zero rung, so it must reproduce cell-blocks BIT-EXACTLY -- not
    approximately. Any cross-cell edge here would silently give the floor control a relay."""
    chunk_ids = _one_token_per_chunk(50, k)
    n_chunks = chunk_ids.shape[1]
    allowed = _allowed(chunk_ids, k=k, bandwidth=0)
    torch.testing.assert_close(allowed, _cell_blocks_reference(n_chunks, k))


@pytest.mark.parametrize("k", [5, 10])
def test_bandwidth_zero_has_no_cross_cell_path(k):
    """The defining property of the floor control: no document may reach a document in another cell,
    at ANY number of hops (the summary spans must not provide a back door)."""
    chunk_ids = _one_token_per_chunk(50, k)
    n_chunks = chunk_ids.shape[1]
    d = _apsp(_allowed(chunk_ids, k=k, bandwidth=0))
    p = k + 1
    docs = [i for i in range(n_chunks) if i % p != k]
    for b in docs:
        for a in docs:
            if a < b and a // p != b // p:
                assert d[b, a] >= INF, f"cell blocks leaked a path {b}->{a}"


# ---------------------------------------------------------------------------
# The ladder: bandwidth moves the bottleneck, not the topology
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bandwidth", [1, 4, 16])
def test_all_positive_bandwidths_share_one_doc_graph(bandwidth):
    """Every ``b > 0`` rung must have an IDENTICAL document-level graph -- only the number of visible
    relay TOKENS may differ. This is what makes the ladder a clean dose-response."""
    k, b_max = 10, 16
    chunk_ids, n_chunks = _realistic_chunk_ids(50, k, b_max)
    ref = _chunk_graph(_allowed(chunk_ids, k=k, bandwidth=1), chunk_ids, n_chunks)
    got = _chunk_graph(_allowed(chunk_ids, k=k, bandwidth=bandwidth), chunk_ids, n_chunks)
    torch.testing.assert_close(got, ref)


@pytest.mark.parametrize("k,expect_2hop", [(5, 0.918), (10, 0.816)])
def test_cross_cell_doc_pairs_are_exactly_two_hops(k, expect_2hop):
    """Every cross-cell document pair sits at exactly 2 hops, and nothing is unreachable. The 2-hop
    fraction equals ``1 - (k-1)/49`` -- the gold-blind stratum the experiment reads."""
    chunk_ids = _one_token_per_chunk(50, k)
    n_chunks = chunk_ids.shape[1]
    d = _apsp(_allowed(chunk_ids, k=k, bandwidth=16))
    p = k + 1
    docs = [i for i in range(n_chunks) if i % p != k]
    n_2hop = n_pairs = 0
    for b in docs:
        for a in docs:
            if a < b:
                n_pairs += 1
                assert d[b, a] < INF, f"doc pair {b}->{a} unreachable"
                assert d[b, a] <= 2, f"doc pair {b}->{a} needs {d[b, a]} hops"
                n_2hop += int(d[b, a] == 2)
                if a // p != b // p:
                    assert d[b, a] == 2
    assert n_2hop / n_pairs == pytest.approx(expect_2hop, abs=0.001)


@pytest.mark.parametrize("bandwidth", [0, 1, 4, 16])
def test_visible_relay_tokens_scale_with_bandwidth(bandwidth):
    """The bandwidth gate exposes exactly the leading ``b`` tokens of each earlier span."""
    k, b_max = 10, 16
    chunk_ids, n_chunks = _realistic_chunk_ids(50, k, b_max)
    allowed = _allowed(chunk_ids, k=k, bandwidth=bandwidth)
    c = chunk_ids[0]
    p = k + 1
    is_sum = (c >= 0) & (c % p == k)
    is_doc = (c >= 0) & (c % p != k)
    # every doc token's visible summary keys, counted per earlier cell
    for q in torch.nonzero(is_doc, as_tuple=True)[0].tolist():
        for cell in range(c[q].item() // p):
            span = torch.nonzero(is_sum & (c == cell * p + k), as_tuple=True)[0]
            n_vis = int(allowed[q][span].sum())
            assert n_vis == min(bandwidth, b_max), (
                f"doc token {q} sees {n_vis} tokens of span {cell}, expected "
                f"{min(bandwidth, b_max)}"
            )


# ---------------------------------------------------------------------------
# Causality: every visible relay token has actually read its cell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [5, 10])
def test_every_summary_token_follows_all_its_cells_docs(k):
    """A relay can only carry information FORWARD, so the span must sit at the cell END: every one of
    its tokens must follow every document of its cell. If this regresses, the bandwidth gate could
    expose a token that read nothing and the whole design is silently vacuous."""
    b_max = 16
    chunk_ids, _ = _realistic_chunk_ids(50, k, b_max)
    c = chunk_ids[0]
    p = k + 1
    for cell in range(50 // k):
        span = torch.nonzero(c == cell * p + k, as_tuple=True)[0]
        cell_docs = torch.nonzero((c >= 0) & (c // p == cell) & (c % p != k), as_tuple=True)[0]
        assert int(span.min()) > int(cell_docs.max())


@pytest.mark.parametrize("k", [5, 10])
def test_summary_span_reads_its_whole_cell(k):
    """The treatment: the span must attend every document of its own cell (that is the relay)."""
    chunk_ids = _one_token_per_chunk(50, k)
    allowed = _allowed(chunk_ids, k=k, bandwidth=16)
    p = k + 1
    for cell in range(50 // k):
        span = cell * p + k
        for slot in range(k):
            assert allowed[span, cell * p + slot], f"span {cell} cannot read doc slot {slot}"


# ---------------------------------------------------------------------------
# The placebo's defining property: it is VACUOUS
# ---------------------------------------------------------------------------


def _doc_content_reaching(chunk_ids, n_chunks, k, *, bandwidth, relay):
    """For each chunk, how many DOCUMENT chunks' content can transitively reach it."""
    d = _apsp(_allowed(chunk_ids, k=k, bandwidth=bandwidth, relay=relay))
    p = k + 1
    is_doc = torch.tensor([i % p != k for i in range(n_chunks)])
    reach = d < INF
    reach.fill_diagonal_(False)
    return (reach & is_doc.unsqueeze(0)).sum(dim=1)


@pytest.mark.parametrize("k", [5, 10])
def test_placebo_is_vacuous_zero_doc_content_reaches_any_span(k):
    """``summary_relay=False`` is the PLACEBO: the span keeps its position, its tokens and every edge
    into it, but reads nothing -- so ZERO document content reaches it, and (by induction through the
    span->span edges) none reaches any span. This is the property that makes it a placebo rather than
    a capacity-matched control, so it must not silently regress."""
    chunk_ids = _one_token_per_chunk(50, k)
    n_chunks = chunk_ids.shape[1]
    p = k + 1
    n = _doc_content_reaching(chunk_ids, n_chunks, k, bandwidth=16, relay=False)
    spans = [i for i in range(n_chunks) if i % p == k]
    assert [int(n[i]) for i in spans] == [0] * len(spans)


@pytest.mark.parametrize("k", [5, 10])
def test_relay_is_not_vacuous_and_reaches_every_earlier_doc(k):
    """The treatment's whole claim: at a document-level out-degree of only (k-1)/2, EVERY earlier
    document's content reaches every document, via a 2-hop relay."""
    chunk_ids = _one_token_per_chunk(50, k)
    n_chunks = chunk_ids.shape[1]
    p = k + 1
    n = _doc_content_reaching(chunk_ids, n_chunks, k, bandwidth=16, relay=True)
    spans = [i for i in range(n_chunks) if i % p == k]
    # span of cell c has read every doc of cells 0..c
    for cell, i in enumerate(spans):
        assert int(n[i]) == k * (cell + 1)
    # and the last document sees every earlier document's content
    last_doc = max(i for i in range(n_chunks) if i % p != k)
    n_earlier_docs = sum(1 for i in range(last_doc) if i % p != k)
    assert int(n[last_doc]) == n_earlier_docs


@pytest.mark.parametrize("k", [5, 10])
def test_placebo_collapses_to_the_bandwidth_zero_doc_graph(k):
    """Because it is vacuous, the placebo carries the same document information as the floor control:
    the document-level reachability is identical to ``bandwidth=0``."""
    chunk_ids = _one_token_per_chunk(50, k)
    n_chunks = chunk_ids.shape[1]
    placebo = _doc_content_reaching(chunk_ids, n_chunks, k, bandwidth=16, relay=False)
    floor = _doc_content_reaching(chunk_ids, n_chunks, k, bandwidth=0, relay=True)
    p = k + 1
    docs = [i for i in range(n_chunks) if i % p != k]
    assert [int(placebo[i]) for i in docs] == [int(floor[i]) for i in docs]


# ---------------------------------------------------------------------------
# Universal invariants under realistic chunk_ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bandwidth", [0, 1, 4, 16])
@pytest.mark.parametrize("relay", [True, False])
def test_realistic_layout_invariants(bandwidth, relay):
    """FREE rows exactly causal-over-non-pad; PAD never attended; no fully-masked row (which would NaN
    the softmax)."""
    k, b_max = 10, 16
    chunk_ids, _ = _realistic_chunk_ids(50, k, b_max)
    allowed = _allowed(chunk_ids, k=k, bandwidth=bandwidth, relay=relay)
    c = chunk_ids[0]
    s = c.shape[0]
    nonpad = c != PAD_CHUNK_ID
    pos = torch.arange(s)

    for q in torch.nonzero(c == FREE_CHUNK_ID, as_tuple=True)[0].tolist():
        expect = (pos <= q)[nonpad]
        torch.testing.assert_close(allowed[q][nonpad], expect)

    assert not bool(allowed[nonpad][:, ~nonpad].any()), "PAD was attended"
    assert bool(allowed.any(dim=1).all()), "a fully-masked query row would NaN the softmax"


@pytest.mark.parametrize("bandwidth", [0, 1, 16])
def test_induced_doc_graph_matches_one_token_per_chunk_idealization(bandwidth):
    """The variable-length realistic layout must induce the same chunk graph as the one-token-per-chunk
    idealization the design's numbers were computed on."""
    k, b_max = 10, 16
    chunk_ids, n_chunks = _realistic_chunk_ids(50, k, b_max)
    induced = _chunk_graph(_allowed(chunk_ids, k=k, bandwidth=bandwidth), chunk_ids, n_chunks)
    ideal = _allowed(_one_token_per_chunk(50, k), k=k, bandwidth=bandwidth)
    torch.testing.assert_close(induced, ideal)


# ---------------------------------------------------------------------------
# Helpers, config validation, and mask_mod parity
# ---------------------------------------------------------------------------


def test_chunk_token_offset_counts_within_each_run():
    chunk_ids = torch.tensor([[FREE_CHUNK_ID, FREE_CHUNK_ID, 0, 0, 0, 1, 2, 2, PAD_CHUNK_ID]])
    torch.testing.assert_close(
        chunk_token_offset(chunk_ids), torch.tensor([[0, 1, 0, 1, 2, 0, 0, 1, 0]])
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"summary_every_k": 0},
        {"summary_every_k": -1},
        {"summary_bandwidth": -1},
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        AttentionPattern(name="summary_attention", **kwargs)


@pytest.mark.parametrize("bandwidth", [0, 1, 4, 16])
@pytest.mark.parametrize("relay", [True, False])
def test_mask_mod_matches_dense_mask(bandwidth, relay):
    """The FlexAttention ``mask_mod`` must be element-identical to the dense materialization, so the
    block-sparse and eager paths compute the same masked softmax."""
    k, b_max = 10, 16
    chunk_ids, _ = _realistic_chunk_ids(50, k, b_max)
    pattern = _pattern(k=k, bandwidth=bandwidth, relay=relay)
    mask_mod = build_chunked_mask_mod(pattern, chunk_ids)
    assert mask_mod is not None, "summary_attention should be mask_mod-expressible"

    s = chunk_ids.shape[1]
    q = torch.arange(s).unsqueeze(1).expand(s, s)
    kv = torch.arange(s).unsqueeze(0).expand(s, s)
    b = torch.zeros_like(q)
    got = mask_mod(b, None, q, kv)
    torch.testing.assert_close(got, build_chunked_allowed_mask(pattern, chunk_ids)[0])


# ---------------------------------------------------------------------------
# Wiring: the knobs must reach the built module's AttentionPattern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bandwidth", [0, 4, 16])
@pytest.mark.parametrize("relay", [True, False])
def test_config_plumbs_summary_knobs_to_the_module(bandwidth, relay):
    """``TransformerConfig.qwen3_0_6B(document_chunked=True, ...)`` must carry the summary knobs all
    the way into each layer's ``AttentionPattern``."""
    from olmo_core.nn.transformer import TransformerConfig

    cfg = TransformerConfig.qwen3_0_6B(
        vocab_size=1024,
        n_layers=2,
        document_chunked=True,
        cross_doc_mode="summary_attention",
        summary_every_k=10,
        summary_bandwidth=bandwidth,
        summary_relay=relay,
    )
    model = cfg.build()
    for block in model.blocks.values():
        pattern = block.attention._pattern
        assert pattern.name == "summary_attention"
        assert pattern.summary_every_k == 10
        assert pattern.summary_bandwidth == bandwidth
        assert pattern.summary_relay is relay


def test_summary_knobs_rejected_without_document_chunked():
    from olmo_core.exceptions import OLMoConfigurationError
    from olmo_core.nn.transformer import TransformerConfig

    with pytest.raises(OLMoConfigurationError, match="summary_every_k"):
        TransformerConfig.qwen3_0_6B(vocab_size=1024, n_layers=2, summary_every_k=10)
