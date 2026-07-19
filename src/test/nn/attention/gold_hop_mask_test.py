"""
CPU tests for the ``"gold_hop_controlled"`` mask -- the gold-aware hop ladder of the multi-hop
gold-routing experiment (``records/multihop-gold-routing-experiment.md`` §"Approach A").

The arm **deletes the direct gold->gold edge** and forces a path of a controlled length, then asks
whether the model still solves the task. Every property the reading depends on is asserted **per
example** here, because each of them, if silently wrong, fakes a *modeling* conclusion:

  * a surviving gold edge would make ``hop2`` secretly ``hop1``;
  * a shortest path of 2 in the ``hop3`` arm would collapse the ladder's rungs together;
  * a surviving path in ``hop_inf`` would destroy the leak-matched control, which is the only thing
    that makes ``hop2 - hop_inf`` interpretable;
  * an edge-count difference across arms would turn the hop contrast into a density contrast;
  * a fully-masked query row NaNs the softmax; PAD leakage / disturbed FREE rows change the task.

Plus the two structural facts the design cannot wish away and must therefore *measure*: the base graph
is genuinely the gold-agnostic ``random_doc`` graph, and ~4% of gold pairs are **adjacent** and so
unroutable by causality -- detected and reported, never mislabelled.
"""

import random

import numpy as np
import pytest
import torch

from olmo_core.nn.attention.chunked_mask import (
    FREE_CHUNK_ID,
    PAD_CHUNK_ID,
    AttentionPattern,
    build_chunked_allowed_mask,
    build_chunked_mask_mod,
    random_doc_nonce,
)
from olmo_core.nn.attention.gold_grad_mask import content_fingerprint
from olmo_core.nn.attention.gold_hop_mask import (
    GOLD_HOP_VALUES,
    GOLD_HOPS_INF,
    build_base_doc_graph,
    edit_doc_graph_for_hops,
    gold_hop_graph_for_row,
    install_gold_hop_mask,
    make_fingerprint_gold_hop_fn,
    shortest_gold_hops,
)

N_DOCS = 50
KEEP_PROB = 0.25
SEED = 42

# The arms that delete the gold edge -- i.e. everything the experiment actually contrasts. hop1 is the
# upper reference and is the one arm that keeps the edge.
DELETING_ARMS = (2, 3, GOLD_HOPS_INF)


def _pairs_for(example: int, n_pairs: int = 3, *, n_docs: int = N_DOCS):
    """Three disjoint gold pairs at assorted distances, mirroring the real task (3 pairs/example)."""
    rng = random.Random(f"pairs:{example}")
    docs = rng.sample(range(n_docs), 2 * n_pairs)
    return [sorted(docs[2 * i : 2 * i + 2]) for i in range(n_pairs)]


def _graph(pairs, hops, *, example: int = 0, keep_prob: float = KEEP_PROB, n_docs: int = N_DOCS):
    return gold_hop_graph_for_row(
        n_docs,
        pairs,
        hops=hops,
        doc_keep_prob=keep_prob,
        seed=SEED,
        nonce=example * 7919 + 13,
        fingerprint=f"fp{example}",
    )


# ---------------------------------------------------------------------------
# The base graph really is the gold-agnostic random_doc graph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("keep_prob", [0.25, 0.5])
@pytest.mark.parametrize("per_example", [False, True])
def test_base_graph_is_bit_identical_to_the_random_doc_pattern(keep_prob, per_example):
    """The camouflage claim ("deleting the gold edge looks like an ordinary miss") only holds if the
    base graph IS ``random_doc``, not merely something similar. One token per doc makes the ``(S, S)``
    mask the doc graph exactly, so the two can be compared elementwise."""
    chunk_ids = torch.arange(N_DOCS).unsqueeze(0)
    nonce = int(random_doc_nonce(chunk_ids)[0].item()) if per_example else None

    ours = build_base_doc_graph(N_DOCS, doc_keep_prob=keep_prob, seed=SEED, nonce=nonce)
    ref = build_chunked_allowed_mask(
        AttentionPattern(
            name="random_doc",
            doc_keep_prob=keep_prob,
            random_seed=SEED,
            random_doc_per_example=per_example,
        ),
        chunk_ids,
    )[0]
    # random_doc's mask adds the self-diagonal and the causal gate; our graph is the strict
    # cross-document part, so compare on the strictly-lower triangle.
    q = torch.arange(N_DOCS).unsqueeze(1)
    k = torch.arange(N_DOCS).unsqueeze(0)
    strictly_earlier = q > k
    torch.testing.assert_close(
        torch.from_numpy(ours)[strictly_earlier], (ref & strictly_earlier)[strictly_earlier]
    )


def test_base_graph_out_degree_matches_the_measured_random_doc_figures():
    """Sanity-check the base against the numbers the design was sized on: p=0.25 -> out-deg ~6.02 of
    24.5, p=0.5 -> ~12.02 (measured at n=50 in the experiment record)."""
    for keep_prob, expected in ((0.25, 6.02), (0.5, 12.02)):
        degs = [
            build_base_doc_graph(N_DOCS, doc_keep_prob=keep_prob, seed=SEED, nonce=i).sum() / N_DOCS
            for i in range(40)
        ]
        assert abs(float(np.mean(degs)) - expected) < 0.4, (keep_prob, np.mean(degs))


# ---------------------------------------------------------------------------
# The gold edge, and the realized path length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hops", DELETING_ARMS)
@pytest.mark.parametrize("example", range(12))
def test_no_gold_edge_survives_in_the_deleting_arms(hops, example):
    """PER EXAMPLE: the direct gold->gold edge is gone in hop2 / hop3 / hop_inf. This is the arm's
    defining property -- the whole experiment is "what happens without the direct edge"."""
    pairs = _pairs_for(example)
    adj, _ = _graph(pairs, hops, example=example)
    for a, b in ((min(p), max(p)) for p in pairs):
        assert not adj[b, a], f"gold edge {b}->{a} survived in hop{hops}"


@pytest.mark.parametrize("hops", [1, 2, 3])
@pytest.mark.parametrize("example", range(12))
def test_shortest_gold_path_is_exactly_the_arm(hops, example):
    """PER EXAMPLE: the realized shortest path equals the arm -- not "at most", not "at least". A
    hop3 arm that leaves a 2-path is silently a hop2 arm, which would merge two rungs of the ladder.

    Pairs the causal structure makes unroutable (an adjacent pair cannot have a 2-hop path) are the
    documented exception: they are honestly forced to hop_inf and reported as such.
    """
    pairs = _pairs_for(example)
    adj, stats = _graph(pairs, hops, example=example)
    for (a, b), realized, unroutable in zip(
        stats.pairs, stats.realized_hops, stats.unroutable
    ):
        assert realized == shortest_gold_hops(adj, a, b)
        if unroutable:
            assert realized == GOLD_HOPS_INF, "an unroutable pair must be honest hop_inf"
            assert (b - a) < hops
        else:
            assert realized == hops, f"pair {(a, b)} realized {realized}, expected {hops}"


@pytest.mark.parametrize("example", range(12))
def test_hop_inf_leaves_no_path_at_all(example):
    """PER EXAMPLE: hop_inf cuts EVERY path, not just the direct edge. It is the leak-matched control
    -- identical gold-edge-absent structure to hop2, differing only in whether a path exists -- so a
    surviving 3-hop path here would quietly contaminate the one contrast that carries the claim."""
    pairs = _pairs_for(example)
    adj, stats = _graph(pairs, GOLD_HOPS_INF, example=example)
    for a, b in stats.pairs:
        assert shortest_gold_hops(adj, a, b) == GOLD_HOPS_INF
    assert all(h == GOLD_HOPS_INF for h in stats.realized_hops)


@pytest.mark.parametrize("example", range(12))
def test_hop1_forces_the_direct_edge(example):
    """hop1 is the ladder's upper reference: the gold edge is present for every pair, always."""
    pairs = _pairs_for(example)
    adj, stats = _graph(pairs, 1, example=example)
    for a, b in stats.pairs:
        assert adj[b, a]
    assert all(h == 1 for h in stats.realized_hops)
    assert not any(stats.unroutable)


# ---------------------------------------------------------------------------
# Density: the arms must differ in structure, not in edge count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("example", range(12))
def test_per_example_edge_count_is_matched_across_arms(example):
    """PER EXAMPLE: every arm carries the same number of doc->doc edges as the untouched base graph,
    and therefore as every other arm.

    This is exact rather than approximate because the base graph is a pure function of
    ``(seed, keep_prob, nonce)`` -- the same example gets the *same* base in every arm -- so the edit
    can be compensated back to that graph's own count. Without this the hop contrast would be a
    density contrast wearing a hop costume.
    """
    pairs = _pairs_for(example)
    counts = {}
    for hops in GOLD_HOP_VALUES:
        adj, stats = _graph(pairs, hops, example=example)
        assert int(adj.sum()) == stats.final_edges
        assert stats.edge_drift == 0, f"hop{hops} drifted {stats.edge_drift} edges from the base"
        counts[hops] = stats.final_edges
    assert len(set(counts.values())) == 1, counts


@pytest.mark.parametrize("keep_prob", [0.25, 0.5, 1.0])
@pytest.mark.parametrize("hops", GOLD_HOP_VALUES)
def test_compensation_never_reintroduces_a_gold_edge_or_a_shorter_path(hops, keep_prob):
    """Compensation adds random NON-gold edges. Each candidate is validated before it is kept, which
    matters concretely: at hop3 an added edge can create a 2-path, and at hop_inf it can reconnect the
    pair -- i.e. the density fix could silently undo the arm.

    Swept over ``keep_prob`` because base density is exactly what decides whether a violation is
    *reachable*: at p=0.25 many of these invariants hold by luck, at p=1.0 they can only hold by
    construction. p=0.5 is the design's documented fallback if the sparse base cannot route at all.
    """
    for example in range(12):
        pairs = _pairs_for(example)
        adj, stats = _graph(pairs, hops, example=example, keep_prob=keep_prob)
        for (a, b), realized, unroutable in zip(stats.pairs, stats.realized_hops, stats.unroutable):
            if hops == 1:
                assert realized == 1
            elif hops == GOLD_HOPS_INF or unroutable:
                assert realized == GOLD_HOPS_INF
            else:
                assert realized == hops
                assert not adj[b, a]


def test_edit_is_deterministic_for_the_same_example():
    """The same example must yield the same graph on every epoch and at eval -- otherwise a held-out
    example is scored on a fresh coin flip rather than a well-defined mask."""
    pairs = _pairs_for(3)
    a1, s1 = _graph(pairs, 2, example=3)
    a2, s2 = _graph(pairs, 2, example=3)
    assert np.array_equal(a1, a2)
    assert s1.realized_hops == s2.realized_hops


# ---------------------------------------------------------------------------
# The adjacent-gold-pair case: detected and reported, never mislabelled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hops", [2, 3])
def test_adjacent_gold_pair_is_detected_and_reported_not_mislabelled(hops):
    """⚠ A causal mask makes a 2-hop path ``b -> m -> a`` need ``a < m < b``, so an ADJACENT gold pair
    has no possible intermediary and is silently hop_inf whatever the arm claims. Measured on the real
    n=50 shard: 4.03% of gold pairs are adjacent (7.82% are at distance <= 2, which also rules out a
    3-hop path).

    The design deliberately does NOT fix this by constraining gold placement -- that would make
    placement gold-aware and re-introduce the leak. So it must be *visible*: flagged per pair, and the
    realized hop recorded as hop_inf rather than reported as hop-h.
    """
    pairs = [[10, 11], [20, 22], [3, 40]]  # distance 1 (never routable), 2 (no 3-hop), 37 (fine)
    # ⚠ keep_prob=1.0 on purpose. At the sparse p=0.25 base an unroutable pair often has no residual
    # path *by luck*, so this test passed against an implementation that deleted the direct edge and
    # then returned early -- leaving a distance-2 pair at a realized 2 hops inside the hop3 arm. A
    # complete base graph guarantees the short path exists, so the cut has to be real.
    adj, stats = _graph(pairs, hops, keep_prob=1.0)

    expected_unroutable = [(b - a) < hops for a, b in stats.pairs]
    assert stats.unroutable == expected_unroutable
    for (a, b), realized, unroutable in zip(stats.pairs, stats.realized_hops, stats.unroutable):
        assert not adj[b, a], "the direct edge goes even for an unroutable pair"
        assert realized == (GOLD_HOPS_INF if unroutable else hops)

    # ...and it is never silently laundered into the arm's nominal hop count.
    assert GOLD_HOPS_INF in stats.realized_hops
    assert stats.distances == [1, 2, 37]


def test_adjacent_pair_fraction_is_reported_by_the_holder_summary():
    """The realized hop distribution must be observable at runtime, so eval can stratify on it and the
    ``hop2`` headline can be reported alongside its mixture-free routable subset."""
    from olmo_core.nn.attention.gold_hop_mask import GoldHopMaskHolder

    holder = GoldHopMaskHolder()
    for example in range(20):
        pairs = _pairs_for(example) + [[7, 8]]  # force an adjacent pair into every example
        _, stats = _graph(pairs, 2, example=example)
        holder.stats[f"fp{example}"] = stats

    hist = holder.hop_histogram()
    assert hist[2] > 0 and hist[GOLD_HOPS_INF] >= 20  # >= one forced-adjacent pair per example
    summary = holder.summary()
    assert "realized_hops" in summary and "unroutable" in summary and "edge_drift" in summary


# ---------------------------------------------------------------------------
# The mask over real token roles
# ---------------------------------------------------------------------------


def _chunk_ids(n_docs: int = N_DOCS, seed: int = 5) -> torch.Tensor:
    """Variable-length docs, a FREE instruction prefix, a FREE query/answer tail, a PAD tail -- i.e.
    what a tokenized contradiction row actually looks like."""
    g = torch.Generator().manual_seed(seed)
    ids: list = [FREE_CHUNK_ID] * 6
    for cid in range(n_docs):
        ids += [cid] * int(torch.randint(2, 6, (1,), generator=g).item())
    ids += [FREE_CHUNK_ID] * 8 + [PAD_CHUNK_ID] * 4
    return torch.tensor(ids).unsqueeze(0)


def _allowed(chunk_ids, adj, hops):
    return build_chunked_allowed_mask(
        AttentionPattern(name="gold_hop_controlled", gold_hops=hops),
        chunk_ids,
        doc_adjacency=torch.from_numpy(adj).unsqueeze(0),
    )


@pytest.mark.parametrize("hops", GOLD_HOP_VALUES)
def test_mask_invariants_free_rows_pad_and_no_fully_masked_row(hops):
    """FREE rows exactly causal-over-non-pad; PAD never attended; no fully-masked row (which would NaN
    the softmax). The FREE bridge is channel (a) and is common to every mask in the family -- if the
    gold edit touched it, the arms would no longer differ only in the doc graph."""
    chunk_ids = _chunk_ids()
    adj, _ = _graph(_pairs_for(0), hops)
    allowed = _allowed(chunk_ids, adj, hops)[0]

    c = chunk_ids[0]
    s = c.shape[0]
    pos = torch.arange(s)
    nonpad = c != PAD_CHUNK_ID

    for q in torch.nonzero(c == FREE_CHUNK_ID).flatten().tolist():
        torch.testing.assert_close(allowed[q][nonpad], (pos <= q)[nonpad])
    assert not bool(allowed[nonpad][:, ~nonpad].any()), "PAD was attended"
    assert bool(allowed.any(dim=-1).all()), "a fully-masked query row would NaN the softmax"
    # Causality is never violated by the gold edit.
    assert not bool((allowed & (pos.unsqueeze(0) > pos.unsqueeze(1))).any())


@pytest.mark.parametrize("hops", DELETING_ARMS)
def test_mask_realizes_the_doc_graph_at_token_level(hops):
    """The token-level mask must be the doc graph, verbatim: a context token may attend an earlier
    document's tokens iff that edge exists. Otherwise the graph-level guarantees above say nothing
    about what the model sees."""
    chunk_ids = _chunk_ids(seed=7)
    pairs = _pairs_for(1)
    adj, _ = _graph(pairs, hops, example=1)
    allowed = _allowed(chunk_ids, adj, hops)[0]

    c = chunk_ids[0]
    ctx = torch.nonzero(c >= 0).flatten().tolist()
    rng = random.Random(0)
    for q in rng.sample(ctx, 60):
        for k in rng.sample(ctx, 60):
            if k >= q:
                continue
            qc, kc = int(c[q]), int(c[k])
            expected = qc == kc or bool(adj[qc, kc])
            assert bool(allowed[q, k]) == expected, (q, k, qc, kc)

    # ...and specifically: no token of gold doc b may attend any token of its partner a.
    for a, b in ((min(p), max(p)) for p in pairs):
        rows = torch.nonzero(c == b).flatten()
        cols = torch.nonzero(c == a).flatten()
        assert not bool(allowed[rows][:, cols].any()), f"gold pair {(a, b)} attends at token level"


def test_mask_without_a_graph_raises_rather_than_silently_degrading():
    """A missing graph is a *different experiment*, not a degraded mask -- the arm is defined by which
    edges are absent. Fail loudly instead of quietly training something unnamed."""
    with pytest.raises(ValueError, match="doc_adjacency"):
        build_chunked_allowed_mask(
            AttentionPattern(name="gold_hop_controlled", gold_hops=2), _chunk_ids()
        )


def test_pattern_rejects_an_unknown_arm():
    with pytest.raises(ValueError, match="gold_hops"):
        AttentionPattern(name="gold_hop_controlled", gold_hops=4)


def test_no_mask_mod_is_offered_so_the_family_stays_on_one_eager_path():
    """The graph comes from a per-forward Python hook, which torch.compile cannot capture. Declining a
    mask_mod keeps every arm on the dense boolean path rather than splitting them across two."""
    assert (
        build_chunked_mask_mod(
            AttentionPattern(name="gold_hop_controlled", gold_hops=2), _chunk_ids()
        )
        is None
    )


# ---------------------------------------------------------------------------
# The non-leaky fingerprint lookup
# ---------------------------------------------------------------------------

DOC_START, DOC_END, EOS = 151648, 151649, 151643


def _row(n_docs: int = 6, *, salt: int = 0):
    ids = [10, 11, 12]
    for d in range(n_docs):
        ids += [DOC_START, 100 + d + salt, 200 + d, DOC_END]
    ids += [20, 21, EOS]
    return ids


def test_fingerprint_lookup_builds_the_right_graph_and_never_sees_a_gold_token():
    """Gold identity must NEVER enter the token stream -- a gold marker would let the model score
    without reading anything. The lookup is keyed on a SHA1 of the row's own content ids, exactly as
    ``gold_grad_mask`` does, so the tokens are untouched."""
    ids = _row(6)
    fp = content_fingerprint(ids)
    fn = make_fingerprint_gold_hop_fn(
        {fp: [[1, 4]]},
        doc_start_id=DOC_START,
        doc_end_id=DOC_END,
        eos_id=EOS,
        hops=2,
        doc_keep_prob=0.5,
        seed=SEED,
    )
    adj = fn(torch.tensor([ids]))
    assert adj.shape == (1, 6, 6)
    assert not bool(adj[0, 4, 1]), "the gold edge survived the lookup path"
    assert shortest_gold_hops(adj[0].numpy(), 1, 4) == 2
    assert fn.hops == 2


def test_fingerprint_miss_leaves_the_row_unmasked():
    """⚠ A miss must degrade to ALL-TRUE (plain causal over context), never all-False. All-False would
    silently turn the row into a ``chunked`` example -- a different arm, reported under this arm's
    name. The trainer's synthetic warm-up mock batch always misses, so this path is always exercised.
    """
    fn = make_fingerprint_gold_hop_fn(
        {},
        doc_start_id=DOC_START,
        doc_end_id=DOC_END,
        eos_id=EOS,
        hops=GOLD_HOPS_INF,
        doc_keep_prob=0.25,
        seed=SEED,
    )
    adj = fn(torch.tensor([_row(5)]))
    assert bool(adj.all()), "a fingerprint miss must not restrict the row"

    chunk_ids = torch.tensor([[FREE_CHUNK_ID] * 2 + [0] * 3 + [1] * 3 + [2] * 3 + [FREE_CHUNK_ID]])
    allowed = build_chunked_allowed_mask(
        AttentionPattern(name="gold_hop_controlled", gold_hops=GOLD_HOPS_INF),
        chunk_ids,
        doc_adjacency=torch.ones(1, 3, 3, dtype=torch.bool),
    )[0]
    ref = build_chunked_allowed_mask(AttentionPattern(name="standard"), chunk_ids)[0]
    torch.testing.assert_close(allowed, ref)


def test_flat_sidecar_is_rejected():
    """``gold_fingerprints.json`` is an unordered SET -- it cannot say which document contradicts
    which, and "delete the gold edge" is meaningless without the partner. That exact defect invalidated
    the first gold-grad arms, so it is refused rather than guessed at."""
    with pytest.raises(ValueError, match="PAIR-preserving"):
        make_fingerprint_gold_hop_fn(
            {"fp": [1, 4, 7]},
            doc_start_id=DOC_START,
            doc_end_id=DOC_END,
            eos_id=EOS,
            hops=2,
            doc_keep_prob=0.25,
        )


def test_batch_rows_get_independent_graphs_and_stats_are_recorded():
    rows = [_row(8, salt=0), _row(8, salt=50)]
    table = {content_fingerprint(r): [[1, 5]] for r in rows}
    stats: dict = {}
    fn = make_fingerprint_gold_hop_fn(
        table,
        doc_start_id=DOC_START,
        doc_end_id=DOC_END,
        eos_id=EOS,
        hops=2,
        doc_keep_prob=0.5,
        seed=SEED,
        stats=stats,
    )
    adj = fn(torch.tensor(rows))
    assert adj.shape == (2, 8, 8)
    assert len(stats) == 2
    for b in range(2):
        assert not bool(adj[b, 5, 1])
        assert shortest_gold_hops(adj[b].numpy(), 1, 5) == 2
    # Per-example resampling is mandatory: two different examples must not share one memorizable graph
    # (against a fixed graph, a MISSING edge announces which document is gold).
    assert not np.array_equal(adj[0].numpy(), adj[1].numpy())


def test_edit_rejects_a_degenerate_pair():
    with pytest.raises(ValueError):
        edit_doc_graph_for_hops(
            build_base_doc_graph(10, doc_keep_prob=0.5, seed=1, nonce=1),
            [[3, 3]],
            2,
            random.Random(0),
        )


# ---------------------------------------------------------------------------
# End to end: the hook must actually reach the attention module
# ---------------------------------------------------------------------------


def _tiny_gold_hop_model(hops: int):
    """A 2-layer document_chunked transformer -- the real module, not a stand-in.

    ⚠ ``document_chunk_attention`` must be set, exactly as the launcher does. Without it the model
    never reconstructs ``chunk_ids`` from the boundary tokens, every layer takes the documented
    "no roles -> plain causal" fallback, and the gold mask is silently inert -- a model that trains
    happily and answers a different question. Found by this test.
    """
    from olmo_core.nn.transformer import TransformerConfig

    cfg = TransformerConfig.llama_like(
        d_model=64,
        n_layers=2,
        n_heads=8,
        vocab_size=151936,
        document_chunked=True,
        cross_doc_mode="gold_hop_controlled",
        gold_hops=hops,
    )
    cfg.document_chunk_attention = {
        "doc_start_id": DOC_START,
        "doc_end_id": DOC_END,
        "eos_id": EOS,
        "mode": "chunked",
    }
    return cfg.build()


def test_end_to_end_hook_feeds_the_real_attention_and_the_arms_differ():
    """The graph-level guarantees are worth nothing if the hook never reaches attention. Run the real
    ``DocumentChunkedAttention`` under each arm and require the outputs to actually differ: same
    weights, same tokens, same positions -- only the gold edge differs, which is exactly the
    experiment's founding claim ("arms differ only in the mask")."""
    ids = _row(8)
    table = {content_fingerprint(ids): [[1, 5]]}
    batch = torch.tensor([ids])

    outs = {}
    for hops in (1, GOLD_HOPS_INF):
        torch.manual_seed(0)
        model = _tiny_gold_hop_model(hops)
        fn = make_fingerprint_gold_hop_fn(
            table,
            doc_start_id=DOC_START,
            doc_end_id=DOC_END,
            eos_id=EOS,
            hops=hops,
            doc_keep_prob=0.5,
            seed=SEED,
        )
        holder = install_gold_hop_mask(model, fn)
        assert holder.n_attached == 2
        model.eval()
        with torch.no_grad():
            outs[hops] = model(input_ids=batch)
        assert holder.adjacency is not None and holder.adjacency.shape == (1, 8, 8)
        assert len(holder.stats) == 1

    assert not torch.allclose(outs[1], outs[GOLD_HOPS_INF]), (
        "hop1 and hop_inf produced identical logits -- the mask is not reaching attention"
    )
    assert torch.isfinite(outs[1]).all() and torch.isfinite(outs[GOLD_HOPS_INF]).all()


def test_end_to_end_without_the_hook_raises_instead_of_training_the_wrong_arm():
    """A ``gold_hop_controlled`` model with no hook installed must refuse to run. Silently falling back
    to some default graph would train an unnamed arm and report it under this one's name."""
    from olmo_core.exceptions import OLMoConfigurationError

    model = _tiny_gold_hop_model(2)
    with pytest.raises(OLMoConfigurationError, match="install_gold_hop_mask"):
        model(input_ids=torch.tensor([_row(6)]))


def test_realized_hops_dump_round_trips_for_eval_stratification(tmp_path):
    """Eval joins on this file to stratify per-example f1 by REALIZED hop distance -- the ``hop2``
    headline is a ~96/4 mixture over it, and the routable subset is the mixture-free number."""
    import json

    from olmo_core.nn.attention.gold_hop_mask import dump_realized_hops

    stats = {}
    for example in range(5):
        pairs = _pairs_for(example) + [[7, 8]]
        _, st = _graph(pairs, 2, example=example)
        stats[f"fp{example}"] = st

    path = str(tmp_path / "realized_hops.json")
    assert dump_realized_hops(stats, path) == 5
    loaded = json.load(open(path))
    assert set(loaded) == set(stats)
    for fp, rec in loaded.items():
        assert rec["realized_hops"] == stats[fp].realized_hops
        assert rec["edge_drift"] == 0 and rec["converged"] is True
        assert GOLD_HOPS_INF in rec["realized_hops"]  # the forced adjacent pair


# ---------------------------------------------------------------------------
# The KV-cached decode path: does the gold edit survive generation?
# ---------------------------------------------------------------------------


def _attach_kv_cache(model, *, batch_size: int, max_seq_len: int):
    from olmo_core.nn.attention.document_chunked import DocumentChunkedAttention
    from olmo_core.nn.attention.kv_cache import KVCacheManager

    n = 0
    for m in model.modules():
        if isinstance(m, DocumentChunkedAttention):
            m.kv_cache_manager = KVCacheManager(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_kv_heads=m.n_kv_heads,
                head_dim=m.head_dim,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            n += 1
    return n


def _fixed_adjacency_fn(adj: np.ndarray, hops: int):
    """A test double that pins ONE graph regardless of input, so the KV-cache question can be asked in
    isolation from the fingerprint question."""

    def fn(input_ids):
        return torch.from_numpy(adj).unsqueeze(0)

    fn.hops = hops
    return fn


def test_kv_cache_retains_the_gold_edit_from_prefill():
    """⚠ The decode path is where a gold-hop eval could silently lose its mask -- and the answer tokens
    (``[[a, b], ...]``) are generated exactly where the cross-doc evidence is needed. So measure it on
    the real cache tensor rather than reasoning about it.

    ``DocumentChunkedAttention.sdpa`` routes a 1-token decode query to the base, UNMASKED path. That is
    correct, not a fallback: under ``allowed = causal & not_pad & (context_ok | q_free | kv_free)`` a
    **FREE** query attends everything causally in *every* arm, and a generated token is FREE. So plain
    causal IS the gold-hop mask for that row.

    Which makes the load-bearing question: do the CONTEXT rows the answer reads carry the gold edit?
    They live in the KV cache, written during the masked prefill and never recomputed. Asserted here:
    the cached K/V for context positions must DIFFER between arms. If they did not, every arm would be
    scored with the mask effectively off for the tokens that matter.
    """
    from olmo_core.nn.attention.document_chunked import DocumentChunkedAttention

    ids = _row(8)
    pairs = [[1, 5]]
    caches = {}
    for hops in (1, GOLD_HOPS_INF):
        torch.manual_seed(0)
        model = _tiny_gold_hop_model(hops)
        adj, _ = _graph(pairs, hops, n_docs=8, keep_prob=0.5)
        install_gold_hop_mask(model, _fixed_adjacency_fn(adj, hops))
        assert _attach_kv_cache(model, batch_size=1, max_seq_len=len(ids) + 4) == 2
        model.eval()
        with torch.no_grad():
            model(input_ids=torch.tensor([ids]))  # prefill: masked attention + populate the cache
        layers = [m for m in model.modules() if isinstance(m, DocumentChunkedAttention)]
        caches[hops] = [m.kv_cache_manager.k_cache[:, : len(ids)].clone() for m in layers]
        # the prefill really did populate the cache
        assert all(int(m.kv_cache_manager.current_position()) == len(ids) for m in layers)

    # Layer 0's K/V come straight from the embeddings, so they are identical by construction -- the
    # mask cannot have acted yet. Layer 1's K/V are computed from layer 0's MASKED output, so they are
    # where the gold edit must show up.
    assert torch.allclose(caches[1][0], caches[GOLD_HOPS_INF][0]), "layer 0 K should not depend on the mask"
    assert not torch.allclose(caches[1][1], caches[GOLD_HOPS_INF][1]), (
        "the cached context K/V are identical across hop1 and hop_inf -- the prefill's gold edit did "
        "NOT reach the cache, so generation would read un-edited context and every arm would land "
        "near the ceiling."
    )


def test_decode_query_is_free_so_plain_causal_is_the_correct_mask():
    """Pins the *reason* decode is unmasked, so a future reader does not 'fix' it into a bug: a
    generated token is FREE, and a FREE row is exactly causal-over-non-pad under every arm -- including
    hop_inf, whose gold pair has no path at all. The FREE bridge is channel (a) and is common to every
    mask in the family."""
    chunk_ids = _chunk_ids()
    c = chunk_ids[0]
    pos = torch.arange(c.shape[0])
    nonpad = c != PAD_CHUNK_ID
    free_rows = torch.nonzero(c == FREE_CHUNK_ID).flatten().tolist()

    for hops in GOLD_HOP_VALUES:
        adj, _ = _graph(_pairs_for(0), hops)
        allowed = _allowed(chunk_ids, adj, hops)[0]
        for q in free_rows:
            torch.testing.assert_close(allowed[q][nonpad], (pos <= q)[nonpad])


def test_decode_row_is_not_counted_as_a_fingerprint_miss():
    """⚠ Under a KV cache, decode feeds ONE token, which carries no documents and whose fingerprint
    cannot match by construction. Counting those as misses would drive the hit rate to ~1/31 on a
    perfectly healthy contradiction eval and fire the hard assert as a false alarm."""
    ids = _row(8)
    table = {content_fingerprint(ids): [[1, 5]]}
    fn = make_fingerprint_gold_hop_fn(
        table,
        doc_start_id=DOC_START,
        doc_end_id=DOC_END,
        eos_id=EOS,
        hops=2,
        doc_keep_prob=0.5,
        seed=SEED,
    )
    fn(torch.tensor([ids]))  # the prefill: a document-bearing row -> a real lookup
    for _ in range(30):  # 30 decode steps, exactly as contradiction generation does
        fn(torch.tensor([[20]]))

    assert fn.counters["graph_rows"] == 1, "decode rows must not enter the hit-rate denominator"
    assert fn.counters["hits"] == 1
    assert fn.misses == []


# ---------------------------------------------------------------------------
# (b) The hard eval assert
# ---------------------------------------------------------------------------


def test_require_full_hit_rate_raises_on_a_wrong_sidecar_and_passes_on_the_right_one():
    """The single most important guard in the eval path: a miss degrades to plain causal = unrestricted
    `standard` (near the ceiling), so the failure LOOKS LIKE SUCCESS. It must be an exception."""
    from olmo_core.nn.attention.gold_hop_mask import GoldHopMaskHolder

    ids = _row(8)
    right = {content_fingerprint(ids): [[1, 5]]}

    for table, should_raise in ((right, False), ({"deadbeef": [[1, 5]]}, True)):
        fn = make_fingerprint_gold_hop_fn(
            table,
            doc_start_id=DOC_START,
            doc_end_id=DOC_END,
            eos_id=EOS,
            hops=2,
            doc_keep_prob=0.5,
            seed=SEED,
        )
        fn(torch.tensor([ids]))
        holder = GoldHopMaskHolder(counters=fn.counters, misses=fn.misses)
        if should_raise:
            assert holder.hit_rate == 0.0
            with pytest.raises(SystemExit, match="MISSED the gold-pairs sidecar"):
                holder.require_full_hit_rate(context="unit test")
        else:
            assert holder.hit_rate == 1.0
            holder.require_full_hit_rate(context="unit test")


def test_require_full_hit_rate_raises_when_the_hook_never_saw_a_document():
    """Zero document-bearing rows means the mask was never applied at all -- nothing was measured. That
    is not a 100% hit rate, it is a vacuous run."""
    from olmo_core.nn.attention.gold_hop_mask import GoldHopMaskHolder

    holder = GoldHopMaskHolder(counters={"graph_rows": 0, "hits": 0})
    with pytest.raises(SystemExit, match="never saw a single document-bearing row"):
        holder.require_full_hit_rate()


# ---------------------------------------------------------------------------
# The decoy fix: distance-matched camouflage for the arm's structural signature
# ---------------------------------------------------------------------------

DECOY_ARMS = (1, 2, GOLD_HOPS_INF)  # hop3 is over-constrained by decoys -- see the record


def test_decoys_are_distance_matched_and_never_gold():
    """The fix only works if the decoys are indistinguishable from gold ON THE FEATURE THE ARM EDITS.
    Distance is that feature: ``P(unreachable | non-gold)`` runs 0.796 at distance 1 down to 0.165 at
    26-49, so an unreachable pair is only suspicious *relative to its own distance*."""
    from olmo_core.nn.attention.gold_hop_mask import sample_distance_matched_decoys

    gold = [(3, 30), (10, 25), (5, 6)]
    decoys = sample_distance_matched_decoys(50, gold, 6, random.Random(0), hops=2)
    gold_set = set(gold)
    gold_dists = {b - a for a, b in gold if (b - a) >= 5}
    assert decoys, "no decoys sampled"
    assert len(decoys) == len(set(decoys)), "decoys must be distinct"
    for a, b in decoys:
        assert (a, b) not in gold_set
        assert 0 <= a < b < 50
        assert (b - a) in gold_dists, "a decoy must sit at one of the GOLD distances"


def test_decoys_skip_short_pairs_where_there_is_no_leak_to_hide():
    """⚠ Two measured reasons, both in the record: (1) at distance 1 the lift is 1.26 -- three quarters
    of short pairs are unreachable by coin flip, so there is nothing to camouflage; (2) 12 decoys at
    distance 1 forbid 12 direct edges in one neighbourhood and the GOLD pairs stop converging."""
    from olmo_core.nn.attention.gold_hop_mask import sample_distance_matched_decoys

    only_short = sample_distance_matched_decoys(50, [(10, 11), (20, 22)], 6, random.Random(0), hops=2)
    assert only_short == [], "short gold pairs must not be decoyed"
    # ...and an arm cannot decoy a pair it could not route anyway.
    assert sample_distance_matched_decoys(50, [(10, 12)], 6, random.Random(0), hops=3) == []


@pytest.mark.parametrize("hops", DECOY_ARMS)
@pytest.mark.parametrize("example", range(8))
def test_gold_contract_survives_the_decoys(hops, example):
    """The camouflage must not cost correctness: with decoys on, every GOLD pair still has no direct
    edge and its exact target hop count, and the edge count is still matched to the base graph."""
    pairs = _pairs_for(example)
    adj, stats = _graph(pairs, hops, example=example)
    adj_d, stats_d = gold_hop_graph_for_row(
        N_DOCS, pairs, hops=hops, doc_keep_prob=KEEP_PROB, seed=SEED,
        nonce=example * 7919 + 13, fingerprint=f"fp{example}", n_decoys=12,
    )
    assert stats_d.converged, "decoys must not break the GOLD pairs"
    assert stats_d.edge_drift == 0, f"decoys drifted the edge count by {stats_d.edge_drift}"
    assert stats_d.n_decoys > 0 and stats_d.decoys_ok == stats_d.n_decoys
    for (a, b), realized, unroutable in zip(stats_d.pairs, stats_d.realized_hops, stats_d.unroutable):
        if hops == 1:
            assert adj_d[b, a] and realized == 1
        else:
            assert not adj_d[b, a]
            assert realized == (GOLD_HOPS_INF if (unroutable or hops == GOLD_HOPS_INF) else hops)
    # the decoys really did change the graph (i.e. the knob is not silently inert)
    assert not np.array_equal(adj, adj_d)


@pytest.mark.parametrize("hops", DECOY_ARMS)
def test_decoys_give_non_gold_pairs_the_arms_signature(hops):
    """The point of the fix, asserted directly: after the edit, the arm's signature names MANY pairs,
    not just the 3 gold ones. Measured effect on a graph-only classifier (debug/gold_hop/leak_probe.py,
    300 real eval examples): hop_inf precision@3 falls 16.2% -> 2.0% (66x -> 8x over chance), and hop2
    (7.3x) and hop_inf (8.2x) become leak-MATCHED -- which is what makes hop2 - hop_inf interpretable.
    """
    pairs = [[3, 30], [8, 35], [12, 40]]  # distances 27, 27, 28 -- long, i.e. the leaky regime
    _, stats = gold_hop_graph_for_row(
        N_DOCS, pairs, hops=hops, doc_keep_prob=KEEP_PROB, seed=SEED,
        nonce=11, fingerprint="fp-decoy", n_decoys=12,
    )
    # ⚠ NOT 3x12=36. Only ``n_docs - d`` pairs exist at distance ``d``, so the decoy pool SHRINKS as
    # distance grows -- at d=48 there are just 2 candidate pairs in the whole example. Long distances
    # are exactly where the leak is worst (lift 45x undecoyed) AND where the disguise is thinnest.
    # That is the fix's ceiling, and it is why hop_inf still measures 6.06x lift at distance 26-49
    # after decoys rather than 1.0x. Asserted here so the limit is visible, not discovered later.
    assert 30 <= stats.n_decoys < 36
    assert stats.decoys_ok == stats.n_decoys, "every sampled decoy must realize the signature"
