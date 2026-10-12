"""
Tests for the per-document summary-token mask
(:mod:`olmo_core.nn.attention.summary_mask`).

The properties being pinned are the two reachability rules the experiment is defined by:

* on a **masked** example the query reads the instruction, its own span and every summary run, but
  **no** raw document content; and a context document reads itself plus strictly earlier summary
  runs only;
* on a **causal** example the mask is exactly plain causal.

Plus a migration check against the legacy ``"summary_attention"``
:class:`~olmo_core.nn.attention.chunked_mask.AttentionPattern`, which this module replaces: at the
configuration the legacy pattern has to be coerced into (``summary_every_k=1``,
``summary_bandwidth=N``, ``summary_relay=True``) the two must be element-identical. That is the proof
the rewrite changed the vocabulary and not the semantics.
"""

import pytest
import torch

from olmo_core.nn.attention.chunked_mask import (
    FREE_CHUNK_ID,
    PAD_CHUNK_ID,
    AttentionPattern,
    build_chunked_allowed_mask,
)
from olmo_core.nn.attention.summary_mask import (
    ROLE_DOC_ID,
    ROLE_KIND,
    ROLE_SUMMARY_OFFSET,
    SummaryMaskSpec,
    TokenKind,
    build_summary_mask_mod,
    build_summary_roles,
    summary_mask_allowed,
)

# Synthetic reserved ids; the real values live in ``olmo_core.data.document_chunk_landmark``.
DOC_START, DOC_END, SUMM, EOS, PAD = 900, 901, 902, 903, 904
N_SUMMARY = 3

IDS_KW = dict(
    doc_start_id=DOC_START, doc_end_id=DOC_END, summary_token_id=SUMM, eos_id=EOS, pad_id=PAD
)


def _build_example(doc_bodies, *, wrap_summary_runs: bool, query_len: int = 3, n_pad: int = 4):
    """``[instruction][<doc><summ>]*[<query>][eos][pad*]`` as a ``(1, T)`` id tensor.

    ``wrap_summary_runs`` toggles whether each summary run gets its own boundary markers. The roles
    builder identifies summary tokens by **id**, so both layouts must give identical roles -- that is
    what lets the data layer choose freely.
    """
    ids = [10, 11, 12]  # instruction prefix
    for body in doc_bodies:
        ids += [DOC_START] + body + [DOC_END]
        run = [SUMM] * N_SUMMARY
        ids += ([DOC_START] + run + [DOC_END]) if wrap_summary_runs else run
    ids += [DOC_START] + [50 + i for i in range(query_len)] + [DOC_END]
    ids += [EOS] + [PAD] * n_pad
    return torch.tensor([ids])


DOC_BODIES = [[20, 21, 22, 23], [30, 31, 32], [40, 41, 42, 43]]


def _roles_for(**kwargs):
    return build_summary_roles(_build_example(DOC_BODIES, **kwargs), **IDS_KW)


def _positions(roles, kind, doc=None):
    k = roles[0, ROLE_KIND]
    d = roles[0, ROLE_DOC_ID]
    return [
        i for i in range(k.shape[0]) if int(k[i]) == int(kind) and (doc is None or int(d[i]) == doc)
    ]


def _dense_mask_mod(roles, spec, causal_example=None):
    """Evaluate the ``mask_mod`` over the full ``(T, T)`` grid so it can be compared elementwise."""
    T = roles.shape[-1]
    mod = build_summary_mask_mod(roles, spec, causal_example=causal_example)
    b = torch.zeros(T, T, dtype=torch.long)
    q = torch.arange(T).view(-1, 1).expand(T, T)
    k = torch.arange(T).view(1, -1).expand(T, T)
    return mod(b, b, q, k)


# ---------------------------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------------------------


def test_summary_runs_are_found_by_id_not_by_span():
    """
    Summary runs are identified by token id, so the data layer may wrap each run in its own
    boundary markers or emit it inline at the end of a document, and the runs come out the same.

    The two layouts are not token-for-token identical -- the wrapped one has extra marker tokens,
    and the marker closing the *final* run falls into the trailing query span -- but those carry no
    information, and the reachability rules hold under both (see
    :func:`test_masked_example_reachability`, which is parametrized over the layout).
    """
    wrapped = _roles_for(wrap_summary_runs=True)
    inline = _roles_for(wrap_summary_runs=False)
    for roles in (wrapped, inline):
        assert len(_positions(roles, TokenKind.SUMMARY)) == N_SUMMARY * len(DOC_BODIES)
        for doc in range(len(DOC_BODIES)):
            run = _positions(roles, TokenKind.SUMMARY, doc=doc)
            assert len(run) == N_SUMMARY
            assert [int(roles[0, ROLE_SUMMARY_OFFSET][i]) for i in run] == list(range(N_SUMMARY))


@pytest.mark.parametrize("wrap_summary_runs", [True, False])
def test_doc_ids_and_offsets(wrap_summary_runs):
    roles = _roles_for(wrap_summary_runs=wrap_summary_runs)
    d, k, off = roles[0, ROLE_DOC_ID], roles[0, ROLE_KIND], roles[0, ROLE_SUMMARY_OFFSET]

    # A document's content and its own summary run share that document's index.
    for doc in range(len(DOC_BODIES)):
        assert _positions(roles, TokenKind.DOC_CONTENT, doc=doc)
        assert len(_positions(roles, TokenKind.SUMMARY, doc=doc)) == N_SUMMARY

    # The trailing query/answer lands one past the last document, which is what makes "an earlier
    # document" a plain integer comparison.
    query = _positions(roles, TokenKind.QUERY)
    assert query and all(int(d[i]) == len(DOC_BODIES) for i in query)

    # Instruction and padding belong to no document.
    for i in _positions(roles, TokenKind.INSTRUCTION) + _positions(roles, TokenKind.PAD):
        assert int(d[i]) == -1

    # Offsets run 0..N-1 inside each summary run.
    for doc in range(len(DOC_BODIES)):
        assert [int(off[i]) for i in _positions(roles, TokenKind.SUMMARY, doc=doc)] == list(
            range(N_SUMMARY)
        )

    # The example terminator belongs to the answer span, not to the globally-readable instruction.
    eos_pos = [
        i
        for i, t in enumerate(_build_example(DOC_BODIES, wrap_summary_runs=wrap_summary_runs)[0])
        if int(t) == EOS
    ]
    assert all(int(k[i]) == TokenKind.QUERY for i in eos_pos)


def test_example_with_no_summary_runs_has_no_query_span():
    """Degenerate input must not classify every document as the trailing query."""
    ids = torch.tensor([[10, 11, DOC_START, 20, 21, DOC_END, EOS, PAD, PAD]])
    roles = build_summary_roles(ids, **IDS_KW)
    kinds = {int(v) for v in roles[0, ROLE_KIND]}
    assert TokenKind.QUERY not in kinds
    assert TokenKind.DOC_CONTENT in kinds


# ---------------------------------------------------------------------------------------------
# The two reachability rules
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("wrap_summary_runs", [True, False])
def test_masked_example_reachability(wrap_summary_runs):
    roles = _roles_for(wrap_summary_runs=wrap_summary_runs)
    spec = SummaryMaskSpec(n_summary_tokens=N_SUMMARY)
    allowed = summary_mask_allowed(roles, spec)[0]

    query = _positions(roles, TokenKind.QUERY)
    summaries = _positions(roles, TokenKind.SUMMARY)
    content = _positions(roles, TokenKind.DOC_CONTENT)
    instruction = _positions(roles, TokenKind.INSTRUCTION)

    # The query reads the instruction, its own span and every summary run...
    assert all(allowed[q, s] for q in query for s in summaries)
    assert all(allowed[q, i] for q in query for i in instruction)
    assert all(allowed[q, p] for q in query for p in query if p <= q)
    # ...and NO raw document content. This is the property the experiment turns on.
    assert not any(allowed[q, c] for q in query for c in content)

    # A document reads itself plus strictly earlier summary runs, and no other document's content.
    own = _positions(roles, TokenKind.DOC_CONTENT, doc=2)
    assert not any(
        allowed[q, c] for q in own for c in _positions(roles, TokenKind.DOC_CONTENT, doc=0)
    )
    assert all(allowed[q, s] for q in own for s in _positions(roles, TokenKind.SUMMARY, doc=0))
    assert all(allowed[q, c] for q in own for c in own if c <= q)

    # A later summary run relays from earlier ones, and reads its own document.
    assert all(
        allowed[q, s]
        for q in _positions(roles, TokenKind.SUMMARY, doc=2)
        for s in _positions(roles, TokenKind.SUMMARY, doc=0)
    )
    assert all(
        allowed[q, c]
        for q in _positions(roles, TokenKind.SUMMARY, doc=0)
        for c in _positions(roles, TokenKind.DOC_CONTENT, doc=0)
    )


def test_causal_example_is_exactly_plain_causal():
    roles = _roles_for(wrap_summary_runs=False)
    spec = SummaryMaskSpec(n_summary_tokens=N_SUMMARY)
    allowed = summary_mask_allowed(roles, spec, causal_example=torch.tensor([True]))[0]

    T = roles.shape[-1]
    not_pad = roles[0, ROLE_KIND] != int(TokenKind.PAD)
    expected = torch.tril(torch.ones(T, T, dtype=torch.bool))
    expected = expected & not_pad.view(-1, 1) & not_pad.view(1, -1)
    expected = expected | torch.eye(T, dtype=torch.bool)
    assert torch.equal(allowed, expected)

    # ...and in particular the query now reaches raw document content.
    query = _positions(roles, TokenKind.QUERY)
    content = _positions(roles, TokenKind.DOC_CONTENT)
    assert all(allowed[q, c] for q in query for c in content)


def test_causal_flag_is_per_example():
    """A mixed batch must mask one example and not the other."""
    ids = _build_example(DOC_BODIES, wrap_summary_runs=False)
    roles = build_summary_roles(torch.cat([ids, ids], dim=0), **IDS_KW)
    spec = SummaryMaskSpec(n_summary_tokens=N_SUMMARY)
    allowed = summary_mask_allowed(roles, spec, causal_example=torch.tensor([False, True]))

    query = _positions(roles, TokenKind.QUERY)
    content = _positions(roles, TokenKind.DOC_CONTENT)
    assert not any(allowed[0, q, c] for q in query for c in content)
    assert all(allowed[1, q, c] for q in query for c in content)


def test_padding_is_never_attended():
    roles = _roles_for(wrap_summary_runs=False)
    spec = SummaryMaskSpec(n_summary_tokens=N_SUMMARY)
    allowed = summary_mask_allowed(roles, spec)[0]
    T = roles.shape[-1]
    for p in _positions(roles, TokenKind.PAD):
        # Only the self-diagonal NaN guard may be set on a pad column.
        assert not any(allowed[q, p] for q in range(T) if q != p)


# ---------------------------------------------------------------------------------------------
# Levers
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("visible", [0, 1, 2, N_SUMMARY, None])
def test_summary_visible_tokens_throttles_the_relay(visible):
    roles = _roles_for(wrap_summary_runs=False)
    spec = SummaryMaskSpec(n_summary_tokens=N_SUMMARY, summary_visible_tokens=visible)
    allowed = summary_mask_allowed(roles, spec)[0]

    query = _positions(roles, TokenKind.QUERY)
    earlier = _positions(roles, TokenKind.SUMMARY, doc=0)
    n_readable = sum(1 for s in earlier if allowed[query[0], s])
    assert n_readable == (N_SUMMARY if visible is None else min(visible, N_SUMMARY))


def test_placebo_summaries_carry_zero_document_content():
    """``summaries_read_own_document=False`` must sever every edge from a run to any content."""
    roles = _roles_for(wrap_summary_runs=False)
    spec = SummaryMaskSpec(
        n_summary_tokens=N_SUMMARY,
        summaries_read_own_document=False,
        summaries_read_earlier_summaries=False,
    )
    allowed = summary_mask_allowed(roles, spec)[0]
    content = _positions(roles, TokenKind.DOC_CONTENT)
    for s in _positions(roles, TokenKind.SUMMARY):
        assert not any(allowed[s, c] for c in content)
    # The runs keep their position and every edge INTO them, so this is a placebo, not a deletion.
    query = _positions(roles, TokenKind.QUERY)
    assert all(allowed[q, s] for q in query for s in _positions(roles, TokenKind.SUMMARY))


def test_relay_off_blocks_summary_to_summary_only():
    roles = _roles_for(wrap_summary_runs=False)
    spec = SummaryMaskSpec(n_summary_tokens=N_SUMMARY, summaries_read_earlier_summaries=False)
    allowed = summary_mask_allowed(roles, spec)[0]
    late = _positions(roles, TokenKind.SUMMARY, doc=2)
    early = _positions(roles, TokenKind.SUMMARY, doc=0)
    assert not any(allowed[q, s] for q in late for s in early)
    # Ordinary documents still relay.
    assert all(
        allowed[q, s] for q in _positions(roles, TokenKind.DOC_CONTENT, doc=2) for s in early
    )


def test_query_reads_documents_makes_the_query_global():
    roles = _roles_for(wrap_summary_runs=False)
    spec = SummaryMaskSpec(n_summary_tokens=N_SUMMARY, query_reads_documents=True)
    allowed = summary_mask_allowed(roles, spec)[0]
    query = _positions(roles, TokenKind.QUERY)
    content = _positions(roles, TokenKind.DOC_CONTENT)
    assert all(allowed[q, c] for q in query for c in content)
    # Documents stay restricted.
    assert not any(
        allowed[q, c]
        for q in _positions(roles, TokenKind.DOC_CONTENT, doc=2)
        for c in _positions(roles, TokenKind.DOC_CONTENT, doc=0)
    )


@pytest.mark.parametrize(
    "spec",
    [
        SummaryMaskSpec(n_summary_tokens=N_SUMMARY),
        SummaryMaskSpec(n_summary_tokens=N_SUMMARY, summary_visible_tokens=1),
        SummaryMaskSpec(n_summary_tokens=N_SUMMARY, summary_visible_tokens=0),
        SummaryMaskSpec(n_summary_tokens=N_SUMMARY, summaries_read_own_document=False),
        SummaryMaskSpec(n_summary_tokens=N_SUMMARY, summaries_read_earlier_summaries=False),
        SummaryMaskSpec(n_summary_tokens=N_SUMMARY, query_reads_documents=True),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_dense_matches_mask_mod(spec, causal):
    """The two renderings of the rule must never drift apart."""
    roles = _roles_for(wrap_summary_runs=False)
    ce = torch.tensor([causal])
    dense = summary_mask_allowed(roles, spec, causal_example=ce)[0]
    assert torch.equal(_dense_mask_mod(roles, spec, causal_example=ce), dense)


@pytest.mark.parametrize("kwargs", [{"n_summary_tokens": 0}, {"summary_visible_tokens": -1}])
def test_spec_validation(kwargs):
    with pytest.raises(ValueError):
        SummaryMaskSpec(**kwargs)


# ---------------------------------------------------------------------------------------------
# Migration: the legacy pattern this module replaces
# ---------------------------------------------------------------------------------------------


def test_matches_legacy_summary_attention():
    """
    Element-identical to ``"summary_attention"`` at the configuration it had to be coerced into.

    The legacy encoding packs the document index and the role into one ``int32``: document ``d``'s
    content is chunk ``2d``, its summary run is ``2d+1``, the query is ``2*n_docs``, the instruction
    is ``FREE`` and padding is ``PAD``. Building that here from the *same* token stream and checking
    the masks agree is what shows the rewrite preserved the semantics.

    Uses the inline layout deliberately: with summary runs wrapped in their own markers, those
    markers are document content under the new roles but part of the summary chunk under the legacy
    encoding, and the two would legitimately disagree about tokens that carry no information either
    way.
    """
    roles = _roles_for(wrap_summary_runs=False)
    kind, doc = roles[0, ROLE_KIND], roles[0, ROLE_DOC_ID]
    n_docs = len(DOC_BODIES)

    chunk_ids = torch.empty_like(kind)
    for i in range(kind.shape[0]):
        k, d = int(kind[i]), int(doc[i])
        if k == TokenKind.PAD:
            chunk_ids[i] = PAD_CHUNK_ID
        elif k == TokenKind.INSTRUCTION:
            chunk_ids[i] = FREE_CHUNK_ID
        elif k == TokenKind.DOC_CONTENT:
            chunk_ids[i] = 2 * d
        elif k == TokenKind.SUMMARY:
            chunk_ids[i] = 2 * d + 1
        else:  # QUERY
            chunk_ids[i] = 2 * n_docs

    legacy = build_chunked_allowed_mask(
        AttentionPattern(
            name="summary_attention",
            summary_every_k=1,
            summary_bandwidth=N_SUMMARY,
            summary_relay=True,
        ),
        chunk_ids.unsqueeze(0),
    )[0]
    new = summary_mask_allowed(roles, SummaryMaskSpec(n_summary_tokens=N_SUMMARY))[0]
    assert torch.equal(new, legacy)
