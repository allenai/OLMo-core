"""
Tests for the summary-token DATA layout
(:func:`~olmo_core.data.document_chunk_landmark.emit_document_chunk_summary`).

The point of these is the **round trip**: what the converter emits must be exactly what
:func:`~olmo_core.nn.attention.summary_mask.build_summary_roles` reads back at training time. Those
two live in different files and are run by different processes months apart, so a drift between them
would not crash -- it would rebind every role and quietly change the experiment.
"""

import pytest

from olmo_core.data.document_chunk_landmark import (
    RESERVED_IDS,
    ChunkSegment,
    emit_document_chunk_dense,
    emit_document_chunk_summary,
)
from olmo_core.nn.attention.summary_mask import (
    ROLE_DOC_ID,
    ROLE_KIND,
    TokenKind,
    build_summary_roles,
)

IDS = RESERVED_IDS["qwen3_5"]
N_SUMMARY = 5


def _segments(n_docs: int = 3, doc_len: int = 4):
    """instruction (free), ``n_docs`` wrapped context documents, then query+answer (free)."""
    segs = [ChunkSegment([10, 11, 12], [False] * 3, False)]
    for d in range(n_docs):
        toks = [IDS.doc_start] + [20 + d] * doc_len + [IDS.doc_end]
        segs.append(ChunkSegment(toks, [False] * len(toks), True))
    segs.append(ChunkSegment([50, 51, 52], [False, True, True], False))
    return segs


def _emit(n_docs: int = 3, doc_len: int = 4, n_summary: int = N_SUMMARY):
    ids, mask = emit_document_chunk_summary(
        _segments(n_docs, doc_len), summary_token_id=IDS.summary, n_summary_tokens=n_summary
    )
    ids.append(IDS.eos)
    mask.append(False)
    return ids, mask


def test_summary_ids_have_headroom_and_are_distinct():
    """Every family's summary id must sit past the real vocab and clash with nothing else."""
    for family, r in RESERVED_IDS.items():
        assert r.summary > 0, f"{family} has no summary id"
        assert r.summary >= r.real_vocab_size, f"{family} summary id is inside the real vocab"
        assert len({r.doc_start, r.doc_end, r.eos, r.landmark, r.pad, r.summary}) == 6, family


@pytest.mark.parametrize("n_docs", [1, 3, 8])
@pytest.mark.parametrize("n_summary", [1, 5])
def test_one_summary_run_per_context_document(n_docs, n_summary):
    ids, mask = _emit(n_docs=n_docs, n_summary=n_summary)
    assert ids.count(IDS.summary) == n_docs * n_summary
    # Each run is contiguous and immediately follows a document's closing marker.
    runs, i = [], 0
    while i < len(ids):
        if ids[i] == IDS.summary:
            j = i
            while j < len(ids) and ids[j] == IDS.summary:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    assert len(runs) == n_docs
    assert all(j - i == n_summary for i, j in runs)
    assert all(ids[i - 1] == IDS.doc_end for i, _ in runs)
    # Summary positions never contribute to the loss.
    assert all(not mask[k] for i, j in runs for k in range(i, j))


def test_summary_layout_is_the_dense_layout_plus_the_runs():
    """Nothing else about the token stream changes, so a dense arm stays a clean comparison."""
    dense_ids, dense_mask = emit_document_chunk_dense(_segments())
    summ_ids, summ_mask = emit_document_chunk_summary(
        _segments(), summary_token_id=IDS.summary, n_summary_tokens=N_SUMMARY
    )
    assert [t for t in summ_ids if t != IDS.summary] == dense_ids
    assert len(summ_ids) == len(dense_ids) + 3 * N_SUMMARY
    assert [m for t, m in zip(summ_ids, summ_mask) if t != IDS.summary] == dense_mask


def test_invalid_summary_count_is_rejected():
    with pytest.raises(ValueError, match="n_summary_tokens"):
        emit_document_chunk_summary(_segments(), summary_token_id=IDS.summary, n_summary_tokens=0)


# ---------------------------------------------------------------------------------------------
# The round trip: emitted data -> roles the model reads back
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("n_docs", [1, 3, 8])
def test_roles_recovered_from_emitted_data(n_docs):
    import torch

    ids, _ = _emit(n_docs=n_docs)
    roles = build_summary_roles(
        torch.tensor([ids]),
        doc_start_id=IDS.doc_start,
        doc_end_id=IDS.doc_end,
        summary_token_id=IDS.summary,
        eos_id=IDS.eos,
        pad_id=IDS.pad,
    )
    kind, doc = roles[0, ROLE_KIND], roles[0, ROLE_DOC_ID]

    # Each document's content and its own summary run share that document's index.
    for d in range(n_docs):
        content = [i for i in range(len(ids)) if kind[i] == TokenKind.DOC_CONTENT and doc[i] == d]
        summary = [i for i in range(len(ids)) if kind[i] == TokenKind.SUMMARY and doc[i] == d]
        assert content, f"document {d} has no content"
        assert len(summary) == N_SUMMARY
        assert max(content) < min(summary), "the summary run must follow its document"

    # The instruction prefix stays free; the trailing query/answer is its own region. Note the
    # emitter never wraps the query -- roles are derived from completed summary RUNS, so everything
    # past the last one is the query region by construction.
    assert [int(k) for k in kind[:3]] == [TokenKind.INSTRUCTION] * 3
    assert int(kind[-1]) == TokenKind.QUERY  # the EOS terminator belongs to the answer span
    assert any(k == TokenKind.QUERY for k in kind)


def test_query_needs_no_wrapping_to_be_recognized():
    """
    The query/answer carries no boundary markers in the emitted stream, and must still come back as
    QUERY rather than as instruction text -- otherwise it would be globally readable and the masked
    arm would leak.
    """
    import torch

    ids, _ = _emit(n_docs=2)
    assert IDS.doc_start not in ids[-5:], "this test assumes an unwrapped trailing query"
    roles = build_summary_roles(
        torch.tensor([ids]),
        doc_start_id=IDS.doc_start,
        doc_end_id=IDS.doc_end,
        summary_token_id=IDS.summary,
        eos_id=IDS.eos,
        pad_id=IDS.pad,
    )
    kind = roles[0, ROLE_KIND]
    tail = [int(k) for k in kind[-4:]]
    assert tail == [TokenKind.QUERY] * 4, tail
