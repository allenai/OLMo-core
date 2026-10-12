"""Tests for the ``"summary_attention"`` DATA layout: one extra **summary span** chunk after every
``summary_every_k`` documents, so chunk indices run on a stride of ``k+1`` and a span is identified by
``(chunk_id % (k+1)) == k`` -- exactly what the ``"summary_attention"`` mask's arithmetic assumes.

The layout is emitted from ``segment_prompt_to_chunks``, the **single source of truth for both training
and eval prefill**. The train/eval parity test below is the guard that keeps them from drifting: a
layout built outside this shared path is what reduced the ``free_pad_repeat`` probe to train-CE-only
with no held-out f1 (``records/free-pad-probe-is-confounded.md``).
"""

import pytest
import torch

from olmo_core.data.document_chunk_landmark import (
    DOC_END_ID,
    DOC_START_ID,
    EOS_TOKEN_ID,
    segment_prompt_to_chunks,
    summary_span_text,
)
from olmo_core.nn.attention.chunked_mask import build_chunk_ids_from_tokens


def _tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    except Exception as e:  # pragma: no cover - network/cache dependent
        pytest.skip(f"Qwen3 tokenizer unavailable: {e}")


def _example(n_docs: int) -> dict:
    return {
        "documents": [
            {"text": f"Claim body number {i} about a distinct medical finding {i}."}
            for i in range(1, n_docs + 1)
        ],
        "queries": ["Which claims contradict?"],
        "answers": ["[[1, 2]]"],
        "gold_doc_indices": [[1, 2]],
    }


def _segment(tok, n_docs, k, *, include_answer=True):
    return segment_prompt_to_chunks(
        tok,
        _example(n_docs),
        "contradiction",
        query_position="both",
        cot_mode="none",
        chunk_by="document",
        include_answer=include_answer,
        summary_every_k=k,
    )


# ---------------------------------------------------------------------------
# summary_span_text
# ---------------------------------------------------------------------------


def test_summary_span_text_restates_its_own_cells_indices():
    assert summary_span_text(1, 10, 50) == (
        "\n\nSummary of claims 11 to 20: [11] [12] [13] [14] [15] [16] [17] [18] [19] [20]"
    )


def test_summary_span_text_clamps_a_short_final_cell():
    assert (
        summary_span_text(2, 10, 25) == "\n\nSummary of claims 21 to 25: [21] [22] [23] [24] [25]"
    )


def test_summary_spans_are_all_textually_distinct():
    """No verbatim repetition across cells -- exactly-repeated text collapses training even under plain
    causal attention (``records/free-pad-probe-is-confounded.md``)."""
    spans = [summary_span_text(c, 10, 50) for c in range(5)]
    assert len(set(spans)) == len(spans)


# ---------------------------------------------------------------------------
# Chunk-id stride: the mask's arithmetic must hold on the real token stream
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_docs,k", [(20, 10), (20, 5), (50, 10)])
def test_emitted_layout_has_one_span_per_cell_on_the_k_plus_1_stride(n_docs, k):
    tok = _tokenizer()
    _, ids, _ = _segment(tok, n_docs, k)
    chunk_ids = build_chunk_ids_from_tokens(
        torch.tensor([ids]), DOC_START_ID, DOC_END_ID, EOS_TOKEN_ID
    )
    ctx = [int(c) for c in chunk_ids[0].unique() if c >= 0]
    n_cells = n_docs // k
    assert len(ctx) == n_docs + n_cells, "expected one extra span chunk per cell"
    assert ctx == list(range(n_docs + n_cells)), "chunk ids must be contiguous"
    # the mask identifies a span purely as (id % (k+1)) == k -- that must match the emitted layout
    p = k + 1
    assert [i for i in ctx if i % p == k] == [c * p + k for c in range(n_cells)]


@pytest.mark.parametrize("n_docs,k", [(20, 10), (50, 10)])
def test_each_span_chunk_decodes_to_its_own_summary_text(n_docs, k):
    """The chunk the mask treats as cell ``c``'s relay must really be cell ``c``'s summary span."""
    tok = _tokenizer()
    _, ids, _ = _segment(tok, n_docs, k)
    chunk_ids = build_chunk_ids_from_tokens(
        torch.tensor([ids]), DOC_START_ID, DOC_END_ID, EOS_TOKEN_ID
    )[0]
    t = torch.tensor(ids)
    p = k + 1
    for cell in range(n_docs // k):
        decoded = tok.decode(t[chunk_ids == cell * p + k])
        assert summary_span_text(cell, k, n_docs) in decoded


@pytest.mark.parametrize("n_docs,k", [(20, 10), (50, 10)])
def test_every_span_token_follows_all_its_cells_document_tokens(n_docs, k):
    """The span sits at the cell END so it has causally read its whole cell. If this regresses the
    relay carries nothing and the design is silently vacuous."""
    tok = _tokenizer()
    _, ids, _ = _segment(tok, n_docs, k)
    chunk_ids = build_chunk_ids_from_tokens(
        torch.tensor([ids]), DOC_START_ID, DOC_END_ID, EOS_TOKEN_ID
    )[0]
    p = k + 1
    for cell in range(n_docs // k):
        span = torch.nonzero(chunk_ids == cell * p + k, as_tuple=True)[0]
        cell_docs = torch.nonzero(
            (chunk_ids >= 0) & (chunk_ids // p == cell) & (chunk_ids % p != k), as_tuple=True
        )[0]
        assert int(span.min()) > int(cell_docs.max())


def test_realized_span_length_supports_the_ladder_rungs():
    """⚠ The design doc originally ESTIMATED ~16 tokens per span; the realized length is larger and
    varies per cell (single- vs double-digit indices). Every ladder rung must be <= the SHORTEST span,
    otherwise a rung would expose fewer keys for some cells than others and the dose axis would not be
    uniform. Assert it rather than assuming it."""
    tok = _tokenizer()
    lens = [
        len(tok(summary_span_text(c, 10, 50), add_special_tokens=False).input_ids) for c in range(5)
    ]
    assert min(lens) >= 16, f"shortest span {min(lens)} cannot serve a bandwidth=16 rung"


def test_summary_every_k_requires_chunk_by_document():
    tok = _tokenizer()
    with pytest.raises(ValueError, match="chunk_by='document'"):
        segment_prompt_to_chunks(
            tok,
            _example(20),
            "contradiction",
            chunk_by="line",
            summary_every_k=10,
        )


def test_zero_summary_every_k_leaves_the_layout_untouched():
    """The knob must be strictly opt-in: every existing shard/pattern is bit-identical at k=0."""
    tok = _tokenizer()
    _, with_zero, _ = _segment(tok, 20, 0)
    _, baseline, _ = segment_prompt_to_chunks(
        tok,
        _example(20),
        "contradiction",
        query_position="both",
        cot_mode="none",
        chunk_by="document",
        include_answer=True,
    )
    assert with_zero == baseline


# ---------------------------------------------------------------------------
# Train / eval parity -- the guard that was missing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [5, 10])
def test_train_and_eval_prefill_layouts_are_token_identical(k):
    """The eval prefill (``include_answer=False``) must be an exact token PREFIX of the training
    sequence (``include_answer=True``) for the same example. If the two layouts drift, the model is
    evaluated on a prompt it was never trained on -- and the arm silently scores garbage."""
    tok = _tokenizer()
    _, train_ids, train_mask = _segment(tok, 50, k, include_answer=True)
    _, eval_ids, _ = _segment(tok, 50, k, include_answer=False)

    assert train_ids[: len(eval_ids)] == eval_ids
    # and the answer really is the only supervised part, i.e. the prefill covers the whole prompt
    assert not any(train_mask[: len(eval_ids)]), "prefill must contain no supervised tokens"
    assert any(train_mask[len(eval_ids) :]), "the answer must be supervised"


@pytest.mark.parametrize("k", [5, 10])
def test_train_and_eval_chunk_ids_agree_on_the_prefill(k):
    """Parity must hold at the ROLE level too: the same token must get the same chunk id at train and
    at eval, or the mask means something different in the two settings."""
    tok = _tokenizer()
    _, train_ids, _ = _segment(tok, 50, k, include_answer=True)
    _, eval_ids, _ = _segment(tok, 50, k, include_answer=False)
    args = (DOC_START_ID, DOC_END_ID, EOS_TOKEN_ID)
    train_c = build_chunk_ids_from_tokens(torch.tensor([train_ids]), *args)[0][: len(eval_ids)]
    eval_c = build_chunk_ids_from_tokens(torch.tensor([eval_ids]), *args)[0]
    torch.testing.assert_close(train_c, eval_c)
