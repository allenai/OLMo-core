"""Regression test for the reorder document-wrapping bug (Stage-2 build log, 2026-07-19,
``src/scripts/data/ctc_suite/BUILD_MATRIX.md``): converted reorder shards wrapped only ~11% of
their documents (9.5 avg docs/ex vs 89.2 expected) because the reorder prompt renderer
(``_format_documents`` in ``corpus_reasoning/lib/data_format.py``) collapses each passage's
internal ``"\\n\\n"`` (Gutenberg's own paragraph breaks) to ``"\\n"`` before embedding it, but
``_wrap_documents`` (``olmo_core/data/document_chunk_landmark.py``) searched for the RAW
(un-collapsed) text -- a verbatim match that silently fails whenever a passage has an internal
blank line, leaving that document FREE (unisolated) instead of chunk-wrapped.

Does not require a tokenizer: exercises ``_wrap_documents`` directly against the rendered prompt
string and counts ``<|box_start|>`` occurrences.
"""

from corpus_reasoning.lib.data_format import build_prompt

from olmo_core.data.document_chunk_landmark import DOC_END_STR, DOC_START_STR, _wrap_documents


def _reorder_example(n_docs: int) -> dict:
    # Every passage has an internal blank line, like real Gutenberg 100-word chunks -- this is
    # exactly the shape that broke verbatim matching pre-fix.
    documents = [
        {
            "text": (
                f"First paragraph of passage {i}, with several words of body text.\n\n"
                f"Second paragraph of passage {i}, continuing the same passage."
            )
        }
        for i in range(1, n_docs + 1)
    ]
    return {
        "documents": documents,
        "queries": [],
        "gold_order": list(range(1, n_docs + 1)),
    }


def test_reorder_wraps_at_least_90pct_of_documents_with_internal_paragraph_breaks():
    n_docs = 12
    example = _reorder_example(n_docs)
    prompt, _answer = build_prompt(
        example,
        task="reorder",
        query_position="both",
        use_alpaca=False,
        cot_mode="none",
        use_titles=False,
    )
    wrapped = _wrap_documents(
        prompt, example["documents"], DOC_START_STR, DOC_END_STR, task="reorder"
    )
    n_wrapped = wrapped.count(DOC_START_STR)
    assert n_wrapped / n_docs >= 0.90, (
        f"only {n_wrapped}/{n_docs} reorder documents were chunk-wrapped; "
        "verbatim match likely broken again by a renderer-side text transform "
        "not mirrored in _RENDERER_TEXT_NORMALIZERS"
    )
    # With the fix, every document should match exactly (no internal-\n\n docs should fail).
    assert n_wrapped == n_docs


def test_reorder_wrapping_was_broken_without_the_normalizer():
    """Sanity-check the regression itself: WITHOUT mirroring the renderer's \\n\\n -> \\n
    collapse (task="" disables the normalizer lookup), the same documents fail to wrap."""
    n_docs = 12
    example = _reorder_example(n_docs)
    prompt, _answer = build_prompt(
        example,
        task="reorder",
        query_position="both",
        use_alpaca=False,
        cot_mode="none",
        use_titles=False,
    )
    wrapped = _wrap_documents(
        prompt, example["documents"], DOC_START_STR, DOC_END_STR, task=""
    )
    n_wrapped = wrapped.count(DOC_START_STR)
    assert n_wrapped == 0, "expected the pre-fix (no-normalizer) path to fail to wrap any document"
