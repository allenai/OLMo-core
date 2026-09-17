"""Converter and native-eval emitter parity without downloading a tokenizer."""
import pytest

from olmo_core.data import document_chunk_landmark as layouts
from scripts.data import convert_unified_to_document_landmark as converter
from corpus_reasoning.eval.eval_lc_native_docchunk import build_eval_prefill


def test_conversion_and_eval_prefill_match(monkeypatch):
    ids = layouts.reserved_ids("qwen3_5")
    prompt = [
        layouts.ChunkSegment([1], [False], False),
        layouts.ChunkSegment([ids.doc_start, 2, ids.doc_end], [False] * 3, True),
        layouts.ChunkSegment([ids.doc_start, 3, 4, ids.doc_end], [False] * 4, True),
        layouts.ChunkSegment([5], [False], False),
    ]

    def segment(*args, include_answer, **kwargs):
        segs = prompt + ([layouts.ChunkSegment([6], [True], False)] if include_answer else [])
        return segs, sum((s.tokens for s in segs), []), []

    monkeypatch.setattr(converter, "segment_prompt_to_chunks", segment)
    monkeypatch.setattr(layouts, "segment_prompt_to_chunks", segment)
    options = dict(
        emit="document_end_landmark",
        query_position="both",
        cot_mode="plan",
        mem_freq=63,
        seq_len=100,
        chunk_by="document",
        item_regex=r"\|\|",
        use_titles=False,
        ids_set=ids,
    )
    train, mask = converter.tokenize_example(None, {}, "contradiction", **options)
    prefill = build_eval_prefill(
        None,
        {},
        "contradiction",
        variant="document_end_landmark",
        doc_start_id=ids.doc_start,
        doc_end_id=ids.doc_end,
        landmark_token_id=ids.landmark,
    )
    assert train.tolist() == prefill + [6, ids.eos]
    assert prefill == [
        1,
        ids.doc_start,
        2,
        ids.doc_end,
        ids.landmark,
        ids.doc_start,
        3,
        4,
        ids.doc_end,
        ids.landmark,
        5,
    ]
    assert mask.tolist() == [False] * len(prefill) + [True, False]
    options["seq_len"] = len(train) - 1
    assert converter.tokenize_example(None, {}, "contradiction", **options) is None
    with pytest.raises(ValueError, match="requires document markers"):
        converter.tokenize_example(None, {}, "contradiction", doc_markers=False, **options)
