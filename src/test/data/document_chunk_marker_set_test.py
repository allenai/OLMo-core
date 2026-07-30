"""Tests for the tokenizer-specific reserved-id ("marker set") parameterization of the
document-chunk converter path (blocker B0 in ``src/scripts/data/ctc_suite/BUILD_MATRIX.md``).

Three guarantees:

* The ``qwen3_5`` marker set really produces Qwen3.5 boundary ids (248049/248050) in the emitted
  dense layout, terminated by the Qwen3.5 EOS (248044).
* Runtime ``chunk_id`` reconstruction (:func:`build_chunk_ids_from_tokens`) with those ids agrees
  with the emitted layout -- the same guarantee the Qwen3 path has always had.
* The ``qwen3`` DEFAULT path is byte-identical to the pre-parameterization behavior (the converter's
  defaults must keep every existing shard reproducible).
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from olmo_core.data.document_chunk_landmark import (
    DOC_END_ID,
    DOC_START_ID,
    EOS_TOKEN_ID,
    LANDMARK_TOKEN_ID,
    PAD_TOKEN_ID,
    REAL_VOCAB_SIZE,
    RESERVED_IDS,
    emit_document_chunk_dense,
    find_chunk_spans,
    reserved_ids,
    segment_prompt_to_chunks,
)
from olmo_core.nn.attention.chunked_mask import (
    FREE_CHUNK_ID,
    build_chunk_ids_from_tokens,
)

_CONVERTER_PATH = (
    Path(__file__).parents[2] / "scripts" / "data" / "convert_unified_to_document_landmark.py"
)


def _converter():
    spec = importlib.util.spec_from_file_location("_docchunk_converter_under_test", _CONVERTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tokenizer(name: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name)
    except Exception as e:  # pragma: no cover - network/cache dependent
        pytest.skip(f"{name} tokenizer unavailable: {e}")


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


def _tokenize(conv, tok, n_docs: int, **kwargs):
    return conv.tokenize_example(
        tok,
        _example(n_docs),
        "contradiction",
        emit=kwargs.pop("emit", "dense"),
        query_position="both",
        cot_mode="none",
        mem_freq=kwargs.pop("mem_freq", 63),
        seq_len=8192,
        chunk_by="document",
        item_regex=r"\|\|",
        use_titles=False,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Registry sanity (no tokenizer needed)
# ---------------------------------------------------------------------------


def test_qwen3_registry_entry_matches_the_module_constants():
    ids = reserved_ids("qwen3")
    assert ids.doc_start == DOC_START_ID == 151648
    assert ids.doc_end == DOC_END_ID == 151649
    assert ids.eos == EOS_TOKEN_ID == 151643
    assert ids.landmark == LANDMARK_TOKEN_ID
    assert ids.pad == PAD_TOKEN_ID
    assert ids.real_vocab_size == REAL_VOCAB_SIZE


def test_qwen3_5_registry_entry_is_internally_consistent():
    ids = reserved_ids("qwen3_5")
    assert (ids.doc_start, ids.doc_end, ids.eos) == (248049, 248050, 248044)
    # Boundary markers and EOS are REAL registered tokens; landmark/pad live PAST the real vocab
    # in the embedding matrix's untrained padded region (Qwen3.5-0.8B-Base embedding = 248320 rows).
    assert max(ids.doc_start, ids.doc_end, ids.eos) < ids.real_vocab_size
    assert min(ids.landmark, ids.pad) >= ids.real_vocab_size
    assert max(ids.landmark, ids.pad) < 248320
    assert len({ids.doc_start, ids.doc_end, ids.eos, ids.landmark, ids.pad}) == 5


def test_unknown_marker_set_raises_loudly():
    with pytest.raises(KeyError, match="unknown model family"):
        reserved_ids("qwen4")


# ---------------------------------------------------------------------------
# qwen3_5 marker set through the converter's dense path
# ---------------------------------------------------------------------------


def test_qwen3_5_dense_layout_uses_qwen3_5_boundary_ids():
    conv = _converter()
    tok = _tokenizer("Qwen/Qwen3.5-0.8B-Base")
    q35 = RESERVED_IDS["qwen3_5"]
    n_docs = 5
    out_ids, out_mask = _tokenize(conv, tok, n_docs, ids_set=q35)
    ids = out_ids.tolist()

    # (a) the Qwen3.5 box ids appear as the chunk boundaries -- one pair per document.
    assert ids.count(q35.doc_start) == n_docs
    assert ids.count(q35.doc_end) == n_docs
    spans = find_chunk_spans(ids, q35.doc_start, q35.doc_end)
    assert len(spans) == n_docs
    # ... and NO Qwen3 ids leaked in (silent id-set mixing is the failure mode B0 guards against).
    assert DOC_START_ID not in ids and DOC_END_ID not in ids and EOS_TOKEN_ID not in ids
    # terminated by the Qwen3.5 EOS, excluded from the loss.
    assert ids[-1] == q35.eos and ids.count(q35.eos) == 1
    assert not out_mask[-1]
    # each wrapped document body round-trips through the Qwen3.5 tokenizer.
    for i, (s, e) in enumerate(spans, start=1):
        assert f"Claim body number {i}" in tok.decode(ids[s : e + 1])


def test_qwen3_5_chunk_ids_reconstruct_the_emitted_layout():
    conv = _converter()
    tok = _tokenizer("Qwen/Qwen3.5-0.8B-Base")
    q35 = RESERVED_IDS["qwen3_5"]
    n_docs = 5
    out_ids, _ = _tokenize(conv, tok, n_docs, ids_set=q35)
    ids = out_ids.tolist()

    # (b) runtime reconstruction with the Qwen3.5 ids matches the emitted spans exactly.
    chunk_ids = build_chunk_ids_from_tokens(
        torch.tensor([ids]), q35.doc_start, q35.doc_end, q35.eos
    )[0]
    ctx = sorted(int(c) for c in chunk_ids.unique() if c >= 0)
    assert ctx == list(range(n_docs)), "one contiguous chunk id per document"
    for want_chunk, (s, e) in enumerate(find_chunk_spans(ids, q35.doc_start, q35.doc_end)):
        span_roles = chunk_ids[s : e + 1]
        assert bool((span_roles == want_chunk).all()), "markers-included span == one chunk"
        # the token just before a span (separator/prefix) must never share the chunk id.
        assert int(chunk_ids[s - 1]) != want_chunk
    # everything outside the spans and before EOS is FREE (instruction / query / answer).
    assert int(chunk_ids[0]) == FREE_CHUNK_ID
    assert int(chunk_ids[-2]) == FREE_CHUNK_ID  # last answer token, just before EOS


def test_qwen3_5_landmark_layout_inserts_qwen3_5_landmark_and_pad_ids():
    conv = _converter()
    tok = _tokenizer("Qwen/Qwen3.5-0.8B-Base")
    q35 = RESERVED_IDS["qwen3_5"]
    mem_freq = 15
    out_ids, out_mask = _tokenize(conv, tok, 5, emit="landmark", mem_freq=mem_freq, ids_set=q35)
    ids = out_ids.tolist()
    # EOS is appended AFTER the packed windows, so the windowed part is ids[:-1].
    assert (len(ids) - 1) % (mem_freq + 1) == 0
    assert ids[-1] == q35.eos
    body = ids[:-1]
    assert set(body[mem_freq :: mem_freq + 1]) == {q35.landmark}, "landmark at every block-end"
    assert q35.pad in body, "short windows are filled with the qwen3_5 pad id"
    assert LANDMARK_TOKEN_ID not in ids and PAD_TOKEN_ID not in ids
    # landmark/pad positions are never supervised.
    mask = out_mask.tolist()
    assert not any(m for t, m in zip(ids, mask) if t in (q35.landmark, q35.pad))


# ---------------------------------------------------------------------------
# Regression guard: the qwen3 DEFAULT path is byte-identical to before
# ---------------------------------------------------------------------------


def test_qwen3_default_path_is_byte_identical_to_the_hardcoded_recipe():
    """The converter's default (``ids_set`` omitted) must reproduce EXACTLY what the pre-B0 hardcoded
    converter emitted: ``segment_prompt_to_chunks`` with the module-constant ids + dense emit + the
    module-constant EOS appended."""
    conv = _converter()
    tok = _tokenizer("Qwen/Qwen3-4B")
    out_ids, out_mask = _tokenize(conv, tok, 5)  # defaults: qwen3

    segments, _, _ = segment_prompt_to_chunks(
        tok,
        _example(5),
        "contradiction",
        query_position="both",
        cot_mode="none",
        chunk_by="document",
        include_answer=True,
        use_titles=False,
        doc_start_id=DOC_START_ID,
        doc_end_id=DOC_END_ID,
    )
    want_ids, want_mask = emit_document_chunk_dense(segments)
    want_ids.append(EOS_TOKEN_ID)
    want_mask.append(False)

    assert out_ids.tolist() == want_ids
    assert out_mask.tolist() == want_mask
    assert out_ids.tolist().count(DOC_START_ID) == 5
    # and passing the qwen3 set explicitly changes nothing.
    exp_ids, exp_mask = _tokenize(conv, tok, 5, ids_set=RESERVED_IDS["qwen3"])
    assert exp_ids.tolist() == out_ids.tolist() and exp_mask.tolist() == out_mask.tolist()
