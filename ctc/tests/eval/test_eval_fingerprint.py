"""
The fingerprint eval actually presents to the guard.

Every other fingerprint test hands ``spec.fingerprint`` the fields it wants compared -- which is
precisely what the evaluator did not do. ``runner.py`` stated only ``query_position``, so
``chunk_layout`` fell to its ``"none"`` default and the guard refused **every** document-chunked
checkpoint against the exact data it was trained on, in the same words a real mismatch uses. The
tests passed throughout, because they were asserting the comparison rule rather than what eval feeds
it.

So these tests do not call ``spec.fingerprint``: they call :func:`ctc.eval.runner.eval_fingerprint`,
the thing the runner actually uses, and compare it against a fingerprint built the way the shard
converter builds one.
"""

from __future__ import annotations

import pytest

import ctc.tasks
from ctc.eval.runner import EvalConfig, eval_fingerprint
from ctc.format import registry
from ctc.format.fingerprint import FormatMismatchError

ctc.tasks.load_all()


def _examples(n_docs=5, n=3):
    return [
        {
            "documents": [{"text": f"Claim {i}."} for i in range(n_docs)],
            "queries": ["Find contradicting claims."],
            "answers": [""],
            "gold_doc_indices": [[1, 2]],
            "source": "pubmed",
        }
        for _ in range(n)
    ]


def _cfg(tmp_path, **kw):
    base = dict(
        ckpt=tmp_path / "ckpt",
        task=registry.get("contradiction"),
        rung="2k",
        data_path=tmp_path / "rung.jsonl",
        query_position="after",
    )
    base.update(kw)
    return EvalConfig(**base)


def _trained(spec, **overrides):
    """A training-time fingerprint, built the way ``convert_to_shards.py`` builds one."""
    base = dict(
        query_position="after",
        chunk_layout="wrap_documents",
        doc_id_range=(1, 5),
        marker_token_ids=(151648, 151649),
        tokenizer="/some/staged/path",
    )
    base.update(overrides)
    return spec.fingerprint(**base)


@pytest.mark.parametrize("attn", ["chunked", "full"])
def test_the_mask_arms_share_one_token_layout(tmp_path, attn):
    """``full`` is a mask, not a token layout: the markers are in both arms, so both must be
    accepted by a ``wrap_documents`` checkpoint. This is the case that was broken."""
    spec = registry.get("contradiction")
    fp = eval_fingerprint(_cfg(tmp_path, attn=attn), _examples())
    assert fp.chunk_layout == "wrap_documents"
    fp.require_compatible_with(_trained(spec))


def test_landmark_is_a_different_token_stream(tmp_path):
    """The one mode whose TOKENS differ, so it must be refused by a wrap_documents checkpoint."""
    spec = registry.get("contradiction")
    fp = eval_fingerprint(_cfg(tmp_path, attn="landmark"), _examples())
    assert fp.chunk_layout == "landmark_documents"
    with pytest.raises(FormatMismatchError, match="chunk_layout"):
        fp.require_compatible_with(_trained(spec))


def test_oolong_takes_its_chunk_unit_from_the_spec(tmp_path):
    """oolong's items are lines inside one context block; a document layout would wrap the whole
    context in a single marker pair. The unit comes from the spec, never from a flag."""
    spec = registry.get("oolong")
    example = {
        "documents": [{"text": "a || b"}],
        "queries": ["q"],
        "answers": [""],
        "gold_doc_indices": [[0]],
    }
    fp = eval_fingerprint(_cfg(tmp_path, task=spec, attn="chunked"), [example])
    assert fp.chunk_layout == "wrap_lines"


def test_the_doc_id_range_is_measured_from_the_rows_being_graded(tmp_path):
    """Unmeasured, it is ``None`` and the containment rule never fires -- and that rule is the
    digit-range failure (training capped at 697, eval reaching 1423) the module was built for."""
    spec = registry.get("contradiction")
    narrow = eval_fingerprint(_cfg(tmp_path, attn="chunked"), _examples(n_docs=5))
    wide = eval_fingerprint(_cfg(tmp_path, attn="chunked"), _examples(n_docs=50))
    assert narrow.doc_id_range == (1, 5)
    assert wide.doc_id_range == (1, 50)

    narrow.require_compatible_with(_trained(spec))
    with pytest.raises(FormatMismatchError, match="doc_id_range"):
        wide.require_compatible_with(_trained(spec))


def test_query_position_still_travels(tmp_path):
    """The one field eval always stated; it must not be lost while adding the others."""
    spec = registry.get("contradiction")
    fp = eval_fingerprint(_cfg(tmp_path, attn="chunked", query_position="both"), _examples())
    assert fp.query_position == "both"
    with pytest.raises(FormatMismatchError, match="query_position"):
        fp.require_compatible_with(_trained(spec))
