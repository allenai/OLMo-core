"""
The chunk layout an eval prefill uses must be the one its training shards were converted with.

This is a small surface with an expensive failure. The layout is not recoverable from the model or
from the data at grading time, so a mismatch is silent: the model is fed a token stream it never
trained on and produces a plausible, wrong number. One task in the suite -- oolong -- is not
document-chunked, which is exactly the case a hardcoded default gets wrong.
"""

from __future__ import annotations

import pytest

import ctc.tasks
from ctc.eval import prefill
from ctc.format import registry

ctc.tasks.load_all()


def test_oolong_declares_line_chunking():
    """An oolong example is one context block of labelled lines, not a list of documents. Document
    chunking would wrap the whole context in a single marker pair."""
    extra = registry.get("oolong").extra
    assert extra["chunk_by"] == "line"
    assert extra["item_regex"] == r"\|\|"


def test_every_other_task_leaves_the_layout_at_the_document_default():
    for name in registry.names():
        if name == "oolong":
            continue
        assert registry.get(name).extra.get("chunk_by", "document") == "document", name


def test_oolong_item_regex_does_not_match_the_empty_string():
    """The shipped bug: a bare ``'||'`` is an alternation of two empty strings, so it matches every
    line and wraps the preamble as chunks. Shards built before 2026-07-26 carry it."""
    import re

    assert re.compile(registry.get("oolong").extra["item_regex"]).match("") is None
    assert re.compile("||").match("") is not None  # the bug, for contrast


def test_structural_prefill_rejects_an_empty_matching_item_regex():
    with pytest.raises(ValueError, match="matches the empty string"):
        prefill.StructuralPrefill(tokenizer=None, task="oolong", chunk_by="line", item_regex="||")


def test_structural_prefill_rejects_an_unknown_chunk_by():
    with pytest.raises(ValueError, match="chunk_by"):
        prefill.StructuralPrefill(tokenizer=None, task="oolong", chunk_by="paragraph")
