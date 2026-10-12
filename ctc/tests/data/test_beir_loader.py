"""
``load_beir``: the one loader whose raw corpus ships undrawable rows.

FiQA's BEIR release contains 38 corpus rows with empty text. The schema refuses an empty document
at build time, but the 2k-32k rungs draw ~80 of 57k documents per example and never sampled one --
the failure surfaced only at the 10M rung, where a single example draws half the corpus and the
build died mid-draw. The pool contract is that every document in it can be drawn.
"""

from __future__ import annotations

import sys
import types

import pytest


class _FakeSplit(list):
    pass


def _fake_datasets(corpus_rows, query_rows, qrel_rows):
    module = types.ModuleType("datasets")

    def load_dataset(path, config=None):
        if path.endswith("-qrels"):
            return {"test": _FakeSplit(qrel_rows)}
        if config == "corpus":
            return {"corpus": _FakeSplit(corpus_rows)}
        if config == "queries":
            return {"queries": _FakeSplit(query_rows)}
        raise AssertionError(f"unexpected load_dataset({path!r}, {config!r})")

    module.load_dataset = load_dataset
    return module


def test_empty_corpus_rows_are_dropped_at_load(monkeypatch):
    corpus = [
        {"_id": "d1", "title": "", "text": "a real passage"},
        {"_id": "d2", "title": "titled", "text": ""},
        {"_id": "d3", "title": "", "text": "   "},
        {"_id": "d4", "title": "", "text": "another real passage"},
    ]
    queries = [{"_id": "q1", "text": "a question"}]
    qrels = [{"query-id": "q1", "corpus-id": "d1", "score": 1}]
    monkeypatch.setitem(sys.modules, "datasets", _fake_datasets(corpus, queries, qrels))

    from ctc.data.sources.beir import load_beir

    loaded, loaded_queries, loaded_qrels = load_beir("fiqa", "test")
    assert [c.id for c in loaded] == ["d1", "d4"]
    assert all(c.text.strip() for c in loaded)
    assert [q["_id"] for q in loaded_queries] == ["q1"]
