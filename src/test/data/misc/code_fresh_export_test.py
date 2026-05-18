import gzip
import json
from pathlib import Path

import numpy as np

from olmo_core.data.misc.code_fresh_export import (
    ExportStats,
    build_documents_and_stats,
    flatten_documents,
    get_export_dir_name,
    process_code_fresh_file_contents,
    write_document_metadata,
    write_memmap,
    write_stats,
)


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return [ord(ch) for ch in text]


def test_process_code_fresh_file_contents():
    tokenizer = FakeTokenizer()
    doc = process_code_fresh_file_contents("  ab\n", tokenizer)
    assert doc is not None
    assert doc.tolist() == [97, 98, 99]


def test_process_code_fresh_file_contents_skips_empty():
    tokenizer = FakeTokenizer()
    assert process_code_fresh_file_contents(" \n\t ", tokenizer) is None


def test_process_code_fresh_file_contents_enforces_cap():
    tokenizer = FakeTokenizer()
    try:
        process_code_fresh_file_contents("abcd", tokenizer, max_doc_tokens_before_eos=3)
    except ValueError as exc:
        assert "exceeds the cap" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_documents_and_stats():
    rows = [{"file_contents": " a "}, {"file_contents": "   "}, {"file_contents": "bc"}]
    docs, stats = build_documents_and_stats(rows, FakeTokenizer(), language="python")
    assert [doc.tolist() for doc in docs] == [[97, 99], [98, 99, 99]]
    assert stats == ExportStats(
        language="python",
        num_docs=2,
        total_tokens=5,
        max_doc_tokens_before_eos=2,
        max_doc_tokens_after_eos=3,
        num_skipped_empty=1,
    )


def test_flatten_documents_and_metadata(tmp_path: Path):
    docs = [np.asarray([1, 2, 99], dtype=np.uint32), np.asarray([3, 99], dtype=np.uint32)]
    tokens, offsets = flatten_documents(docs)
    assert tokens.tolist() == [1, 2, 99, 3, 99]
    assert offsets == [(0, 3), (3, 5)]

    token_path = tmp_path / "part-0-00000.npy"
    metadata_path = tmp_path / "part-0-00000.csv.gz"
    stats_path = tmp_path / "stats.json"
    write_memmap(token_path, tokens)
    write_document_metadata(metadata_path, offsets)
    write_stats(
        stats_path,
        ExportStats(
            language="python",
            num_docs=2,
            total_tokens=5,
            max_doc_tokens_before_eos=2,
            max_doc_tokens_after_eos=3,
            num_skipped_empty=0,
        ),
    )

    loaded = np.memmap(token_path, mode="r", dtype=np.uint32)
    assert loaded.tolist() == [1, 2, 99, 3, 99]
    with gzip.open(metadata_path, "rt") as f:
        assert [line.strip() for line in f] == ["0,3", "3,5"]
    assert json.loads(stats_path.read_text())["language"] == "python"


def test_get_export_dir_name():
    assert (
        get_export_dir_name("allenai/dolma2-tokenizer") == "code_fresh_0825_1225_dolma2-tokenizer"
    )
