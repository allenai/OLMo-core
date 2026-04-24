"""Tests for :class:`PerFileChunkedInstanceSource`.

The primary regression covered here is **cross-file alignment preservation**:
when individual ``.npy`` files have token counts that are not multiples of
``sequence_length``, chunks returned by this source must never span file
boundaries. The comparable :class:`ConcatAndChunkInstanceSource` + flat-concat
reader does span boundaries, which is what motivated this source's existence
(see the ICL-overlap pipeline alignment investigation, 2026-04-23).
"""
from pathlib import Path

import numpy as np
import pytest

from olmo_core.data.composable.concat_and_chunk_instance_source import (
    ConcatAndChunkInstanceSource,
)
from olmo_core.data.composable.numpy_document_source import NumpyDocumentSource
from olmo_core.data.composable.per_file_chunked_instance_source import (
    PerFileChunkedInstanceSource,
    PerFileChunkedInstanceSourceConfig,
)
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.data.types import NumpyDatasetDType


SEQ_LEN = 8


def _write_shard(path: Path, start: int, num_tokens: int, dtype=np.uint16) -> None:
    """Write a tokens-only .npy file containing `num_tokens` sequential uint16s starting at `start`."""
    np.arange(start, start + num_tokens, dtype=dtype).tofile(path)


def test_single_aligned_file(tmp_path: Path):
    """Single file exactly N * seq_len tokens: yields exactly N chunks, all from that file."""
    shard = tmp_path / "shard-00.npy"
    _write_shard(shard, 0, 3 * SEQ_LEN)

    src = PerFileChunkedInstanceSource(
        source_paths=[str(shard)],
        token_dtype=np.uint16,
        sequence_length=SEQ_LEN,
        work_dir=tmp_path,
    )
    assert len(src) == 3
    assert list(src[0]["input_ids"]) == list(range(0, SEQ_LEN))
    assert list(src[1]["input_ids"]) == list(range(SEQ_LEN, 2 * SEQ_LEN))
    assert list(src[2]["input_ids"]) == list(range(2 * SEQ_LEN, 3 * SEQ_LEN))


def test_tails_dropped_per_file(tmp_path: Path):
    """Files with non-seq_len-multiple sizes have their tails silently dropped (per-file floor)."""
    shard0 = tmp_path / "shard-00.npy"
    shard1 = tmp_path / "shard-01.npy"
    # shard0: 2 complete chunks + 3-token tail
    _write_shard(shard0, 0, 2 * SEQ_LEN + 3)
    # shard1: 1 complete chunk + 5-token tail
    _write_shard(shard1, 1000, SEQ_LEN + 5)

    src = PerFileChunkedInstanceSource(
        source_paths=[str(shard0), str(shard1)],
        token_dtype=np.uint16,
        sequence_length=SEQ_LEN,
        work_dir=tmp_path,
    )
    assert len(src) == 3  # 2 from shard0, 1 from shard1 — tails dropped
    # The key assertion: shard1's first chunk starts at shard1 byte 0, NOT at
    # shard0's 3-token tail + first 5 tokens of shard1 (which is what the
    # ConcatAndChunk / NumpyDocumentSource cross-file reader would do).
    assert list(src[2]["input_ids"]) == list(range(1000, 1000 + SEQ_LEN))


def test_every_chunk_stays_within_one_file(tmp_path: Path):
    """Exhaustively verify no chunk spans a file boundary for a multi-shard layout."""
    paths = []
    # 3 shards with mixed alignment: 24 toks (aligned), 19 toks (tail 3), 13 toks (tail 5)
    for i, (start, n) in enumerate([(0, 3 * SEQ_LEN), (1000, 2 * SEQ_LEN + 3), (2000, SEQ_LEN + 5)]):
        p = tmp_path / f"shard-{i:02d}.npy"
        _write_shard(p, start, n)
        paths.append(str(p))

    src = PerFileChunkedInstanceSource(
        source_paths=paths,
        token_dtype=np.uint16,
        sequence_length=SEQ_LEN,
        work_dir=tmp_path,
    )
    # Expected: 3 + 2 + 1 = 6 instances; all chunks stay within their file's
    # contiguous range (start, start+n) and are exactly seq_len tokens.
    assert len(src) == 6
    for idx in range(len(src)):
        ids = list(src[idx]["input_ids"])
        assert len(ids) == SEQ_LEN
        # Identify which source file this chunk came from by start-value.
        origin_start = (ids[0] // 1000) * 1000
        # Every token in the chunk must be within the same file's value range.
        assert all(origin_start <= t < origin_start + 3 * SEQ_LEN for t in ids), (
            f"chunk {idx} spans files: {ids}"
        )


def test_regression_vs_concat_and_chunk(tmp_path: Path):
    """
    Document the per-file-vs-flat-concat distinction.

    Given the same non-seq_len-aligned shard layout:
      * PerFileChunkedInstanceSource drops tails, yields fewer instances, but
        every chunk aligns with a file's natural chunk boundaries.
      * ConcatAndChunkInstanceSource (over NumpyDocumentSource) treats all
        files as a flat concat stream — produces a chunk whose first bytes
        are from shard0's tail and remaining bytes are from shard1's head.
    This test locks in that divergent behaviour so future changes are visible.
    """
    # shard0: 1 complete chunk + 3-token tail
    shard0 = tmp_path / "shard-00.npy"
    _write_shard(shard0, 0, SEQ_LEN + 3)
    # shard1: 2 complete chunks aligned
    shard1 = tmp_path / "shard-01.npy"
    _write_shard(shard1, 1000, 2 * SEQ_LEN)

    per_file = PerFileChunkedInstanceSource(
        source_paths=[str(shard0), str(shard1)],
        token_dtype=np.uint16,
        sequence_length=SEQ_LEN,
        work_dir=tmp_path,
    )
    # PerFile: 1 from shard0 + 2 from shard1 = 3 instances.
    assert len(per_file) == 3
    # shard1's first chunk starts cleanly at token 1000 (no phase shift from
    # shard0's tail).
    assert list(per_file[1]["input_ids"])[0] == 1000

    # Now the ConcatAndChunk over a single NumpyDocumentSource wrapping both
    # shards: flat-concat reader will happily produce a frankenstein chunk.
    tokenizer = TokenizerConfig.dolma2()
    doc_source = NumpyDocumentSource(
        source_paths=[str(shard0), str(shard1)],
        dtype=np.uint16,
        tokenizer=tokenizer,
        work_dir=tmp_path,
    )
    cc = ConcatAndChunkInstanceSource(
        doc_source,
        sequence_length=SEQ_LEN,
        max_sequence_length=SEQ_LEN,
        work_dir=tmp_path,
    )
    # Flat concat: (SEQ_LEN + 3) + 2*SEQ_LEN = 3*SEQ_LEN + 3 total tokens
    # -> floor(28/SEQ_LEN=8) = 3 instances (3-token overall tail dropped).
    assert len(cc) == 3
    # Instance 1 starts at byte SEQ_LEN=8 in the flat concat space:
    #   bytes [8, 11) are shard0's tail (indices 8,9,10 -> values 8,9,10)
    #   bytes [11, 16) are shard1 first 5 bytes -> values 1000..1004
    # This is the frankenstein behaviour that motivates PerFileChunked.
    cc_chunk_1 = list(cc[1]["input_ids"])
    assert cc_chunk_1[:3] == [8, 9, 10]  # shard0 tail
    assert cc_chunk_1[3:] == [1000, 1001, 1002, 1003, 1004]  # shard1 head


def test_visualize_single_file(tmp_path: Path, capsys):
    """Smoke test: source participates in the visualize() tree without error."""
    shard = tmp_path / "shard-00.npy"
    _write_shard(shard, 0, 4 * SEQ_LEN)
    src = PerFileChunkedInstanceSource(
        source_paths=[str(shard)],
        token_dtype=np.uint16,
        sequence_length=SEQ_LEN,
        work_dir=tmp_path,
    )
    src.visualize(icons=False)
    captured = capsys.readouterr()
    assert "PerFileChunkedInstanceSource" in captured.out


def test_config_requires_dtype_or_tokenizer(tmp_path: Path):
    """
    Neither ``token_dtype`` nor ``tokenizer`` set -> the config must raise
    at build time rather than silently default to a wrong dtype.

    Regression: the initial version of this source silently defaulted to
    uint16. Dolma2-tokenized shards are uint32 (vocab 100278), so the wrong
    default caused the model to train on low/high-byte fragments of real
    tokens — observed in the 2026-04-24 align-fix reruns, where the
    treatment effect got 3-4x WORSE than broken-alignment until this was
    fixed.
    """
    shard = tmp_path / "shard-00.npy"
    _write_shard(shard, 0, 2 * SEQ_LEN)
    cfg = PerFileChunkedInstanceSourceConfig(
        source_paths=[str(shard)], sequence_length=SEQ_LEN,
    )
    with pytest.raises(ValueError, match="token_dtype.*tokenizer"):
        cfg.build(tmp_path)


def test_config_infers_dtype_from_tokenizer(tmp_path: Path):
    """Passing a tokenizer should auto-detect dtype from vocab_size."""
    shard = tmp_path / "shard-00.npy"
    # uint32 data (vocab > 2^16)
    np.arange(2 * SEQ_LEN, dtype=np.uint32).tofile(shard)

    # Dolma2 vocab (100278) requires uint32. Build from config with tokenizer
    # only (no explicit token_dtype).
    cfg = PerFileChunkedInstanceSourceConfig(
        source_paths=[str(shard)],
        sequence_length=SEQ_LEN,
        tokenizer=TokenizerConfig.dolma2(),
    )
    src = cfg.build(tmp_path)
    assert src._token_dtype == np.uint32
    # And reading recovers the original uint32 values.
    assert list(src[0]["input_ids"]) == list(range(SEQ_LEN))
    assert list(src[1]["input_ids"]) == list(range(SEQ_LEN, 2 * SEQ_LEN))


def test_config_explicit_dtype_wins_over_tokenizer(tmp_path: Path):
    """If both are given, explicit ``token_dtype`` takes priority."""
    shard = tmp_path / "shard-00.npy"
    np.arange(2 * SEQ_LEN, dtype=np.uint16).tofile(shard)

    cfg = PerFileChunkedInstanceSourceConfig(
        source_paths=[str(shard)],
        sequence_length=SEQ_LEN,
        token_dtype=NumpyDatasetDType.uint16,
        tokenizer=TokenizerConfig.dolma2(),  # would imply uint32; we override.
    )
    src = cfg.build(tmp_path)
    assert src._token_dtype == np.uint16
