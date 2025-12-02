from pathlib import Path

import numpy as np

from olmo_core.data.composable import NumpyDocumentSource
from olmo_core.data.composable.sampling_document_source import SamplingDocumentSource
from olmo_core.data.tokenizer import TokenizerConfig


def _write_mmap(path, data, dtype):
    mmap = np.memmap(path, mode="w+", dtype=dtype, shape=(len(data),))
    mmap[:] = data
    mmap.flush()


def test_sampling_document_source(tmp_path: Path):
    dtype = np.uint16

    path, data = tmp_path / "mmap1.npy", [1, 1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 0, 4, 0]
    _write_mmap(path, data, dtype)

    og_source = NumpyDocumentSource(
        source_paths=[path],
        dtype=dtype,
        work_dir=tmp_path / "work_dir",
        tokenizer=TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1),
    )
    assert og_source.num_tokens == len(data) == 14

    sampled_source = SamplingDocumentSource(og_source, max_tokens=13, work_dir=tmp_path, seed=None)
    # Last doc will be excluded from sample.
    assert sampled_source.num_docs == 3
    assert sampled_source.num_tokens == 12

    assert list(sampled_source.get_document_offsets()) == [(0, 5), (5, 9), (9, 12)]
    assert list(sampled_source.get_token_range(0, 5)["input_ids"]) == [1, 1, 1, 1, 0]
    assert list(sampled_source.get_token_range(5, 12)["input_ids"]) == [2, 2, 2, 0, 3, 3, 0]
    assert list(sampled_source.get_token_range(2, 7)["input_ids"]) == [1, 1, 0, 2, 2]
    assert list(sampled_source.get_token_range(0, 12)["input_ids"]) == [
        1,
        1,
        1,
        1,
        0,
        2,
        2,
        2,
        0,
        3,
        3,
        0,
    ]


def test_sampling_document_source_with_repetition(tmp_path: Path):
    dtype = np.uint16

    path, data = tmp_path / "mmap1.npy", [1, 1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 0, 4, 0]
    _write_mmap(path, data, dtype)

    og_source = NumpyDocumentSource(
        source_paths=[path],
        dtype=dtype,
        work_dir=tmp_path / "work_dir",
        tokenizer=TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1),
    )
    assert og_source.num_tokens == len(data) == 14

    sampled_source = SamplingDocumentSource(
        og_source,
        max_tokens=20,
        work_dir=tmp_path,
        seed=None,
    )
    assert sampled_source.num_docs == 5  # first doc will be repeated
    assert sampled_source.num_tokens == 19

    assert list(sampled_source.get_document_offsets()) == [
        (0, 5),
        (5, 9),
        (9, 12),
        (12, 14),
        (14, 19),
    ]
    assert list(sampled_source[:]["input_ids"]) == data + [1, 1, 1, 1, 0]


def test_sampling_document_source_random_sample(tmp_path: Path):
    dtype = np.uint16

    path, data = tmp_path / "mmap1.npy", [1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3, 0]
    _write_mmap(path, data, dtype)

    og_source = NumpyDocumentSource(
        source_paths=[path],
        dtype=dtype,
        work_dir=tmp_path / "work_dir",
        tokenizer=TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1),
    )
    assert og_source.num_tokens == len(data) == 12

    sampled_source = SamplingDocumentSource(og_source, max_tokens=9, work_dir=tmp_path, seed=0)
    # Only 2 docs will be included.
    assert sampled_source.num_docs == 2
    assert sampled_source.num_tokens == 8

    assert list(sampled_source.get_document_offsets()) == [(0, 4), (4, 8)]
    assert list(sampled_source.get_token_range(0, 4)["input_ids"]) in [
        [1, 1, 1, 0],
        [2, 2, 2, 0],
        [3, 3, 3, 0],
    ]
    assert list(sampled_source.get_token_range(4, 8)["input_ids"]) in [
        [1, 1, 1, 0],
        [2, 2, 2, 0],
        [3, 3, 3, 0],
    ]
