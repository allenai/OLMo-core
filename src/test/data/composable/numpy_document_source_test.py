from pathlib import Path

import numpy as np

from olmo_core.data.composable import ConcatenatedDocumentSource
from olmo_core.data.composable.numpy_document_source import NumpyDocumentSource
from olmo_core.data.tokenizer import TokenizerConfig


def _write_mmap(path, data, dtype):
    mmap = np.memmap(path, mode="w+", dtype=dtype, shape=(len(data),))
    mmap[:] = data
    mmap.flush()


def test_numpy_document_source(tmp_path: Path):
    dtype = np.uint16

    path1, data1 = tmp_path / "mmap1.npy", [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
    _write_mmap(path1, data1, dtype)
    mask_path1, mask1 = tmp_path / "mmap_mask1.npy", [True] * len(data1)
    mask1[1] = False
    mask1[-1] = False
    _write_mmap(mask_path1, mask1, np.bool_)

    path2, data2 = tmp_path / "mmap2.npy", [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 21, 22, 0]
    _write_mmap(path2, data2, dtype)
    mask_path2, mask2 = tmp_path / "mmap_mask2.npy", [True] * len(data2)
    mask2[-1] = False
    mask2[-4] = False
    _write_mmap(mask_path2, mask2, np.bool_)

    source = NumpyDocumentSource(
        source_paths=[path1, path2],
        label_mask_paths=[mask_path1, mask_path2],
        dtype=dtype,
        work_dir=tmp_path / "work_dir",
        tokenizer=TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1),
    )
    assert isinstance(source.fingerprint, str)
    assert source.num_tokens == len(data1) + len(data2)

    assert list(source.get_token_range(0, 4)["input_ids"]) == [1, 0, 2, 3]
    assert list(source.get_token_range(0, 4).get("label_mask", [])) == [True, False, True, True]
    # Same thing but with slicing.
    assert list(source[0:4]["input_ids"]) == [1, 0, 2, 3]
    assert list(source[0:4].get("label_mask", [])) == [True, False, True, True]
    assert list(source[0:4:2]["input_ids"]) == [1, 2]
    assert list(source[0]["input_ids"]) == [1]

    assert list(source.get_token_range(10, 14)["input_ids"]) == [10, 0, 11, 12]
    assert list(source.get_token_range(10, 14).get("label_mask", [])) == [True, False, True, True]

    assert list(source.get_token_range(12, 16)["input_ids"]) == [11, 12, 13, 14]
    assert list(source.get_token_range(12, 16).get("label_mask", [])) == [True, True, True, True]

    assert list(source[-3:]["input_ids"]) == [21, 22, 0]

    assert list(source.get_document_offsets()) == [(0, 2), (2, 12), (12, 23), (23, 26)]


def test_numpy_document_source_concatenated(tmp_path: Path):
    dtype = np.uint16
    tokenizer = TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1)

    path1 = tmp_path / "mmap1.npy"
    data1 = [1, 2, 3, 0, 11, 12, 13, 14, 15, 0, 21, 22, 23, 24, 25, 0, 31, 32, 33, 0, 41, 42, 0]
    _write_mmap(path1, data1, dtype)

    path2 = tmp_path / "mmap2.npy"
    data2 = [51, 52, 0, 61, 62, 63, 64, 65, 66, 67, 0]
    _write_mmap(path2, data2, dtype)

    path3 = tmp_path / "mmap3.npy"
    data3 = [71, 72, 73, 74, 75, 76, 77, 78, 79, 0, 81, 82, 0]
    _write_mmap(path3, data3, dtype)

    path4 = tmp_path / "mmap4.npy"
    data4 = [91, 92, 93, 94, 0]
    _write_mmap(path4, data4, dtype)

    source = ConcatenatedDocumentSource(
        NumpyDocumentSource(
            source_paths=[path1],
            dtype=dtype,
            tokenizer=tokenizer,
            work_dir=tmp_path,
        ),
        NumpyDocumentSource(
            source_paths=[path2],
            dtype=dtype,
            tokenizer=tokenizer,
            work_dir=tmp_path,
        ),
        NumpyDocumentSource(
            source_paths=[path3],
            dtype=dtype,
            tokenizer=tokenizer,
            work_dir=tmp_path,
        ),
        NumpyDocumentSource(
            source_paths=[path4],
            dtype=dtype,
            tokenizer=tokenizer,
            work_dir=tmp_path,
        ),
        work_dir=tmp_path,
    )

    assert source.num_tokens == len(data1) + len(data2) + len(data3) + len(data4)
    assert list(source.get_document_offsets()) == (
        list(source.sources[0].get_document_offsets())
        + [
            (start_idx + len(data1), end_idx + len(data1))
            for (start_idx, end_idx) in source.sources[1].get_document_offsets()
        ]
        + [
            (start_idx + len(data1) + len(data2), end_idx + len(data1) + len(data2))
            for (start_idx, end_idx) in source.sources[2].get_document_offsets()
        ]
        + [
            (
                start_idx + len(data1) + len(data2) + len(data3),
                end_idx + len(data1) + len(data2) + len(data3),
            )
            for (start_idx, end_idx) in source.sources[3].get_document_offsets()
        ]
    )
