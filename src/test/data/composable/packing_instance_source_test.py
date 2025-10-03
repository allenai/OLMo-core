from pathlib import Path

import numpy as np
import pytest

from olmo_core.data import TokenizerConfig
from olmo_core.data.composable import (
    LongDocStrategy,
    NumpyDocumentSource,
    PackingInstanceSource,
)


def _write_mmap(path, data, dtype):
    mmap = np.memmap(path, mode="w+", dtype=dtype, shape=(len(data),))
    mmap[:] = data
    mmap.flush()


@pytest.mark.parametrize("long_doc_strategy", [LongDocStrategy.truncate, LongDocStrategy.fragment])
def test_packing_intance_source(tmp_path: Path, long_doc_strategy: LongDocStrategy):
    dtype = np.uint16
    tokenizer = TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1)

    path1 = tmp_path / "mmap1.npy"
    data1 = [1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 0]
    _write_mmap(path1, data1, dtype)

    path2 = tmp_path / "mmap2.npy"
    data2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 0]
    _write_mmap(path2, data2, dtype)

    instances = PackingInstanceSource(
        sources=[
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
        ],
        sequence_length=8,
        work_dir=tmp_path,
        tokenizer=tokenizer,
        long_doc_strategy=long_doc_strategy,
    )

    assert len(instances) == 6

    assert instances[0]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 0]  # type: ignore
    assert instances[0]["label_mask"].tolist() == [True] * 8  # type: ignore

    assert instances[1]["input_ids"].tolist() == [1, 2, 3, 4, 5, 0, -1, -1]  # type: ignore
    assert instances[1]["label_mask"].tolist() == [True] * 6 + [False] * 2  # type: ignore

    assert instances[3]["input_ids"].tolist() == [1, 2, 3, 0, 1, 2, 0, -1]  # type: ignore
    assert instances[3]["label_mask"].tolist() == [True] * 7 + [False]  # type: ignore

    if long_doc_strategy == LongDocStrategy.truncate:
        assert instances[5]["input_ids"].tolist() == [1, 2, 0, -1, -1, -1, -1, -1]  # type: ignore
        assert instances[5]["label_mask"].tolist() == [True] * 3 + [False] * 5  # type: ignore
    elif long_doc_strategy == LongDocStrategy.fragment:
        assert instances[5]["input_ids"].tolist() == [9, 10, 0, 1, 2, 0, -1, -1]  # type: ignore
        assert instances[5]["label_mask"].tolist() == [True] * 6 + [False] * 2  # type: ignore
    else:
        raise ValueError(long_doc_strategy)


@pytest.mark.parametrize("long_doc_strategy", [LongDocStrategy.truncate, LongDocStrategy.fragment])
def test_packing_instance_source_with_label_mask(
    tmp_path: Path, long_doc_strategy: LongDocStrategy
):
    dtype = np.uint16
    tokenizer = TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1)

    path1 = tmp_path / "mmap1.npy"
    data1 = [1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 0]
    _write_mmap(path1, data1, dtype)
    mask_path1 = tmp_path / "mmap_mask1.npy"
    mask1 = [True] * len(data1)
    mask1[1] = False
    _write_mmap(mask_path1, mask1, np.bool_)

    path2 = tmp_path / "mmap2.npy"
    data2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 0]
    _write_mmap(path2, data2, dtype)
    mask_path2 = tmp_path / "mmap_mask2.npy"
    mask2 = [True] * len(data2)
    mask2[8] = False
    _write_mmap(mask_path2, mask2, np.bool_)

    instances = PackingInstanceSource(
        sources=[
            NumpyDocumentSource(
                source_paths=[path1],
                dtype=dtype,
                tokenizer=tokenizer,
                work_dir=tmp_path,
                label_mask_paths=[mask_path1],
            ),
            NumpyDocumentSource(
                source_paths=[path2],
                dtype=dtype,
                tokenizer=tokenizer,
                work_dir=tmp_path,
                label_mask_paths=[mask_path2],
            ),
        ],
        sequence_length=8,
        work_dir=tmp_path,
        tokenizer=tokenizer,
        long_doc_strategy=long_doc_strategy,
    )

    assert len(instances) == 6

    assert instances[0]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 0]  # type: ignore
    assert instances[0]["label_mask"].tolist() == [True, False] + [True] * 6  # type: ignore

    assert instances[1]["input_ids"].tolist() == [1, 2, 3, 4, 5, 0, -1, -1]  # type: ignore
    assert instances[1]["label_mask"].tolist() == [True] * 6 + [False] * 2  # type: ignore

    assert instances[3]["input_ids"].tolist() == [1, 2, 3, 0, 1, 2, 0, -1]  # type: ignore
    assert instances[3]["label_mask"].tolist() == [True] * 7 + [False]  # type: ignore

    if long_doc_strategy == LongDocStrategy.truncate:
        assert instances[5]["input_ids"].tolist() == [1, 2, 0, -1, -1, -1, -1, -1]  # type: ignore
        assert instances[5]["label_mask"].tolist() == [True] * 3 + [False] * 5  # type: ignore
    elif long_doc_strategy == LongDocStrategy.fragment:
        assert instances[5]["input_ids"].tolist() == [9, 10, 0, 1, 2, 0, -1, -1]  # type: ignore
        assert instances[5]["label_mask"].tolist() == [False, True] + [True] * 4 + [False] * 2  # type: ignore
    else:
        raise ValueError(long_doc_strategy)


def test_packing_intance_source_with_grouping(tmp_path: Path):
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

    instances = PackingInstanceSource(
        sources=[
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
        ],
        sequence_length=8,
        work_dir=tmp_path,
        tokenizer=tokenizer,
        source_group_size=2,
    )

    # NOTE: potentially brittle test here!
    # Hard-coding exactly what the instances should be to ensure it's deterministic.
    assert len(instances) == 7
    assert instances[0]["input_ids"].tolist() == [61, 62, 63, 64, 65, 66, 67, 0]  # type: ignore
    assert instances[1]["input_ids"].tolist() == [11, 12, 13, 14, 15, 0, -1, -1]  # type: ignore
    assert instances[2]["input_ids"].tolist() == [21, 22, 23, 24, 25, 0, -1, -1]  # type: ignore
    assert instances[3]["input_ids"].tolist() == [1, 2, 3, 0, 31, 32, 33, 0]  # type: ignore
    assert instances[4]["input_ids"].tolist() == [41, 42, 0, 51, 52, 0, -1, -1]  # type: ignore
    assert instances[5]["input_ids"].tolist() == [71, 72, 73, 74, 75, 76, 77, 78]  # type: ignore
    assert instances[6]["input_ids"].tolist() == [91, 92, 93, 94, 0, 81, 82, 0]  # type: ignore
