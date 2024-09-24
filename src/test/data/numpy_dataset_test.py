from pathlib import Path
from typing import List

import numpy as np

from olmo_core.data import (
    NumpyDatasetConfig,
    NumpyFSLDataset,
    NumpyPaddedFSLDataset,
    NumpyVSLDataset,
    TokenizerConfig,
)
from olmo_core.data.utils import get_document_indices, write_document_indices


def test_numpy_fsl_dataset(tmp_path: Path):
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(16,))
    mmap1[:] = list(range(16))
    mmap1.flush()

    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(16,))
    mmap2[:] = list(range(16, 32))
    mmap2.flush()

    ds = NumpyFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=4,
        pad_token_id=-1,
        eos_token_id=-1,
    )
    assert ds[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert ds[1]["input_ids"].tolist() == [4, 5, 6, 7]
    assert ds[7]["input_ids"].tolist() == [28, 29, 30, 31]
    assert len(ds) == 8


def test_numpy_padded_fsl_dataset(tmp_path: Path):
    data1 = [1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 0]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 21, 22, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    ds = NumpyPaddedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=8,
        pad_token_id=0,
        eos_token_id=0,
    )
    ds.prepare()
    assert ds[0]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 0]
    assert ds[0]["label_mask"].tolist() == [True] * 8
    assert ds[1]["input_ids"].tolist() == [8, 9, 10, 0, 0, 0, 0, 0]
    assert ds[1]["label_mask"].tolist() == [True] * 4 + [False] * 4
    assert ds[2]["input_ids"].tolist() == [11, 12, 13, 14, 15, 16, 17, 18]
    assert ds[3]["input_ids"].tolist() == [21, 22, 0, 0, 0, 0, 0, 0]
    assert len(ds) == 4


def write_data_file(data: List[int], path: Path, dtype, eos_token_id: int):
    path.parent.mkdir(exist_ok=True, parents=True)
    mmap = np.memmap(path, mode="w+", dtype=dtype, shape=(len(data),))
    mmap[:] = data
    mmap.flush()
    write_document_indices(path, dtype=dtype, eos_token_id=eos_token_id)


def test_numpy_vsl_dataset(tmp_path: Path):
    eos_token_id = 0
    pad_token_id = 0
    dtype = np.uint32

    # Write some fake data.
    data1 = [1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 0]
    data1_path = tmp_path / "data" / "part-1-00000.npy"
    write_data_file(data1, data1_path, dtype, eos_token_id)
    assert get_document_indices(data1_path) == [(0, 9), (9, len(data1))]

    data2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0]
    data2_path = tmp_path / "data" / "part-2-00000.npy"
    write_data_file(data2, data2_path, dtype, eos_token_id)
    assert get_document_indices(data2_path) == [(0, len(data2))]

    ds = NumpyVSLDataset(
        data1_path,
        data2_path,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        max_sequence_length=8,
        min_sequence_length=2,
        dtype=dtype,
    )
    ds.work_dir = tmp_path
    ds.prepare()

    assert len(ds) == 5
    assert ds[0]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert ds[1]["input_ids"].tolist() == [1, 2, 3, 4]
    assert ds[2]["input_ids"].tolist() == [5, 0]
    assert ds[3]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert ds[4]["input_ids"].tolist() == [9, 10]

    assert ds.get_instance_lengths().tolist() == [8, 4, 2, 8, 2]
    buckets = ds.get_instance_buckets()
    assert len(buckets) == 3  # for each power of 2 from 2**1 = 2 through 2**3 = 8
    assert buckets[0][1].tolist() == [2, 4]  # instances of length 2
    assert buckets[1][1].tolist() == [1]  # instances of length 4
    assert buckets[2][1].tolist() == [0, 3]  # instances of length 8

    assert ds.instances_per_bucket == ((2, 2), (4, 1), (8, 2))


def test_guess_dtype():
    config = NumpyDatasetConfig(paths=[], sequence_length=1024, tokenizer=TokenizerConfig.gpt2())
    assert config.get_dtype() == np.uint16

    config = NumpyDatasetConfig(paths=[], sequence_length=1024, tokenizer=TokenizerConfig.dolma2())
    assert config.get_dtype() == np.uint32
