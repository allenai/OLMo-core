import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from olmo_core.data import (
    LongDocStrategy,
    NumpyDatasetConfig,
    NumpyFSLDataset,
    NumpyPackedFSLDataset,
    NumpyPaddedFSLDataset,
    NumpyVSLDataset,
    TokenizerConfig,
)
from olmo_core.data.numpy_dataset import NumpyInterleavedFSLDataset
from olmo_core.data.source_mixture import (
    SourceMixtureConfig,
    SourceMixtureDatasetConfig,
)
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.data.utils import get_document_indices, write_document_indices

from .utils import mk_mmaps


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
        vocab_size=32_000,
    )
    assert ds[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert ds[1]["input_ids"].tolist() == [4, 5, 6, 7]
    assert ds[7]["input_ids"].tolist() == [28, 29, 30, 31]
    assert len(ds) == 8


def test_numpy_fsl_dataset_with_label_mask(tmp_path: Path):
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(16,))
    mmap1[:] = list(range(16))
    mmap1.flush()

    mmap1_mask = np.memmap(tmp_path / "mmap1_mask.npy", mode="w+", dtype=np.bool_, shape=(16,))
    mmap1_mask[:] = True
    mmap1_mask[7] = False
    mmap1_mask.flush()

    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(16,))
    mmap2[:] = list(range(16, 32))
    mmap2.flush()

    mmap2_mask = np.memmap(tmp_path / "mmap2_mask.npy", mode="w+", dtype=np.bool_, shape=(16,))
    mmap2_mask[:] = True
    mmap2_mask[0:3] = False
    mmap2_mask.flush()

    ds = NumpyFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=4,
        pad_token_id=-1,
        eos_token_id=-1,
        vocab_size=32_000,
        label_mask_paths=[tmp_path / "mmap1_mask.npy", tmp_path / "mmap2_mask.npy"],
    )
    assert len(ds) == 8

    assert ds[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert ds[0]["label_mask"].tolist() == [True, True, True, True]

    assert ds[1]["input_ids"].tolist() == [4, 5, 6, 7]
    assert ds[1]["label_mask"].tolist() == [True, True, True, False]

    assert ds[4]["input_ids"].tolist() == [16, 17, 18, 19]
    assert ds[4]["label_mask"].tolist() == [False, False, False, True]

    assert ds[7]["input_ids"].tolist() == [28, 29, 30, 31]
    assert ds[7]["label_mask"].tolist() == [True, True, True, True]


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
        vocab_size=32_000,
    )
    ds.prepare()
    assert ds[0]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 0]
    assert ds[0]["label_mask"].tolist() == [True] * 8
    assert ds[1]["input_ids"].tolist() == [8, 9, 10, 0, 0, 0, 0, 0]
    assert ds[1]["label_mask"].tolist() == [True] * 4 + [False] * 4
    assert ds[2]["input_ids"].tolist() == [11, 12, 13, 14, 15, 16, 17, 18]
    assert ds[3]["input_ids"].tolist() == [21, 22, 0, 0, 0, 0, 0, 0]
    assert len(ds) == 4


def test_numpy_padded_fsl_dataset_with_label_mask(tmp_path: Path):
    data1 = [1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 0]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data1_mask = [False, True, True, True, True, True, True, True] + [True, True, True, True]
    mmap1_mask = np.memmap(
        tmp_path / "mmap1_mask.npy", mode="w+", dtype=np.bool_, shape=(len(data1_mask),)
    )
    mmap1_mask[:] = data1_mask
    mmap1_mask.flush()

    data2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 21, 22, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    data2_mask = [True, True, True, True, True, True, True, True, True, True, True] + [
        True,
        True,
        True,
    ]
    mmap2_mask = np.memmap(
        tmp_path / "mmap2_mask.npy", mode="w+", dtype=np.bool_, shape=(len(data2_mask),)
    )
    mmap2_mask[:] = data2_mask
    mmap2_mask.flush()

    ds = NumpyPaddedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=8,
        pad_token_id=0,
        eos_token_id=0,
        vocab_size=32_000,
        label_mask_paths=[tmp_path / "mmap1_mask.npy", tmp_path / "mmap2_mask.npy"],
    )

    ds.prepare()
    assert len(ds) == 4

    assert ds[0]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 0]
    assert ds[0]["label_mask"].tolist() == [False] + [True] * 7

    assert ds[1]["input_ids"].tolist() == [8, 9, 10, 0, 0, 0, 0, 0]
    assert ds[1]["label_mask"].tolist() == [True] * 4 + [False] * 4

    assert ds[2]["input_ids"].tolist() == [11, 12, 13, 14, 15, 16, 17, 18]
    assert ds[3]["input_ids"].tolist() == [21, 22, 0, 0, 0, 0, 0, 0]


@pytest.mark.parametrize("long_doc_strategy", [LongDocStrategy.truncate, LongDocStrategy.fragment])
def test_numpy_packed_fsl_dataset(tmp_path: Path, long_doc_strategy):
    data1 = np.array(
        [1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 0]
    )
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    ds = NumpyPackedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=8,
        pad_token_id=-1,
        eos_token_id=0,
        vocab_size=32_000,
        long_doc_strategy=long_doc_strategy,
    )
    ds.prepare()
    assert len(ds) == 6

    assert ds[0]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 0]
    assert ds[0]["label_mask"].tolist() == [True] * 8

    assert ds[1]["input_ids"].tolist() == [1, 2, 3, 4, 5, 0, -1, -1]
    assert ds[1]["label_mask"].tolist() == [True] * 6 + [False] * 2

    assert ds[3]["input_ids"].tolist() == [1, 2, 3, 0, 1, 2, 0, -1]
    assert ds[3]["label_mask"].tolist() == [True] * 7 + [False]

    if long_doc_strategy == LongDocStrategy.truncate:
        assert ds[5]["input_ids"].tolist() == [1, 2, 0, -1, -1, -1, -1, -1]
        assert ds[5]["label_mask"].tolist() == [True] * 3 + [False] * 5
    elif long_doc_strategy == LongDocStrategy.fragment:
        assert ds[5]["input_ids"].tolist() == [9, 10, 0, 1, 2, 0, -1, -1]
        assert ds[5]["label_mask"].tolist() == [True] * 6 + [False] * 2
    else:
        raise ValueError(long_doc_strategy)


@pytest.mark.parametrize("long_doc_strategy", [LongDocStrategy.truncate, LongDocStrategy.fragment])
def test_numpy_packed_fsl_dataset_with_label_mask(tmp_path: Path, long_doc_strategy):
    data1 = [1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 0, 1, 2, 0]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data1_mask = [True] * len(data1)
    data1_mask[1] = False
    mmap1_mask = np.memmap(
        tmp_path / "mmap1_mask.npy", mode="w+", dtype=np.bool_, shape=(len(data1_mask),)
    )
    mmap1_mask[:] = data1_mask
    mmap1_mask.flush()

    data2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    data2_mask = [True] * len(data2)
    data2_mask[8] = False
    mmap2_mask = np.memmap(
        tmp_path / "mmap2_mask.npy", mode="w+", dtype=np.bool_, shape=(len(data2_mask),)
    )
    mmap2_mask[:] = data2_mask
    mmap2_mask.flush()

    ds = NumpyPackedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=8,
        pad_token_id=-1,
        eos_token_id=0,
        vocab_size=32_000,
        label_mask_paths=[tmp_path / "mmap1_mask.npy", tmp_path / "mmap2_mask.npy"],
        long_doc_strategy=long_doc_strategy,
    )
    ds.prepare()
    assert len(ds) == 6

    assert ds[0]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 0]
    assert ds[0]["label_mask"].tolist() == [True, False] + [True] * 6

    assert ds[1]["input_ids"].tolist() == [1, 2, 3, 4, 5, 0, -1, -1]
    assert ds[1]["label_mask"].tolist() == [True] * 6 + [False] * 2

    assert ds[3]["input_ids"].tolist() == [1, 2, 3, 0, 1, 2, 0, -1]
    assert ds[3]["label_mask"].tolist() == [True] * 7 + [False]

    if long_doc_strategy == LongDocStrategy.truncate:
        assert ds[5]["input_ids"].tolist() == [1, 2, 0, -1, -1, -1, -1, -1]
        assert ds[5]["label_mask"].tolist() == [True] * 3 + [False] * 5
    elif long_doc_strategy == LongDocStrategy.fragment:
        assert ds[5]["input_ids"].tolist() == [9, 10, 0, 1, 2, 0, -1, -1]
        assert ds[5]["label_mask"].tolist() == [False, True] + [True] * 4 + [False] * 2
    else:
        raise ValueError(long_doc_strategy)


def test_numpy_fsl_mixture_dataset(tmp_path: Path):
    # NOTE: At small token counts the take_ratio can be finicky so we test at small but real world-ish scale
    npdtype = np.uint16
    seed = 42
    mmap1 = mk_mmaps(tmp_path, "mmap1", 1, 20 * 1000, npdtype, eos=0, seed=seed)
    mmap2 = mk_mmaps(tmp_path, "mmap2", 1, 20 * 1000, npdtype, eos=0, seed=seed * 2)

    sequence_length = 4
    tokenizer = TokenizerConfig(
        vocab_size=32_000,
        eos_token_id=0,
        pad_token_id=-1,
    )

    bsz = 32
    max_tokens = 10_000

    mixture_config = SourceMixtureDatasetConfig(
        render_tables=False,
        max_tokens=max_tokens,
        sequence_length=sequence_length,
        source_configs=[
            SourceMixtureConfig(
                source_name="mmap1",
                paths=[str(i[0]) for i in mmap1],
                target_ratio=0.8,
            ),
            SourceMixtureConfig(
                source_name="mmap2",
                paths=[str(i[0]) for i in mmap2],
                target_ratio=0.2,
            ),
        ],
        dtype=NumpyDatasetDType.uint16,
        processes=1,
        seed=seed,
        global_batch_size=sequence_length * bsz,
    )

    ds = NumpyDatasetConfig(
        source_mixture_config=mixture_config,
        sequence_length=sequence_length,
        tokenizer=tokenizer,
        include_instance_metadata=False,
    ).build()
    ds.prepare()

    first_ds_item = ds[0]["input_ids"].tolist()

    # NOTE: This is commented out until we fix behavior of the source mixture dataset
    # first_src_sequence = mmap1[0][1][:sequence_length].tolist()
    # Note that changing the seed here could result in the inclusion of the first sequence from the mock data.
    # assert not np.array_equal(first_src_sequence, first_ds_item)
    expected = "2b5a2c"
    assert ds.fingerprint.endswith(
        expected
    ), f"Fingerprint mismatch, expected {expected}, got {ds.fingerprint[-6:]}...Do you need to update expected fingerprint?"
    assert first_ds_item == [56423, 24546, 15796, 52203]  # stable because we pass a seed
    assert ds.num_tokens == 10_112  # oversamples to handle rounding error
    assert len(ds) == 2528
    assert len(ds) / bsz >= math.ceil(max_tokens / (sequence_length * bsz))


def test_numpy_fsl_mixture_dataset_with_repetition(tmp_path: Path):
    # NOTE: At small token counts the take_ratio can be finicky so we test at small but real world-ish scale
    npdtype = np.uint16
    seed = 42
    # Only 10k tokens in mmap1 so we have to upsample to meet 0.8 target below
    mmap1 = mk_mmaps(
        tmp_path=tmp_path,
        prefix="mmap1",
        num_files=1,
        size=10 * 1000,
        dtype=npdtype,
        eos=0,
        seed=72,
    )
    mmap2 = mk_mmaps(
        tmp_path=tmp_path,
        prefix="mmap2",
        num_files=1,
        size=20 * 1000,
        dtype=npdtype,
        eos=0,
        seed=27,
    )

    sequence_length = 4
    tokenizer = TokenizerConfig(
        vocab_size=32_000,
        eos_token_id=0,
        pad_token_id=-1,
    )

    source1_paths = [str(i[0]) for i in mmap1] * 2  # duplicate the paths

    bsz = 32
    max_tokens = 40_000

    mixture_config = SourceMixtureDatasetConfig(
        render_tables=False,
        max_tokens=max_tokens,
        sequence_length=sequence_length,
        source_configs=[
            SourceMixtureConfig(
                source_name="mmap1", paths=source1_paths, target_ratio=0.8, max_repetition_ratio=2.0
            ),
            SourceMixtureConfig(
                source_name="mmap2",
                paths=[str(i[0]) for i in mmap2],
                target_ratio=0.2,
            ),
        ],
        dtype=NumpyDatasetDType.uint16,
        processes=1,
        seed=seed,
        global_batch_size=sequence_length * bsz,  # 10k sequences of length 4
    )

    ds = NumpyDatasetConfig(
        source_mixture_config=mixture_config,
        sequence_length=sequence_length,
        tokenizer=tokenizer,
        include_instance_metadata=False,
    ).build()
    ds.prepare()

    expected_fingerprint = "60cd21"
    first_ds_item = ds[0]["input_ids"].tolist()

    # NOTE: This is commented out until we fix behavior of the source mixture dataset
    # first_src_sequence = mmap1[0][1][:sequence_length].tolist()
    # Note that changing the seed here could result in the inclusion of the first sequence from the mock data.
    # assert not np.array_equal(first_src_sequence, first_ds_item)

    assert ds.fingerprint.endswith(
        expected_fingerprint
    ), f"Fingerprint mismatch, expected {expected_fingerprint}, got {ds.fingerprint[-6:]}...Do you need to update expected fingerprint?"
    assert first_ds_item == [
        12761,
        6996,
        63252,
        65373,
    ]  # stable because we pass a seed
    assert ds.num_tokens == 40_064  # oversamples to handle rounding error
    assert len(ds) / bsz == math.ceil(max_tokens / (sequence_length * bsz))


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
        vocab_size=32_000,
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


def test_numpy_interleaved_fsl_dataset(tmp_path: Path):
    data1 = [1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 0]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 21, 22, 23, 24, 25, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    ds = NumpyInterleavedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=16,
        pad_token_id=-1,
        eos_token_id=0,
        vocab_size=32_000,
        seed=2,
        docs_per_instance=2,
        chunks_per_doc=4,
    )
    ds.work_dir = tmp_path
    ds.prepare()

    assert ds[0]["input_ids"].tolist() == [
        21,
        22,
        11,
        12,
        23,
        13,
        14,
        24,
        15,
        16,
        25,
        17,
        18,
        0,
        -1,
        -1,
    ]
    assert ds[0]["label_mask"].tolist() == [True] * 14 + [False] * 2
    assert ds[1]["input_ids"].tolist() == [1, 2, 8, 3, 4, 9, 5, 6, 10, 7, 0, -1, -1, -1, -1, -1]
    assert ds[1]["label_mask"].tolist() == [True] * 11 + [False] * 5
    assert len(ds) == 2


def test_numpy_interleaved_fsl_dataset_with_label_mask(tmp_path: Path):
    data1 = [1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 0]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data1_mask = [False, True, True, True, True, True, True, True] + [True, True, True, True]
    mmap1_mask = np.memmap(
        tmp_path / "mmap1_mask.npy", mode="w+", dtype=np.bool_, shape=(len(data1_mask),)
    )
    mmap1_mask[:] = data1_mask
    mmap1_mask.flush()

    data2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 21, 22, 23, 24, 25, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    data2_mask = [True, True, True, True, True, True, True, True, True, True, True] + [
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    mmap2_mask = np.memmap(
        tmp_path / "mmap2_mask.npy", mode="w+", dtype=np.bool_, shape=(len(data2_mask),)
    )
    mmap2_mask[:] = data2_mask
    mmap2_mask.flush()

    ds = NumpyInterleavedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=16,
        pad_token_id=-1,
        eos_token_id=0,
        vocab_size=32_000,
        seed=2,
        docs_per_instance=2,
        chunks_per_doc=4,
        label_mask_paths=[tmp_path / "mmap1_mask.npy", tmp_path / "mmap2_mask.npy"],
    )

    ds.work_dir = tmp_path
    ds.prepare()

    assert ds[0]["input_ids"].tolist() == [
        21,
        22,
        11,
        12,
        23,
        13,
        14,
        24,
        15,
        16,
        25,
        17,
        18,
        0,
        -1,
        -1,
    ]
    assert ds[0]["label_mask"].tolist() == [True] * 14 + [False] * 2
    assert ds[1]["input_ids"].tolist() == [1, 2, 8, 3, 4, 9, 5, 6, 10, 7, 0, -1, -1, -1, -1, -1]
    assert ds[1]["label_mask"].tolist() == [False] + [True] * 10 + [False] * 5
    assert len(ds) == 2


def test_numpy_interleaved_fsl_dataset_with_bos_token(tmp_path: Path):
    data1 = [99, 1, 2, 3, 4, 5, 6, 7, 0, 99, 8, 9, 10, 0]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data2 = [99, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 99, 21, 22, 23, 24, 25, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    ds = NumpyInterleavedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=16,
        pad_token_id=-1,
        eos_token_id=0,
        vocab_size=32_000,
        seed=2,
        docs_per_instance=2,
        chunks_per_doc=4,
        bos_token_id=99,
    )
    ds.work_dir = tmp_path
    ds.prepare()

    assert ds[0]["input_ids"].tolist() == [
        99,
        21,
        22,
        11,
        12,
        23,
        13,
        14,
        24,
        15,
        16,
        25,
        17,
        0,
        -1,
        -1,
    ]
    assert ds[0]["label_mask"].tolist() == [True] * 14 + [False] * 2
    assert ds[1]["input_ids"].tolist() == [99, 1, 2, 8, 3, 4, 9, 5, 6, 10, 7, 0, -1, -1, -1, -1]
    assert ds[1]["label_mask"].tolist() == [True] * 12 + [False] * 4
    assert len(ds) == 2


def test_numpy_interleaved_fsl_dataset_with_bos_token_and_label_mask(tmp_path: Path):
    data1 = [99, 1, 2, 3, 4, 5, 6, 7, 0, 99, 8, 9, 10, 0]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data1_mask = [True, False, True, True, True, True, True, True, True] + [
        True,
        True,
        True,
        True,
        True,
    ]
    mmap1_mask = np.memmap(
        tmp_path / "mmap1_mask.npy", mode="w+", dtype=np.bool_, shape=(len(data1_mask),)
    )
    mmap1_mask[:] = data1_mask
    mmap1_mask.flush()

    data2 = [99, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 99, 21, 22, 23, 24, 25, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    data2_mask = [True, True, True, True, True, True, True, True, True, True, True, True] + [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
    ]
    mmap2_mask = np.memmap(
        tmp_path / "mmap2_mask.npy", mode="w+", dtype=np.bool_, shape=(len(data2_mask),)
    )
    mmap2_mask[:] = data2_mask
    mmap2_mask.flush()

    ds = NumpyInterleavedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=16,
        pad_token_id=-1,
        eos_token_id=0,
        vocab_size=32_000,
        seed=2,
        docs_per_instance=2,
        chunks_per_doc=4,
        label_mask_paths=[tmp_path / "mmap1_mask.npy", tmp_path / "mmap2_mask.npy"],
        bos_token_id=99,
    )

    ds.work_dir = tmp_path
    ds.prepare()

    assert ds[0]["input_ids"].tolist() == [
        99,
        21,
        22,
        11,
        12,
        23,
        13,
        14,
        24,
        15,
        16,
        25,
        17,
        0,
        -1,
        -1,
    ]
    assert ds[0]["label_mask"].tolist() == [True] * 14 + [False] * 2
    assert ds[1]["input_ids"].tolist() == [99, 1, 2, 8, 3, 4, 9, 5, 6, 10, 7, 0, -1, -1, -1, -1]
    assert ds[1]["label_mask"].tolist() == [True] + [False] + [True] * 10 + [False] * 4
    assert len(ds) == 2


def test_numpy_interleaved_fsl_dataset_with_interleaving_exempt_paths(tmp_path: Path):
    data1 = [1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 0]
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(len(data1),))
    mmap1[:] = data1
    mmap1.flush()

    data2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 21, 22, 23, 24, 25, 0]
    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(len(data2),))
    mmap2[:] = data2
    mmap2.flush()

    ds = NumpyInterleavedFSLDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=16,
        pad_token_id=-1,
        eos_token_id=0,
        vocab_size=32_000,
        seed=3,
        docs_per_instance=2,
        chunks_per_doc=4,
        interleaving_exempt_paths=[tmp_path / "mmap1.npy"],
    )
    ds.work_dir = tmp_path
    ds.prepare()

    assert ds[0]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 7, 0] + [-1] * 8
    assert ds[1]["input_ids"].tolist() == [8, 9, 10, 0] + [-1] * 12

    assert ds[2]["input_ids"].tolist() == [
        21,
        22,
        11,
        12,
        23,
        13,
        14,
        24,
        15,
        16,
        25,
        17,
        18,
        0,
        -1,
        -1,
    ]
    assert ds[2]["label_mask"].tolist() == [True] * 14 + [False] * 2
    assert len(ds) == 3


def test_guess_dtype():
    config = NumpyDatasetConfig(paths=[], sequence_length=1024, tokenizer=TokenizerConfig.gpt2())
    assert config.get_dtype() == np.uint16

    config = NumpyDatasetConfig(paths=[], sequence_length=1024, tokenizer=TokenizerConfig.dolma2())
    assert config.get_dtype() == np.uint32
