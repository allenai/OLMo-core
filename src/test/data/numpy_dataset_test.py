from pathlib import Path

import numpy as np

from olmo_core.data import NumpyDataset, NumpyDatasetConfig, TokenizerConfig


def test_numpy_dataset(tmp_path: Path):
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(16,))
    mmap1[:] = np.array(list(range(16)), dtype=np.uint16)
    mmap1.flush()

    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(16,))
    mmap2[:] = np.array(list(range(16, 32)), dtype=np.uint16)
    mmap2.flush()

    ds = NumpyDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=4,
        pad_token_id=-1,
        eos_token_id=-1,
    )
    assert ds[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert ds[1]["input_ids"].tolist() == [4, 5, 6, 7]
    assert ds[7]["input_ids"].tolist() == [28, 29, 30, 31]


def test_numpy_dataset_with_label_mask(tmp_path: Path):
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(16,))
    mmap1[:] = np.array(list(range(16)), dtype=np.uint16)
    mmap1.flush()

    mask1 = [True] * 16
    mask1[1] = False
    mask_mmap1 = np.memmap(tmp_path / "mask_mmap1.npy", mode="w+", dtype=np.bool_, shape=(16,))
    mask_mmap1[:] = np.array(mask1, dtype=np.bool_)
    mask_mmap1.flush()

    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(16,))
    mmap2[:] = np.array(list(range(16, 32)), dtype=np.uint16)
    mmap2.flush()

    mask2 = [True] * 16
    mask2[-1] = False
    mask_mmap2 = np.memmap(tmp_path / "mask_mmap2.npy", mode="w+", dtype=np.bool_, shape=(16,))
    mask_mmap2[:] = np.array(mask2, dtype=np.bool_)
    mask_mmap2.flush()

    ds = NumpyDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        sequence_length=4,
        label_mask_paths=[tmp_path / "mask_mmap1.npy", tmp_path / "mask_mmap2.npy"],
        pad_token_id=-1,
        eos_token_id=-1,
    )
    assert ds[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert ds[0]["label_mask"].tolist() == [True, False, True, True]
    assert ds[1]["input_ids"].tolist() == [4, 5, 6, 7]
    assert ds[7]["input_ids"].tolist() == [28, 29, 30, 31]
    assert ds[7]["label_mask"].tolist() == [True, True, True, False]


def test_concat_numpy_datasets(tmp_path: Path):
    # Write some data to disk.
    mmap1 = np.memmap(tmp_path / "tokens1.npy", dtype=np.uint16, mode="w+", shape=(16,))
    mmap1[:] = list(range(16))
    mmap1.flush()
    mmap2 = np.memmap(tmp_path / "tokens2.npy", dtype=np.uint16, mode="w+", shape=(8,))
    mmap2[:] = list(range(8))
    mmap2.flush()
    del mmap1, mmap2

    # Initialize two datasets, one for each file.
    ds1 = NumpyDataset(
        tmp_path / "tokens1.npy",
        sequence_length=3,
        metadata={"label": "test1"},
        pad_token_id=-1,
        eos_token_id=-1,
    )
    assert len(ds1) == 5
    ds2 = NumpyDataset(
        tmp_path / "tokens2.npy",
        sequence_length=3,
        metadata={"label": "test2"},
        pad_token_id=-1,
        eos_token_id=-1,
    )
    assert len(ds2) == 2

    # Now concatenate them.
    ds = ds1 + ds2
    assert len(ds) == 7
    assert ds[0]["input_ids"].tolist() == [0, 1, 2]
    assert ds[0]["metadata"]["label"] == "test1"
    assert ds[6]["input_ids"].tolist() == [3, 4, 5]
    # Should get the same with negative index.
    assert ds[-1]["input_ids"].tolist() == [3, 4, 5]
    assert ds[-1]["metadata"]["label"] == "test2"


def test_guess_dtype():
    config = NumpyDatasetConfig(paths=[], sequence_length=1024, tokenizer=TokenizerConfig.gpt2())
    assert config.get_dtype() == np.uint16

    config = NumpyDatasetConfig(paths=[], sequence_length=1024, tokenizer=TokenizerConfig.dolma2())
    assert config.get_dtype() == np.uint32
