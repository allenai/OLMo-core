from pathlib import Path

import numpy as np

from olmo_core.data import (
    NumpyFSLDataset,
    NumpyFSLDatasetConfig,
    NumpyVSLDataset,
    TokenizerConfig,
)


def test_numpy_fsl_dataset(tmp_path: Path):
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(16,))
    mmap1[:] = np.array(list(range(16)), dtype=np.uint16)
    mmap1.flush()

    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(16,))
    mmap2[:] = np.array(list(range(16, 32)), dtype=np.uint16)
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


def test_numpy_vsl_dataset(tmp_path: Path):
    data_path = "s3://ai2-llm/preprocessed/proof-pile-2/v0_decontaminated/algebraic-stack/train/allenai/dolma2-tokenizer/part-15-00000.npy"
    tokenizer_config = TokenizerConfig.dolma2()
    dtype = np.uint32
    ds = NumpyVSLDataset(
        data_path,
        pad_token_id=tokenizer_config.pad_token_id,
        eos_token_id=tokenizer_config.eos_token_id,
        max_sequence_length=1024,
        dtype=dtype,
    )
    ds.work_dir = tmp_path
    ds.prepare()
    assert len(ds) > 71788
    assert ds[0]["input_ids"].shape[0] <= 1024
    assert ds[-1]["input_ids"].shape[0] <= 1024


def test_guess_dtype():
    config = NumpyFSLDatasetConfig(paths=[], sequence_length=1024, tokenizer=TokenizerConfig.gpt2())
    assert config.get_dtype() == np.uint16

    config = NumpyFSLDatasetConfig(
        paths=[], sequence_length=1024, tokenizer=TokenizerConfig.dolma2()
    )
    assert config.get_dtype() == np.uint32
