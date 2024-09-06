from typing import List

import numpy as np
import pytest

from olmo_core.data import IterableDataset, NumpyDataset


@pytest.mark.parametrize(
    "shuffle", [pytest.param(True, id="shuffle"), pytest.param(False, id="no-shuffle")]
)
def test_restart_with_seq_len_warmup(tmp_path, shuffle):
    max_target_sequence_length = 8

    # Write some data to disk.
    mmap1 = np.memmap(tmp_path / "tokens1.npy", dtype=np.uint16, mode="w+", shape=(18,))
    mmap1[:] = list(range(18))
    mmap1.flush()
    mmap2 = np.memmap(tmp_path / "tokens2.npy", dtype=np.uint16, mode="w+", shape=(8,))
    mmap2[:] = list(range(18, 26))
    mmap2.flush()
    del mmap1, mmap2

    def get_all_tokens(seq_len: int, start_index: int = 0) -> List[int]:
        dataset = NumpyDataset(
            tmp_path / "tokens1.npy",
            tmp_path / "tokens2.npy",
            sequence_length=seq_len,
            pad_token_id=-1,
            eos_token_id=-1,
            max_target_sequence_length=max_target_sequence_length,
        )
        iter_dataset = IterableDataset(
            dataset,
            shuffle=shuffle,
            chunk_size=max_target_sequence_length // seq_len,
            start_index=start_index,
            num_threads=0,
        )
        all_tokens = []
        for instance in iter_dataset:
            all_tokens.extend(instance["input_ids"].tolist())
        return all_tokens

    assert get_all_tokens(2) == get_all_tokens(4) == get_all_tokens(8)
    assert get_all_tokens(2, 4) == get_all_tokens(4, 2) == get_all_tokens(8, 1)
