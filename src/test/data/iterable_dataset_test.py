from pathlib import Path
from typing import List

import numpy as np
import pytest
from torch.utils.data import DataLoader

from olmo_core.data import (
    DataCollator,
    IterableFSLDataset,
    IterableVSLDataset,
    NumpyFSLDataset,
    NumpyVSLDataset,
    VSLCurriculum,
    VSLGrowP2Curriculum,
    VSLNaturalCurriculum,
)


@pytest.mark.parametrize(
    "num_tokens, sequence_length, world_size, num_workers, batch_size",
    [
        (100, 4, 2, 2, 8),  # 2 instances per batch, 12 instances total
    ],
)
def test_iterable_fsl_dataset_with_dataloader(
    tmp_path: Path,
    num_tokens: int,
    sequence_length: int,
    world_size: int,
    num_workers: int,
    batch_size: int,  # in tokens
):
    assert batch_size % sequence_length == 0
    assert batch_size % world_size == 0
    rank_batch_size = batch_size // world_size
    assert rank_batch_size > 0
    num_batches = num_tokens // batch_size

    # Write some data to disk.
    mmap = np.memmap(tmp_path / "tokens.npy", dtype=np.uint16, mode="w+", shape=(num_tokens,))
    mmap[:] = list(range(num_tokens))
    mmap.flush()

    def get_all_batches() -> List[List[int]]:
        all_batches: List[List[int]] = [[] for _ in range(num_batches)]
        dataset = NumpyFSLDataset(
            tmp_path / "tokens.npy",
            sequence_length=sequence_length,
            pad_token_id=-1,
            eos_token_id=-1,
        )
        for rank in range(world_size):
            iter_dataset = IterableFSLDataset(
                dataset,
                rank_batch_size=rank_batch_size,
                collator=DataCollator(pad_token_id=-1),
                shuffle=False,
                num_threads=0,
                work_dir=tmp_path,
                dp_rank=rank,
                dp_world_size=world_size,
            )
            iter_dataset.build_and_save_global_indices()
            data_loader = DataLoader(
                iter_dataset,
                batch_size=None,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=False,
                timeout=0,
            )
            batches = list(data_loader)
            assert len(batches) == num_batches
            for i, batch in enumerate(batches):
                for instance in batch["input_ids"]:
                    all_batches[i].extend(instance.tolist())
        return all_batches

    all_batches = get_all_batches()
    all_tokens = []
    assert len(all_batches) == num_batches
    for batch in all_batches:
        assert len(batch) == batch_size
        all_tokens.extend(batch)

    assert len(all_tokens) == num_batches * batch_size
    assert set(all_tokens) == set(range(len(all_tokens)))


@pytest.mark.parametrize(
    "shuffle", [pytest.param(True, id="shuffle"), pytest.param(False, id="no-shuffle")]
)
def test_restart_with_seq_len_warmup(tmp_path: Path, shuffle: bool):
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
        dataset = NumpyFSLDataset(
            tmp_path / "tokens1.npy",
            tmp_path / "tokens2.npy",
            sequence_length=seq_len,
            pad_token_id=-1,
            eos_token_id=-1,
            max_target_sequence_length=max_target_sequence_length,
        )
        iter_dataset = IterableFSLDataset(
            dataset,
            rank_batch_size=seq_len,
            collator=DataCollator(pad_token_id=-1),
            shuffle=shuffle,
            chunk_size=max_target_sequence_length // seq_len,
            start_index=start_index,
            num_threads=0,
            work_dir=tmp_path,
        )
        iter_dataset.build_and_save_global_indices()
        all_tokens = []
        for instance in iter_dataset:
            all_tokens.extend(instance["input_ids"].flatten().tolist())
        return all_tokens

    assert get_all_tokens(2) == get_all_tokens(4) == get_all_tokens(8)
    assert get_all_tokens(2, 4) == get_all_tokens(4, 2) == get_all_tokens(8, 1)


@pytest.mark.parametrize(
    "shuffle, curriculum",
    [
        pytest.param(True, VSLNaturalCurriculum(), id="shuffle-natural"),
        pytest.param(True, VSLNaturalCurriculum(), id="no-shuffle-natural"),
        pytest.param(True, VSLGrowP2Curriculum(num_cycles=1, balanced=True), id="grow-p2-balanced"),
        pytest.param(
            True, VSLGrowP2Curriculum(num_cycles=1, balanced=False), id="grow-p2-unbalanced"
        ),
    ],
)
def test_variable_sequence_length_dataset(tmp_path: Path, shuffle: bool, curriculum: VSLCurriculum):
    data1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 13, 14, 15, 16, 17, 0, 18, 19, 0])
    mmap = np.memmap(tmp_path / "tokens1.npy", dtype=np.uint16, mode="w+", shape=data1.shape)
    mmap[:] = data1
    mmap.flush()

    data2 = np.array([20, 21, 22, 0, 23, 24, 0, 25, 26, 27, 28, 29, 30, 0])
    mmap2 = np.memmap(tmp_path / "tokens2.npy", dtype=np.uint16, mode="w+", shape=data2.shape)
    mmap2[:] = data2
    mmap2.flush()

    dataset = NumpyVSLDataset(
        tmp_path / "tokens1.npy",
        tmp_path / "tokens2.npy",
        pad_token_id=0,
        eos_token_id=0,
        min_sequence_length=2,
        max_sequence_length=4,
        dtype=np.uint16,
        curriculum=curriculum,
    )
    dataset.work_dir = tmp_path
    dataset.prepare()

    world_size = 2
    global_batch_size = 8

    iter_dataset = IterableVSLDataset(
        dataset,
        shuffle=shuffle,
        num_threads=0,
        work_dir=tmp_path,
        dp_world_size=world_size,
        collator=DataCollator(pad_token_id=0),
        rank_batch_size=global_batch_size // world_size,
    )
    iter_dataset.build_and_save_global_indices()

    all_tokens = []
    for rank in range(world_size):
        iter_dataset.dp_rank = rank
        n_batches = 0
        for batch in iter_dataset:
            n_batches += 1
            assert batch["input_ids"].numel() == global_batch_size // world_size
            all_tokens.extend(batch["input_ids"].flatten().tolist())

        assert n_batches == iter_dataset.total_batches

    assert len(all_tokens) == iter_dataset.total_batches * global_batch_size

    # Make sure batches were unique.
    all_tokens_less_eos = [t for t in all_tokens if t != 0]
    assert len(set(all_tokens_less_eos)) == len(all_tokens_less_eos)
