from pathlib import Path
from typing import List

import numpy as np
import pytest

from olmo_core.data import (
    DataCollator,
    NumpyDataLoaderBase,
    NumpyFSLDataLoader,
    NumpyFSLDataset,
    NumpyVSLDataLoader,
    NumpyVSLDataset,
    VSLCurriculum,
    VSLGrowP2Curriculum,
    VSLNaturalCurriculum,
)


@pytest.mark.parametrize(
    "num_tokens, sequence_length, world_size, num_workers, num_threads, batch_size",
    [
        (100, 4, 2, 2, 2, 8),  # 2 instances per batch, 12 instances total
    ],
)
def test_fsl_data_loader(
    tmp_path: Path,
    num_tokens: int,
    sequence_length: int,
    world_size: int,
    num_workers: int,
    num_threads: int,
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
            vocab_size=32_000,
        )
        for rank in range(world_size):
            data_loader = NumpyFSLDataLoader(
                dataset,
                global_batch_size=batch_size,
                collator=DataCollator(pad_token_id=-1),
                shuffle=False,
                num_threads=num_threads,
                work_dir=tmp_path,
                dp_rank=rank,
                dp_world_size=world_size,
                num_workers=num_workers,
            )
            data_loader.reshuffle(epoch=1)
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
    "num_tokens, sequence_length, num_workers, batch_size",
    [
        (100, 4, 1, 8),  # 2 instances per batch, 12 instances total
    ],
)
def test_fsl_data_loader_multiple_epochs(
    tmp_path: Path,
    num_tokens: int,
    sequence_length: int,
    num_workers: int,
    batch_size: int,  # in tokens
):
    assert batch_size % sequence_length == 0
    num_batches = num_tokens // batch_size

    # Write some data to disk.
    mmap = np.memmap(tmp_path / "tokens.npy", dtype=np.uint16, mode="w+", shape=(num_tokens,))
    mmap[:] = list(range(num_tokens))
    mmap.flush()

    dataset = NumpyFSLDataset(
        tmp_path / "tokens.npy",
        sequence_length=sequence_length,
        pad_token_id=-1,
        eos_token_id=-1,
        vocab_size=32_000,
    )
    data_loader = NumpyFSLDataLoader(
        dataset,
        global_batch_size=batch_size,
        collator=DataCollator(pad_token_id=-1),
        shuffle=False,
        num_threads=0,
        work_dir=tmp_path,
        num_workers=num_workers,
    )

    ## First epoch ##

    data_loader.reshuffle(1)

    num_batches = 0
    num_tokens = 0
    for batch in data_loader:
        num_batches += 1
        num_tokens += batch["input_ids"].numel()
        assert data_loader.batches_processed == num_batches
        assert data_loader.tokens_processed == num_tokens
    assert num_batches == data_loader.total_batches

    data_loader.reset()

    assert data_loader.batches_processed == 0
    assert data_loader.tokens_processed == 0

    ## Second epoch ##

    data_loader.reshuffle(2)

    num_batches = 0
    num_tokens = 0
    for batch in data_loader:
        num_batches += 1
        num_tokens += batch["input_ids"].numel()
        assert data_loader.batches_processed == num_batches
        assert data_loader.tokens_processed == num_tokens
        # Stop after first batch to checkpoint.
        break

    # Create a new data loader and restart from the same spot.
    state_dict = data_loader.state_dict()
    data_loader = NumpyFSLDataLoader(
        dataset,
        global_batch_size=batch_size,
        collator=DataCollator(pad_token_id=-1),
        shuffle=False,
        num_threads=0,
        work_dir=tmp_path,
        num_workers=num_workers,
    )
    data_loader.load_state_dict(state_dict)
    assert data_loader.epoch == 2
    assert data_loader.batches_processed == 1
    assert data_loader.tokens_processed == batch_size

    # Continue the epoch.
    data_loader.reshuffle(2)
    for batch in data_loader:
        num_batches += 1
        num_tokens += batch["input_ids"].numel()
        assert data_loader.batches_processed == num_batches
        assert data_loader.tokens_processed == num_tokens

    assert num_batches == data_loader.total_batches

    data_loader.reset()

    assert data_loader.batches_processed == 0
    assert data_loader.tokens_processed == 0


@pytest.mark.parametrize(
    "shuffle", [pytest.param(True, id="shuffle"), pytest.param(False, id="no-shuffle")]
)
def test_fsl_data_loader_with_seq_len_warmup(tmp_path: Path, shuffle: bool):
    max_target_sequence_length = 8

    # Write some data to disk.
    mmap1 = np.memmap(tmp_path / "tokens1.npy", dtype=np.uint16, mode="w+", shape=(18,))
    mmap1[:] = list(range(18))
    mmap1.flush()
    mmap2 = np.memmap(tmp_path / "tokens2.npy", dtype=np.uint16, mode="w+", shape=(8,))
    mmap2[:] = list(range(18, 26))
    mmap2.flush()
    del mmap1, mmap2

    def get_all_tokens(seq_len: int, tokens_processed: int = 0) -> List[int]:
        dataset = NumpyFSLDataset(
            tmp_path / "tokens1.npy",
            tmp_path / "tokens2.npy",
            sequence_length=seq_len,
            pad_token_id=-1,
            eos_token_id=-1,
            vocab_size=32_000,
            max_target_sequence_length=max_target_sequence_length,
        )
        data_loader = NumpyFSLDataLoader(
            dataset,
            global_batch_size=seq_len,
            collator=DataCollator(pad_token_id=-1),
            shuffle=shuffle,
            chunk_size=max_target_sequence_length // seq_len,
            num_threads=0,
            work_dir=tmp_path,
        )
        data_loader.tokens_processed = tokens_processed
        data_loader.reshuffle(epoch=1)
        all_tokens = []
        for instance in data_loader:
            all_tokens.extend(instance["input_ids"].flatten().tolist())
        return all_tokens

    assert get_all_tokens(2) == get_all_tokens(4) == get_all_tokens(8)
    assert get_all_tokens(2, 8) == get_all_tokens(4, 8) == get_all_tokens(8, 8)


@pytest.mark.parametrize(
    "num_workers", [pytest.param(0, id="no-workers"), pytest.param(1, id="with-workers")]
)
def test_numpy_data_loader_while_changing_batch_size(
    tmp_path: Path, num_workers: int, num_tokens: int = 1024, sequence_length: int = 8
):
    mmap1 = np.memmap(tmp_path / "tokens.npy", dtype=np.uint16, mode="w+", shape=(num_tokens,))
    mmap1[:] = list(range(num_tokens))
    mmap1.flush()

    dataset = NumpyFSLDataset(
        tmp_path / "tokens.npy",
        sequence_length=sequence_length,
        pad_token_id=-1,
        eos_token_id=-1,
        vocab_size=32_000,
    )

    batch_size = sequence_length * 4

    data_loader = NumpyDataLoaderBase.wrap_numpy_dataset(
        dataset,
        global_batch_size=batch_size,
        collator=DataCollator(pad_token_id=-1),
        num_workers=num_workers,
        num_threads=0,
        work_dir=tmp_path,
    )
    data_loader.reshuffle()

    assert len(data_loader) == num_tokens // batch_size  # 32
    batch_iterator = iter(data_loader)

    # Process some batches with starting batch size.
    for _ in range(4):
        batch = next(batch_iterator)
        assert batch["input_ids"].numel() == batch_size
    assert data_loader.batches_processed == 4

    # Double batch size. This is allowed because we've processed 4 batches so far, which is divisible by 2.
    batch_size *= 2
    data_loader.global_batch_size = batch_size

    assert len(data_loader) == num_tokens // batch_size  # 16
    assert isinstance(data_loader.batches_processed, int)
    assert data_loader.batches_processed == 2

    for _ in range(6):
        batch = next(batch_iterator)
        assert batch["input_ids"].numel() == batch_size

    assert data_loader.batches_processed == 8

    # Now cut batch size in half.
    batch_size //= 2
    data_loader.global_batch_size = batch_size

    assert len(data_loader) == num_tokens // batch_size  # 16
    assert isinstance(data_loader.batches_processed, int)
    assert data_loader.batches_processed == 16


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
@pytest.mark.parametrize(
    "num_threads", [pytest.param(2, id="2-threads"), pytest.param(0, id="no-threads")]
)
@pytest.mark.parametrize(
    "in_memory", [pytest.param(True, id="in-memory"), pytest.param(False, id="on-disk")]
)
def test_vsl_data_loader(
    tmp_path: Path, shuffle: bool, num_threads: int, curriculum: VSLCurriculum, in_memory: bool
):
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
        vocab_size=32_000,
        min_sequence_length=2,
        max_sequence_length=4,
        dtype=np.uint16,
        curriculum=curriculum,
    )
    dataset.work_dir = tmp_path
    dataset.prepare()

    world_size = 2
    global_batch_size = 8

    data_loader = NumpyVSLDataLoader(
        dataset,
        shuffle=shuffle,
        num_threads=num_threads,
        work_dir=tmp_path,
        dp_world_size=world_size,
        collator=DataCollator(pad_token_id=0),
        global_batch_size=global_batch_size,
    )
    data_loader.reshuffle(epoch=1, in_memory=in_memory)

    all_tokens = []
    for rank in range(world_size):
        data_loader.dp_rank = rank
        n_batches = 0
        for batch in data_loader:
            n_batches += 1
            assert batch["input_ids"].numel() == global_batch_size // world_size
            all_tokens.extend(batch["input_ids"].flatten().tolist())
        data_loader.reset()

        assert n_batches == data_loader.total_batches

    assert len(all_tokens) == data_loader.total_batches * global_batch_size

    # Make sure batches were unique.
    all_tokens_less_eos = [t for t in all_tokens if t != 0]
    assert len(set(all_tokens_less_eos)) == len(all_tokens_less_eos)
