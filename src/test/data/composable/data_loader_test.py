from pathlib import Path

import pytest

from olmo_core.data.collator import DataCollator
from olmo_core.data.composable.concat_and_chunk_instance_source import (
    ConcatAndChunkInstanceSource,
)
from olmo_core.data.composable.data_loader import ComposableDataLoader
from olmo_core.data.composable.token_source import InMemoryTokenSource
from olmo_core.data.tokenizer import TokenizerConfig


@pytest.mark.parametrize("num_workers", [0, 2])
def test_data_loader(tmp_path: Path, num_workers: int):
    tokens1 = InMemoryTokenSource(tokens=list(range(64)), work_dir=tmp_path)
    tokens2 = InMemoryTokenSource(tokens=list(range(64, 128)), work_dir=tmp_path)
    instances = ConcatAndChunkInstanceSource(tokens1, tokens2, sequence_length=8, work_dir=tmp_path)
    tokenizer = TokenizerConfig(vocab_size=32000, eos_token_id=0, pad_token_id=-1)
    data_loader = ComposableDataLoader(
        instances,
        collator=DataCollator(pad_token_id=tokenizer.pad_token_id),
        tokenizer=tokenizer,
        work_dir=tmp_path,
        global_batch_size=16,
        num_workers=num_workers,
    )
    assert data_loader.sequence_length == 8
    assert data_loader.max_sequence_length == 8
    assert data_loader.total_instances == 16
    assert data_loader.total_tokens == 128
    assert data_loader.total_batches == len(data_loader) == 8

    batch = data_loader.get_mock_batch()
    assert batch["input_ids"].shape == (2, 8)

    data_loader.reshuffle()
    all_tokens = set()
    n_batches = 0
    for batch in data_loader:
        assert batch["input_ids"].shape == (2, 8)
        for seq in batch["input_ids"]:
            for token in seq.tolist():
                all_tokens.add(token)
        n_batches += 1
    assert n_batches == len(data_loader) == data_loader.batches_processed
    assert all_tokens == set(range(128))
    data_loader.reset()
