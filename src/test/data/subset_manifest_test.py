"""
Tests for ``olmo-token-subset-v1`` manifest datasets, per-epoch dynamic document repacking, and
the data-order controls (fixed order, batch shuffling) on the numpy FSL data loader.
"""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from olmo_core.data import (
    DataCollator,
    NumpyDataLoaderConfig,
    NumpyFSLDataLoader,
    NumpyFSLDataset,
    NumpyFSLDatasetConfig,
    NumpyFSLDynamicRepackedDataset,
    NumpyFSLSubsetDataset,
    TokenizerConfig,
)
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.exceptions import OLMoConfigurationError


def _entries_digest(entries) -> str:
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_tokens(path: Path, tokens) -> None:
    mmap = np.memmap(path, mode="w+", dtype=np.uint16, shape=(len(tokens),))
    mmap[:] = tokens
    mmap.flush()
    del mmap


def test_numpy_fsl_subset_manifest(tmp_path: Path):
    _write_tokens(tmp_path / "mmap1.npy", list(range(24)))
    _write_tokens(tmp_path / "mmap2.npy", list(range(24, 48)))

    entries = [
        {"path": "mmap1.npy", "start_instance": 1, "num_instances": 2},
        {"path": "mmap2.npy", "start_instance": 3, "num_instances": 1},
    ]
    manifest = {
        "format": "olmo-token-subset-v1",
        "source": {"dtype": "uint16"},
        "selection": {"sequence_length": 4, "selected_instances": 3},
        "entries_sha256": _entries_digest(entries),
        "entries": entries,
    }
    manifest_path = tmp_path / "subset.json"
    manifest_path.write_text(json.dumps(manifest))

    ds = NumpyFSLDatasetConfig(
        tokenizer=TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1),
        mix_base_dir=str(tmp_path),
        subset_manifest=str(manifest_path),
        sequence_length=4,
        dtype=NumpyDatasetDType.uint16,
        include_instance_metadata=False,
    ).build()

    assert isinstance(ds, NumpyFSLSubsetDataset)
    assert len(ds) == 3
    assert ds.num_tokens == 12
    assert ds[0]["input_ids"].tolist() == [4, 5, 6, 7]
    assert ds[1]["input_ids"].tolist() == [8, 9, 10, 11]
    assert ds[2]["input_ids"].tolist() == [36, 37, 38, 39]


def test_numpy_fsl_subset_manifest_masks_declared_alignment_tail(tmp_path: Path):
    _write_tokens(tmp_path / "mmap.npy", list(range(12)))

    entries = [{"path": "mmap.npy", "start_instance": 0, "num_instances": 3}]
    manifest = {
        "format": "olmo-token-subset-v1",
        "source": {"dtype": "uint16"},
        "selection": {
            "sequence_length": 4,
            "selected_instances": 3,
            "selected_real_document_tokens": 10,
            "padding_eos_tokens": 2,
            "selected_tokens": 12,
        },
        "entries_sha256": _entries_digest(entries),
        "entries": entries,
    }
    manifest_path = tmp_path / "subset.json"
    manifest_path.write_text(json.dumps(manifest))

    ds = NumpyFSLDatasetConfig(
        tokenizer=TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1),
        mix_base_dir=str(tmp_path),
        subset_manifest=str(manifest_path),
        sequence_length=4,
        dtype=NumpyDatasetDType.uint16,
        include_instance_metadata=False,
    ).build()

    assert ds[0]["label_mask"].tolist() == [True, True, True, True]
    assert ds[1]["label_mask"].tolist() == [True, True, True, True]
    assert ds[2]["label_mask"].tolist() == [True, True, False, False]


def test_numpy_fsl_subset_manifest_rejects_multiple_sources(tmp_path: Path):
    with pytest.raises(OLMoConfigurationError, match="exactly one of"):
        NumpyFSLDatasetConfig(
            tokenizer=TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=-1),
            mix_base_dir=str(tmp_path),
            subset_manifest=str(tmp_path / "subset.json"),
            paths=[str(tmp_path / "mmap.npy")],
            sequence_length=4,
        ).validate()


def test_numpy_fsl_subset_manifest_dynamic_repacking(tmp_path: Path):
    documents = [
        [11, 12, 0],
        [21, 0],
        [31, 32, 33, 0],
        [41, 42, 43, 44, 0],
        [51, 0],
        [61, 62, 0],
    ]
    real_tokens = [token for document in documents for token in document]
    sequence_length = 4
    padding_tokens = (-len(real_tokens)) % sequence_length
    all_tokens = real_tokens + [0] * padding_tokens
    _write_tokens(tmp_path / "tokens.npy", all_tokens)

    metadata_path = tmp_path / "tokens.csv.gz"
    offset = 0
    with gzip.open(metadata_path, "wt") as metadata_file:
        for document in documents:
            metadata_file.write(f"{offset},{offset + len(document)}\n")
            offset += len(document)
        if padding_tokens:
            metadata_file.write(f"{offset},{offset + padding_tokens}\n")

    entries = [
        {
            "path": "tokens.npy",
            "start_instance": 0,
            "num_instances": len(all_tokens) // sequence_length,
        }
    ]
    manifest = {
        "format": "olmo-token-subset-v1",
        "source": {"dtype": "uint16"},
        "selection": {
            "sequence_length": sequence_length,
            "selected_instances": len(all_tokens) // sequence_length,
            "selected_documents": len(documents),
            "selected_real_document_tokens": len(real_tokens),
            "padding_eos_tokens": padding_tokens,
            "selected_tokens": len(all_tokens),
        },
        "materialized": {"document_metadata_path": "tokens.csv.gz"},
        "entries_sha256": _entries_digest(entries),
        "entries": entries,
    }
    manifest_path = tmp_path / "subset.json"
    manifest_path.write_text(json.dumps(manifest))

    config = NumpyFSLDatasetConfig(
        tokenizer=TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=0),
        mix_base_dir=str(tmp_path),
        subset_manifest=str(manifest_path),
        sequence_length=sequence_length,
        dtype=NumpyDatasetDType.uint16,
        include_instance_metadata=False,
        dynamic_repacking=True,
    )
    ds = config.build()
    assert isinstance(ds, NumpyFSLDynamicRepackedDataset)
    ds.prepare()

    def epoch_stream(epoch: int):
        ds.repack_for_epoch(epoch, seed=17)
        instances = [ds[i] for i in range(len(ds))]
        stream = torch.cat([instance["input_ids"] for instance in instances])
        label_mask = torch.cat([instance["label_mask"] for instance in instances])
        assert label_mask.tolist() == [True] * len(real_tokens) + [False] * padding_tokens
        return stream[: len(real_tokens)].tolist()

    epoch1 = epoch_stream(1)
    epoch1_again = epoch_stream(1)
    epoch2 = epoch_stream(2)
    assert epoch1 == real_tokens
    assert epoch1 == epoch1_again
    assert epoch1 != epoch2

    def split_documents(stream):
        result = []
        current = []
        for token in stream:
            current.append(token)
            if token == 0:
                result.append(current)
                current = []
        assert current == []
        return sorted(result)

    assert split_documents(epoch1) == sorted(documents)
    assert split_documents(epoch2) == sorted(documents)

    loader = NumpyFSLDataLoader(
        ds,
        global_batch_size=sequence_length,
        collator=DataCollator(pad_token_id=0),
        shuffle=False,
        num_threads=0,
        num_workers=0,
        work_dir=tmp_path,
    )
    loader.reshuffle(epoch=3, in_memory=True)
    assert ds.repacking_epoch == 3
    loader_stream = torch.cat([batch["input_ids"].reshape(-1) for batch in loader])[
        : len(real_tokens)
    ].tolist()
    assert split_documents(loader_stream) == sorted(documents)


def test_numpy_fsl_subset_manifest_default_does_not_repack(tmp_path: Path):
    _write_tokens(tmp_path / "tokens.npy", list(range(8)))
    entries = [{"path": "tokens.npy", "start_instance": 0, "num_instances": 2}]
    manifest_path = tmp_path / "subset.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "olmo-token-subset-v1",
                "source": {"dtype": "uint16"},
                "selection": {"sequence_length": 4, "selected_instances": 2},
                "entries_sha256": _entries_digest(entries),
                "entries": entries,
            }
        )
    )
    ds = NumpyFSLDatasetConfig(
        tokenizer=TokenizerConfig(vocab_size=32_000, eos_token_id=0, pad_token_id=0),
        mix_base_dir=str(tmp_path),
        subset_manifest=str(manifest_path),
        sequence_length=4,
        dtype=NumpyDatasetDType.uint16,
        include_instance_metadata=False,
    ).build()
    assert not isinstance(ds, NumpyFSLDynamicRepackedDataset)
    assert ds[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert ds[1]["input_ids"].tolist() == [4, 5, 6, 7]


def test_fsl_data_loader_fixed_data_order(tmp_path: Path):
    """Fixed order freezes the ordinary epoch-1 permutation and exact batch grouping."""
    num_tokens = 256
    sequence_length = 4
    token_path = tmp_path / "fixed-order-tokens.npy"
    _write_tokens(token_path, list(range(num_tokens)))

    dataset = NumpyFSLDataset(
        token_path,
        sequence_length=sequence_length,
        pad_token_id=-1,
        eos_token_id=-1,
        vocab_size=32_000,
    )

    def get_order(*, shuffle: bool, reshuffle_each_epoch: bool, epoch: int) -> np.ndarray:
        loader = NumpyFSLDataLoader(
            dataset,
            global_batch_size=sequence_length * 8,
            collator=DataCollator(pad_token_id=-1),
            seed=17,
            shuffle=shuffle,
            reshuffle_each_epoch=reshuffle_each_epoch,
            num_threads=0,
            work_dir=tmp_path,
        )
        loader.reshuffle(epoch=epoch, in_memory=True)
        return loader.get_global_indices().copy()

    ordinary_epoch1 = get_order(shuffle=True, reshuffle_each_epoch=True, epoch=1)
    ordinary_epoch2 = get_order(shuffle=True, reshuffle_each_epoch=True, epoch=2)
    fixed_epoch1 = get_order(shuffle=True, reshuffle_each_epoch=False, epoch=1)
    fixed_epoch2 = get_order(shuffle=True, reshuffle_each_epoch=False, epoch=2)
    raw_epoch1 = get_order(shuffle=False, reshuffle_each_epoch=True, epoch=1)
    raw_epoch2 = get_order(shuffle=False, reshuffle_each_epoch=True, epoch=2)

    assert np.array_equal(fixed_epoch1, ordinary_epoch1)
    assert np.array_equal(fixed_epoch2, ordinary_epoch1)
    assert not np.array_equal(ordinary_epoch2, ordinary_epoch1)
    assert np.array_equal(raw_epoch1, np.arange(len(dataset), dtype=np.uint32))
    assert np.array_equal(raw_epoch2, raw_epoch1)

    # A data-order intervention that branches from an old checkpoint must be able to retain
    # the newly configured mode instead of restoring the historical per-epoch reshuffle.
    fixed_loader = NumpyFSLDataLoader(
        dataset,
        global_batch_size=sequence_length * 8,
        collator=DataCollator(pad_token_id=-1),
        seed=17,
        shuffle=True,
        reshuffle_each_epoch=False,
        restore_data_order_from_state=False,
        num_threads=0,
        work_dir=tmp_path,
    )
    old_checkpoint_state = fixed_loader.state_dict()
    old_checkpoint_state.pop("shuffle")
    old_checkpoint_state.pop("reshuffle_each_epoch")
    fixed_loader.load_state_dict(old_checkpoint_state)
    assert fixed_loader.shuffle is True
    assert fixed_loader.reshuffle_each_epoch is False


def test_fsl_data_loader_batch_shuffling(tmp_path: Path):
    """Batch shuffling preserves epoch-1 batch membership and permutes only batch rows."""
    num_tokens = 512
    sequence_length = 4
    instances_per_batch = 8
    token_path = tmp_path / "batch-shuffling-tokens.npy"
    _write_tokens(token_path, list(range(num_tokens)))

    dataset = NumpyFSLDataset(
        token_path,
        sequence_length=sequence_length,
        pad_token_id=-1,
        eos_token_id=-1,
        vocab_size=32_000,
    )

    configured_loader = NumpyDataLoaderConfig(
        global_batch_size=sequence_length * instances_per_batch,
        seed=17,
        batch_shuffling=True,
        work_dir=str(tmp_path),
    ).build(dataset, collator=DataCollator(pad_token_id=-1))
    assert isinstance(configured_loader, NumpyFSLDataLoader)
    assert configured_loader.batch_shuffling is True

    def make_loader(*, batch_shuffling: bool, restore: bool = True) -> NumpyFSLDataLoader:
        return NumpyFSLDataLoader(
            dataset,
            global_batch_size=sequence_length * instances_per_batch,
            collator=DataCollator(pad_token_id=-1),
            seed=17,
            shuffle=True,
            reshuffle_each_epoch=True,
            batch_shuffling=batch_shuffling,
            restore_data_order_from_state=restore,
            chunk_size=2,
            num_threads=0,
            work_dir=tmp_path,
        )

    def order(loader: NumpyFSLDataLoader, epoch: int) -> np.ndarray:
        loader.reshuffle(epoch=epoch, in_memory=True)
        return loader.get_global_indices().copy()

    ordinary_epoch1 = order(make_loader(batch_shuffling=False), 1)
    batch_loader = make_loader(batch_shuffling=True)
    batch_epoch1 = order(batch_loader, 1)
    batch_epoch2 = order(batch_loader, 2)
    batch_epoch3 = order(batch_loader, 3)

    assert np.array_equal(batch_epoch1, ordinary_epoch1)
    assert not np.array_equal(batch_epoch2, batch_epoch1)
    assert not np.array_equal(batch_epoch3, batch_epoch2)

    def batch_rows(indices: np.ndarray) -> set:
        return set(map(tuple, indices.reshape(-1, instances_per_batch).tolist()))

    expected_batches = batch_rows(batch_epoch1)
    assert batch_rows(batch_epoch2) == expected_batches
    assert batch_rows(batch_epoch3) == expected_batches

    # New checkpoints durably restore the mode and reconstruct the same deterministic order.
    batch_loader.reshuffle(epoch=2, in_memory=True)
    state = batch_loader.state_dict()
    restored_loader = make_loader(batch_shuffling=False)
    restored_loader.load_state_dict(state)
    assert restored_loader.batch_shuffling is True
    assert np.array_equal(order(restored_loader, 2), batch_epoch2)

    # Branching from a historical checkpoint can intentionally retain the newly configured mode.
    old_state = state.copy()
    old_state.pop("batch_shuffling")
    branch_loader = make_loader(batch_shuffling=True, restore=False)
    branch_loader.load_state_dict(old_state)
    assert branch_loader.batch_shuffling is True


def test_fsl_data_loader_batch_shuffling_rejects_incompatible_settings(tmp_path: Path):
    token_path = tmp_path / "batch-shuffling-invalid.npy"
    _write_tokens(token_path, list(range(32)))
    dataset = NumpyFSLDataset(
        token_path,
        sequence_length=4,
        pad_token_id=-1,
        eos_token_id=-1,
        vocab_size=32_000,
    )

    with pytest.raises(OLMoConfigurationError, match="batch_shuffling.*shuffle=True"):
        NumpyFSLDataLoader(
            dataset,
            global_batch_size=8,
            collator=DataCollator(pad_token_id=-1),
            shuffle=False,
            batch_shuffling=True,
            work_dir=tmp_path,
        )
    with pytest.raises(OLMoConfigurationError, match="batch_shuffling.*reshuffle_each_epoch=True"):
        NumpyFSLDataLoader(
            dataset,
            global_batch_size=8,
            collator=DataCollator(pad_token_id=-1),
            reshuffle_each_epoch=False,
            batch_shuffling=True,
            work_dir=tmp_path,
        )
