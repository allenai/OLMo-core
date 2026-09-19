"""Focused CPU regressions for legacy metadata and the streaming warm-start copy."""

import pickle
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from olmoe3_integration_sft_audit import verify_source_weights
from olmoe3_integration_sft_repack import repack, validate_copy
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from olmo_core.distributed.checkpoint.filesystem import (
    RemoteFileSystemReader,
    _StorageInfo,
)


def make_source(root):
    directory = root / "model_and_optim"
    directory.mkdir(parents=True)
    (root / "config.json").write_text('{"model": {}}')
    metadata = Metadata(state_dict_metadata={}, storage_data={}, planner_data={})
    values = {}
    with (directory / "__0.distcp").open("wb") as handle:
        for i, length in enumerate((1, 7, 8, 17, 64, 101)):
            key = f"module.blocks.{i}.q_norm.weight.main"
            value = torch.arange(length, dtype=torch.float32) / 16
            start = handle.tell()
            torch.save(value, handle)
            chunk = ChunkStorageMetadata(torch.Size([0]), value.size())
            metadata.state_dict_metadata[key] = TensorStorageMetadata(
                TensorProperties(dtype=value.dtype), value.size(), [chunk]
            )
            metadata.storage_data[MetadataIndex(key, chunk.offsets, 0)] = _StorageInfo(
                "__0.distcp", start, handle.tell() - start
            )
            metadata.planner_data[key] = (key,)
            values[key] = value
    (directory / ".metadata").write_bytes(pickle.dumps(metadata))
    return values


def trainer_for(values):
    parameters = {
        k.removeprefix("module.").removesuffix(".main"): v.bfloat16() for k, v in values.items()
    }
    return SimpleNamespace(
        train_module=SimpleNamespace(
            model=SimpleNamespace(named_parameters=lambda: parameters.items())
        ),
        bookkeeping_pg=None,
    )


def test_legacy_audit_and_error_broadcast(tmp_path):
    source = tmp_path / "source"
    values = make_source(source)
    state = {k: torch.empty_like(v) for k, v in values.items()}
    # Confirm this fixture reproduces the actual failure in the generic reader.
    with pytest.raises(CheckpointException, match="transform_descriptors"):
        dcp.load(
            state, storage_reader=dcp.FileSystemReader(source / "model_and_optim"), no_dist=True
        )
    with patch("olmoe3_integration_sft_audit.dist.get_rank", return_value=0), patch(
        "olmoe3_integration_sft_audit.dist.broadcast_object_list"
    ) as broadcast:
        verify_source_weights(trainer_for(values), source)
        assert broadcast.call_args.args[0][0]["passed"]
        with patch(
            "olmoe3_integration_sft_audit.dcp.load", side_effect=CheckpointException("read", {})
        ):
            with pytest.raises(AssertionError):
                verify_source_weights(trainer_for(values), source)
            assert not broadcast.call_args.args[0][0]["passed"]


def test_repack_roundtrip_and_no_overwrite(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "copy"
    values = make_source(source)
    before = (source / "model_and_optim/.metadata").read_bytes()
    proof = repack(source, destination, 8)
    assert proof["all_slices_verified"]
    assert (source / "model_and_optim/.metadata").read_bytes() == before
    assert repack(source, destination, 8) == proof  # Valid complete copy is reusable.
    state = {k: torch.empty_like(v) for k, v in values.items()}
    reader = RemoteFileSystemReader(destination / "model_and_optim", thread_count=1)
    dcp.load(state, storage_reader=reader, no_dist=True)
    for key, value in values.items():
        assert torch.equal(state[key], value)
    for value in state.values():
        value.zero_()
    dcp.load(
        state, storage_reader=RemoteFileSystemReader(destination, thread_count=1), no_dist=True
    )
    for key, value in values.items():
        assert torch.equal(state[key], value)
    with (destination / "model_and_optim/__warmstart_0.distcp").open("ab") as handle:
        handle.write(b"broken")
    with pytest.raises(AssertionError):
        validate_copy(source, destination)
    partial = tmp_path / "partial"
    partial.mkdir()
    with pytest.raises(FileExistsError):
        repack(source, partial)


def distributed_copy_worker(rank, destination, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=8,
        timeout=timedelta(seconds=90),
    )
    try:
        mesh = init_device_mesh("cpu", (8,))
        reader = RemoteFileSystemReader(Path(destination) / "model_and_optim", thread_count=1)
        metadata = reader.read_metadata()
        state = {}
        expected = {}
        for key, meta in metadata.state_dict_metadata.items():
            count = meta.size[0]
            chunk = (count + 7) // 8
            first = min(rank * chunk, count)
            last = min(first + chunk, count)
            local = torch.empty(last - first, dtype=torch.float32)
            state[key] = DTensor.from_local(local, mesh, (Shard(0),), shape=meta.size, stride=(1,))
            expected[key] = torch.arange(first, last, dtype=torch.float32) / 16
        dcp.load(state, storage_reader=reader)
        for key, value in state.items():
            assert torch.equal(value.to_local(), expected[key]), (rank, key)
    finally:
        dist.destroy_process_group()


def test_eight_rank_resharded_load(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "copy"
    make_source(source)
    repack(source, destination)
    torch.multiprocessing.spawn(
        distributed_copy_worker,
        args=(str(destination), str(tmp_path / "rendezvous")),
        nprocs=8,
        join=True,
    )
