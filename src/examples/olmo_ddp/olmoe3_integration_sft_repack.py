"""CPU-only, bit-exact warm-start copy of the integration checkpoint.

Read one legacy tensor at a time, write eight independently serialized slices,
and verify every slice after writing. Never mutate the original checkpoint.
This deliberately omits optimizer moments/trainer state: it is an SFT warm start,
not a resumable copy of the original LC run.
"""

import copy
import hashlib
import io
import json
import math
import os
import pickle
import time
from contextlib import ExitStack
from pathlib import Path

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
)

from olmo_core.distributed.checkpoint.filesystem import (
    RemoteFileSystemReader,
    _StorageInfo,
)
from olmo_core.io import get_bytes_range

READY = "_SFT_SOURCE_COPY_READY.json"


def digest(tensor):
    """Hash contiguous tensor bytes without another tensor-sized copy."""
    return hashlib.sha256(memoryview(tensor.numpy()).cast("B")).hexdigest()


def validate_copy(source, destination, shards=8):
    """Require a complete, verified copy tied to the unchanged original metadata."""
    source, destination = Path(source), Path(destination)
    proof = json.loads((destination / READY).read_text())
    assert proof["source"] == str(source.resolve()) and proof["shards"] == shards
    assert proof["weights_only"] and proof["all_slices_verified"]
    for relative, expected in proof["source_hashes"].items():
        assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == expected
    assert (
        hashlib.sha256((destination / "model_and_optim/.metadata").read_bytes()).hexdigest()
        == proof["metadata_sha256"]
    )
    assert (
        hashlib.sha256((destination / ".metadata").read_bytes()).hexdigest()
        == proof["root_metadata_sha256"]
    )
    assert (
        hashlib.sha256((destination / "config.json").read_bytes()).hexdigest()
        == proof["source_hashes"]["config.json"]
    )
    for name, size in proof["files"].items():
        assert (destination / "model_and_optim" / name).stat().st_size == size
    return proof


@torch.no_grad()
def repack(source, destination, shards=8):
    """Stream FP32 main parameters into a new eight-shard DCP warm-start copy."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    assert source != destination and source not in destination.parents
    assert destination not in source.parents and shards > 0
    if (destination / READY).exists():
        return validate_copy(source, destination, shards)
    # A partial copy is never silently overwritten or treated as complete.
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / "model_and_optim"
    output.mkdir()
    started = time.monotonic()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    reader = RemoteFileSystemReader(source / "model_and_optim", thread_count=1)
    old = reader.read_metadata()
    reader.set_up_storage_reader(old, True)
    keys = sorted(k for k in old.state_dict_metadata if k.endswith(".main"))
    assert keys and not any(
        k.startswith("model.") for k in old.state_dict_metadata
    ), "Review persistent model buffers before using this legacy-only converter"
    new = Metadata(state_dict_metadata={}, planner_data={}, storage_data={})
    proof = {
        "source": str(source),
        "destination": str(destination),
        "shards": shards,
        "weights_only": True,
        "all_slices_verified": False,
        "tensors": {},
        "files": {},
        "source_hashes": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("config.json", "model_and_optim/.metadata")
        },
    }
    total_read = 0
    with ExitStack() as stack:
        handles = [
            stack.enter_context((output / f"__warmstart_{rank}.distcp").open("x+b"))
            for rank in range(shards)
        ]
        for number, key in enumerate(keys):
            tick = time.monotonic()
            meta = old.state_dict_metadata[key]
            assert len(meta.size) == 1 and meta.properties.dtype == torch.float32
            assert len(meta.chunks) == 1 and meta.chunks[0].offsets == torch.Size([0])
            assert meta.chunks[0].sizes == meta.size
            index = MetadataIndex(key, meta.chunks[0].offsets, 0)
            info = old.storage_data[index]
            content = get_bytes_range(
                source / "model_and_optim" / info.relative_path, info.offset, info.length
            )
            total_read += len(content)
            tensor = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
            del content
            assert tensor.shape == meta.size and tensor.dtype == meta.properties.dtype
            source_read_seconds = time.monotonic() - tick
            shard_size = math.ceil(tensor.numel() / shards)
            chunks, hashes = [], []
            for rank, handle in enumerate(handles):
                offset = rank * shard_size
                if offset >= tensor.numel():
                    break
                # Clone to avoid torch.save serializing the full source storage.
                part = tensor[offset : min(offset + shard_size, tensor.numel())].clone()
                expected = digest(part)
                file_offset = handle.tell()
                torch.save(part, handle)
                length = handle.tell() - file_offset
                handle.flush()
                # Verify the actual serialized bytes, not just the in-memory slice.
                handle.seek(file_offset)
                restored = torch.load(
                    io.BytesIO(handle.read(length)), map_location="cpu", weights_only=False
                )
                assert restored.shape == part.shape and restored.dtype == part.dtype
                assert digest(restored) == expected, (key, rank)
                chunk = ChunkStorageMetadata(torch.Size([offset]), part.size())
                chunks.append(chunk)
                new.storage_data[MetadataIndex(key, chunk.offsets, rank)] = _StorageInfo(
                    Path(handle.name).name, file_offset, length
                )
                hashes.append(expected)
                del part, restored
            new_meta = copy.deepcopy(meta)
            new_meta.chunks = chunks
            new.state_dict_metadata[key] = new_meta
            new.planner_data[key] = (key,)
            proof["tensors"][key] = {"numel": tensor.numel(), "shard_sha256": hashes}
            del tensor
            print(
                json.dumps(
                    {
                        "event": "SOURCE_TENSOR_REPACKED",
                        "tensor": key,
                        "count": number + 1,
                        "total": len(keys),
                        "read_bytes": total_read,
                        "source_read_seconds": source_read_seconds,
                        "tensor_seconds": time.monotonic() - tick,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
        for handle in handles:
            handle.flush()
            os.fsync(handle.fileno())
            proof["files"][Path(handle.name).name] = handle.tell()
    metadata_bytes = pickle.dumps(new)
    with (output / ".metadata").open("xb") as handle:
        handle.write(metadata_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    # A standalone weight checkpoint has a root .metadata, not fake trainer
    # state. Keep nested metadata too for the existing SFT source audit.
    root_metadata = copy.deepcopy(new)
    for info in root_metadata.storage_data.values():
        info.relative_path = "model_and_optim/" + info.relative_path
    root_metadata_bytes = pickle.dumps(root_metadata)
    with (destination / ".metadata").open("xb") as handle:
        handle.write(root_metadata_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    with (destination / "config.json").open("xb") as handle:
        handle.write((source / "config.json").read_bytes())
        handle.flush()
        os.fsync(handle.fileno())
    proof.update(
        all_slices_verified=True,
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        root_metadata_sha256=hashlib.sha256(root_metadata_bytes).hexdigest(),
        read_bytes=total_read,
        elapsed_seconds=time.monotonic() - started,
    )
    with (destination / READY).open("x") as handle:
        json.dump(proof, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    for directory in (output, destination):
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    validate_copy(source, destination, shards)
    print(
        json.dumps(
            {
                "event": "SOURCE_COPY_READY",
                "destination": str(destination),
                "tensors": len(keys),
                "read_bytes": total_read,
                "elapsed_seconds": proof["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    return proof


if __name__ == "__main__":
    from olmoe3_hero_sft_plan import LEGACY_SOURCE, SOURCE

    repack(LEGACY_SOURCE, SOURCE)
