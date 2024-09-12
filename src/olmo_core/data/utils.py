import gzip
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
import torch

from olmo_core.aliases import PathOrStr
from olmo_core.io import get_bytes_range, resource_path


def split_batch(batch: Dict[str, Any], num_microbatch_instances: int) -> List[Dict[str, Any]]:
    """
    Split a batch (such as one generated by the :class:`DataCollator`) into a list of micro-batches.
    """
    batch_size = batch["input_ids"].shape[0]
    if batch_size <= num_microbatch_instances:
        return [batch]
    else:
        micro_batches = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                micro_batches[key] = value.split(num_microbatch_instances, dim=0)
            elif isinstance(value, list):
                micro_batches[key] = [
                    value[
                        num_microbatch_instances * i : num_microbatch_instances * i
                        + num_microbatch_instances
                    ]
                    for i in range(math.ceil(batch_size / num_microbatch_instances))
                ]
            else:
                raise RuntimeError(f"unexpected item in batch: '{key}={value}'")
        return [
            {key: value[i] for key, value in micro_batches.items()}
            for i in range(len(micro_batches["input_ids"]))
        ]


def melt_batch(batch: Dict[str, Any], target_sequence_length: int) -> Dict[str, Any]:
    """
    "Melts" a batch by shortening the sequence length and proportionally increasing the number
    of instances.
    """
    current_batch_size, current_sequence_length = batch["input_ids"].shape
    if current_sequence_length <= target_sequence_length:
        return batch

    if current_sequence_length % target_sequence_length != 0:
        raise RuntimeError(
            "current sequence of batch must be a multiple of the target sequence length "
            "in order to 'melt' the batch"
        )

    ratio = current_sequence_length // target_sequence_length

    new_batch: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.shape == (current_batch_size, current_sequence_length):
                new_batch[key] = value.reshape(-1, target_sequence_length)
            elif value.shape == (current_batch_size,) or value.shape == (current_batch_size, 1):
                new_batch[key] = value.repeat_interleave(ratio)
            else:
                raise RuntimeError(
                    f"unable to melt '{key}' tensor in batch with shape '{value.shape}'"
                )
        elif isinstance(value, list) and len(value) > 0:
            new_batch[key] = []
            for item in value:
                if isinstance(item, list):
                    if len(item) != current_sequence_length:
                        raise RuntimeError(f"unexpected item length for '{key}' in batch")
                    for i in range(ratio):
                        new_batch[key].append(item[i * ratio : i * ratio + target_sequence_length])
                else:
                    for _ in range(ratio):
                        new_batch[key].append(item)
        else:
            raise RuntimeError(f"unexpected item in batch: '{key}={value}'")

    return new_batch


def truncate_batch(batch: Dict[str, Any], target_sequence_length: int) -> Dict[str, Any]:
    """
    Truncate the instances in a batch to ``target_sequence_length``.
    """
    current_batch_size, current_sequence_length = batch["input_ids"].shape
    if current_sequence_length <= target_sequence_length:
        return batch

    new_batch: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if value.shape == (current_batch_size, current_sequence_length):
                new_batch[key] = value[:, :target_sequence_length]
            elif value.shape == (current_batch_size,) or value.shape == (current_batch_size, 1):
                new_batch[key] = value
            else:
                raise RuntimeError(
                    f"unable to truncate '{key}' tensor in batch with shape '{value.shape}'"
                )
        elif isinstance(value, list) and len(value) > 0:
            new_batch[key] = []
            for item in value:
                if isinstance(item, list):
                    if len(item) != current_sequence_length:
                        raise RuntimeError(f"unexpected item length for '{key}' in batch")
                    new_batch[key].append(item[:target_sequence_length])
                else:
                    new_batch[key].append(item)
        else:
            raise RuntimeError(f"unexpected item in batch: '{key}={value}'")

    return new_batch


def write_document_indices(data_path: Path, *, dtype, eos_token_id: int) -> Path:
    """
    Given a local ".npy" data path from the Dolma toolkit, write a metadata file with start/end indices
    of each document within the array.
    """
    token_ids = np.memmap(data_path, mode="r", dtype=dtype)
    eos_token_locations = (token_ids == eos_token_id).nonzero()[0]
    metadata_path = data_path.with_suffix(".csv.gz")
    with gzip.open(metadata_path, mode="wt") as f:
        start_idx = 0
        for eos_token_location in eos_token_locations:
            end_idx = eos_token_location + 1
            f.write(f"{start_idx},{end_idx}\n")
            start_idx = end_idx
    return metadata_path


def iter_document_indices(
    data_path: PathOrStr, local_cache: Optional[PathOrStr] = None
) -> Generator[Tuple[int, int], None, None]:
    """
    Given a ".npy" data path from the Dolma toolkit, get the list of document start/end indices within
    the array.

    :param data_path: Path to a ".npy" Dolma toolkit data file.
    :param local_cache: Local directory to put downloads into.
    """
    metadata_path = resource_path(
        os.path.dirname(data_path),
        os.path.basename(data_path).replace(".npy", ".csv.gz"),
        local_cache=local_cache,
    )
    with gzip.open(metadata_path, "rt") as f:
        for line in f:
            start_index, end_index, *_ = line.split(",")
            yield int(start_index), int(end_index)


def get_document_indices(
    data_path: PathOrStr, local_cache: Optional[PathOrStr] = None
) -> List[Tuple[int, int]]:
    """
    Like :func:`iter_document_indices` but returns a list.
    """
    return list(iter_document_indices(data_path, local_cache=local_cache))


def load_array_slice(
    path: PathOrStr,
    start_idx: int,
    end_idx: int,
    dtype: Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64], Type[np.bool_]],
) -> np.ndarray:
    """
    Load a slice from a numpy array on disk.

    :param path: The path/URL to the array.
    :param start_idx: The start index (0-based) of the slice within the array.
    :param end_idx: The end index (0-based, exclusive) of the slice within the array.
    :param dtype: The numpy datatype of the array.
    """
    item_size = dtype(0).itemsize
    bytes_start = start_idx * item_size
    num_bytes = (end_idx - start_idx) * item_size
    buffer = get_bytes_range(path, bytes_start, num_bytes)
    return np.frombuffer(buffer, dtype=dtype)


def load_array_slice_into_tensor(
    path: PathOrStr,
    start_idx: int,
    end_idx: int,
    dtype: Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64], Type[np.bool_]],
) -> torch.Tensor:
    """
    Read a chunk from a numpy array, returning the chunk as a :class:`torch.Tensor`.

    :param path: The path/URL to the array.
    :param start_idx: The start index (0-based) of the chunk within the array.
    :param end_idx: The end index (0-based, exclusive) of the chunk within the array.
    :param dtype: The numpy datatype of the array.
    """
    array = load_array_slice(path, start_idx, end_idx, dtype)
    if dtype == np.bool_:
        return torch.tensor(array)
    else:
        return torch.tensor(array.astype(np.int_), dtype=torch.long)


def get_document_lengths(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """
    Get the length of documents.

    :param input_ids: An integer-type tensor of token IDs.
    :param eos_token_id: The ID of the EOS token (use to denote document boundaries).
    """
    doc_boundaries = torch.cat(
        [
            torch.tensor([-1], dtype=torch.int32),
            (input_ids == eos_token_id).nonzero(as_tuple=True)[0].to(dtype=torch.int32),
            torch.tensor(
                [] if input_ids[-1] == eos_token_id else [input_ids.shape[0] - 1], dtype=torch.int32
            ),
        ]
    )
    return doc_boundaries[1:] - doc_boundaries[:-1]


def get_cumulative_document_lengths(doc_lens: torch.Tensor) -> torch.Tensor:
    """
    Transform a batched tensor of document lengths into a 1D tensor of cumulative document
    lengths for the whole batch.

    :param doc_lens: The document lengths, such as those returned by :func:`get_document_lengths`.
    """
    return torch.cat(
        [
            torch.tensor([0], dtype=torch.int32, device=doc_lens.device),
            torch.cumsum(doc_lens.masked_select(doc_lens != 0), 0, dtype=torch.int32),
        ]
    )


def iter_batched(
    iterable: Iterable[Dict[str, Any]], batch_num_tokens: int
) -> Iterable[Tuple[Dict[str, Any], ...]]:
    batch: List[Dict[str, Any]] = []
    tokens = 0
    for x in iter(iterable):
        x_num_tokens = x["input_ids"].numel()
        assert x_num_tokens <= batch_num_tokens, f"{x_num_tokens} > {batch_num_tokens}"
        if (tokens + x_num_tokens) > batch_num_tokens:
            yield tuple(batch)
            batch.clear()
            tokens = 0
        tokens += x_num_tokens
        batch.append(x)

    if batch:
        yield tuple(batch)


@contextmanager
def memmap_to_write(
    path: Path,
    *,
    shape: Tuple[int, ...],
    dtype: Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64], Type[np.bool_]],
) -> Generator[np.ndarray, None, None]:
    """
    A context manager for safely writing a numpy memory-mapped array to disk.
    The memory-mapped ndarray returned by the context manager will be mapped to a temporary
    file until the context exists successfully.
    """
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp_path = path.with_suffix(".npy.tmp")
    mmap = np.memmap(tmp_path, dtype=dtype, mode="w+", shape=shape)
    yield mmap
    mmap.flush()
    del mmap
    tmp_path.replace(path)


def divide_into_buckets(n: int, b: int) -> List[int]:
    buckets: List[int] = []
    while (buckets_remaining := b - len(buckets)) > 0:
        c = math.ceil(n / buckets_remaining)
        n -= c
        buckets.append(c)
    return buckets


def chunk_array(arr: np.ndarray, chunk_sizes: Sequence[int]) -> List[np.ndarray]:
    assert len(arr.shape) == 1
    assert sum(chunk_sizes) == arr.shape[0]
    offset = 0
    chunks = []
    for n in chunk_sizes:
        chunks.append(arr[offset : offset + n])
        offset += n
    return chunks


def get_rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed=seed))
