from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import math
import os
import tempfile
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import torch
from torch.utils.data import Dataset

from olmo_core.exceptions import OLMoConfigurationError, OLMoEnvironmentError

from ..aliases import PathOrStr
from ..config import Config, StrEnum
from ..distributed.utils import barrier, get_fs_local_rank
from ..io import _get_s3_client, get_file_size
from .mixes import DataMix
from .tokenizer import TokenizerConfig
from .utils import (
    bucket_documents,
    chunk_array,
    divide_into_buckets,
    get_doc_lengths_from_indices,
    get_document_lengths,
    get_rng,
    load_array_slice_into_tensor,
    memmap_to_write,
    segment_documents_into_instances,
)

__all__ = [
    "NumpyDatasetBase",
    "NumpyFSLDataset",
    "NumpyPaddedFSLDataset",
    "VSLCurriculum",
    "VSLNaturalCurriculum",
    "VSLGrowthCurriculum",
    "VSLGrowP2Curriculum",
    "VSLGrowLinearCurriculum",
    "NumpyVSLDataset",
    "NumpyDatasetType",
    "NumpyDatasetConfig",
    "VSLCurriculumType",
    "VSLCurriculumConfig",
    "NumpyDatasetDType",
]


log = logging.getLogger(__name__)


T = TypeVar("T")


class NumpyDatasetBase(ABC):
    """
    An abstract base class for datasets backed by numpy arrays on disk of token IDs.

    In general the instances that these datasets produce are sequences of token IDs from one
    or more numpy arrays, sometimes with additional metadata attached.
    The way those instances are formed depends on the implementation details of the subclass.

    .. warning::
        When using :class:`NumpyDatasetBase` implementations in a distributed setting be sure
        that the :data:`work_dir` is shared among all local ranks and :data:`fs_local_rank` is set
        accordingly. Once those fields are set you should then call :meth:`prepare()` in the
        main process before doing anything else.

    .. tip::
        Use :class:`NumpyDatasetConfig` to configure and construct datasets instead of constructing
        them directly.
    """

    def __init__(
        self,
        *paths: PathOrStr,
        pad_token_id: int,
        eos_token_id: int,
        dtype: Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]] = np.uint16,
    ):
        if not paths:
            raise OLMoConfigurationError("At least one path is required")

        self._array_paths = tuple(paths)
        self._pad_token_id = pad_token_id
        self._eos_token_id = eos_token_id
        self._dtype = dtype
        self._fs_local_rank = get_fs_local_rank()
        self._work_dir: Optional[Path] = None
        self._work_dir_set = False
        self._array_file_sizes: Optional[Tuple[int, ...]] = None

    @property
    def paths(self) -> Tuple[PathOrStr, ...]:
        """
        Paths and/or URLs to the numpy arrays.
        """
        return self._array_paths

    @property
    def file_sizes(self) -> Tuple[int, ...]:
        """
        The size, in bytes, of each numpy array.
        """
        if self._array_file_sizes is None:
            self._array_file_sizes = tuple(self.map(get_file_size))
        return self._array_file_sizes

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def dtype(
        self,
    ) -> Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]]:
        """
        The numpy datatype of the arrays.
        """
        return self._dtype

    @property
    def fingerprint_version(self) -> str:
        """
        The version of the :data:`fingerprint`.
        """
        return "v1.1"

    @property
    def fingerprint(self) -> str:
        """
        Can be used to identify/compare the contents of a dataset.
        """
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            f"pad_token_id={self.pad_token_id},"
            f"eos_token_id={self.eos_token_id},"
            f"dtype={self.dtype}".encode()
        )
        for path, size in zip(self.paths, self.file_sizes):
            sha256_hash.update(f"name={os.path.basename(path)},size={size}".encode())
        return sha256_hash.hexdigest()

    @property
    def fs_local_rank(self) -> int:
        return self._fs_local_rank

    @fs_local_rank.setter
    def fs_local_rank(self, fs_local_rank: int):
        self._fs_local_rank = fs_local_rank

    @property
    def work_dir(self) -> Path:
        if self._work_dir is not None:
            return self._work_dir
        else:
            return Path(tempfile.gettempdir())

    @work_dir.setter
    def work_dir(self, work_dir: PathOrStr):
        self._work_dir = Path(work_dir)
        self._work_dir_set = True

    @property
    def work_dir_set(self) -> bool:
        """
        Check if the working directory was explicitly set.
        """
        return self._work_dir_set

    def _get_file_size(self, path: PathOrStr):
        path_idx = self.paths.index(path)
        return self.file_sizes[path_idx]

    def _warmup_clients(self):
        # Maybe create client up front to work around a threading issue in boto.
        if any(str(p).startswith("s3://") for p in self.paths):
            _get_s3_client("s3")

        if any(str(p).startswith("r2://") for p in self.paths):
            try:
                _get_s3_client("r2")
            except OLMoEnvironmentError:
                # R2 might not be needed, so ignore this error. We will get an error
                # later if R2 is needed.
                pass

        if any(str(p).startswith("weka://") for p in self.paths):
            try:
                _get_s3_client("weka")
            except OLMoEnvironmentError:
                # Weka might not be needed, so ignore this error. We will get an error
                # later if Weka is needed.
                pass

    def map(
        self,
        func: Callable[[PathOrStr], T],
        *,
        max_workers: Optional[int] = None,
        method: Literal["threads", "processes"] = "threads",
        _paths: Optional[Sequence[PathOrStr]] = None,
    ) -> List[T]:
        """
        Call a function on each path in the dataset, returning a list of the results, in order.

        :param func: The function to map to the paths.
        :param max_workers: The number of workers threads/processes. Set to 0 to execute synchronously
            in the main thread/process.
        :param method: Whether to use multi-threading or multi-processing.

        :returns: The results, in the same order as :data:`paths`.
        """
        paths = _paths or self.paths

        if max_workers == 0:
            return [func(path) for path in paths]

        executor_class: Union[
            Type[concurrent.futures.ThreadPoolExecutor],
            Type[concurrent.futures.ProcessPoolExecutor],
        ]
        if method == "threads":
            self._warmup_clients()
            executor_class = concurrent.futures.ThreadPoolExecutor
        elif method == "processes":
            executor_class = concurrent.futures.ProcessPoolExecutor
        else:
            raise ValueError(method)

        with executor_class(max_workers=max_workers) as executor:
            path_to_future = {}
            for path in paths:
                if path not in path_to_future:
                    path_to_future[path] = executor.submit(func, path)

            results = []
            for path in paths:
                results.append(path_to_future[path].result())

        return results

    def prepare(self):
        """
        Perform any necessary preparation.

        .. warning::
            Be sure to set :data:`work_dir` properly before calling this and only call this from the
            main process (not a worker process).
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Get the number of instances in the dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get an instance from the dataset. At a minimum this will contain the field "input_ids", a
        integer tensor of token IDs.
        """
        raise NotImplementedError


class NumpyFSLDataset(NumpyDatasetBase, Dataset[Dict[str, Any]]):
    """
    A fixed sequence length (FSL) numpy array-backed dataset.

    In this implementation the token IDs from all arrays are concatenated together and then chunked
    into contiguous blocks of ``sequence_length`` tokens to create instances. Therefore documents
    may be split over multiple instances.

    .. seealso::
        :class:`NumpyPaddedFSLDataset`

    .. important::
        If the length of an array is not a multiple of ``sequence_length`` or
        ``max_target_sequence_length`` the remainder of the tokens will be ignored.

    .. important::
        No special tokens are added to the input IDs so it's assumed that if you want
        EOS tokens between documents, for example, those will already be in the array.

    :param paths: Paths or URLs to numpy token ID arrays.
    :param sequence_length: The number of tokens to chunk together into a single instance.
        Generally this should correspond to your model's maximum input length.
    :param pad_token_id: The ID of the padding token.
    :param eos_token_id: The ID of the EOS token.
    :param dtype: The numpy datatype of the arrays.
    :param metadata: Metadata to add to each item. This should be a dictionary or a list of dictionaries
        with the same number of items as there are paths.
    :param include_instance_metadata: If ``True`` (the default), each instance returned from
        :meth:`__getitem__()` will include the metadata from its source.
    :param max_target_sequence_length: If using sequence length warm-up throughput training, this
        should be set to the maximum/final target sequence length to ensure consistent
        data order.
    """

    def __init__(
        self,
        *paths: PathOrStr,
        sequence_length: int,
        pad_token_id: int,
        eos_token_id: int,
        dtype: Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]] = np.uint16,
        metadata: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        include_instance_metadata: Optional[bool] = None,
        generate_doc_lengths: bool = False,
        max_target_sequence_length: Optional[int] = None,
    ):
        if max_target_sequence_length is not None and (
            max_target_sequence_length < sequence_length
            or max_target_sequence_length % sequence_length != 0
        ):
            raise OLMoConfigurationError(
                "'max_target_sequence_length' should be a multiple of 'sequence_length'"
            )

        if include_instance_metadata is None and metadata:
            include_instance_metadata = True

        if isinstance(metadata, list):
            if len(metadata) != len(paths):
                raise OLMoConfigurationError(
                    "'metadata' should have the same length as the number of file paths"
                )
        else:
            metadata = [metadata or {}] * len(paths)

        super().__init__(*paths, pad_token_id=pad_token_id, eos_token_id=eos_token_id, dtype=dtype)
        self._metadata = tuple(metadata)
        self._sequence_length = sequence_length
        self._max_target_sequence_length = max_target_sequence_length
        self._array_offsets: Optional[Tuple[Tuple[int, int], ...]] = None
        self._num_instances: Optional[int] = None
        self._include_instance_metadata = include_instance_metadata
        self._generate_doc_lengths = generate_doc_lengths

    @property
    def num_tokens(self) -> int:
        return len(self) * self.sequence_length

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def max_target_sequence_length(self) -> Optional[int]:
        return self._max_target_sequence_length

    @property
    def file_sizes(self) -> Tuple[int, ...]:
        """
        The size, in bytes, of each numpy array.
        """
        return self._sizes_and_offsets[0]

    @property
    def offsets(self) -> Tuple[Tuple[int, int], ...]:
        """
        Gives the global start and end instance indices for each data file in the dataset.
        """
        return self._sizes_and_offsets[1]

    @property
    def metadata(self) -> Tuple[Dict[str, Any], ...]:
        return self._metadata

    def prepare(self):
        len(self)

    def __len__(self) -> int:
        if self._num_instances is None:
            self._num_instances = self.offsets[-1][1]
        return self._num_instances

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)  # in case this is a numpy int type.
        pos_index = index if index >= 0 else len(self) + index

        # The index of the array within 'self.paths'.
        array_index: Optional[int] = None
        # The index within the corresponding array.
        array_local_index: Optional[int] = None
        for i, (offset_start, offset_end) in enumerate(self.offsets):
            if offset_start <= pos_index < offset_end:
                array_index = i
                array_local_index = pos_index - offset_start
                break

        if array_index is None or array_local_index is None:
            raise IndexError(f"{index} is out of bounds for dataset of size {len(self)}")

        # Read the data from file.
        input_ids = self._read_chunk_from_array(self.paths[array_index], array_local_index)
        out: Dict[str, Any] = {"input_ids": input_ids}

        if self._include_instance_metadata:
            metadata = self._metadata[array_index]
            out["metadata"] = deepcopy(metadata)

        if self._generate_doc_lengths:
            out["doc_lens"] = get_document_lengths(input_ids, self.eos_token_id)

        return out

    @property
    def _sizes_and_offsets(self) -> Tuple[Tuple[int, ...], Tuple[Tuple[int, int], ...]]:
        if self._array_offsets is None or self._array_file_sizes is None:
            array_offsets: List[Tuple[int, int]] = []
            array_file_sizes: List[int] = []

            start_offset = 0
            for size, length in self.map(self._get_file_size_and_length):
                end_offset = start_offset + length
                array_offsets.append((start_offset, end_offset))
                array_file_sizes.append(size)
                start_offset += length

            self._array_offsets = tuple(array_offsets)
            self._array_file_sizes = tuple(array_file_sizes)

        return self._array_file_sizes, self._array_offsets

    def _read_chunk_from_array(self, path: PathOrStr, index: int) -> torch.Tensor:
        start_idx = index * self.sequence_length
        return load_array_slice_into_tensor(
            path, start_idx, start_idx + self.sequence_length, self.dtype
        )

    def _get_file_size_and_length(self, path, dtype=None) -> Tuple[int, int]:
        dtype = dtype or self.dtype
        item_size = dtype(0).itemsize
        file_size = get_file_size(path)
        if (
            self.max_target_sequence_length is None
            or self.max_target_sequence_length == self.sequence_length
        ):
            return file_size, file_size // (item_size * self.sequence_length)
        elif self.max_target_sequence_length > self.sequence_length:
            num_max_seq_len_instances = file_size // (item_size * self.max_target_sequence_length)
            return (
                file_size,
                num_max_seq_len_instances
                * (self.max_target_sequence_length // self.sequence_length),
            )
        else:
            raise RuntimeError("invalid 'max_target_sequence_length' or 'sequence_length'")


class NumpyPaddedFSLDataset(NumpyFSLDataset):
    """
    A version of :class:`NumpyFSLDataset` that creates a single instance from each document.
    The resulting instances may be padded by the :class:`~olmo_core.data.collator.DataCollator`.
    """

    def __init__(
        self,
        *paths: PathOrStr,
        sequence_length: int,
        pad_token_id: int,
        eos_token_id: int,
        dtype: Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]] = np.uint16,
        metadata: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        include_instance_metadata: Optional[bool] = None,
    ):
        super().__init__(
            *paths,
            sequence_length=sequence_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            dtype=dtype,
            metadata=metadata,
            include_instance_metadata=include_instance_metadata,
        )
        self._array_instance_offsets: Optional[Tuple[Tuple[int, int], ...]] = None

    @property
    def offsets(self) -> Tuple[Tuple[int, int], ...]:
        if self._array_instance_offsets is None:
            item_size = self.indices_dtype(0).itemsize
            num_instances_per_path = self.map(
                lambda path: get_file_size(self._get_instance_indices_path(path)) // (item_size * 2)
            )
            array_instance_offsets = []
            start_offset = 0
            for num_instances in num_instances_per_path:
                array_instance_offsets.append((start_offset, start_offset + num_instances))
                start_offset += num_instances
            self._array_instance_offsets = tuple(array_instance_offsets)
        return self._array_instance_offsets

    @property
    def indices_dtype(
        self,
    ) -> Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]]:
        return np.uint32

    def prepare(self):
        if self.fs_local_rank == 0:
            log.info("Gathering dataset document indices...")
            self._write_instance_indices()
        len(self)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = super().__getitem__(index)
        item["label_mask"] = torch.ones_like(item["input_ids"])
        return item

    def _read_chunk_from_array(self, path: PathOrStr, index: int) -> torch.Tensor:
        indices_path = self._get_instance_indices_path(path)
        indices = load_array_slice_into_tensor(
            indices_path, index * 2, index * 2 + 2, self.indices_dtype
        )
        start_idx, end_idx = indices
        data = load_array_slice_into_tensor(path, int(start_idx), int(end_idx), self.dtype)
        return data

    def _get_instance_indices_path(self, path: PathOrStr) -> Path:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(str(path).encode())
        sha256_hash.update(str(self._get_file_size(path)).encode())
        path_hash = sha256_hash.hexdigest()
        return (
            self.work_dir
            / "dataset-common"
            / f"instance-indices-{self.sequence_length}-{path_hash}.npy"
        )

    def _write_instance_indices(self):
        paths_needed: List[PathOrStr] = []
        for path in self.paths:
            indices_path = self._get_instance_indices_path(path)
            if indices_path.is_file():
                log.info(f"Reusing instance indices for '{path}' at:\n'{indices_path}'")
            elif path not in paths_needed:
                paths_needed.append(path)

        if paths_needed:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                for path in paths_needed:
                    indices_path = self._get_instance_indices_path(path)
                    log.info(f"Gathering instance indices for '{path}'...")
                    future = executor.submit(
                        segment_documents_into_instances,
                        path,
                        indices_path,
                        max_sequence_length=self.sequence_length,
                        eos_token_id=self.eos_token_id,
                        dtype=self.dtype,
                        indices_dtype=self.indices_dtype,
                    )
                    futures.append(future)

                concurrent.futures.wait(futures, return_when="ALL_COMPLETED")

                # Log results.
                for path, future in zip(paths_needed, futures):
                    total_og_docs, total_instances = future.result()
                    log.info(
                        f"Created {total_instances:,d} instances of sequence length up to "
                        f"{self.sequence_length} from "
                        f"{total_og_docs:,d} original documents in '{path}'"
                    )


@dataclass
class VSLCurriculum:
    """
    Base class for variable sequence length curriculums. These determine the sampling
    probability of batches from each bucket throughout training with a :class:`NumpyVSLDataset`.
    """

    @abstractmethod
    def batches_per_bucket(
        self, dataset: NumpyVSLDataset, global_batch_size: int
    ) -> List[Tuple[int, int]]:
        raise NotImplementedError

    @abstractmethod
    def get_batch_indices(
        self, batches_per_bucket: Sequence[Tuple[int, int]], seed: int
    ) -> np.ndarray:
        raise NotImplementedError

    def get_total_batches(self, batches_per_bucket: Sequence[Tuple[int, int]]) -> int:
        return sum([batches for _, batches in batches_per_bucket])

    def log_buckets(
        self,
        dataset: NumpyVSLDataset,
        global_batch_size: int,
        batches_per_bucket: Sequence[Tuple[int, int]],
    ):
        natural_batches_per_bucket = VSLNaturalCurriculum().batches_per_bucket(
            dataset, global_batch_size
        )
        for i, (seq_len, num_batches) in enumerate(batches_per_bucket):
            num_natural_batches = natural_batches_per_bucket[i][1]
            if num_batches != num_natural_batches:
                log.info(
                    f"- bucket {i}: sequence length {seq_len}, using {num_batches:,d} batches out of "
                    f"{num_natural_batches:,d} total"
                )
            else:
                log.info(f"- bucket {i}: sequence length {seq_len}, {num_batches:,d} batches")

    @property
    @abstractmethod
    def short_str(self) -> str:
        """
        Return a unique human-readable identifier for the instance.
        """
        raise NotImplementedError


@dataclass
class VSLNaturalCurriculum(VSLCurriculum):
    """
    Implements the natural curriculum from
    `Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum
    <https://arxiv.org/pdf/2405.13226>`_.
    """

    def batches_per_bucket(
        self, dataset: NumpyVSLDataset, global_batch_size: int
    ) -> List[Tuple[int, int]]:
        batches_per_bucket = []
        for seq_len, num_instances in dataset.instances_per_bucket:
            instances_per_batch = global_batch_size // seq_len
            batches = num_instances // instances_per_batch
            batches_per_bucket.append((seq_len, batches))
        return batches_per_bucket

    def get_batch_indices(
        self, batches_per_bucket: Sequence[Tuple[int, int]], seed: int
    ) -> np.ndarray:
        total_batches = self.get_total_batches(batches_per_bucket)
        batch_indices = np.arange(total_batches, dtype=np.uint32)
        rng = get_rng(seed)
        # Put a batch with the largest sequence length first to catch OOMs early.
        idx = rng.integers(total_batches - batches_per_bucket[-1][1], total_batches)
        batch = batch_indices[idx]
        batch_indices[idx] = batch_indices[0]
        batch_indices[0] = batch
        rng.shuffle(batch_indices[1:])
        return batch_indices

    @property
    def short_str(self) -> str:
        return "vsl-natural"


@dataclass
class VSLGrowthCurriculum(VSLCurriculum):
    """
    A base class for growth curriculums, like :class:`VSLGrowP2Curriculum` and :class:`VSLGrowLinearCurriculum`.
    """

    num_cycles: int = 8
    """
    The number of cycles in the curriculum.
    """
    balanced: bool = False
    """
    Whether or not to balance the number of batches in each bucket.

    .. note::
        Balancing the number of batches requires dropping more data.
    """

    def batches_per_bucket(
        self, dataset: NumpyVSLDataset, global_batch_size: int
    ) -> List[Tuple[int, int]]:
        actual_batches_per_bucket = VSLNaturalCurriculum().batches_per_bucket(
            dataset, global_batch_size
        )
        if self.balanced:
            batches_per_bucket = min([batches for _, batches in actual_batches_per_bucket])
            batches_per_bucket = self.num_cycles * (batches_per_bucket // self.num_cycles)
            return [(seq_len, batches_per_bucket) for seq_len, _ in actual_batches_per_bucket]
        else:
            return [
                (seq_len, self.num_cycles * (batches_per_bucket // self.num_cycles))
                for seq_len, batches_per_bucket in actual_batches_per_bucket
            ]

    def get_cycle_distribution(
        self, indices: np.ndarray, batches_per_bucket: Sequence[Tuple[int, int]], cycle: int = 0
    ) -> List[List[int]]:
        cycle_length = indices.shape[0] // self.num_cycles
        cycle_indices = indices[cycle * cycle_length : (cycle * cycle_length) + cycle_length]
        distribution: List[List[int]] = []
        for subcycle in np.array_split(cycle_indices, len(batches_per_bucket)):
            distribution.append([])
            bucket_offset_start = 0
            bucket_offset_end = 0
            for _, num_batches in batches_per_bucket:
                bucket_offset_end += num_batches
                count = ((subcycle >= bucket_offset_start) & (subcycle < bucket_offset_end)).sum()
                distribution[-1].append(count)
                bucket_offset_start += num_batches
        return distribution

    def get_batch_indices(
        self, batches_per_bucket: Sequence[Tuple[int, int]], seed: int
    ) -> np.ndarray:
        # Shortest sequence length first.
        assert batches_per_bucket[0][0] < batches_per_bucket[-1][0]

        rng = get_rng(seed)
        num_buckets = len(batches_per_bucket)

        log.info(f"Constructing {self.__class__.__name__} curriculum with {num_buckets} buckets")

        cycles: List[np.ndarray] = []
        for cycle in range(self.num_cycles):
            # Now we need to chunk the batch indices *within* each bucket in this cycle into the batch
            # indices for each sub-cycle.
            # At the same time we'll translate those *within* bucket indices into global batch indices
            # by adding the right offset for each bucket.
            all_bucket_subcycle_batches: List[List[np.ndarray]] = []
            for bucket in range(num_buckets):
                # This is how many batches we'll pull from this bucket for each cycle.
                batch_counts_per_cycle_this_bucket = divide_into_buckets(
                    batches_per_bucket[bucket][1], self.num_cycles
                )
                # These are the batch indices *within* this bucket that we'll use for this cycle.
                batches_this_cycle_this_bucket = chunk_array(
                    np.arange(0, batches_per_bucket[bucket][1], dtype=np.uint32),
                    batch_counts_per_cycle_this_bucket,
                )[cycle]
                bucket_offset = sum([b for _, b in batches_per_bucket[:bucket]])
                bucket_subcycle_batch_counts = self._get_num_bucket_batches_for_cycle(
                    bucket, num_buckets, batch_counts_per_cycle_this_bucket[cycle]
                )
                bucket_subcycle_batches = chunk_array(
                    bucket_offset + batches_this_cycle_this_bucket, bucket_subcycle_batch_counts
                )
                all_bucket_subcycle_batches.append(bucket_subcycle_batches)

            # Now we'll build each full syb-cycle by concatenating all of the bucket sub-cycle batches
            # together and shuffling.
            all_subsycles: List[np.ndarray] = []
            for subcycle in range(num_buckets):
                subsycle_batches: List[np.ndarray] = []
                for bucket in range(num_buckets):
                    subsycle_batches.append(all_bucket_subcycle_batches[bucket][subcycle])
                res = np.concatenate(subsycle_batches)
                rng.shuffle(res)
                all_subsycles.append(res)
            del all_bucket_subcycle_batches

            # Finally we can concatenate all of the subsycles together to form the complete cycle.
            cycles.append(np.concatenate(all_subsycles))
            del all_subsycles

        indices = np.concatenate(cycles)
        del cycles

        # Make sure the very first batch has the longest sequence length (is from the last bucket).
        # That way OOMs should happen right away.
        final_bucket_start = sum([b for _, b in batches_per_bucket[:-1]])
        first_long_seq_len_batch = np.argmax(indices >= final_bucket_start)
        batch = indices[first_long_seq_len_batch]
        indices[first_long_seq_len_batch] = indices[0]
        indices[0] = batch

        assert indices.shape[0] == self.get_total_batches(batches_per_bucket)
        return indices

    @classmethod
    @abstractmethod
    def _get_bucket_odds_for_cycle(cls, bucket_idx: int, num_buckets: int) -> List[int]:
        raise NotImplementedError

    @classmethod
    def _get_num_bucket_batches_for_cycle(
        cls, bucket_idx: int, num_buckets: int, num_batches: int
    ) -> List[int]:
        odds = cls._get_bucket_odds_for_cycle(bucket_idx, num_buckets)
        divisor = sum(odds)
        props = [o / divisor for o in odds]
        out = []
        total = 0
        for p in props:
            n = round(p * num_batches)
            total += n
            out.append(n)
        if total < num_batches:
            out[-1] += num_batches - total
        return out


@dataclass
class VSLGrowP2Curriculum(VSLGrowthCurriculum):
    """
    Implements the "Grow-P2" curriculum from
    `Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum
    <https://arxiv.org/pdf/2405.13226>`_.
    """

    @classmethod
    def _get_bucket_odds_for_cycle(cls, bucket_idx: int, num_buckets: int) -> List[int]:
        all_odds = []
        start_odds = num_buckets - bucket_idx
        for cycle in range(num_buckets):
            exp = (
                start_odds + cycle
                if start_odds + cycle <= num_buckets
                else start_odds - ((start_odds + cycle) % num_buckets)
            )
            all_odds.append(2 ** (exp - 1))
        return all_odds

    @property
    def short_str(self) -> str:
        return f"vsl-grow-p2-{self.num_cycles}-cycle{'-balanced' if self.balanced else ''}"


@dataclass
class VSLGrowLinearCurriculum(VSLGrowthCurriculum):
    """
    Implements the "Grow-Linear" curriculum from
    `Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum
    <https://arxiv.org/pdf/2405.13226>`_.
    """

    @classmethod
    def _get_bucket_odds_for_cycle(cls, bucket_idx: int, num_buckets: int) -> List[int]:
        all_odds = []
        start_odds = num_buckets - bucket_idx
        for cycle in range(num_buckets):
            odds = (
                start_odds + cycle
                if start_odds + cycle <= num_buckets
                else start_odds - ((start_odds + cycle) % num_buckets)
            )
            all_odds.append(odds)
        return all_odds

    @property
    def short_str(self) -> str:
        return f"vsl-grow-linear-{self.num_cycles}-cycle{'-balanced' if self.balanced else ''}"


class NumpyVSLDataset(NumpyDatasetBase, Dataset[Dict[str, Any]]):
    """
    A variable sequence length (VSL) numpy array-backed dataset. This is used to inject a sequence
    length-based curriculum during training as introduced in
    `Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum
    <https://arxiv.org/pdf/2405.13226>`_.

    This dataset creates instances of token IDs with lengths that are powers of 2
    between ``min_sequence_length`` (which must be a power of 2) and ``max_sequence_length``
    (also a power a 2). Some tokens will be discarded unless ``min_sequence_length`` is 1.

    .. important::
        No special tokens are added to the input IDs so it's assumed that if you want
        EOS tokens between documents, for example, those will already be in the array.

    :param paths: Paths or URLs to numpy token ID arrays.
    :param pad_token_id: The ID of the padding token.
    :param eos_token_id: The ID of the EOS token.
    :param max_sequence_length: The maximum allowed sequence length. A power of 2, e.g. '4096'.
    :param min_sequence_length: The minimum allowed sequence length. A power of 2, e.g. '256'.
    :param curriculum: The variable sequence length curriculum. Determines the sampling
        probability of batches from each bucket throughout training.
    :param dtype: The numpy datatype of the arrays.
    :param metadata: Metadata to add to each item. This should be a dictionary or a list of dictionaries
        with the same number of items as there are paths.
    :param include_instance_metadata: If ``True`` (the default), each instance returned from
        :meth:`__getitem__()` will include the metadata from its source.
    """

    def __init__(
        self,
        *paths: PathOrStr,
        pad_token_id: int,
        eos_token_id: int,
        max_sequence_length: int,
        min_sequence_length: int = 256,
        curriculum: Optional[VSLCurriculum] = None,
        dtype: Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]] = np.uint16,
        metadata: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        include_instance_metadata: Optional[bool] = None,
    ):
        if math.log(max_sequence_length, 2) % 1 != 0:
            raise OLMoConfigurationError("'max_sequence_length' must be a power of 2")

        if math.log(min_sequence_length, 2) % 1 != 0:
            raise OLMoConfigurationError("'min_sequence_length' must be a power of 2")

        if max_sequence_length <= min_sequence_length:
            raise OLMoConfigurationError(
                "'max_sequence_length' should be bigger than 'min_sequence_length'"
            )

        if include_instance_metadata is None and metadata:
            include_instance_metadata = True

        if isinstance(metadata, list):
            if len(metadata) != len(paths):
                raise OLMoConfigurationError(
                    "'metadata' should have the same length as the number of file paths"
                )
        else:
            metadata = [metadata or {}] * len(paths)

        super().__init__(*paths, pad_token_id=pad_token_id, eos_token_id=eos_token_id, dtype=dtype)
        self._metadata = metadata
        self._include_instance_metadata = include_instance_metadata
        self._max_sequence_length = max_sequence_length
        self._min_sequence_length = min_sequence_length
        self._curriculum = curriculum or VSLNaturalCurriculum()
        self._num_instances: Optional[int] = None
        self._array_offsets: Optional[Tuple[Tuple[int, int], ...]] = None
        self._lengths_dtype: Optional[
            Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]]
        ] = None
        self._instances_per_bucket: Optional[Tuple[Tuple[int, int], ...]] = None

    @property
    def fingerprint_version(self) -> str:
        return "v1.1"

    @property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            f"min_sequence_length={self.min_sequence_length},"
            f"max_sequence_length={self.max_sequence_length},"
            f"pad_token_id={self.pad_token_id},"
            f"eos_token_id={self.eos_token_id},"
            f"dtype={self.dtype}".encode()
        )
        for path, size in zip(self.paths, self.file_sizes):
            sha256_hash.update(f"name={os.path.basename(path)},size={size}".encode())
        return sha256_hash.hexdigest()

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    @property
    def min_sequence_length(self) -> int:
        return self._min_sequence_length

    @property
    def curriculum(self) -> VSLCurriculum:
        return self._curriculum

    @property
    def all_sequence_lengths(self) -> List[int]:
        min_exp = int(math.log(self.min_sequence_length, 2))
        max_exp = int(math.log(self.max_sequence_length, 2))
        return [2**exp for exp in range(min_exp, max_exp + 1)]

    @property
    def offsets(self) -> Tuple[Tuple[int, int], ...]:
        """
        Gives the global start and end instance indices for each data file in the dataset.
        """
        if self._array_offsets is None:
            array_offsets = []
            item_size = self.indices_dtype(0).itemsize
            start_offset = 0
            for path in self.paths:
                doc_indices_path = self._get_document_indices_path(path)
                instances_in_file = (get_file_size(doc_indices_path) // item_size) // 2
                end_offset = start_offset + instances_in_file
                array_offsets.append((start_offset, end_offset))
                start_offset += instances_in_file
            self._array_offsets = tuple(array_offsets)
        return self._array_offsets

    def prepare(self):
        if self.fs_local_rank == 0:
            log.info("Gathering dataset document indices and buckets...")
            self._write_document_indices()
            self._write_instance_lengths()
            self._write_instance_buckets(self.get_instance_lengths())
        barrier()
        len(self)

    def __len__(self):
        if self._num_instances is None:
            self._num_instances = self.offsets[-1][1]
        return self._num_instances

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)  # in case this is a numpy int type.
        pos_index = index if index >= 0 else len(self) + index

        # The index of the array within 'self.paths'.
        array_index: Optional[int] = None
        # The index within the corresponding array.
        array_local_index: Optional[int] = None
        for i, (offset_start, offset_end) in enumerate(self.offsets):
            if offset_start <= pos_index < offset_end:
                array_index = i
                array_local_index = pos_index - offset_start
                break

        if array_index is None or array_local_index is None:
            raise IndexError(f"{index} is out of bounds for dataset of size {len(self)}")

        # Read the data from file.
        input_ids = self._read_chunk_from_array(self.paths[array_index], array_local_index)
        out: Dict[str, Any] = {"input_ids": input_ids}

        if self._include_instance_metadata:
            metadata = self._metadata[array_index]
            out["metadata"] = deepcopy(metadata)

        return out

    def _read_chunk_from_array(self, path: PathOrStr, index: int) -> torch.Tensor:
        indices_path = self._get_document_indices_path(path)
        indices = load_array_slice_into_tensor(
            indices_path, index * 2, index * 2 + 2, self.indices_dtype
        )
        start_idx, end_idx = indices
        data = load_array_slice_into_tensor(path, int(start_idx), int(end_idx), self.dtype)
        return data

    def _get_document_indices_path(self, path: PathOrStr) -> Path:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(str(path).encode())
        sha256_hash.update(str(self._get_file_size(path)).encode())
        for seq_len in self.all_sequence_lengths:
            sha256_hash.update(str(seq_len).encode())
        path_hash = sha256_hash.hexdigest()
        return self.work_dir / "dataset-common" / f"bucketed-doc-indices-{path_hash}.npy"

    def _get_instance_lengths_path(self) -> Path:
        return self.work_dir / f"dataset-{self.fingerprint}" / "instance-lengths.npy"

    def _get_instance_bucket_path(self, seq_len: int) -> Path:
        return self.work_dir / f"dataset-{self.fingerprint}" / f"bucket{seq_len}-indices.npy"

    def _write_document_indices(self):
        paths_needed: List[PathOrStr] = []
        for path in self.paths:
            indices_path = self._get_document_indices_path(path)
            if indices_path.is_file():
                log.info(f"Reusing document indices for '{path}' at:\n'{indices_path}'")
            elif path not in paths_needed:
                paths_needed.append(path)

        if paths_needed:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = []
                for path in paths_needed:
                    indices_path = self._get_document_indices_path(path)
                    log.info(f"Gathering document indices for '{path}'...")
                    future = executor.submit(
                        bucket_documents,
                        path,
                        indices_path,
                        buckets=self.all_sequence_lengths,
                        eos_token_id=self.eos_token_id,
                        dtype=self.dtype,
                        indices_dtype=self.indices_dtype,
                    )
                    futures.append(future)

                concurrent.futures.wait(futures, return_when="ALL_COMPLETED")

                # Log results.
                for path, future in zip(paths_needed, futures):
                    total_og_docs, total_bucketed_docs = future.result()
                    log.info(
                        f"Created {total_bucketed_docs:,d} bucketed documents by sequence length from "
                        f"{total_og_docs:,d} original documents in '{path}'"
                    )

    def _write_instance_lengths(self):
        instance_lengths_path = self._get_instance_lengths_path()
        if instance_lengths_path.is_file():
            log.info(f"Reusing all instance lengths at:\n'{instance_lengths_path}'")
        else:
            log.info(f"Gathering all instance lengths to:\n'{instance_lengths_path}...")
            with memmap_to_write(
                instance_lengths_path, dtype=self.lengths_dtype, shape=(len(self),)
            ) as instance_lengths:
                for path, (offset_start, offset_end) in zip(self.paths, self.offsets):
                    indices_path = self._get_document_indices_path(path)
                    indices_mmap = np.memmap(indices_path, dtype=self.indices_dtype, mode="r")
                    instance_lengths[offset_start:offset_end] = get_doc_lengths_from_indices(
                        indices_mmap
                    )
                    del indices_mmap

    def _write_instance_buckets(self, instance_lengths: np.ndarray):
        for seq_len in self.all_sequence_lengths:
            bucket_path = self._get_instance_bucket_path(seq_len)
            if bucket_path.is_file():
                log.info(
                    f"Reusing instance indices for seq len {seq_len} bucket at:\n'{bucket_path}'"
                )
            else:
                log.info(f"Gathering instance indices for seq len {seq_len} bucket...")
                bucket_path.parent.mkdir(exist_ok=True, parents=True)
                instance_indices = (instance_lengths == seq_len).nonzero()[0]
                with memmap_to_write(
                    bucket_path,
                    dtype=self.indices_dtype,
                    shape=instance_indices.shape,
                ) as bucket:
                    bucket[:] = instance_indices
                log.info(
                    f"Instance indices for seq len {seq_len} bucket written to:\n'{bucket_path}'"
                )

    def get_instance_lengths(self) -> np.ndarray:
        """
        Get a numpy memory-mapped array with the length of every instance in the dataset.
        """
        return np.memmap(self._get_instance_lengths_path(), dtype=self.lengths_dtype, mode="r")

    def get_instance_bucket(self, seq_len: int) -> np.ndarray:
        """
        Get the instance indices in a bucket.
        """
        return np.memmap(
            self._get_instance_bucket_path(seq_len), dtype=self.indices_dtype, mode="r"
        )

    def get_instance_buckets(self) -> List[Tuple[int, np.ndarray]]:
        """
        Get the buckets of instance indices that all have the same length.
        The buckets will be sorted from smallest sequence length to longest.
        """
        buckets = []
        for seq_len in self.all_sequence_lengths:
            buckets.append((seq_len, self.get_instance_bucket(seq_len)))
        return buckets

    @property
    def instances_per_bucket(self) -> Tuple[Tuple[int, int], ...]:
        """
        The number of instances in each bucket.
        """
        if self._instances_per_bucket is None:
            instances_per_bucket = []
            item_size = self.indices_dtype(0).itemsize
            for seq_len in self.all_sequence_lengths:
                instances_per_bucket.append(
                    (seq_len, get_file_size(self._get_instance_bucket_path(seq_len)) // item_size)
                )
            self._instances_per_bucket = tuple(instances_per_bucket)
        return self._instances_per_bucket

    @property
    def indices_dtype(
        self,
    ) -> Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]]:
        return np.uint32

    @property
    def lengths_dtype(
        self,
    ) -> Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]]:
        if self._lengths_dtype is None:
            for dtype in (
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ):
                if (self.max_sequence_length - 1) <= np.iinfo(dtype).max:
                    self._lengths_dtype = dtype
                    break
            assert self._lengths_dtype is not None
        return self._lengths_dtype


class NumpyDatasetType(StrEnum):
    """
    An enumeration of the different :class:`NumpyDatasetBase` implementations.
    """

    fsl = "fsl"
    """
    Fixed sequenced length ➡️ :class:`NumpyFSLDataset`.
    """

    padded_fsl = "padded_fsl"
    """
    Padded fixed sequence length ➡️ :class:`NumpyPaddedFSLDataset`.
    """

    vsl = "vsl"
    """
    Variable sequenced length ➡️ :class:`NumpyVSLDataset`.
    """


class NumpyDatasetDType(StrEnum):
    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"

    def as_np_dtype(
        self,
    ) -> Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]]:
        return getattr(np, str(self))


class VSLCurriculumType(StrEnum):
    """
    An enumeration of the different VSL curriculum implementations.
    """

    natural = "natural"
    """
    The natural curriculum ➡️ :class:`VSLNaturalCurriculum`.
    """

    grow_p2 = "grow_p2"
    """
    The "Grow-P2" curriculum ➡️ :class:`VSLGrowP2Curriculum`.
    """

    grow_linear = "grow_linear"
    """
    The "Grow-Linear" curriculum ➡️ :class:`VSLGrowLinearCurriculum`.
    """


@dataclass
class VSLCurriculumConfig(Config):
    name: VSLCurriculumType = VSLCurriculumType.natural
    num_cycles: Optional[int] = None
    balanced: Optional[bool] = None

    def build(self) -> VSLCurriculum:
        """
        Build the VSL curriculum.
        """
        if self.name == VSLCurriculumType.natural:
            if self.num_cycles is not None:
                raise OLMoConfigurationError(
                    f"'num_cycles' is not a valid field for the {self.name} curriculum"
                )
            elif self.balanced is not None:
                raise OLMoConfigurationError(
                    f"'balanced' is not a valid field for the {self.name} curriculum"
                )
            return VSLNaturalCurriculum()
        elif self.name == VSLCurriculumType.grow_p2:
            if self.num_cycles is None:
                raise OLMoConfigurationError(
                    f"'num_cycles' is required for the {self.name} curriculum"
                )
            elif self.balanced is None:
                raise OLMoConfigurationError(
                    f"'balanced' is required for the {self.name} curriculum"
                )
            return VSLGrowP2Curriculum(num_cycles=self.num_cycles, balanced=self.balanced)
        elif self.name == VSLCurriculumType.grow_linear:
            if self.num_cycles is None:
                raise OLMoConfigurationError(
                    f"'num_cycles' is required for the {self.name} curriculum"
                )
            elif self.balanced is None:
                raise OLMoConfigurationError(
                    f"'balanced' is required for the {self.name} curriculum"
                )
            return VSLGrowLinearCurriculum(num_cycles=self.num_cycles, balanced=self.balanced)
        else:
            raise NotImplementedError(self.name)


@dataclass
class NumpyDatasetConfig(Config):
    """
    A config class for easily building :class:`NumpyDatasetBase` classes.
    """

    tokenizer: TokenizerConfig
    """
    The tokenizer config.
    """
    name: NumpyDatasetType = NumpyDatasetType.fsl
    """
    The type of dataset.
    """
    sequence_length: Optional[int] = None
    """
    The sequence length for a :class:`NumpyFSLDataset`.
    """
    max_target_sequence_length: Optional[int] = None
    """
    The max target sequene length for a :class:`NumpyFSLDataset`.
    """
    max_sequence_length: Optional[int] = None
    """
    The max sequence length for a :class:`NumpyVSLDataset`.
    """
    min_sequence_length: Optional[int] = None
    """
    The min sequence length for a :class:`NumpyVSLDataset`.
    """
    vsl_curriculum: Optional[VSLCurriculumConfig] = None
    """
    The variable sequence length (VSL) curriculum for a :class:`NumpyVSLDataset`.
    """
    paths: Optional[List[str]] = None
    """
    The paths/URLs to the numpy token ID arrays.
    """
    mix: Optional[DataMix] = None
    """
    The name of a data mix.
    """
    mix_base_dir: Optional[str] = None
    """
    The base directory for the data mix.
    """
    dtype: Optional[NumpyDatasetDType] = None
    """
    The numpy datatype of the token ID arrays.
    """
    metadata: Optional[List[Dict[str, Any]]] = None
    """
    Metadata for the numpy arrays.
    """
    include_instance_metadata: bool = True
    """
    Whether or not to include the :data:`metadata` in the instances returned from
    :meth:`NumpyDatasetBase.__getitem__()`.
    """
    generate_doc_lengths: bool = False
    """
    Include individual document lengths in the instances returned from
    :meth:`NumpyDatasetBase.__getitem__()`.
    """
    expand_glob: bool = False
    """
    Treat the :data:`paths` as globs.
    """
    work_dir: Optional[str] = None
    """
    The dataset working directory. This is used to cache working files like shuffled indices,
    instance buckets, etc.

    .. tip::
        You can save a lot of time and disk space by setting this to a common directory across
        all of you runs.
    """

    @property
    def effective_sequence_length(self) -> int:
        if self.sequence_length is not None:
            return self.sequence_length
        elif self.max_sequence_length is not None:
            return self.max_sequence_length
        else:
            raise ValueError("missing 'sequence_length' or 'max_sequence_length'")

    @classmethod
    def glob(cls, *glob_paths: str, **kwargs) -> NumpyDatasetConfig:
        """
        Initialize a dataset config with glob paths.

        .. note::
            Globs are not expanded until :meth:`build()` is called.
            If any of the globs don't expand to any matches a :class:`FileNotFoundError`
            error is raised

        :returns: A new dataset config.
        """
        return cls(paths=list(glob_paths), expand_glob=True, **kwargs)

    @classmethod
    def from_data_mix(
        cls, mix: DataMix, *, tokenizer: TokenizerConfig, **kwargs
    ) -> "NumpyDatasetConfig":
        """
        Initialize a dataset config from an official data mix.

        :param mix: The data mix.
        :param tokenizer: The tokenizer config.

        :returns: A new dataset config.
        """
        if tokenizer.identifier is None:
            raise OLMoConfigurationError(
                "Missing tokenizer identifier required to construct data mix"
            )
        return cls(mix=mix, tokenizer=tokenizer, **kwargs)

    def get_dtype(
        self,
    ) -> Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]]:
        if self.dtype is not None:
            return NumpyDatasetDType(self.dtype).as_np_dtype()

        # Guess based on vocab size.
        for dtype in (
            NumpyDatasetDType.uint8,
            NumpyDatasetDType.uint16,
            NumpyDatasetDType.uint32,
            NumpyDatasetDType.uint64,
        ):
            if (self.tokenizer.vocab_size - 1) <= np.iinfo(dtype.as_np_dtype()).max:
                log.info(f"Assuming dtype '{dtype}' based on vocab size")
                return dtype.as_np_dtype()

        raise ValueError("vocab size too big!")

    def build(self) -> NumpyDatasetBase:
        """
        Construct the corresponding :class:`NumpyDatasetBase`.
        """
        if (self.paths is None) == (self.mix is None):
            raise OLMoConfigurationError("Exactly one of 'paths' or 'mix' is required")

        paths: List[str] = []
        if self.paths and self.expand_glob:
            from glob import glob

            for glob_path in self.paths:
                log.info(f"Expanding '{glob_path}'...")
                matches = sorted(glob(glob_path))
                if not matches:
                    raise FileNotFoundError(glob_path)
                for path in matches:
                    log.info(f" - '{path}'")
                paths.extend(matches)
        elif self.paths:
            paths = self.paths
        else:
            assert self.mix is not None
            if self.mix_base_dir is None:
                raise OLMoConfigurationError(
                    "'mix_base_dir' is required to build a dataset from a mix"
                )
            if self.tokenizer.identifier is None:
                raise OLMoConfigurationError(
                    "Missing tokenizer identifier required to construct data mix"
                )
            paths = self.mix.build(self.mix_base_dir, self.tokenizer.identifier)

        dataset: NumpyDatasetBase
        if self.name == NumpyDatasetType.fsl:
            if self.sequence_length is None:
                raise OLMoConfigurationError("'sequence_length' is required for FSL dataset")
            if self.max_sequence_length is not None:
                if self.max_target_sequence_length is None:
                    raise OLMoConfigurationError(
                        "'max_sequence_length' is only a valid field for VSL datasets, "
                        "did you mean to set 'max_target_sequence_length' instead?"
                    )
                else:
                    raise OLMoConfigurationError(
                        "'max_sequence_length' is only a valid field for VSL datasets"
                    )
            if self.min_sequence_length is not None:
                raise OLMoConfigurationError(
                    "'min_sequence_length' is only a valid field for VSL datasets"
                )
            if self.vsl_curriculum is not None:
                raise OLMoConfigurationError(
                    "'vsl_curriculum' is only a valid field for VSL datasets"
                )
            dataset = NumpyFSLDataset(
                *paths,
                sequence_length=self.sequence_length,
                max_target_sequence_length=self.max_target_sequence_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                dtype=self.get_dtype(),
                metadata=self.metadata,
                include_instance_metadata=self.include_instance_metadata,
                generate_doc_lengths=self.generate_doc_lengths,
            )
        elif self.name == NumpyDatasetType.padded_fsl:
            if self.sequence_length is None:
                raise OLMoConfigurationError("'sequence_length' is required for padded FSL dataset")
            if self.max_target_sequence_length is not None:
                raise OLMoConfigurationError(
                    "'max_target_sequence_length' is only valid for the (non-padded) FSL dataset"
                )
            if self.generate_doc_lengths is not None:
                raise OLMoConfigurationError(
                    "'generate_doc_lengths' is only valid for the (non-padded) FSL dataset"
                )
            if self.max_sequence_length is not None:
                if self.max_target_sequence_length is None:
                    raise OLMoConfigurationError(
                        "'max_sequence_length' is only a valid field for VSL datasets, "
                        "did you mean to set 'max_target_sequence_length' instead?"
                    )
                else:
                    raise OLMoConfigurationError(
                        "'max_sequence_length' is only a valid field for VSL datasets"
                    )
            if self.min_sequence_length is not None:
                raise OLMoConfigurationError(
                    "'min_sequence_length' is only a valid field for VSL datasets"
                )
            if self.vsl_curriculum is not None:
                raise OLMoConfigurationError(
                    "'vsl_curriculum' is only a valid field for VSL datasets"
                )
            dataset = NumpyPaddedFSLDataset(
                *paths,
                sequence_length=self.sequence_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                dtype=self.get_dtype(),
                metadata=self.metadata,
                include_instance_metadata=self.include_instance_metadata,
            )
        elif self.name == NumpyDatasetType.vsl:
            if self.max_sequence_length is None:
                raise OLMoConfigurationError("'max_sequence_length' is required for VSL datasets")
            if self.min_sequence_length is None:
                raise OLMoConfigurationError("'min_sequence_length' is required for VSL datasets")
            if self.sequence_length is not None:
                raise OLMoConfigurationError(
                    "'sequence_length' is only a valid field for FSL datasets"
                )
            if self.generate_doc_lengths:
                raise OLMoConfigurationError(
                    "'generate_doc_lengths' is only valid for FSL datasets"
                )
            dataset = NumpyVSLDataset(
                *paths,
                max_sequence_length=self.max_sequence_length,
                min_sequence_length=self.min_sequence_length,
                curriculum=None if self.vsl_curriculum is None else self.vsl_curriculum.build(),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                dtype=self.get_dtype(),
                metadata=self.metadata,
                include_instance_metadata=self.include_instance_metadata,
            )
        else:
            raise NotImplementedError(self.name)

        if self.work_dir is not None:
            dataset.work_dir = Path(self.work_dir)

        return dataset
