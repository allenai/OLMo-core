from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import math
import tempfile
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from olmo_core.exceptions import OLMoConfigurationError, OLMoEnvironmentError

from ..aliases import PathOrStr
from ..config import Config, StrEnum
from ..distributed.utils import barrier, get_fs_local_rank
from ..io import _get_s3_client, get_file_size
from ..utils import capped_powers_of_2
from .mixes import DataMix
from .tokenizer import TokenizerConfig
from .utils import get_document_lengths, iter_document_indices, read_chunk_from_array

__all__ = [
    "NumpyDatasetBase",
    "NumpyFSLDatasetConfig",
    "NumpyFSLDataset",
    "NumpyDatasetDType",
    "NumpyVSLDataset",
]


log = logging.getLogger(__name__)


T = TypeVar("T")


class NumpyDatasetDType(StrEnum):
    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"

    def as_np_dtype(
        self,
    ) -> Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]]:
        return getattr(np, str(self))


class NumpyDatasetBase(ABC):
    """
    A base class for datasets backed from numpy arrays of token IDs.

    .. warning::
        When using subclasses in a distributed setting be sure that the :data:`work_dir` is shared
        among all local ranks. Then you should then call :meth:`prepare()` in the main process
        before doing anything else.
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
        self._array_file_sizes: Optional[Tuple[int, ...]] = None

    @property
    def paths(self) -> Tuple[PathOrStr, ...]:
        """
        Paths to the numpy arrays.
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
        return "v1"

    @property
    def fingerprint(self) -> str:
        """
        Can be used to identify/compare the contents of a dataset.
        """
        sha256_hash = hashlib.sha256()
        for size in self.file_sizes:
            sha256_hash.update(f"size={size}".encode())
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
    def work_dir(self, work_dir: Path):
        self._work_dir = work_dir

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

    def map(self, func: Callable[[PathOrStr], T], max_workers: Optional[int] = None) -> List[T]:
        """
        Call a function on each path in the dataset, returning a list of the results, in order.
        """
        self._warmup_clients()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            path_to_future = {}
            for path in self.paths:
                if path not in path_to_future:
                    path_to_future[path] = executor.submit(func, path)

            results = []
            for path in self.paths:
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
        Get an instance from the dataset.
        """
        raise NotImplementedError

    def state_dict(self, instances_processed: int) -> Dict[str, Any]:
        """
        Given the number of instances processed so far (e.g. during training), get a state dict
        for checkpointing.
        """
        return {
            "fingerprint_version": self.fingerprint_version,
            "fingerprint": self.fingerprint,
            "instances_processed": instances_processed,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> int:
        """
        Given a state dict from :meth:`state_dict()`, return the number of instances processed
        so far (e.g. during training).
        """
        if state_dict["fingerprint_version"] != self.fingerprint_version:
            log.warning(
                "Dataset fingerprint version does not match the version in the checkpoint, "
                "this could mean the data has changed"
            )
        elif state_dict["fingerprint"] != self.fingerprint:
            raise RuntimeError(
                "Restoring state from a different dataset is not supported! (fingerprint doesn't match)"
            )
        return state_dict["instances_processed"]


@dataclass
class NumpyFSLDatasetConfig(Config):
    """
    A config class for easily building :class:`NumpyFSLDataset` classes.
    """

    sequence_length: int
    tokenizer: TokenizerConfig
    paths: Optional[List[str]] = None
    mix: Optional[DataMix] = None
    mix_base_dir: Optional[str] = None
    max_target_sequence_length: Optional[int] = None
    dtype: Optional[NumpyDatasetDType] = None
    metadata: Optional[List[Dict[str, Any]]] = None
    include_instance_metadata: bool = True
    generate_doc_lengths: bool = False
    expand_glob: bool = False

    @classmethod
    def glob(cls, *glob_paths: str, **kwargs) -> NumpyFSLDatasetConfig:
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
    ) -> "NumpyFSLDatasetConfig":
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

    def build(self) -> NumpyFSLDataset:
        """
        Construct the corresponding :class:`NumpyFSLDataset`.
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

        return NumpyFSLDataset(
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


class NumpyFSLDataset(NumpyDatasetBase, Dataset[Dict[str, Any]]):
    """
    A fixed sequence length (FSL) numpy array-backed dataset.

    Token IDs from all arrays are concatenated together and then chunked into contiguous blocks of
    ``sequence_length`` to create instances.

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
        self._metadata = metadata
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

    def state_dict(self, instances_processed: int) -> Dict[str, Any]:
        state_dict = super().state_dict(instances_processed)
        state_dict["sequence_length"] = self.sequence_length
        state_dict["max_target_sequence_length"] = self.max_target_sequence_length
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> int:
        instances_processed = super().load_state_dict(state_dict)

        if state_dict["max_target_sequence_length"] != self.max_target_sequence_length:
            raise RuntimeError(
                "Restoring dataset state with a different 'max_target_sequence_length' is not supported"
            )

        if self.sequence_length == state_dict["sequence_length"]:
            return instances_processed
        elif self.sequence_length % state_dict["sequence_length"] == 0:
            # Adjust for current larger sequence length.
            ratio = self.sequence_length // state_dict["sequence_length"]
            return instances_processed // ratio
        elif state_dict["sequence_length"] % self.sequence_length == 0:
            # Adjust for current smaller sequence length.
            ratio = state_dict["sequence_length"] // self.sequence_length
            return instances_processed * ratio
        else:
            raise RuntimeError(
                "You can only restore dataset state from a sequence length that is a multiple "
                "of the current configured sequence length, or vice versa."
            )

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

    def _read_chunk_from_array(self, path: PathOrStr, index: int, dtype=None) -> torch.Tensor:
        dtype = dtype or self.dtype
        start_idx = index * self.sequence_length
        return read_chunk_from_array(path, start_idx, start_idx + self.sequence_length, dtype)

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


class NumpyVSLDataset(NumpyDatasetBase, Dataset[Dict[str, Any]]):
    """
    A variable sequence length (VSL) numpy array-backed dataset. Use with
    :class:`IterableVSLDataset` to implement a sequence length-based curriculum during training as
    introduced in `Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum
    <https://arxiv.org/pdf/2405.13226>`_.

    This dataset creates instances of token IDs with lengths that are powers of 2
    between ``min_sequence_length`` (which must be a power of 2) and ``max_sequence_length``
    (also a power a 2). Some tokens will be discarded unless ``min_sequence_length`` is 1.

    :param paths: Paths or URLs to numpy token ID arrays.
    :param pad_token_id: The ID of the padding token.
    :param eos_token_id: The ID of the EOS token.
    :param max_sequence_length: The maximum allowed sequence length. A power of 2, e.g. '4096'.
    :param min_sequence_length: The minimum allowed sequence length. A power of 2, e.g. '64'.
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
        self._num_instances: Optional[int] = None
        self._array_offsets: Optional[Tuple[Tuple[int, int], ...]] = None
        self._lengths_dtype: Optional[
            Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]]
        ] = None
        self._instances_per_bucket: Optional[Tuple[Tuple[int, int], ...]] = None

    @property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(f"min_sequence_length={self.min_sequence_length}".encode())
        sha256_hash.update(f"max_sequence_length={self.max_sequence_length}".encode())
        for size in self.file_sizes:
            sha256_hash.update(f"size={size}".encode())
        return sha256_hash.hexdigest()

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    @property
    def min_sequence_length(self) -> int:
        return self._min_sequence_length

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
            self.map(self._write_document_indices, max_workers=8)
            instance_lengths = self._write_instance_lengths()
            self._write_instance_buckets(instance_lengths)
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
        indices = read_chunk_from_array(indices_path, index * 2, index * 2 + 2, self.indices_dtype)
        start_idx, end_idx = indices
        data = read_chunk_from_array(path, int(start_idx), int(end_idx), self.dtype)
        return data

    def _get_document_indices_path(self, path: PathOrStr) -> Path:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(str(path).encode())
        path_hash = sha256_hash.hexdigest()
        return self.work_dir / f"dataset-{self.fingerprint}" / f"{path_hash}.npy"

    def _get_instance_lengths_path(self) -> Path:
        return self.work_dir / f"dataset-{self.fingerprint}" / "instance-lengths.npy"

    def _get_instance_bucket_path(self, seq_len: int) -> Path:
        return self.work_dir / f"dataset-{self.fingerprint}" / f"bucket{seq_len}-indices.npy"

    def _write_document_indices(self, path: PathOrStr) -> Path:
        indices_path = self._get_document_indices_path(path)
        if not indices_path.is_file():
            indices_path.parent.mkdir(exist_ok=True, parents=True)
            indices = []
            for start_idx, end_idx in iter_document_indices(path):
                bin_decomp = capped_powers_of_2(end_idx - start_idx, self.max_sequence_length)
                for x in bin_decomp:
                    if x < self.min_sequence_length:
                        break
                    indices.append(start_idx)
                    indices.append(start_idx + x)
                    start_idx += x

            indices_mmap = np.memmap(
                indices_path, dtype=self.indices_dtype, mode="w+", shape=(len(indices),)
            )
            indices_mmap[:] = indices
            indices_mmap.flush()
            del indices_mmap
        return indices_path

    def _write_instance_lengths(self) -> np.ndarray:
        instance_lengths_path = self._get_instance_lengths_path()
        if not instance_lengths_path.is_file():
            instance_lengths_path.parent.mkdir(exist_ok=True, parents=True)
            instance_lengths = np.memmap(
                instance_lengths_path,
                dtype=self.lengths_dtype,
                mode="w+",
                shape=(len(self),),
            )
            for path, (offset_start, offset_end) in zip(self.paths, self.offsets):
                indices_path = self._get_document_indices_path(path)
                indices_mmap = np.memmap(indices_path, dtype=self.indices_dtype, mode="r")
                instance_lengths[offset_start:offset_end] = indices_mmap[1::2] - indices_mmap[::2]
                del indices_mmap
            instance_lengths.flush()
            return instance_lengths
        else:
            return self.get_instance_lengths()

    def _write_instance_buckets(self, instance_lengths: np.ndarray):
        for seq_len in self.all_sequence_lengths:
            bucket_path = self._get_instance_bucket_path(seq_len)
            if not bucket_path.is_file():
                bucket_path.parent.mkdir(exist_ok=True, parents=True)
                instance_indices = (instance_lengths == seq_len).nonzero()[0]
                bucket = np.memmap(
                    bucket_path,
                    dtype=self.indices_dtype,
                    mode="w+",
                    shape=instance_indices.shape,
                )
                bucket[:] = instance_indices

    def get_instance_lengths(self) -> np.ndarray:
        """
        Get a numpy memory-mapped array with the length of every instance in the dataset.
        """
        return np.memmap(self._get_instance_lengths_path(), dtype=self.lengths_dtype, mode="r")

    def get_instance_buckets(self) -> List[Tuple[int, np.ndarray]]:
        """
        Get the buckets of instance indices that all have the same length.
        The buckets will be sorted from smallest sequence length to longest.
        """
        buckets = []
        for seq_len in self.all_sequence_lengths:
            buckets.append(
                (
                    seq_len,
                    np.memmap(
                        self._get_instance_bucket_path(seq_len), dtype=self.indices_dtype, mode="r"
                    ),
                )
            )
        return buckets

    @property
    def instances_per_bucket(self) -> Tuple[Tuple[int, int], ...]:
        """
        The number of instances in each bucket of equal sequence length instances.
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
