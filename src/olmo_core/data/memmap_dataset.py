from __future__ import annotations

import hashlib
import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from olmo_core.exceptions import OLMoConfigurationError, OLMoEnvironmentError

from ..aliases import PathOrStr
from ..config import Config, StrEnum
from ..io import _get_s3_client, file_size, get_bytes_range
from ..utils import get_document_lengths
from .mixes import DataMix
from .tokenizer import TokenizerConfig

__all__ = ["MemMapDatasetConfig", "MemMapDataset"]


log = logging.getLogger(__name__)


class MemMapDType(StrEnum):
    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"

    def as_np_dtype(
        self,
    ) -> Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]]:
        return getattr(np, str(self))


@dataclass
class MemMapDatasetConfig(Config):
    """
    A config class for easily building :class:`MemMapDataset` classes.
    """

    sequence_length: int
    tokenizer: TokenizerConfig
    paths: Optional[List[str]] = None
    mix: Optional[DataMix] = None
    mix_base_dir: Optional[str] = None
    memmap_dtype: Optional[MemMapDType] = None
    metadata: Optional[List[Dict[str, Any]]] = None
    include_instance_metadata: bool = True
    generate_attention_mask: bool = False
    generate_doc_lengths: bool = False
    label_mask_paths: Optional[List[str]] = None
    expand_glob: bool = False

    @classmethod
    def glob(cls, *glob_paths: str, **kwargs) -> "MemMapDatasetConfig":
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
    ) -> "MemMapDatasetConfig":
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

    def get_memmap_dtype(
        self,
    ) -> Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]]:
        if self.memmap_dtype is not None:
            return MemMapDType(self.memmap_dtype).as_np_dtype()

        # Guess based on vocab size.
        for dtype in (
            MemMapDType.uint8,
            MemMapDType.uint16,
            MemMapDType.uint32,
            MemMapDType.uint64,
        ):
            if (self.tokenizer.vocab_size - 1) <= np.iinfo(dtype.as_np_dtype()).max:
                log.info(f"Assuming memmap dtype {dtype} based on vocab size")
                return dtype.as_np_dtype()

        raise ValueError("vocab size too big!")

    def build(self) -> MemMapDataset:
        """
        Construct the corresponding :class:`MemMapDataset`.
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

        dataset = MemMapDataset(
            *paths,
            sequence_length=self.sequence_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            memmap_dtype=self.get_memmap_dtype(),
            metadata=self.metadata,
            include_instance_metadata=self.include_instance_metadata,
            generate_attention_mask=self.generate_attention_mask,
            generate_doc_lengths=self.generate_doc_lengths,
            label_mask_paths=cast(Optional[List[PathOrStr]], self.label_mask_paths),
        )
        log.info(f"Built dataset with {len(dataset):,d} examples of length {self.sequence_length}")

        return dataset


class MemMapDataset(Dataset[Dict[str, Any]]):
    """
    A PyTorch :class:`~torch.utils.data.Dataset` backed by one or more numpy memory-mapped arrays
    of token IDs. Token IDs are chunked together into contiguous blocks of ``sequence_length``
    to create instances.

    .. important::
        If the length of a memory-mapped array is not a multiple of ``sequence_length`` the
        remainder of the tokens will be ignored.

    .. important::
        No special tokens are added to the input IDs so it's assumed that if you want
        EOS tokens between documents, for example, those will already be in the memory-mapped array.

    :param paths: Paths to memory-mapped token arrays.
    :param sequence_length: The number of tokens to chunk together into a single instance.
        Generally this should correspond to your model's maximum input length.
    :param pad_token_id: The ID of the padding token.
    :param eos_token_id: The ID of the EOS token.
    :param memmap_dtype: The numpy datatype of the memory-mapped array.
    :param metadata: Metadata to add to each item. This should be a dictionary or a list of dictionaries
        with the same number of items as there are paths.
    :param include_instance_metadata: If ``True`` (the default), each instance returned from ``__getitem__`` will
        include the metadata from its source.
    :param generate_attention_mask: If ``True``, each instance returned from ``__getitem__`` will include an
        attention mask generated by masking each padding token.
    :param label_mask_paths: Optional paths to ``np.bool_`` memory-mapped arrays of label masks.
    """

    def __init__(
        self,
        *paths: PathOrStr,
        sequence_length: int,
        pad_token_id: int,
        eos_token_id: int,
        memmap_dtype: Union[
            Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]
        ] = np.uint16,
        metadata: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        include_instance_metadata: bool = True,
        generate_attention_mask: bool = False,
        generate_doc_lengths: bool = False,
        label_mask_paths: Optional[List[PathOrStr]] = None,
    ):
        if not paths:
            raise ValueError("At least one path is required")

        if label_mask_paths and len(label_mask_paths) != len(paths):
            raise ValueError(
                "There must be the same number of 'label_mask_paths' as there are 'paths'"
            )

        if isinstance(metadata, list):
            if len(metadata) != len(paths):
                raise ValueError(
                    "'metadata' should have the same length as the number of file paths"
                )
        else:
            metadata = [metadata or {}] * len(paths)

        self._memmap_dtype = memmap_dtype
        self._memmap_paths = paths
        self._metadata = metadata
        self._label_mask_paths = label_mask_paths
        self._sequence_length = sequence_length
        self._mmap_offsets: Optional[List[Tuple[int, int]]] = None
        self._num_instances: Optional[int] = None
        self._include_instance_metadata = include_instance_metadata
        self._generate_attention_mask = generate_attention_mask
        self._generate_doc_lengths = generate_doc_lengths
        self._pad_token_id = pad_token_id
        self._eos_token_id = eos_token_id

    @property
    def fingerprint(self) -> str:
        """
        A fingerprint for the dataset. Can be used to validate that a dataset is the same.
        """
        sha256_hash = hashlib.sha256()
        for offset_start, offset_end in self.offsets:
            sha256_hash.update(f"(start={offset_start}, end={offset_end})".encode())
        return sha256_hash.hexdigest()

    @property
    def memmap_dtype(
        self,
    ) -> Union[Type[np.uint8], Type[np.uint16], Type[np.uint32], Type[np.uint64]]:
        return self._memmap_dtype

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def offsets(self) -> List[Tuple[int, int]]:
        if self._mmap_offsets is None:
            import concurrent.futures

            # Maybe create client up front to work around a threading issue in boto.
            if any(str(p).startswith("s3://") for p in self._memmap_paths):
                _get_s3_client("s3")

            if any(str(p).startswith("r2://") for p in self._memmap_paths):
                try:
                    _get_s3_client("r2")
                except OLMoEnvironmentError:
                    # R2 might not be needed, so ignore this error. We will get an error
                    # later if R2 is needed.
                    pass

            if any(str(p).startswith("weka://") for p in self._memmap_paths):
                try:
                    _get_s3_client("weka")
                except OLMoEnvironmentError:
                    # Weka might not be needed, so ignore this error. We will get an error
                    # later if Weka is needed.
                    pass

            self._mmap_offsets = []

            path_to_length: Dict[PathOrStr, int] = {}
            path_to_mask_path: Dict[PathOrStr, PathOrStr] = {}
            mask_path_to_length: Dict[PathOrStr, int] = {}

            with concurrent.futures.ThreadPoolExecutor() as executor:
                path_futures = []
                mask_path_futures = []
                for i, path in enumerate(self._memmap_paths):
                    path_futures.append(executor.submit(self._get_file_length, path))
                    if self._label_mask_paths is not None:
                        mask_path = self._label_mask_paths[i]
                        path_to_mask_path[path] = mask_path
                        mask_path_futures.append(
                            executor.submit(self._get_file_length, mask_path, np.bool_)
                        )

                for future in concurrent.futures.as_completed(path_futures):
                    path, length = future.result()
                    path_to_length[path] = length

                for future in concurrent.futures.as_completed(mask_path_futures):
                    path, length = future.result()
                    mask_path_to_length[path] = length

            start_offset = 0
            for path in self._memmap_paths:
                length = path_to_length[path]
                if mask_path_to_length:
                    mask_path = path_to_mask_path[path]
                    if length != mask_path_to_length[mask_path]:
                        raise ValueError(
                            f"masking file '{mask_path}' should be the same size as '{path}'"
                        )
                end_offset = start_offset + length
                self._mmap_offsets.append((start_offset, end_offset))
                start_offset += length
        return self._mmap_offsets

    def _read_chunk_from_memmap(self, path: PathOrStr, index: int, dtype=None) -> torch.Tensor:
        dtype = dtype or self.memmap_dtype
        item_size = dtype(0).itemsize
        bytes_start = index * item_size * self._sequence_length
        num_bytes = item_size * self._sequence_length
        buffer = get_bytes_range(path, bytes_start, num_bytes)
        array = np.frombuffer(buffer, dtype=dtype)
        if dtype == np.bool_:
            return torch.tensor(array)
        else:
            return torch.tensor(array.astype(np.int_), dtype=torch.long)

    def _get_file_length(self, path, dtype=None) -> Tuple[PathOrStr, int]:
        dtype = dtype or self.memmap_dtype
        item_size = dtype(0).itemsize
        return path, file_size(path) // (item_size * self._sequence_length)

    def __len__(self) -> int:
        if self._num_instances is None:
            self._num_instances = self.offsets[-1][1]
        return self._num_instances

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)  # in case this is a numpy int type.
        pos_index = index if index >= 0 else len(self) + index

        # The index of the memmap array within 'self.memmaps'
        memmap_index: Optional[int] = None
        # The 'index' relative to the corresponding memmap array.
        memmap_local_index: Optional[int] = None
        for i, (offset_start, offset_end) in enumerate(self.offsets):
            if offset_start <= pos_index < offset_end:
                memmap_index = i
                memmap_local_index = pos_index - offset_start

        if memmap_index is None or memmap_local_index is None:
            raise IndexError(f"{index} is out of bounds for dataset of size {len(self)}")

        # Read the data from file.
        input_ids = self._read_chunk_from_memmap(
            self._memmap_paths[memmap_index], memmap_local_index
        )
        out: Dict[str, Any] = {"input_ids": input_ids}

        if self._label_mask_paths is not None:
            label_mask = self._read_chunk_from_memmap(
                self._label_mask_paths[memmap_index], memmap_local_index, dtype=np.bool_
            )
            out["label_mask"] = label_mask

        if self._include_instance_metadata:
            metadata = self._metadata[memmap_index]
            out["metadata"] = deepcopy(metadata)

        if self._generate_attention_mask:
            attn_mask = torch.ones_like(input_ids)
            attn_mask.masked_fill_(input_ids == self.pad_token_id, 0)
            out["attention_mask"] = attn_mask

        if self._generate_doc_lengths:
            out["doc_lens"] = get_document_lengths(input_ids, self.eos_token_id)

        return out

    def __add__(self, other: MemMapDataset) -> MemMapDataset:
        """
        Concatenate one :class:`MemMapDataset` with another.
        """
        if not isinstance(other, MemMapDataset):
            raise NotImplementedError(f"Expected another MemMapDataset but got {type(other)}")
        return MemMapDataset(
            *(self._memmap_paths + other._memmap_paths),
            sequence_length=self.sequence_length,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            memmap_dtype=self.memmap_dtype,
            metadata=self._metadata + other._metadata,
        )
