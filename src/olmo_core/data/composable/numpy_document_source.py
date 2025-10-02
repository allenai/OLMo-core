import functools as ft
import hashlib
import typing
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

import olmo_core.distributed.utils as dist_utils
import olmo_core.io as io
from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError

from ..tokenizer import TokenizerConfig
from ..types import NumpyUIntTypes
from ..utils import iter_document_indices, load_array_slice
from .token_source import DocumentSource, TokenRange
from .utils import path_map


class NumpyDocumentSource(DocumentSource):
    """
    A :class:`DocumentSource` that reads tokens from one or more tokenized numpy source files.
    """

    def __init__(
        self,
        *,
        source_paths: Sequence[PathOrStr],
        dtype: NumpyUIntTypes,
        work_dir: PathOrStr,
        tokenizer: TokenizerConfig,
        label_mask_paths: Optional[Sequence[PathOrStr]] = None,
    ):
        super().__init__(work_dir=work_dir)
        self.source_paths = tuple((io.normalize_path(p) for p in source_paths))
        if not self.source_paths:
            raise OLMoConfigurationError("'source_paths' must contain at least one path.")
        self.label_mask_paths = (
            None
            if label_mask_paths is None
            else tuple((io.normalize_path(p) for p in label_mask_paths))
        )
        if self.label_mask_paths is not None and len(self.label_mask_paths) != len(
            self.source_paths
        ):
            raise OLMoConfigurationError(
                "'label_mask_paths' should have the same length as 'source_paths'."
            )
        self._dtype = dtype
        self._tokenizer = tokenizer

        source_sizes: List[int]
        if self.rank == 0:
            item_size = self.dtype(0).itemsize
            source_sizes = path_map(lambda p: io.get_file_size(p) // item_size, self.source_paths)
        else:
            source_sizes = []
        source_sizes = dist_utils.scatter_object(source_sizes)
        assert len(source_sizes) == len(self.source_paths)
        self.source_sizes = source_sizes

        self.label_mask_sizes: Optional[Tuple[int]] = None
        if self.label_mask_paths is not None:
            label_mask_sizes: List[int]
            if self.rank == 0:
                item_size = np.bool_(0).itemsize
                label_mask_sizes = path_map(
                    lambda p: io.get_file_size(p) // item_size, self.label_mask_paths
                )
            else:
                label_mask_sizes = []
            label_mask_sizes = dist_utils.scatter_object(label_mask_sizes)
            assert len(label_mask_sizes) == len(self.label_mask_paths)
            self.label_mask_sizes = tuple(label_mask_sizes)
            for label_path, label_mask_size, source_path, source_size in zip(
                self.label_mask_paths, self.label_mask_sizes, self.source_paths, self.source_sizes
            ):
                if label_mask_size != source_size:
                    raise OLMoConfigurationError(
                        "Each file in 'label_mask_paths' should have the same number of items as the corresponding file in 'source_paths', "
                        f"but found {label_mask_size:,d} in '{label_path}' vs {source_size:,d} in '{source_path}'.",
                    )

    @property
    def dtype(self) -> NumpyUIntTypes:
        return self._dtype

    @property
    def tokenizer(self) -> TokenizerConfig:
        return self._tokenizer

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.tokenizer.bos_token_id

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            (
                f"class={self.__class__.__name__},"
                f"{self.dtype=},"
                f"{self.eos_token_id=},"
                f"{self.bos_token_id=},"
            ).encode()
        )
        # NOTE: it's too expensive to hash the contents of the source files, so we take a shortcut
        # by hashing their paths and sizes instead. This should be sufficient to detect changes 99.99% of the time.
        for path, size in zip(self.source_paths, self.source_sizes):
            sha256_hash.update(f"{path=},{size=},".encode())
        if self.label_mask_paths is not None:
            for label_path, size in zip(self.label_mask_paths, self.source_sizes):
                sha256_hash.update(f"{label_path=},{size=},".encode())
        return sha256_hash.hexdigest()

    @ft.cached_property
    def num_tokens(self) -> int:
        return sum(self.source_sizes)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}{self.source_paths}"

    def get_token_range(self, start_idx: int, count: int) -> TokenRange:
        assert count >= 0
        if start_idx < 0:
            start_idx = self.num_tokens + start_idx
        end_idx = start_idx + count
        if start_idx >= self.num_tokens or end_idx > self.num_tokens:
            raise IndexError(f"Token range [{start_idx}, {end_idx}) is out of bounds.")

        token_chunks: List[np.ndarray] = []
        mask_chunks: List[np.ndarray] = []
        source_start_offset = 0
        for i, (source_path, source_size) in enumerate(zip(self.source_paths, self.source_sizes)):
            source_end_offset = source_start_offset + source_size

            if source_start_offset <= start_idx < source_end_offset:
                token_chunk = load_array_slice(
                    source_path,
                    start_idx - source_start_offset,
                    min(end_idx - source_start_offset, source_size),
                    self.dtype,
                )
                token_chunks.append(token_chunk)

                if self.label_mask_paths is not None:
                    mask_path = self.label_mask_paths[i]
                    mask_chunk = load_array_slice(
                        mask_path,
                        start_idx - source_start_offset,
                        min(end_idx - source_start_offset, source_size),
                        np.bool_,
                    )
                    mask_chunks.append(mask_chunk)

                if end_idx - source_start_offset <= source_size:
                    break
                else:
                    start_idx = source_end_offset

            source_start_offset = source_end_offset
        else:
            raise IndexError(f"Failed to find tokens in range [{start_idx}, {end_idx}).")

        input_ids = np.concatenate(token_chunks)
        out: TokenRange = {"input_ids": typing.cast(Sequence[int], input_ids)}
        if mask_chunks:
            out["label_mask"] = typing.cast(Sequence[bool], np.concatenate(mask_chunks))

        return out

    def get_document_offsets(self) -> Iterable[tuple[int, int]]:
        start_offset = 0
        for source_path, source_size in zip(self.source_paths, self.source_sizes):
            for doc_start, doc_end in iter_document_indices(
                source_path,
                eos_token_id=self.eos_token_id,
                bos_token_id=self.bos_token_id,
                dtype=self.dtype,
            ):
                yield doc_start + start_offset, doc_end + start_offset
            start_offset += source_size
