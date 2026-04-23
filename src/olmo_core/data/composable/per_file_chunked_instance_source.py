import functools as ft
import hashlib
import logging
import typing
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np

import olmo_core.distributed.utils as dist_utils
import olmo_core.io as io
from olmo_core.aliases import PathOrStr

from ..types import NumpyDatasetDType, NumpyUIntTypes
from ..utils import load_array_slice
from .instance_source import Instance, InstanceSource, InstanceSourceConfig
from .utils import path_map

log = logging.getLogger(__name__)


@dataclass
class PerFileChunkedInstanceSourceConfig(InstanceSourceConfig):
    """
    Config for :class:`PerFileChunkedInstanceSource`.

    Reads tokens-only ``.npy`` files and chunks each file independently into
    ``sequence_length``-sized instances. Does **not** cross file boundaries
    (unlike :class:`ConcatAndChunkInstanceSource`), so pre-written instance
    boundaries are preserved even when individual files have token counts that
    are not multiples of ``sequence_length``.
    """

    source_paths: List[str]
    """Glob patterns for token ``.npy`` files."""

    sequence_length: int
    """The number of tokens per instance."""

    token_dtype: Optional[NumpyDatasetDType] = None
    """The numpy dtype for token arrays. Defaults to uint16."""

    label: Optional[str] = None
    """An optional label for this source."""

    def build(self, work_dir: PathOrStr) -> "PerFileChunkedInstanceSource":
        token_dtype = (
            self.token_dtype.as_np_dtype() if self.token_dtype is not None else np.uint16
        )
        source_paths = self._expand_all_globs(self.source_paths)
        return PerFileChunkedInstanceSource(
            source_paths=source_paths,
            token_dtype=token_dtype,
            sequence_length=self.sequence_length,
            work_dir=work_dir,
            label=self.label,
        )

    @staticmethod
    def _expand_all_globs(patterns: Sequence[str]) -> List[str]:
        """Expand glob patterns, distributing the work from rank 0."""
        expanded: List[str]
        if dist_utils.get_rank() == 0:
            result: List[str] = []
            for pattern in patterns:
                if "*" in pattern:
                    matches = io.deterministic_glob_directory(pattern)
                    if not matches:
                        raise FileNotFoundError(f"No files matched pattern: {pattern}")
                    result.extend(matches)
                else:
                    result.append(pattern)
            expanded = result
        else:
            expanded = []
        expanded = dist_utils.broadcast_object(expanded)
        return expanded


class PerFileChunkedInstanceSource(InstanceSource):
    """
    An :class:`InstanceSource` that chunks each ``.npy`` file independently into
    ``sequence_length``-sized instances, never crossing file boundaries.

    This is the tokens-only sibling of :class:`PreChunkedInstanceSource` (which
    also requires parallel ``pos_ids`` and ``vis_limit`` files for tree-attention
    data). It is equivalent in per-file semantics to
    :class:`~olmo_core.data.numpy_dataset.NumpyFSLDatasetMixture` with
    ``chunk_based_mixture=True``, but in the composable API.

    Use this for pre-chunked tokens-only data where each file contains a
    concatenation of pre-built ``sequence_length``-sized instances and files may
    have short tails at worker-budget exhaustion. The alternative
    :class:`ConcatAndChunkInstanceSource` treats all files as one flat token
    stream: any non-aligned tail in an earlier file propagates a phase shift
    across every subsequent file, bisecting pre-written instance boundaries.

    Per-file residual bytes (``file_tokens % sequence_length``) are silently
    dropped via integer division in :data:`num_instances`.
    """

    Config: typing.ClassVar[Type[PerFileChunkedInstanceSourceConfig]] = (
        PerFileChunkedInstanceSourceConfig
    )
    DISPLAY_ICON = ""

    def __init__(
        self,
        *,
        source_paths: Sequence[str],
        token_dtype: NumpyUIntTypes,
        sequence_length: int,
        work_dir: PathOrStr,
        label: Optional[str] = None,
    ):
        super().__init__(sequence_length=sequence_length, work_dir=work_dir, label=label)
        self._source_paths: Tuple[str, ...] = tuple(source_paths)
        self._token_dtype = token_dtype

        item_size = token_dtype(0).itemsize
        if dist_utils.get_rank() == 0:
            source_sizes = path_map(
                lambda p: io.get_file_size(p) // item_size, self._source_paths
            )
        else:
            source_sizes = []
        self._source_sizes: Tuple[int, ...] = tuple(
            dist_utils.broadcast_object(source_sizes)
        )

        n_unaligned = sum(
            1 for s in self._source_sizes if s % self.sequence_length != 0
        )
        tail_tokens = sum(s % self.sequence_length for s in self._source_sizes)
        log.info(
            f"PerFileChunkedInstanceSource: {len(self._source_paths)} files, "
            f"{sum(self._source_sizes):,d} total tokens, "
            f"{len(self):,d} instances of length {sequence_length}"
        )
        if n_unaligned > 0:
            log.info(
                f"PerFileChunkedInstanceSource: {tail_tokens:,d} tail tokens "
                f"across {n_unaligned} non-{sequence_length}-aligned files dropped"
            )

    @ft.cached_property
    def _instances_per_file(self) -> Tuple[int, ...]:
        return tuple(size // self.sequence_length for size in self._source_sizes)

    @ft.cached_property
    def num_tokens(self) -> int:
        return self.num_instances * self.sequence_length

    @ft.cached_property
    def num_instances(self) -> int:
        return sum(self._instances_per_file)

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            f"class={self.__class__.__name__},seq_len={self.sequence_length},".encode()
        )
        for path, size in zip(self._source_paths, self._source_sizes):
            sha256_hash.update(f"path={path},size={size},".encode())
        return sha256_hash.hexdigest()

    def __len__(self) -> int:
        return self.num_instances

    def __getitem__(self, idx: int) -> Instance:
        idx = self.validate_index(idx)

        file_start = 0
        for file_idx, n_instances in enumerate(self._instances_per_file):
            file_end = file_start + n_instances
            if file_start <= idx < file_end:
                local_idx = idx - file_start
                start_token = local_idx * self.sequence_length
                end_token = start_token + self.sequence_length
                input_ids = load_array_slice(
                    self._source_paths[file_idx],
                    start_token,
                    end_token,
                    self._token_dtype,
                )
                return {"input_ids": typing.cast(Sequence[int], input_ids)}
            file_start = file_end

        raise IndexError(
            f"Index {idx} out of range for {self.num_instances} instances."
        )

    def children(self) -> Iterable["PerFileChunkedInstanceSource"]:
        return []
