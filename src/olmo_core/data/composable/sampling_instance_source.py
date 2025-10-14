import functools as ft
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import olmo_core.distributed.utils as dist_utils
from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError

from ..utils import load_array_slice, write_array_to_disk
from .instance_source import Instance, InstanceSource, InstanceSourceConfig
from .utils import build_global_indices


@dataclass
class SamplingInstanceSourceConfig(InstanceSourceConfig):
    """Config for :class:`SamplingInstanceSource`."""

    sources: List[InstanceSourceConfig]
    max_tokens: Optional[int] = None
    max_instances: Optional[int] = None
    seed: Optional[int] = None
    label: Optional[str] = None

    def __post_init__(self):
        if (self.max_tokens is None) == (self.max_instances is None):
            raise OLMoConfigurationError(
                "Either max_tokens or max_instances must be set, but not both."
            )

    def build(self, work_dir: PathOrStr) -> "SamplingInstanceSource":
        return SamplingInstanceSource(
            *[source.build(work_dir) for source in self.sources],
            max_tokens=self.max_tokens,
            max_instances=self.max_instances,
            work_dir=work_dir,
            seed=self.seed,
            label=self.label,
        )


class SamplingInstanceSource(InstanceSource):
    """
    An instance source that samples instances from other instance sources.
    This can be used to adjust the effective size of a source.

    .. seealso::
        - :class:`SamplingTokenSource`
        - :class:`SamplingDocumentSource`

    :param sources: The sources to sample instances from.
    :param max_tokens: The maximum number of tokens to sample. Alternatively you can specify
      ``max_instances``.
    :param max_instances: The maximum number of instances to sample. Mutually exclusive with
      ``max_tokens``.
    :param seed: A optional seed for sampling. If ``None``, the first ``N_s`` instances are taken
      from each source where ``N_s`` is proportional to the size of the source.
    """

    Config = SamplingInstanceSourceConfig

    DISPLAY_ICON = "\uedec"

    def __init__(
        self,
        *sources: InstanceSource,
        max_tokens: Optional[int] = None,
        max_instances: Optional[int] = None,
        work_dir: PathOrStr,
        seed: Optional[int] = None,
        label: Optional[str] = None,
    ):
        if not sources:
            raise OLMoConfigurationError("At least one source must be provided.")

        sequence_length = sources[0].sequence_length
        max_sequence_length = sources[0].max_sequence_length
        for source in sources:
            if source.sequence_length != sequence_length:
                raise OLMoConfigurationError("All sources must have the same sequence length.")
            if source.max_sequence_length != max_sequence_length:
                raise OLMoConfigurationError("All sources must have the same max sequence length.")

        if (max_tokens is None) == (max_instances is None):
            raise OLMoConfigurationError(
                "Either max_tokens or max_instances must be set, but not both."
            )
        elif max_tokens is not None:
            assert max_tokens > 0
            max_instances = max_tokens // sequence_length
        elif max_instances is not None:
            assert max_instances > 0

        assert max_instances is not None

        super().__init__(
            work_dir=work_dir,
            sequence_length=sequence_length,
            max_sequence_length=max_sequence_length,
            label=label,
        )
        self._sources = sources
        self._max_instances = max_instances
        self._seed = seed
        self._dtype = np.uint32

        # Determine how many instances to sample from each source.
        total_instances = sum(len(source) for source in self.sources)
        chunk_size = self.max_sequence_length // self.sequence_length
        source_sample_sizes: List[int] = []
        for source in sources:
            # We want `len(source) / total_instances ~= source_sample_size / max_instances`,
            # so `source_sample_size = max_instances * (len(source) / total_instances)`.
            source_sample_size = int(max_instances * (len(source) / total_instances))
            # Adjust to be a multiple of chunk_size.
            source_sample_size = chunk_size * (source_sample_size // chunk_size)
            source_sample_sizes.append(source_sample_size)
        self._source_sample_sizes = tuple(source_sample_sizes)

        # Sample indices from each source.
        source_sample_paths: List[PathOrStr] = []
        for i, (source, sample_size) in enumerate(zip(sources, source_sample_sizes)):
            source_sample_path = (
                self.work_dir / f"{self.fingerprint}-{source.fingerprint}-indices.npy"
            )
            source_sample_paths.append(source_sample_path)
            if self.fs_local_rank == 0:
                n_repetitions = sample_size // len(source)
                remaining_sample_size = sample_size % len(source)
                source_indices = build_global_indices(
                    len(source),
                    sequence_length=self.sequence_length,
                    max_sequence_length=self.max_sequence_length,
                    seed=None if seed is None else seed + i,
                    dtype=self._dtype,
                )
                sample_indices = source_indices[:remaining_sample_size]
                source_sample_indices = np.concatenate(
                    [np.tile(source_indices, n_repetitions), sample_indices]
                )
                write_array_to_disk(source_sample_indices, source_sample_path)
        self._source_sample_paths = tuple(source_sample_paths)
        dist_utils.barrier()

    @property
    def sources(self) -> Tuple[InstanceSource, ...]:
        return self._sources

    @property
    def max_instances(self) -> int:
        return self._max_instances

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @property
    def source_sample_sizes(self) -> Tuple[int, ...]:
        return self._source_sample_sizes

    @ft.cached_property
    def num_instances(self) -> int:
        return sum(self.source_sample_sizes)

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update((f"class={self.__class__.__name__},{self.seed=},").encode())
        for source, sample_size in zip(self.sources, self.source_sample_sizes):
            chunk_size = self.max_sequence_length // self.sequence_length
            sample_size_chunk_size_ratio = sample_size // chunk_size
            sha256_hash.update(
                f"source={source.fingerprint},{sample_size_chunk_size_ratio=}".encode()
            )
        return sha256_hash.hexdigest()

    def children(self):
        return self.sources

    def __len__(self) -> int:
        return self.num_instances

    def __getitem__(self, idx: int) -> Instance:
        idx = self.validate_index(idx)

        source_start_offset = 0
        for source, source_sample_size, source_sample_indices_path in zip(
            self.sources, self.source_sample_sizes, self._source_sample_paths
        ):
            source_end_offset = source_start_offset + source_sample_size
            if source_start_offset <= idx < source_end_offset:
                idx_in_source = load_array_slice(
                    source_sample_indices_path,
                    idx - source_start_offset,
                    idx - source_start_offset + 1,
                    self._dtype,
                )[0]
                return source[int(idx_in_source)]
            source_start_offset = source_end_offset

        raise IndexError(f"{idx} is out of bounds for source of size {len(self)}")
