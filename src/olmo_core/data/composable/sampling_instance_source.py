import functools as ft
import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import olmo_core.distributed.utils as dist_utils
from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError

from ..utils import get_rng, load_array_slice, write_array_to_disk
from .instance_source import Instance, InstanceSource, InstanceSourceConfig


@dataclass
class SamplingInstanceSourceConfig(InstanceSourceConfig):
    """Config for :class:`SamplingInstanceSource`."""

    sources: List[InstanceSourceConfig]
    max_instances: int
    seed: Optional[int] = None

    def build(self, work_dir: PathOrStr) -> "SamplingInstanceSource":
        return SamplingInstanceSource(
            *[source.build(work_dir) for source in self.sources],
            max_instances=self.max_instances,
            work_dir=work_dir,
            seed=self.seed,
        )


class SamplingInstanceSource(InstanceSource):
    """
    An instance source that samples instances from other instance sources.

    :param sources: The sources to sample instances from.
    :param max_instances: The maximum number of instances to sample.
    :param seed: A optional seed for sampling. If ``None``, the first ``N_s`` instances are taken
      from each source where ``N_s`` is proportional to the size of the source.
    """

    Config = SamplingInstanceSourceConfig

    def __init__(
        self,
        *sources: InstanceSource,
        max_instances: int,
        work_dir: PathOrStr,
        seed: Optional[int] = None,
    ):
        if not sources:
            raise OLMoConfigurationError("At least one source must be provided.")
        assert max_instances > 0

        sequence_length = sources[0].sequence_length
        max_sequence_length = sources[0].max_sequence_length
        for source in sources:
            if source.sequence_length != sequence_length:
                raise OLMoConfigurationError("All sources must have the same sequence length.")
            if source.max_sequence_length != max_sequence_length:
                raise OLMoConfigurationError("All sources must have the same max sequence length.")

        super().__init__(
            work_dir=work_dir,
            sequence_length=sequence_length,
            max_sequence_length=max_sequence_length,
        )
        self._sources = sources
        self._max_instances = max_instances
        self._seed = seed

        # Determine how many instances to sample from each source.
        total_instances = sum(len(source) for source in self.sources)
        max_instances = min(max_instances, total_instances)
        source_sample_sizes: List[int] = []
        for source in sources:
            if max_instances == total_instances:
                source_sample_sizes.append(len(source))
            else:
                # We want `len(source) / total_instances ~= source_sample_size / max_instances`.
                source_sample_sizes.append(int(max_instances * (len(source) / total_instances)))
        self._source_sample_sizes = tuple(source_sample_sizes)

        # Sample indices from each source.
        rng = None if seed is None else get_rng(seed)
        source_sample_paths: List[PathOrStr] = []
        for source, sample_size in zip(sources, source_sample_sizes):
            source_sample_path = (
                self.work_dir / f"{self.fingerprint}-{source.fingerprint}-indices.npy"
            )
            source_sample_paths.append(source_sample_path)
            if self.fs_local_rank == 0:
                if rng is None:
                    source_sample_indices = np.arange(sample_size, dtype=np.uint64)
                else:
                    source_sample_indices = np.arange(len(source), dtype=np.uint64)
                    rng.shuffle(source_sample_indices)
                    source_sample_indices = source_sample_indices[:sample_size]
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
        sha256_hash.update(
            (f"class={self.__class__.__name__},{self.max_instances=},{self.seed=},").encode()
        )
        for source in self.sources:
            sha256_hash.update(f"source={source.fingerprint},".encode())
        return sha256_hash.hexdigest()

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
                    np.uint64,
                )[0]
                return source[int(idx_in_source)]
            source_start_offset = source_end_offset

        raise IndexError(f"{idx} is out of bounds for source of size {len(self)}")
