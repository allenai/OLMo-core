import functools as ft
import hashlib
import typing
from dataclasses import dataclass
from typing import List, Optional, Tuple

from olmo_core.aliases import PathOrStr
from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError

from .instance_source import (
    ConcatenatedInstanceSource,
    Instance,
    InstanceSource,
    InstanceSourceConfig,
)
from .sampling_instance_source import SamplingInstanceSource
from .utils import calculate_sample_sizes


@dataclass
class MixingInstanceSourceSpecConfig(Config):
    """Config for :class:`MixingInstanceSourceSpec`."""

    source: InstanceSourceConfig
    ratio: float
    max_repetition: float = 1.0

    def __post_init__(self):
        if self.ratio <= 0:
            raise OLMoConfigurationError("Ratio must be greater than 0.")
        if self.max_repetition < 1.0:
            raise OLMoConfigurationError("Max repetition must be at least 1.0.")

    def build(self, work_dir: PathOrStr) -> "MixingInstanceSourceSpec":
        return MixingInstanceSourceSpec(
            source=self.source.build(work_dir),
            ratio=self.ratio,
            max_repetition=self.max_repetition,
        )


@dataclass
class MixingInstanceSourceConfig(InstanceSourceConfig):
    """A config for :class:`MixingInstanceSource`."""

    source_specs: List[MixingInstanceSourceSpecConfig]
    """Mixing source specs."""
    seed: Optional[int] = None
    """A random seed for sampling."""

    def build(self, work_dir: PathOrStr) -> "MixingInstanceSource":  # type: ignore[override]
        source_specs = [spec.build(work_dir) for spec in self.source_specs]
        return MixingInstanceSource(
            *source_specs,
            work_dir=work_dir,
            seed=self.seed,
        )


@dataclass
class MixingInstanceSourceSpec:
    """Defines a source and its associated mixing ratio for :class:`MixingInstanceSource`."""

    source: InstanceSource
    """The source."""
    ratio: float
    """The relative ratio for this source."""
    max_repetition: float = 1.0
    """
    The maximum amount of repetition allowed, expressed as a factor greater than or equal to 1.0.
    A factor of 1.0 means no repetition is allowed. A factor of 2.0 means each instance could be
    repeated at most once (i.e., seen twice).
    """

    def __post_init__(self):
        if self.ratio <= 0:
            raise OLMoConfigurationError("Ratio must be greater than 0.")
        if self.max_repetition < 1.0:
            raise OLMoConfigurationError("Max repetition must be at least 1.0.")


class MixingInstanceSource(InstanceSource):
    """
    An instance source for mixing other instance sources together with arbitrary ratios.

    :param source_specs: The sources and how to sample from them.
    """

    Config = MixingInstanceSourceConfig

    def __init__(
        self,
        *source_specs: MixingInstanceSourceSpec,
        work_dir: PathOrStr,
        seed: Optional[int] = None,
    ):
        if not source_specs:
            raise OLMoConfigurationError("At least one source spec must be provided.")

        sources = [spec.source for spec in source_specs]
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

        # Determine the number of instances to sample from each source.
        sample_sizes = calculate_sample_sizes(
            [len(source) for source in sources],
            [spec.ratio for spec in source_specs],
            [spec.max_repetition for spec in source_specs],
        )

        # Sample instances from each source.
        sampled_sources: List[SamplingInstanceSource] = []
        for i, (source, sample_size) in enumerate(zip(sources, sample_sizes)):
            sampled_sources.append(
                SamplingInstanceSource(
                    source,
                    max_instances=int(sample_size),
                    work_dir=work_dir,
                    seed=None if seed is None else seed + i,
                    allow_repetition=sample_size > len(source),
                )
            )
        self._source = ConcatenatedInstanceSource(*sampled_sources, work_dir=work_dir)

    @property
    def sampled_sources(self) -> Tuple[SamplingInstanceSource, ...]:
        return typing.cast(Tuple[SamplingInstanceSource, ...], self._source.sources)

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update((f"class={self.__class__.__name__},source={self._source}").encode())
        return sha256_hash.hexdigest()

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, idx: int) -> Instance:
        return self._source[idx]
