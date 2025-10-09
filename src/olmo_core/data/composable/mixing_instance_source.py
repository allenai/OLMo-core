import functools as ft
import hashlib
import logging
import typing
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple, Type

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

log = logging.getLogger(__name__)


@dataclass
class MixingInstanceSourceSpecConfig(Config):
    """Config for :class:`MixingInstanceSourceSpec`."""

    source: InstanceSourceConfig
    ratio: float
    size_adjustment_factor: float = 1.0
    max_repetition_factor: float = 1.0
    label: Optional[str] = None

    def __post_init__(self):
        if self.ratio <= 0:
            raise OLMoConfigurationError("Ratio must be greater than 0.")
        if self.size_adjustment_factor <= 0:
            raise OLMoConfigurationError("Size adjustment factor must be greater than 0.")
        if self.max_repetition_factor < 1.0:
            raise OLMoConfigurationError("Max repetition must be at least 1.0.")

    def build(self, work_dir: PathOrStr) -> "MixingInstanceSourceSpec":
        return MixingInstanceSourceSpec(
            source=self.source.build(work_dir),
            ratio=self.ratio,
            size_adjustment_factor=self.size_adjustment_factor,
            max_repetition_factor=self.max_repetition_factor,
            label=self.label,
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

    Config: ClassVar[Type["MixingInstanceSourceSpecConfig"]] = MixingInstanceSourceSpecConfig
    """The config class for this spec."""

    source: InstanceSource
    """The source."""
    ratio: float
    """The relative target ratio for this source."""
    size_adjustment_factor: float = 1.0
    """
    An optional factor to adjust the effective size of this source prior to determining how many
    instances to sample. A factor less than 1.0 makes the source smaller, while a factor greater
    than 1.0 makes it larger by oversampling.
    """
    max_repetition_factor: float = 1.0
    """
    The maximum amount of repetition allowed after applying the ``size_adjustment_factor``,
    expressed as a factor greater than or equal to 1.0.
    A factor of 1.0 means no repetition is allowed. A factor of 2.0 means each instance could be
    repeated at most once (i.e., seen twice).
    """
    label: Optional[str] = None
    """An optional label for this source."""

    def __post_init__(self):
        if self.ratio <= 0:
            raise OLMoConfigurationError("Ratio must be greater than 0.")
        if self.size_adjustment_factor <= 0:
            raise OLMoConfigurationError("Size adjustment factor must be greater than 0.")
        if self.max_repetition_factor < 1.0:
            raise OLMoConfigurationError("Max repetition must be at least 1.0.")


class MixingInstanceSource(InstanceSource):
    """
    An instance source for mixing other instance sources together with arbitrary ratios.

    :param source_specs: The sources and how to sample from them.
    """

    Config: ClassVar[Type[MixingInstanceSourceConfig]] = MixingInstanceSourceConfig
    """The config class for this source."""

    Spec = MixingInstanceSourceSpec
    """The mixing spec class for this source."""

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
            [int(len(spec.source) * spec.size_adjustment_factor) for spec in source_specs],
            [spec.ratio for spec in source_specs],
            [spec.max_repetition_factor for spec in source_specs],
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

        summary_lines = []
        for i, (source, sampled_source, spec) in enumerate(
            zip(sources, self.sampled_sources, source_specs)
        ):
            summary_lines.append(
                f" ❯ {100 * len(sampled_source) / len(self):0.3f}% {spec.label or ('source ' + str(i))}, "
                f"{len(sampled_source):,d} sampled instances from {len(source):,d} source instances"
            )
        summary = "\n".join(summary_lines)
        log.info(f"Created instance mixture consisting of:\n{summary}")

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
