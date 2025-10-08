import functools as ft
import hashlib
import typing
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

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


@dataclass
class MixingInstanceSourceSpec(Config):
    """Defines a source and its associated mixing ratio for :class:`MixingInstanceSource`."""

    ratio: float
    """The relative ratio for this source."""
    source: InstanceSourceConfig
    """The source."""


@dataclass
class MixingInstanceSourceConfig(InstanceSourceConfig):
    """A config for :class:`MixingInstanceSource`."""

    source_specs: List[MixingInstanceSourceSpec]
    """Mixing source specs."""
    seed: Optional[int] = None
    """A random seed for sampling."""
    allow_repetition: bool = False
    """allow repetition of instances when sampling from sources."""

    def build(self, work_dir: PathOrStr) -> "MixingInstanceSource":  # type: ignore[override]
        source_specs = [(spec.ratio, spec.source.build(work_dir)) for spec in self.source_specs]
        return MixingInstanceSource(
            *source_specs, work_dir=work_dir, seed=self.seed, allow_repetition=self.allow_repetition
        )


class MixingInstanceSource(InstanceSource):
    """
    An instance source for mixing other instance sources together with arbitrary ratios.

    :param source_specs: Tuples of ``(ratio, source)`` where `ratio` is a float indicating the relative
      proportion of instances to sample from `source`.
    """

    Config = MixingInstanceSourceConfig

    def __init__(
        self,
        *source_specs: Tuple[float, InstanceSource],
        work_dir: PathOrStr,
        seed: Optional[int] = None,
        allow_repetition: bool = False,
    ):
        if not source_specs:
            raise OLMoConfigurationError("At least one source spec must be provided.")

        sources = [spec[1] for spec in source_specs]
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

        if allow_repetition:
            # TODO: implement this.
            raise NotImplementedError("allow_repetition=True is not implemented yet.")

        # Normalize ratios.
        ratios = np.array([spec[0] for spec in source_specs])
        ratios = ratios / ratios.sum()
        sizes = np.array([len(source) for source in sources])

        # Determine the number of instances to sample from each source.
        # NOTE: this is tricky because the sources may have different sizes, yet we want to stay
        # true to the sampling ratios.
        total_instances = sizes.sum()
        ideal_sample_sizes = total_instances * ratios
        adjustment_factor = min(1.0, (sizes / ideal_sample_sizes).min())
        actual_sample_sizes = ideal_sample_sizes * adjustment_factor

        # Sanity check.
        # Sample sizes should stay true to target ratios.
        assert np.allclose(ratios, actual_sample_sizes / actual_sample_sizes.sum())
        # And sample sizes shouldn't be larger than the number of instances available.
        actual_sample_sizes = actual_sample_sizes.astype(np.uint64)
        assert (actual_sample_sizes <= sizes).all()

        # Sample instances from each source.
        sampled_sources: List[SamplingInstanceSource] = []
        for source, sample_size in zip(sources, actual_sample_sizes):
            sampled_sources.append(
                SamplingInstanceSource(
                    source, max_instances=int(sample_size), work_dir=work_dir, seed=seed
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
