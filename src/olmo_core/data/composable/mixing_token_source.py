import functools as ft
import hashlib
import typing
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple, Type

from olmo_core.aliases import PathOrStr
from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError

from .sampling_token_source import SamplingTokenSource
from .token_source import (
    ConcatenatedTokenSource,
    TokenRange,
    TokenSource,
    TokenSourceConfig,
)
from .utils import calculate_sample_sizes


@dataclass
class MixingTokenSourceSpecConfig(Config):
    """Config for :class:`MixingTokenSourceSpec`."""

    source: TokenSourceConfig
    ratio: float
    max_repetition: float = 1.0

    def __post_init__(self):
        if self.ratio <= 0:
            raise OLMoConfigurationError("Ratio must be greater than 0.")
        if self.max_repetition < 1.0:
            raise OLMoConfigurationError("Max repetition must be at least 1.0.")

    def build(self, work_dir: PathOrStr) -> "MixingTokenSourceSpec":
        sources = self.source.build(work_dir)
        source = (
            sources[0]
            if len(sources) == 1
            else ConcatenatedTokenSource(*sources, work_dir=work_dir)
        )
        return MixingTokenSourceSpec(
            source=source,
            ratio=self.ratio,
            max_repetition=self.max_repetition,
        )


@dataclass
class MixingTokenSourceConfig(TokenSourceConfig):
    """A config for :class:`MixingTokenSource`."""

    source_specs: List[MixingTokenSourceSpecConfig]
    """Mixing source specs."""
    seed: Optional[int] = None
    """A random seed for sampling."""

    def build(self, work_dir: PathOrStr) -> List["MixingTokenSource"]:  # type: ignore[override]
        source_specs = [spec.build(work_dir) for spec in self.source_specs]
        return [
            MixingTokenSource(
                *source_specs,
                work_dir=work_dir,
                seed=self.seed,
            )
        ]


@dataclass
class MixingTokenSourceSpec:
    """Defines a source and its associated mixing ratio for :class:`MixingTokenSource`."""

    Config: ClassVar[Type["MixingTokenSourceSpecConfig"]] = MixingTokenSourceSpecConfig
    """The config class for this spec."""

    source: TokenSource
    """The source."""
    ratio: float
    """The relative ratio for this source."""
    max_repetition: float = 1.0
    """
    The maximum amount of repetition allowed, expressed as a factor greater than or equal to 1.0.
    A factor of 1.0 means no repetition is allowed. A factor of 2.0 means each token could be
    repeated at most once (i.e., seen twice).
    """

    def __post_init__(self):
        if self.ratio <= 0:
            raise OLMoConfigurationError("Ratio must be greater than 0.")
        if self.max_repetition < 1.0:
            raise OLMoConfigurationError("Max repetition must be at least 1.0.")


class MixingTokenSource(TokenSource):
    """
    A token source for mixing other token sources together with arbitrary ratios.

    :param source_specs: The sources and how to sample from them.
    """

    Config = MixingTokenSourceConfig
    """The config class for this source."""

    Spec = MixingTokenSourceSpec
    """The mixing spec class for this source."""

    def __init__(
        self,
        *source_specs: MixingTokenSourceSpec,
        work_dir: PathOrStr,
        seed: Optional[int] = None,
    ):
        if not source_specs:
            raise OLMoConfigurationError("At least one source spec must be provided.")

        super().__init__(work_dir=work_dir)

        sources = [spec.source for spec in source_specs]

        # Determine the number of tokens to sample from each source.
        sample_sizes = calculate_sample_sizes(
            [len(source) for source in sources],
            [spec.ratio for spec in source_specs],
            [spec.max_repetition for spec in source_specs],
        )

        # Sample tokens from each source.
        sampled_sources: List[SamplingTokenSource] = []
        for i, (source, sample_size) in enumerate(zip(sources, sample_sizes)):
            sampled_sources.append(
                SamplingTokenSource(
                    source,
                    max_tokens=int(sample_size),
                    work_dir=work_dir,
                    seed=None if seed is None else seed + i,
                    allow_repetition=sample_size > len(source),
                )
            )
        self._source = ConcatenatedTokenSource(*sampled_sources, work_dir=work_dir)

    @property
    def sampled_sources(self) -> Tuple[SamplingTokenSource, ...]:
        return typing.cast(Tuple[SamplingTokenSource, ...], self._source.sources)

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update((f"class={self.__class__.__name__},source={self._source}").encode())
        return sha256_hash.hexdigest()

    @property
    def num_tokens(self) -> int:
        return self._source.num_tokens

    def get_token_range(self, start_idx: int, end_idx: int) -> TokenRange:
        return self._source.get_token_range(start_idx, end_idx)
