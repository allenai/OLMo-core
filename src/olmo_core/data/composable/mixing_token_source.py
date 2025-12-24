import dataclasses
import functools as ft
import hashlib
import logging
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
from .utils import (
    SEED_NOT_SET,
    calculate_sample_sizes,
    format_token_count,
    resolve_seed,
)

log = logging.getLogger(__name__)


@dataclass
class MixingTokenSourceSpecConfig(Config):
    """Config for :class:`MixingTokenSourceSpec`."""

    source: TokenSourceConfig
    ratio: float
    max_repetition_factor: float = 1.0
    label: Optional[str] = None

    def __post_init__(self):
        if self.ratio <= 0:
            raise OLMoConfigurationError("Ratio must be greater than 0.")
        if self.max_repetition_factor < 1.0:
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
            max_repetition_factor=self.max_repetition_factor,
            label=self.label,
        )


@dataclass
class MixingTokenSourceConfig(TokenSourceConfig):
    """A config for :class:`MixingTokenSource`."""

    source_specs: List[MixingTokenSourceSpecConfig]
    """Mixing source specs."""
    seed: Optional[int] = dataclasses.field(default_factory=lambda: resolve_seed(SEED_NOT_SET))
    """A random seed for sampling."""
    label: Optional[str] = None
    """An optional label for this source."""
    num_tokens: Optional[int] = None
    """An optional target number of tokens for the mixed source."""

    def build(self, work_dir: PathOrStr) -> List["MixingTokenSource"]:  # type: ignore[override]
        source_specs = [spec.build(work_dir) for spec in self.source_specs]
        return [
            MixingTokenSource(
                *source_specs,
                work_dir=work_dir,
                seed=self.seed,
                label=self.label,
                num_tokens=self.num_tokens,
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
    """
    The relative target ratio for this source. If the ratios across all source specs don't sum
    to 1.0 then they'll be normalized.
    """
    max_repetition_factor: float = 1.0
    """
    The maximum amount of repetition allowed, expressed as a factor greater than or equal to 1.0.
    A factor of 1.0 means no repetition is allowed. A factor of 2.0 means each token could be
    repeated at most once (i.e., seen twice).
    """
    label: Optional[str] = None
    """An optional label for this source."""

    def __post_init__(self):
        if self.ratio <= 0:
            raise OLMoConfigurationError("Ratio must be greater than 0.")
        if self.max_repetition_factor < 1.0:
            raise OLMoConfigurationError("Max repetition must be at least 1.0.")


class MixingTokenSource(TokenSource):
    """
    A token source for mixing other token sources together with arbitrary ratios.
    Sampling within each source is done using :class:`SamplingTokenSource`, which samples a consecutive
    chunk of tokens.

    .. seealso::
        - :class:`MixingDocumentSource` for mixing document sources by sampling whole documents.
        - :class:`MixingInstanceSource` for mixing instance sources.

    .. important::
        Sampling is done in a way that minimizes the number of dropped and repeated tokens while
        matching the target ratios and respecting the :data:`MixingTokenSourceSpec.max_repetition_factor`
        values.

        If ``num_tokens`` is not specified, then the number of tokens this source produces will always
        be less than or equal to the sum of tokens across all of its immediate children defined in
        the ``source_specs``.

        If ``num_tokens`` is specified, this class will try to match that size
        but may raise an :class:`~olmo_core.exceptions.OLMoConfigurationError` if it's not possible
        with the given ``max_repetition_factor`` values.

    :param source_specs: The sources and how to sample from them.
    :param num_tokens: An optional target number of tokens for the mixed source.
    """

    Config = MixingTokenSourceConfig
    """The config class for this source."""

    Spec = MixingTokenSourceSpec
    """The mixing spec class for this source."""

    DISPLAY_ICON = "\uf074"

    def __init__(
        self,
        *source_specs: MixingTokenSourceSpec,
        work_dir: PathOrStr,
        seed: Optional[int] = SEED_NOT_SET,
        label: Optional[str] = None,
        num_tokens: Optional[int] = None,
    ):
        if not source_specs:
            raise OLMoConfigurationError("At least one source spec must be provided.")

        sources = [spec.source for spec in source_specs]

        super().__init__(work_dir=work_dir, label=label)

        # Determine the number of tokens to sample from each source.
        sample_sizes = calculate_sample_sizes(
            [len(spec.source) for spec in source_specs],
            [spec.ratio for spec in source_specs],
            [spec.max_repetition_factor for spec in source_specs],
            target_size=num_tokens,
            labels=[
                spec.label or spec.source.label or str(i) for i, spec in enumerate(source_specs)
            ],
        )

        # Sample tokens from each source.
        seed = resolve_seed(seed)
        sampled_sources: List[SamplingTokenSource] = []
        for i, (spec, sample_size) in enumerate(zip(source_specs, sample_sizes)):
            sampled_sources.append(
                SamplingTokenSource(
                    spec.source,
                    max_tokens=int(sample_size),
                    work_dir=work_dir,
                    seed=None if seed is None else seed + i,
                    label=spec.label or spec.source.label,
                )
            )
        self._source = ConcatenatedTokenSource(*sampled_sources, work_dir=work_dir)

        summary_lines = []
        for i, (source, sampled_source, spec) in enumerate(
            zip(sources, self.sampled_sources, source_specs)
        ):
            summary_lines.append(
                f" ❯ {100 * len(sampled_source) / len(self):0.3f}% {spec.label or ('source ' + str(i))}, "
                f"{format_token_count(len(sampled_source))} sampled tokens from {format_token_count(len(source))} source tokens"
            )
        summary_lines.append(f"Total: {format_token_count(self.num_tokens)} tokens")
        summary = "\n".join(summary_lines)
        log.info(f"Created token mixture consisting of:\n{summary}")

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

    def children(self):
        return self.sampled_sources
