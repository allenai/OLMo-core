import functools as ft
import hashlib
import logging
import typing
from dataclasses import dataclass
from typing import ClassVar, Iterable, List, Optional, Tuple, Type

from olmo_core.aliases import PathOrStr
from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError

from .sampling_document_source import SamplingDocumentSource
from .token_source import (
    ConcatenatedDocumentSource,
    DocumentSource,
    DocumentSourceConfig,
    TokenRange,
)
from .utils import calculate_sample_sizes, format_token_count

log = logging.getLogger(__name__)


@dataclass
class MixingDocumentSourceSpecConfig(Config):
    """Config for :class:`MixingDocumentSourceSpec`."""

    source: DocumentSourceConfig
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

    def build(self, work_dir: PathOrStr) -> "MixingDocumentSourceSpec":
        sources = self.source.build(work_dir)
        source = (
            sources[0]
            if len(sources) == 1
            else ConcatenatedDocumentSource(*sources, work_dir=work_dir)
        )
        return MixingDocumentSourceSpec(
            source=source,
            ratio=self.ratio,
            size_adjustment_factor=self.size_adjustment_factor,
            max_repetition_factor=self.max_repetition_factor,
            label=self.label,
        )


@dataclass
class MixingDocumentSourceConfig(DocumentSourceConfig):
    """A config for :class:`MixingDocumentSource`."""

    source_specs: List[MixingDocumentSourceSpecConfig]
    """Mixing source specs."""
    seed: Optional[int] = None
    """A random seed for sampling."""
    label: Optional[str] = None

    def build(self, work_dir: PathOrStr) -> List["MixingDocumentSource"]:  # type: ignore[override]
        source_specs = [spec.build(work_dir) for spec in self.source_specs]
        return [
            MixingDocumentSource(
                *source_specs,
                work_dir=work_dir,
                seed=self.seed,
                label=self.label,
            )
        ]


@dataclass
class MixingDocumentSourceSpec:
    """Defines a source and its associated mixing ratio for :class:`MixingDocumentSource`."""

    Config: ClassVar[Type["MixingDocumentSourceSpecConfig"]] = MixingDocumentSourceSpecConfig
    """The config class for this spec."""

    source: DocumentSource
    """The source."""
    ratio: float
    """The relative target ratio for this source."""
    size_adjustment_factor: float = 1.0
    """
    An optional factor to adjust the effective size of this source prior to determining how many
    tokens to sample. A factor less than 1.0 makes the source smaller, while a factor greater
    than 1.0 makes it larger by oversampling.

    Equivalently you could wrap the source in a :class:`SamplingDocumentSource` to adjust its size.
    """
    max_repetition_factor: float = 1.0
    """
    The maximum amount of repetition allowed after applying the ``size_adjustment_factor``,
    expressed as a factor greater than or equal to 1.0.
    A factor of 1.0 means no repetition is allowed. A factor of 2.0 means each document could be
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


class MixingDocumentSource(DocumentSource):
    """
    A document source for mixing other document sources together with arbitrary ratios.

    .. important::
        Sampling is done in a way that minimizes the number of dropped and repeated tokens while
        matching the target ratios and respecting the :data:`MixingDocumentSourceSpec.max_repetition_factor`
        values.

        The number of tokens this source produces will always be less than or equal to the
        sum of tokens across all of its immediate children defined in the ``source_specs``,
        after applying their respective :data:`MixingDocumentSourceSpec.size_adjustment_factor` values.

        You can always adjust the final size of a source by wrapping it in a :class:`SamplingDocumentSource`.

    :param source_specs: The sources and how to sample from them.
    """

    Config = MixingDocumentSourceConfig
    """The config class for this source."""

    Spec = MixingDocumentSourceSpec
    """The mixing spec class for this source."""

    DISPLAY_ICON = "\uf074"

    def __init__(
        self,
        *source_specs: MixingDocumentSourceSpec,
        work_dir: PathOrStr,
        seed: Optional[int] = None,
        label: Optional[str] = None,
    ):
        if not source_specs:
            raise OLMoConfigurationError("At least one source spec must be provided.")

        super().__init__(work_dir=work_dir, label=label)

        sources = [spec.source for spec in source_specs]

        # Determine the number of tokens to sample from each source.
        sample_sizes = calculate_sample_sizes(
            [int(len(spec.source) * spec.size_adjustment_factor) for spec in source_specs],
            [spec.ratio for spec in source_specs],
            [spec.max_repetition_factor for spec in source_specs],
        )

        # Sample documents from each source.
        sampled_sources: List[SamplingDocumentSource] = []
        for i, (spec, sample_size) in enumerate(zip(source_specs, sample_sizes)):
            sampled_sources.append(
                SamplingDocumentSource(
                    spec.source,
                    max_tokens=int(sample_size),
                    work_dir=work_dir,
                    seed=None if seed is None else seed + i,
                    label=spec.label or spec.source.label,
                )
            )
        self._source = ConcatenatedDocumentSource(*sampled_sources, work_dir=work_dir)

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
        log.info(f"Created document mixture consisting of:\n{summary}")

    @property
    def sampled_sources(self) -> Tuple[SamplingDocumentSource, ...]:
        return typing.cast(Tuple[SamplingDocumentSource, ...], self._source.sources)

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

    def get_document_offsets(self) -> Iterable[tuple[int, int]]:
        return self._source.get_document_offsets()

    def children(self):
        return self.sampled_sources
