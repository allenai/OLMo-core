import dataclasses
import logging
from abc import abstractmethod
from dataclasses import dataclass

import yaml
from dataclass_extensions import decode
from typing_extensions import Self

from olmo_core.aliases import PathOrStr
from olmo_core.config import Config, Registrable
from olmo_core.exceptions import OLMoConfigurationError

from ..tokenizer import TokenizerConfig
from ..types import LongDocStrategy
from .concat_and_chunk_instance_source import ConcatAndChunkInstanceSourceConfig
from .instance_source import InstanceSourceConfig
from .numpy_document_source import NumpyDocumentSourceConfig
from .sampling_document_source import SamplingDocumentSourceConfig
from .sampling_token_source import SamplingTokenSourceConfig
from .token_source import TokenSourceConfig

log = logging.getLogger(__name__)


@dataclass
class _MixtureSourceSpec:
    name: str
    weight: float
    paths: list[str]
    repetition_factor: float = 1.0


@dataclass
class _MixtureSourceCategories:
    name: str
    weight: float
    categories: list[Self | _MixtureSourceSpec]


@dataclass
class _MixtureSources:
    sources: list[_MixtureSourceCategories | _MixtureSourceSpec]


def _flatten_sources(
    sources: list[_MixtureSourceCategories | _MixtureSourceSpec],
    prefix: str = "",
    remaining_weight: float = 1.0,
) -> list[_MixtureSourceSpec]:
    """
    Flatten a hierarchical collection of sources.
    """
    total_weight = 0.0
    out: list[_MixtureSourceSpec] = []
    for source in sources:
        if isinstance(source, _MixtureSourceSpec):
            total_weight += source.weight
            out.append(
                dataclasses.replace(
                    source,
                    name=prefix + source.name,
                    weight=remaining_weight * source.weight,
                )
            )
        elif isinstance(source, _MixtureSourceCategories):
            total_weight += source.weight
            sub_group_prefix = prefix + source.name + " ❯ "
            out.extend(
                _flatten_sources(
                    source.categories,  # type: ignore
                    sub_group_prefix,
                    remaining_weight=remaining_weight * source.weight,
                )
            )
        else:
            raise TypeError(source)

    if round(total_weight, 4) != 1.0:
        raise OLMoConfigurationError(
            f"weights within {prefix} do not sum to 1.0! Got {total_weight:.4f}."
        )
    return out


@dataclass
class SamplingStrategy(Registrable, Config):
    """
    A strategy for sampling from a mixture of sources.
    """

    @abstractmethod
    def build_sources(
        self,
        source_groups: dict[str, list[str]],
        *,
        sequence_length: int,
        tokenizer: TokenizerConfig,
    ) -> dict[str, NumpyDocumentSourceConfig]:
        """
        Build a source to sample from.
        """
        pass

    @abstractmethod
    def sample_source(
        self,
        source: NumpyDocumentSourceConfig,
        *,
        label: str,
        sequence_length: int,
        num_tokens_needed: int,
        repetition_factor: float,
    ) -> TokenSourceConfig:
        """
        Sample from the given source according to this strategy.
        """
        pass


@SamplingStrategy.register("documents")
@dataclass
class DocumentSamplingStrategy(SamplingStrategy):
    long_doc_strategy: LongDocStrategy = LongDocStrategy.fragment

    def build_sources(
        self,
        source_groups: dict[str, list[str]],
        *,
        sequence_length: int,
        tokenizer: TokenizerConfig,
    ) -> dict[str, NumpyDocumentSourceConfig]:
        return NumpyDocumentSourceConfig.from_source_groups(
            source_groups,  # type: ignore[arg-type]
            tokenizer=tokenizer,
            max_document_length=sequence_length,
            long_doc_strategy=self.long_doc_strategy,
        )

    def sample_source(
        self,
        source: NumpyDocumentSourceConfig,
        *,
        label: str,
        sequence_length: int,
        num_tokens_needed: int,
        repetition_factor: float,
    ) -> SamplingDocumentSourceConfig:
        if repetition_factor == 1.0:
            return SamplingDocumentSourceConfig(
                sources=[source], max_tokens=num_tokens_needed, label=label
            )
        else:
            num_tokens_to_sample = sequence_length * round(
                num_tokens_needed / repetition_factor / sequence_length
            )
            return SamplingDocumentSourceConfig(
                sources=[SamplingDocumentSourceConfig([source], max_tokens=num_tokens_to_sample)],
                factor=repetition_factor,
                label=label,
            )


@SamplingStrategy.register("contiguous_chunks")
@dataclass
class ContiguousChunksSamplingStrategy(SamplingStrategy):
    def build_sources(
        self,
        source_groups: dict[str, list[str]],
        *,
        sequence_length: int,
        tokenizer: TokenizerConfig,
    ) -> dict[str, NumpyDocumentSourceConfig]:
        return NumpyDocumentSourceConfig.from_source_groups(
            source_groups,  # type: ignore[arg-type]
            tokenizer=tokenizer,
            max_document_length=sequence_length,
        )

    def sample_source(
        self,
        source: NumpyDocumentSourceConfig,
        *,
        label: str,
        sequence_length: int,
        num_tokens_needed: int,
        repetition_factor: float,
    ) -> SamplingTokenSourceConfig:
        if repetition_factor == 1.0:
            return SamplingTokenSourceConfig(
                sources=[source], max_tokens=num_tokens_needed, label=label
            )
        else:
            num_tokens_to_sample = sequence_length * round(
                num_tokens_needed / repetition_factor / sequence_length
            )
            return SamplingTokenSourceConfig(
                sources=[SamplingDocumentSourceConfig([source], max_tokens=num_tokens_to_sample)],
                factor=repetition_factor,
                label=label,
            )


def build_mixture_from_file(
    path: PathOrStr,
    *,
    tokenizer: TokenizerConfig,
    total_tokens: int,
    sequence_length: int,
    sampling_strategy: SamplingStrategy | str,
) -> InstanceSourceConfig:
    if isinstance(sampling_strategy, str):
        sampling_strategy = SamplingStrategy.get_registered_class(sampling_strategy)()
    assert isinstance(sampling_strategy, SamplingStrategy)

    with open(path) as f:
        raw_config = yaml.safe_load(f)
        mixture_specs = _flatten_sources(
            decode(_MixtureSources, {"sources": raw_config["mix"]}).sources
        )

    sources = sampling_strategy.build_sources(
        {source_spec.name: source_spec.paths for source_spec in mixture_specs},
        sequence_length=sequence_length,
        tokenizer=tokenizer,
    )

    sampled_sources: list[TokenSourceConfig] = []
    for source_spec in mixture_specs:
        source = sources[source_spec.name]
        num_tokens = source.get_num_tokens()
        num_tokens_needed = sequence_length * round(
            total_tokens * source_spec.weight / sequence_length
        )
        if num_tokens * source_spec.repetition_factor < num_tokens_needed:
            raise OLMoConfigurationError(
                f"'{source_spec.name}' doesn't have enough tokens to fulfill the mix"
            )

        if source_spec.weight == 0.0:
            log.warning(f"the mix requests 0 weight from '{source_spec.name}'")
            continue

        if num_tokens_needed == 0:
            log.warning(f"the mix requests 0 tokens from '{source_spec.name}' after rounding")
            continue

        sampled_source = sampling_strategy.sample_source(
            source,
            label=source_spec.name,
            sequence_length=sequence_length,
            num_tokens_needed=num_tokens_needed,
            repetition_factor=source_spec.repetition_factor,
        )
        sampled_sources.append(sampled_source)

    return ConcatAndChunkInstanceSourceConfig(
        sources=sampled_sources,
        sequence_length=sequence_length,
    )
