import dataclasses
import logging
from abc import abstractmethod
from dataclasses import dataclass

import yaml
from cached_path import cached_path
from olmo_core.aliases import PathOrStr
from olmo_core.config import Config, Registrable
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import add_cached_path_clients
from typing_extensions import Self

from dataclass_extensions import decode

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
    """Set to -1.0 to allow for as much repetition as needed to fulfill the mix, or to a value
    greater than 1.0 to get an exact amount of repetition."""


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
class NumpySamplingStrategy(Registrable, Config):
    """
    A strategy for sampling from a mixture of numpy sources.
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


@NumpySamplingStrategy.register("documents")
@dataclass
class NumpyDocumentSamplingStrategy(NumpySamplingStrategy):
    """
    Samples documents from the sources, and concatenates them until reaching the target number of tokens.
    """

    long_doc_strategy: LongDocStrategy = LongDocStrategy.fragment
    """How to handle long documents."""

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
        if repetition_factor == 1.0 or repetition_factor == -1.0:
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


@NumpySamplingStrategy.register("contiguous_chunks")
@dataclass
class NumpyContiguousChunksSamplingStrategy(NumpySamplingStrategy):
    """
    Samples contiguous chunks of tokens from the sources until reaching the target number of tokens.
    """

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
        if repetition_factor == 1.0 or repetition_factor == -1.0:
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


def build_numpy_mixture_from_yaml_spec(
    spec_path: PathOrStr,
    *,
    tokenizer: TokenizerConfig,
    total_tokens: int,
    sequence_length: int,
    sampling_strategy: NumpySamplingStrategy | str,
) -> InstanceSourceConfig:
    """
    This is a convenience function for building a mixture of numpy sources from a YAML specification file.

    :param tokenizer: The config of the tokenizer uses to create the numpy sources.
    :param total_tokens: The total target number of tokens to sample.
    :param sequence_length: The sequence length to use when sampling from the sources.
    :param sampling_strategy: The strategy to use when sampling from the sources.
        Can be either a string (the name of a registered strategy), e.g. "documents" or "contiguous_chunks",
        or an instance of `NumpySamplingStrategy`.

    The YAML file must have a top-level key "mix", which should be a list of sources or source groups.
    Each source (leaf node) should have the following keys:

    - name: A unique name for the source.
    - weight: The weight of the source in the mixture (should sum to 1.0 across all sources *within its parent group*).
    - paths: A list of file paths to sample from.

    And each source group (non-leaf node) should have the following keys:

    - name: A unique name for the group.
    - weight: The weight of the group in the mixture (should sum to 1.0 across all groups *within its parent group*).
    - categories: A list of sources or more source groups.

    .. code-block:: yaml

        mix:
        - name: science_math_and_technology
          weight: 0.50
          categories:
          - name: high quality
            weight: 0.5
            paths:
            - s3://ai2-llm/preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer/science_math_and_technology/vigintile_0020/*.npy
          - name: med quality
            weight: 0.3
            paths:
            - s3://ai2-llm/preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer/science_math_and_technology/vigintile_0018/*.npy
          - name: low quality
            weight: 0.2
            paths:
            - s3://ai2-llm/preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer/science_math_and_technology/vigintile_0017/*.npy
            - s3://ai2-llm/preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer/science_math_and_technology/vigintile_0016/*.npy
        - name: software_development
          weight: 0.50
          categories:
          - name: high quality
            weight: 0.5
            paths:
            - s3://ai2-llm/preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer/software_development/vigintile_0020/*.npy
          - name: med quality
            weight: 0.3
            paths:
            - s3://ai2-llm/preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer/software_development/vigintile_0018/*.npy
          - name: low quality
            weight: 0.2
            paths:
            - s3://ai2-llm/preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer/software_development/vigintile_0017/*.npy
            - s3://ai2-llm/preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer/software_development/vigintile_0016/*.npy
    """
    if isinstance(sampling_strategy, str):
        sampling_strategy = NumpySamplingStrategy.get_registered_class(sampling_strategy)()
    assert isinstance(sampling_strategy, NumpySamplingStrategy)

    if spec_path.startswith("weka://"):
        add_cached_path_clients()
    with cached_path(spec_path).open() as f:
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
        if (
            num_tokens * source_spec.repetition_factor < num_tokens_needed
            and source_spec.repetition_factor != -1.0
        ):
            raise OLMoConfigurationError(
                f"'{source_spec.name}' doesn't have enough tokens to fulfill the mix. "
                "You can allow for some repetition to get the desired number of tokens by setting "
                "the 'repetition_factor' for this source to a value greater than 1.0 "
                "(e.g. 2.0 to allow for up to 2 repetitions of the source), or to -1.0 to use just as "
                "much repetition as needed."
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
