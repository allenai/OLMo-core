import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.text import Text

from olmo_core.aliases import PathOrStr
from olmo_core.config import Config
from olmo_core.data.types import NumpyUIntTypes
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import file_exists, get_file_size, glob_directory

__all__ = [
    "SourceMixtureConfig",
    "SourceMixtureList",
    "SourceMixtureDatasetConfig",
]

log = logging.getLogger(__name__)


@dataclass
class SourceMixtureConfig(Config):
    """
    Configuration for a single data source within a mixture.

    This class defines how a data source should be sampled and weighted when
    creating a training dataset from multiple sources. It allows control over
    the target proportion, repetition limits, and maximum usage fraction of
    the source data.
    """

    source_name: str
    """
    The name of the source.
    """
    target_ratio: float
    """
    The target ratio of the source in the mixture.
    """
    paths: List[str]
    """
    A list of paths to the source data.
    """
    max_repetition_ratio: float = 1.0
    """
    The maximum ratio of repetitions of the source data to include in the mixture.
    This can be used to upsample the source data by setting the repetition ratio > 1.
    """
    max_source_fraction: float = 1.0
    """
    The maximum ratio of the source data to include in the mixture.
    """

    _resolved_paths: Optional[List[str]] = None

    def validate(self):
        if self.target_ratio:
            if not 0 < self.target_ratio <= 1:
                raise OLMoConfigurationError("target_ratio must be > 0 and <= 1")
            if not 0 < self.max_source_fraction <= 1:
                raise OLMoConfigurationError("max_source_fraction must > 0 and <= 1")

        if self.max_repetition_ratio < 1:
            raise OLMoConfigurationError("max_repetition_ratio must be >= 1")

        if not self.paths:
            raise OLMoConfigurationError("paths must not be empty")

        if not 0 <= self.max_source_fraction <= 1:
            raise OLMoConfigurationError("max_source_fraction must be in the range [0, 1]")

    @property
    def resolved_paths(self) -> List[str]:
        """
        Resolve the paths, expanding any globs and validating existence.
        Caches the result after the first access.
        """
        if self._resolved_paths is not None:
            return self._resolved_paths

        resolved: List[str] = []
        for path in self.paths:
            path_str = str(path)
            if "*" in path_str:
                matches = sorted(glob_directory(path_str))
                if not matches:
                    raise FileNotFoundError(f"Glob pattern '{path_str}' did not match any files")
                resolved.extend(matches)
            else:
                if not file_exists(path_str):
                    raise FileNotFoundError(f"Path '{path_str}' does not exist")
                resolved.append(path_str)

        self._resolved_paths = resolved
        return resolved


@dataclass
class SourceMixtureList(Config):
    """
    A list of source configurations for building a mixture dataset.
    This class ensures that the target ratios of the sources sum to 1.0.

    The purpose of this class is to make managing sources independent from the details of
    materializing those sources with SourceMixtureDatasetConfig.build().

    With this separation, we can define a list of sources in a YAML file without also needing to
    specify parameters like requested_tokens, global_batch_size, or processes.
    """

    sources: List[SourceMixtureConfig]

    def validate(self):
        if not self.sources:
            raise OLMoConfigurationError("sources must not be empty")

        summed_weights = np.sum([source.target_ratio for source in self.sources])

        if not np.allclose(summed_weights, 1.0):
            raise OLMoConfigurationError(f"target_ratios must sum to 1.0, got {summed_weights}")


@dataclass
class SourceTokenDetails:
    """
    A class to hold intermediate selection details for a mixture source.
    """

    config: SourceMixtureConfig
    """
    The configuration object associated with the source.
    """
    population: int
    """
    The total number of tokens available for the source.
    """
    num_selected: int
    """
    The number of tokens to select for the source.
    """

    def for_table(self, requested_tokens: int) -> Dict:
        return {
            "source_name": self.config.source_name,
            "source_population": f"{self.population:.2e}",
            "num_selected": f"{self.num_selected:.2e}",
            "target_ratio": str(self.config.target_ratio),
            "max_repetion_ratio": str(self.config.max_repetition_ratio),
            "max_source_fraction": str(self.config.max_source_fraction),
            "observed_source_ratio": f"{(self.num_selected / self.population):.4}",
            "observed_global_ratio": f"{(self.num_selected / requested_tokens):.4}",
        }


@dataclass
class SourcePathTokens:
    path: str
    tokens: int
    max_tokens: int


@dataclass
class SourceMixtureOutcome:
    name: str
    """
    The name of the source.
    """
    path_tokens: List[SourcePathTokens]
    """
    A list of paths and the associated token counts.
    """


@dataclass
class SourceMixtureDataset:  # Note: "dataset" naming is a bit inconsistent with the rest of the codebase
    """
    A container for a fractionalized mixture of data sources. Do not construct directly,
    use :class:`SourceMixtureDatasetConfig` instead.

    See also :class:`~olmo_core.data.numpy_dataset.NumpyFSLDatasetMixture`, the downstream
    consumer of this dataset.
    """

    sources: List[SourceMixtureOutcome]
    """
    A list of sources and their associated paths and token counts.
    """

    def to_index(self) -> Dict[Tuple[str, int], int]:
        """
        Convert the dataset to an indexed array of dict((int, path), int).
        """
        return {
            (str(outcome.path), idx): outcome.tokens
            for idx, outcome in enumerate(
                list(chain.from_iterable([outcome.path_tokens for outcome in self.sources]))
            )
        }

    def to_paths(self) -> List[PathOrStr]:
        """
        Convert the dataset to a list of paths while maintaining stable ordering.
        """
        return [
            item.path
            for item in list(chain.from_iterable([outcome.path_tokens for outcome in self.sources]))
        ]


@dataclass
class SourceMixtureDatasetConfig(Config):
    """
    Configuration for building a dataset from a fractionalized mixture of sources.

    This class manages the creation of training datasets by combining multiple data sources
    according to specified target ratios. It handles token counting, source selection,
    and ensures the final mixture meets the requested dataset size while maintaining
    the desired proportions across sources.

    The build process will:
    1. Count available tokens in each source
    2. Calculate token allocations based on target ratios
    3. Validate that sources have sufficient data
    4. Generate a mixture that respects repetition and fraction limits
    """

    source_list: SourceMixtureList
    """
    A list of source configurations contained in a SourceMixtureList.
    """
    requested_tokens: int
    """
    The desired dataset size, in tokens. This is used to determine the number of tokens to select from each source.
    The total dataset size will be greater than or equal to this value, depending on rounding.
    """
    global_batch_size: int
    """
    The global batch size for training, in tokens. Used to determine the total number of requested instances.
    """
    processes: int = 1
    """
    The number of processes to use for counting tokens in parallel.
    """
    seed: int = 42
    """
    The seed used to generate the dataset.
    """
    render_tables: bool = True
    """
    Whether to render tables of the mixture outcome.
    """
    quiet: bool = False

    def validate(self):
        if self.requested_tokens <= 0:
            raise OLMoConfigurationError("requested_tokens must be > 0")
        self.source_list.validate()

    def build(self, *, npdtype: NumpyUIntTypes, sequence_length: int) -> SourceMixtureDataset:
        self.validate()
        random.seed(self.seed)
        available_tokens_by_source: Dict[str, int] = {}

        log.info("---------------------------------------------------------")
        log.info("Generating a source mixture from configurations:")
        log.info(self.source_list.sources)

        # Count the number of tokens available for each source
        for source_config in self.source_list.sources:
            log.info(f"Counting tokens for source: {source_config.source_name}")
            available_tokens_by_source[source_config.source_name] = self._count_tokens_for_paths(
                paths=cast(List[PathOrStr], source_config.resolved_paths),
                source=source_config.source_name,
                npdtype=npdtype,
            )

        tokens_details_by_source: List[SourceTokenDetails] = []
        max_tokens_cap_by_source: Dict[str, int] = {}

        # Calculate the number of tokens available and to include for each source
        for source_config in self.source_list.sources:
            num_for_source = available_tokens_by_source[source_config.source_name]
            needed_for_source = int(self.requested_tokens * source_config.target_ratio)
            max_for_source = int(
                (num_for_source * source_config.max_source_fraction)
                * source_config.max_repetition_ratio
            )

            # Ensure that the max tokens for a source meet the target ratio requirement
            if max_for_source < needed_for_source:
                raise OLMoConfigurationError(
                    f"Insufficient tokens for source: {source_config.source_name} @ target global ratio: {source_config.target_ratio} :: {max_for_source} < {needed_for_source}"
                )

            max_tokens_cap_by_source[source_config.source_name] = max_for_source

            tokens_details_by_source.append(
                SourceTokenDetails(
                    config=source_config,
                    population=num_for_source,
                    num_selected=needed_for_source,
                )
            )

        completed: List[SourceMixtureOutcome] = []
        tokens_per_path_per_source: Dict[str, List[SourcePathTokens]] = {}
        for source in tokens_details_by_source:
            source_path_tokens = self.get_paths_and_tokens_for_source(
                source_config=source.config, token_details=source, npdtype=npdtype
            )
            tokens_per_path_per_source[source.config.source_name] = source_path_tokens

        # We adjust the number of tokens per path so that we can complete the desired number of training steps while still retaining the target ratios.
        training_steps = math.ceil(self.requested_tokens / self.global_batch_size)
        assert self.global_batch_size % sequence_length == 0, (
            "global_batch_size must be multiple of sequence_length"
        )
        num_instances_per_batch = self.global_batch_size // sequence_length
        requested_instances = training_steps * num_instances_per_batch

        all_entries: List[Tuple[str, SourcePathTokens]] = []
        for source_name, source_path_tokens in tokens_per_path_per_source.items():
            for path_token in source_path_tokens:
                all_entries.append((source_name, path_token))

        base_instances: List[int] = []
        remainders: List[float] = []
        max_instances_available: List[int] = []
        for source_name, path_token in all_entries:
            max_instances = path_token.max_tokens // sequence_length
            max_instances_available.append(max_instances)
            desired_instances = path_token.tokens // sequence_length
            assigned_instances = min(desired_instances, max_instances)
            base_instances.append(assigned_instances)
            remainder = 0.0
            if max_instances > 0 and assigned_instances < max_instances:
                remainder = (path_token.tokens % sequence_length) / sequence_length
            remainders.append(remainder)

        source_tokens_assigned: Dict[str, int] = {name: 0 for name in tokens_per_path_per_source}
        for (source_name, _), instances in zip(all_entries, base_instances):
            source_tokens_assigned[source_name] += instances * sequence_length

        source_remaining_tokens: Dict[str, int] = {
            name: max(0, max_tokens_cap_by_source[name] - source_tokens_assigned[name])
            for name in source_tokens_assigned
        }

        additional_instances_needed = requested_instances - sum(base_instances)

        if additional_instances_needed > 0:
            fractional_candidates = [
                (remainders[idx], idx)
                for idx in range(len(base_instances))
                if remainders[idx] > 0
                and base_instances[idx] < max_instances_available[idx]
                and source_remaining_tokens[all_entries[idx][0]] >= sequence_length
            ]
            fractional_candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
            for _, idx in fractional_candidates:
                if additional_instances_needed == 0:
                    break
                source_name, _ = all_entries[idx]
                if source_remaining_tokens[source_name] < sequence_length:
                    continue
                base_instances[idx] += 1
                source_remaining_tokens[source_name] -= sequence_length
                additional_instances_needed -= 1

            while additional_instances_needed > 0:
                capacity_indices = [
                    idx
                    for idx in range(len(base_instances))
                    if base_instances[idx] < max_instances_available[idx]
                    and source_remaining_tokens[all_entries[idx][0]] >= sequence_length
                ]
                if not capacity_indices:
                    break
                for idx in capacity_indices:
                    if additional_instances_needed == 0:
                        break
                    source_name, _ = all_entries[idx]
                    if base_instances[idx] >= max_instances_available[idx]:
                        continue
                    if source_remaining_tokens[source_name] < sequence_length:
                        continue
                    base_instances[idx] += 1
                    source_remaining_tokens[source_name] -= sequence_length
                    additional_instances_needed -= 1

            if additional_instances_needed > 0:
                log.warning(
                    "Unable to allocate %d additional instances due to capacity constraints.",
                    additional_instances_needed,
                )
        elif additional_instances_needed < 0:
            reduction_needed = -additional_instances_needed
            removal_candidates = [
                (remainders[idx], idx)
                for idx in range(len(base_instances))
                if base_instances[idx] > 0
            ]
            removal_candidates.sort(key=lambda item: (item[0], -item[1]))
            while reduction_needed > 0 and removal_candidates:
                for _, idx in removal_candidates:
                    if reduction_needed == 0:
                        break
                    if base_instances[idx] == 0:
                        continue
                    source_name, _ = all_entries[idx]
                    base_instances[idx] -= 1
                    source_remaining_tokens[source_name] += sequence_length
                    reduction_needed -= 1
                removal_candidates = [
                    (remainders[idx], idx)
                    for idx in range(len(base_instances))
                    if base_instances[idx] > 0
                ]
                removal_candidates.sort(key=lambda item: (item[0], -item[1]))
            if reduction_needed > 0:
                log.warning(
                    "Unable to remove %d excess instances due to minimum capacity constraints.",
                    reduction_needed,
                )

        final_tokens_per_path = [instances * sequence_length for instances in base_instances]

        for (source_name, path_token), tokens in zip(all_entries, final_tokens_per_path):
            path_token.tokens = tokens

        final_token_distribution: Dict[str, float] = {}
        for source_name, source_path_tokens in tokens_per_path_per_source.items():
            completed.append(
                SourceMixtureOutcome(
                    name=source_name,
                    path_tokens=source_path_tokens,
                )
            )
            final_token_distribution[source_name] = sum(path.tokens for path in source_path_tokens)

        total_tokens = sum(final_token_distribution.values())
        final_token_distribution = {
            k: v / total_tokens for k, v in final_token_distribution.items()
        }

        if self.render_tables:
            self.render_mixture_outcome_tables(tokens_details_by_source)

        for outcome in completed:
            for item in outcome.path_tokens:
                log.info(f"Selected {item.tokens} tokens from {outcome.name} at {item.path}")

        token_difference = total_tokens - self.requested_tokens
        percent_difference = (token_difference / self.requested_tokens) * 100
        log.info(
            f"Total tokens in mixture: {total_tokens} "
            f"(requested: {self.requested_tokens}, diff: {token_difference:+} tokens, "
            f"{percent_difference:+.2f}%)"
        )

        original_token_distribution = {
            source_config.source_name: source_config.target_ratio
            for source_config in self.source_list.sources
        }
        for source_name, ratio in original_token_distribution.items():
            diff = np.abs(final_token_distribution.get(source_name, 0) - ratio)
            log.info(f"{source_name}: {diff:.4f} difference from target ratio {ratio:.4f}")

        return SourceMixtureDataset(sources=completed)

    def get_paths_and_tokens_for_source(
        self,
        source_config: SourceMixtureConfig,
        token_details: SourceTokenDetails,
        npdtype: NumpyUIntTypes,
    ) -> List[SourcePathTokens]:
        """
        Get the paths and resulting token count for a source.
        """
        take_ratio = token_details.num_selected / token_details.population
        path_tokens: List[SourcePathTokens] = []

        resolved_paths = source_config.resolved_paths
        token_counts_by_path = {
            path: self._count_tokens_for_file(path, npdtype) for path in resolved_paths
        }

        # When we need more than 1 repetition of the source data we have a take ration > 1
        if take_ratio > 1:
            take_ratios = []
            remaining = take_ratio

            while remaining > 0:
                chunk = min(1.0, remaining)
                take_ratios.append(chunk)
                remaining -= chunk

            for ratio in take_ratios:
                for path in resolved_paths:
                    available_tokens = token_counts_by_path[path]
                    tokens_to_keep = int(math.ceil(available_tokens * ratio))
                    path_tokens.append(
                        SourcePathTokens(
                            path=path,
                            tokens=tokens_to_keep,
                            max_tokens=available_tokens,
                        )
                    )

            return path_tokens

        for path in resolved_paths:
            available_tokens = token_counts_by_path[path]
            tokens_to_keep = int(math.ceil(available_tokens * take_ratio))
            path_tokens.append(
                SourcePathTokens(
                    path=path,
                    tokens=tokens_to_keep,
                    max_tokens=available_tokens,
                )
            )

        return path_tokens

    def _count_tokens_for_paths(
        self, paths: List[PathOrStr], source: Optional[str], npdtype: NumpyUIntTypes
    ) -> int:
        """
        Count the number of tokens for a set of source files in parallel.

        Args:
            source_config: The source configuration.
            dtype: The data type of the source tokens.
        """

        with ThreadPoolExecutor(max_workers=self.processes) as executor:
            futures = []
            for path in paths:
                futures.append(
                    executor.submit(self._count_tokens_for_file, path=path, npdtype=npdtype)
                )

            with Progress(disable=self.quiet) as progress:
                results = []
                task = progress.add_task(
                    f"Counting available tokens for source: {source}", total=len(futures)
                )
                for future in as_completed(futures):
                    progress.update(task, advance=1)
                    results.append(future.result())

            return sum(results)

    def _count_tokens_for_file(self, path: PathOrStr, npdtype: NumpyUIntTypes) -> int:
        return self._bytes_to_tokens(get_file_size(path), npdtype=npdtype)

    def _bytes_to_tokens(self, num_bytes: int, npdtype: NumpyUIntTypes) -> int:
        """
        Convert bytes to tokens based on the dtype.
        """
        return num_bytes // npdtype(int(0)).itemsize

    def render_mixture_outcome_tables(self, results: List[SourceTokenDetails]) -> None:
        """
        Render tables enumerating the global and per-source mixture outcomes.
        """
        source_rows = [item.for_table(self.requested_tokens) for item in results]
        source_headers = source_rows[0].keys()

        source_table = Table(title="Outcome by source")
        for header in source_headers:
            source_table.add_column(header)

        for row in source_rows:
            source_table.add_row(*[row[header] for header in source_headers])

        log.info(self.table_to_text(source_table))

        total_tokens = sum([item.population for item in results])
        selected_tokens = sum([item.num_selected for item in results])
        observed_global_ratio = f"{(selected_tokens / total_tokens):.4}"

        global_table = Table(title="Global outcome")
        global_headers = [
            "total_tokens",
            "selected_tokens",
            "observed_global_ratio",
        ]

        for header in global_headers:
            global_table.add_column(header)

        global_table.add_row(f"{total_tokens:.2e}", f"{selected_tokens:.2e}", observed_global_ratio)
        log.info(self.table_to_text(global_table))

    def table_to_text(self, table: Table) -> Text:
        """Generate an ascii formatted presentation of a Rich table
        Eliminates column styling
        """
        console = Console(width=250)
        with console.capture() as capture:
            table.width = 250
            console.print(table)

        return Text.from_ansi(capture.get())
