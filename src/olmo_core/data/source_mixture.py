import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Tuple

import numpy as np
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.text import Text

from olmo_core.aliases import PathOrStr
from olmo_core.config import Config
from olmo_core.data.types import NumpyUIntTypes
from olmo_core.data.utils import bytes_to_tokens
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import get_file_size

__all__ = [
    "SourceMixtureConfig",
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
    paths: List[str]
    """
    A list of paths (or globs) specifiying the pre-tokenized numpy files that make up the source.
    """
    target_ratio: float
    """
    The target ratio of this source in the final mixture.
    """
    max_repetitions: float = 1.0
    """
    The maximum number of repetitions of the source data to include in the mixture.
    This can be used to upsample the source data by setting the number of repetitions > 1
    and allocating a target_ratio that requires >1 repetitions.
    """

    def validate(self):
        if not self.paths:
            raise OLMoConfigurationError("paths must not be empty")
        if not 0 < self.target_ratio <= 1:
            raise OLMoConfigurationError("target_ratio must be > 0 and <= 1")
        if self.max_repetitions < 1:
            raise OLMoConfigurationError("max_repetitions must be >= 1")


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
            "max_repetion_ratio": str(self.config.max_repetitions),
            "max_source_fraction": str(self.config.max_source_fraction),
            "observed_source_ratio": f"{(self.num_selected / self.population):.4}",
            "observed_global_ratio": f"{(self.num_selected / requested_tokens):.4}",
        }


@dataclass
class SourcePathTokens:
    path: str
    tokens: int


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

    source_configs: List[SourceMixtureConfig]
    """
    A list of source configurations.
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
    num_worker_threads: int = 1
    """
    The number of threads to use for counting tokens in parallel across source files.
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

        if not self.source_configs:
            raise OLMoConfigurationError("source_configs must not be empty")

        summed_weights = np.sum([source.target_ratio for source in self.source_configs])

        if not np.allclose(summed_weights, 1.0):
            raise OLMoConfigurationError(f"target_ratios must sum to 1.0, got {summed_weights}")

    def build(self, *, npdtype: NumpyUIntTypes, sequence_length: int) -> SourceMixtureDataset:
        self.validate()
        random.seed(self.seed)  # but we've already run seed_all()...

        log.info(f"Generating a mixture from {len(self.source_configs)} sources")

        # Calculate the number of tokens available and to include for each source
        tokens_details_by_source: List[SourceTokenDetails] = []
        for source_config in self.source_configs:
            tokens_in_source = self._count_tokens_for_source(
                source_config=source_config, npdtype=npdtype
            )
            needed_for_source = int(self.requested_tokens * source_config.target_ratio)
            max_for_source = int(
                tokens_in_source
                * source_config.max_source_fraction
                # why does max_repetitions affect the max number of tokens to take?
                # it should affect the max available to take!
                * source_config.max_repetitions
            )

            # Ensure that the max tokens for a source meet the target ratio requirement
            if max_for_source < needed_for_source:
                raise OLMoConfigurationError(
                    f"Insufficient tokens for source: {source_config.source_name} @ target global ratio: {source_config.target_ratio} :: {max_for_source} < {needed_for_source}"
                )

            tokens_details_by_source.append(
                SourceTokenDetails(
                    config=source_config,
                    population=tokens_in_source,
                    num_selected=needed_for_source,
                )
            )

        tokens_per_path_per_source: Dict[str, List[SourcePathTokens]] = {}
        for source in tokens_details_by_source:
            tokens_per_path = self.get_tokens_per_path(
                source_config=source.config, token_details=source, npdtype=npdtype
            )
            tokens_per_path_per_source[source.config.source_name] = tokens_per_path
        all_tokens_per_path = [
            path.tokens for paths in tokens_per_path_per_source.values() for path in paths
        ]

        # We adjust the number of tokens per individual path so that we can complete the desired
        # number of training steps while still retaining the target ratios at the source level.

        total_batches = math.ceil(self.requested_tokens / self.global_batch_size)
        assert self.global_batch_size % sequence_length == 0

        num_instances_per_batch = self.global_batch_size // sequence_length
        requested_instances = total_batches * num_instances_per_batch

        # determine how many instances we still need to allocate
        int_instances = []
        remainders = []
        for tokens_per_path in all_tokens_per_path:
            int_part = tokens_per_path // sequence_length
            remainder = (tokens_per_path % sequence_length) / sequence_length
            int_instances.append(int_part)
            remainders.append(remainder)

        # we apply Hamilton's method for rounding (see https://mathematics-democracy-institute.org/apportionment/)
        # that is, we only round up the requested instances for the paths that have the largest remainders
        # there is a lot of literature on how any other allocation of increments is suboptimal
        # basically, the requested tokens that are closest to a full instance are rounded up first
        additional_instances_needed = requested_instances - sum(int_instances)
        if additional_instances_needed > 0:
            # Distribute additional instances evenly across all paths first (base),
            # then allocate any remaining instances (leftover) to paths with largest remainders
            #
            # TODO: does this actually work? Can the dataset simply read the remainders like this?
            # Seems we're pretending these full instances actually exist?
            base, leftover = divmod(additional_instances_needed, len(remainders))
            if base:
                int_instances = [n + base for n in int_instances]
            if leftover:
                largest_remainder_indicies = sorted(
                    range(len(remainders)), key=remainders.__getitem__, reverse=True
                )
                for idx in largest_remainder_indicies[:leftover]:
                    int_instances[idx] += 1

        final_tokens_per_path = [inst * sequence_length for inst in int_instances]

        i = 0
        final_token_distribution: Dict[str, float] = {}
        completed: List[SourceMixtureOutcome] = []
        for source_name, source_path_tokens in tokens_per_path_per_source.items():
            for j in range(len(source_path_tokens)):
                source_path_tokens[j] = SourcePathTokens(
                    path=source_path_tokens[j].path, tokens=final_tokens_per_path[i]
                )
                i += 1

            completed.append(SourceMixtureOutcome(name=source_name, path_tokens=source_path_tokens))
            final_token_distribution[source_name] = sum(
                [path.tokens for path in source_path_tokens]
            )

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
            for source_config in self.source_configs
        }
        for source_name, ratio in original_token_distribution.items():
            diff = np.abs(final_token_distribution.get(source_name, 0) - ratio)
            log.info(f"{source_name}: {diff:.4f} difference from target ratio {ratio:.4f}")

        return SourceMixtureDataset(sources=completed)

    def get_tokens_per_path(
        self,
        source_config: SourceMixtureConfig,
        token_details: SourceTokenDetails,
        npdtype: NumpyUIntTypes,
    ) -> List[SourcePathTokens]:
        """
        Get the paths and resulting token count for a source.
        """
        take_ratio = token_details.num_selected / token_details.population
        path_tokens = []

        # When we need more than 1 repetition of the source data we have a take ratio > 1
        if take_ratio > 1:
            take_ratios = []
            remaining = take_ratio

            while remaining > 0:
                chunk = min(1.0, remaining)
                take_ratios.append(chunk)
                remaining -= chunk

            for ratio in take_ratios:
                for path in source_config.paths:
                    tokens_to_keep = int(
                        math.ceil(self._count_tokens_for_file(path, npdtype) * ratio)
                    )
                    path_tokens.append(SourcePathTokens(path=path, tokens=tokens_to_keep))

            return path_tokens

        for path in source_config.paths:
            # TODO: should this round to the nearest sequence length?
            tokens_to_keep = int(math.ceil(self._count_tokens_for_file(path, npdtype) * take_ratio))
            path_tokens.append(SourcePathTokens(path=path, tokens=tokens_to_keep))

        return path_tokens

    def _count_tokens_for_source(
        self, source_config: SourceMixtureConfig, npdtype: NumpyUIntTypes
    ) -> int:
        """
        Count the total number of tokens available for a source, across all files.

        Args:
            source_config: The source configuration.
            dtype: The data type of the source tokens.
        """
        log.info(f"Counting tokens for source: {source_config.source_name}")
        with ThreadPoolExecutor(max_workers=self.num_worker_threads) as executor:
            futures = []
            for path in source_config.paths:
                futures.append(
                    executor.submit(self._count_tokens_for_file, path=path, npdtype=npdtype)
                )

            with Progress(disable=self.quiet) as progress:
                results = []
                task = progress.add_task(
                    f"Counting available tokens for source: {source_config.source_name}",
                    total=len(futures),
                )
                for future in as_completed(futures):
                    progress.update(task, advance=1)
                    results.append(future.result())

            return sum(results)

    def _count_tokens_for_file(self, path: PathOrStr, npdtype: NumpyUIntTypes) -> int:
        return bytes_to_tokens(get_file_size(path), dtype=npdtype)

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
