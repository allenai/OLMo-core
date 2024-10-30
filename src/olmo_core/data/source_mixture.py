import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from olmo_core.aliases import PathOrStr
from olmo_core.config import Config
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import get_file_size

__all__ = [
    "SourceMixtureConfig",
    "SourceMixtureDataset",
    "SourceMixtureDatasetConfig",
]

log = logging.getLogger(__name__)


@dataclass
class SourceMixtureConfig(Config):
    """
    A configuration class for building a source mixture.
    """

    source_name: str
    """
    The name of the source.
    """
    target_ratio: float
    """
    The target ratio of the source in the mixture.
    """
    paths: List[PathOrStr]
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

    def validate(self):
        if self.target_ratio:
            if not 0 <= self.target_ratio <= 1:
                raise OLMoConfigurationError("target_ratio must be in the range [0, 1]")
            if not 0 <= self.max_source_fraction <= 1:
                raise OLMoConfigurationError("max_source_fraction must be in the range [0, 1]")
            if self.max_source_fraction < self.target_ratio:
                raise OLMoConfigurationError("max_source_fraction must be >= target_ratio")

        if self.max_repetition_ratio < 1:
            raise OLMoConfigurationError("max_repetition_ratio must be >= 1")

        if not self.paths:
            raise OLMoConfigurationError("paths must not be empty")

        if not 0 <= self.max_source_fraction <= 1:
            raise OLMoConfigurationError("max_source_fraction must be in the range [0, 1]")


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

    def for_table(self, max_tokens: int) -> Dict:
        return {
            "source_name": self.config.source_name,
            "source_population": f"{self.population:.2e}",
            "num_selected": f"{self.num_selected:.2e}",
            "target_ratio": str(self.config.target_ratio),
            "max_repetion_ratio": str(self.config.max_repetition_ratio),
            "max_source_fraction": str(self.config.max_source_fraction),
            "observed_source_ratio": f"{(self.num_selected / self.population):.4}",
            "observed_global_ratio": f"{(self.num_selected / max_tokens):.4}",
        }


@dataclass
class SourcePathTokens:
    path: PathOrStr
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
class SourceMixtureDataset:
    """
    A dataset consisting of a fractionalized mixture of data sources.
    """

    seed: int
    """
    The seed used to generate the dataset.
    """
    sources: List[SourceMixtureOutcome]
    """
    A list of sources and the associated paths and token counts.
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
    A configuration class for building a dataset from a fractionalized mixture of sources.
    """

    max_tokens: int
    """
    The maximum number of tokens to include in the dataset.
    """
    source_configs: List[SourceMixtureConfig]
    """
    A list of source configurations.
    """
    sequence_length: int
    """
    The instance sequence length of the dataset.
    """
    dtype: NumpyDatasetDType
    """
    The data type of the dataset.
    """
    processes: int = 1
    """
    The number of processes to use for counting tokens in parallel.
    """
    seed: int = 42
    """
    The seed used to generate the dataset.
    """

    def validate(self):
        if self.max_tokens <= 0:
            raise OLMoConfigurationError("max_tokens must be > 0")

        if not self.source_configs:
            raise OLMoConfigurationError("source_configs must not be empty")

        if (total := sum([source.target_ratio for source in self.source_configs])) != 1.0:
            raise OLMoConfigurationError(f"target_ratios must sum to 1, got {total}")

    def build(self) -> SourceMixtureDataset:
        self.validate()
        random.seed(self.seed)
        available_tokens_by_source: Dict[str, int] = {}

        log.info("---------------------------------------------------------")
        log.info("Generating a source mixture from configurations:")
        log.info(self.source_configs)

        # Count the number of tokens available for each source
        for source_config in self.source_configs:
            log.info(f"Counting tokens for source: {source_config.source_name}")
            available_tokens_by_source[source_config.source_name] = self._count_tokens_for_paths(
                paths=source_config.paths, source=source_config.source_name
            )

        tokens_details_by_source: List[SourceTokenDetails] = []

        # Calculate the number of tokens available and to include for each source
        for source_config in self.source_configs:
            num_for_source = available_tokens_by_source[source_config.source_name]
            needed_for_source = int(self.max_tokens * source_config.target_ratio)
            max_for_source = int(
                (num_for_source * source_config.max_source_fraction)
                * source_config.max_repetition_ratio
            )

            # Ensure that the max tokens for a source meet the target ratio requirement
            if max_for_source < needed_for_source:
                raise OLMoConfigurationError(
                    f"Insufficient tokens for source: {source_config.source_name} @ target global ratio: {source_config.target_ratio} :: {max_for_source} < {needed_for_source}"
                )

            tokens_details_by_source.append(
                SourceTokenDetails(
                    config=source_config,
                    population=num_for_source,
                    num_selected=needed_for_source,
                )
            )

        completed: List[SourceMixtureOutcome] = []
        for source in tokens_details_by_source:
            completed.append(
                SourceMixtureOutcome(
                    name=source.config.source_name,
                    path_tokens=self.get_paths_and_tokens_for_source(
                        source_config=source.config,
                        token_details=source,
                    ),
                )
            )

        self.render_mixture_outcome_tables(tokens_details_by_source)

        for outcome in completed:
            for item in outcome.path_tokens:
                log.info(f"Selected {item.tokens} tokens from {outcome.name} at {item.path}")

        return SourceMixtureDataset(seed=self.seed, sources=completed)

    def get_paths_and_tokens_for_source(
        self, source_config: SourceMixtureConfig, token_details: SourceTokenDetails
    ) -> List[SourcePathTokens]:
        """
        Get the paths and resulting token count for a source.
        """
        take_ratio = token_details.num_selected / token_details.population
        path_tokens = []

        # When we need more than 1 repetition of the source data we have a take ration > 1
        if take_ratio > 1:
            take_ratios = []
            remaining = take_ratio

            while remaining > 0:
                chunk = min(1.0, remaining)
                take_ratios.append(chunk)
                remaining -= chunk

            for ratio in take_ratios:
                for path in source_config.paths:
                    tokens_to_keep = int(math.ceil(self._count_tokens_for_file(path) * ratio))
                    path_tokens.append(SourcePathTokens(path=path, tokens=tokens_to_keep))

            return path_tokens

        for path in source_config.paths:
            tokens_to_keep = int(math.ceil(self._count_tokens_for_file(path) * take_ratio))
            path_tokens.append(SourcePathTokens(path=path, tokens=tokens_to_keep))

        return path_tokens

    def _count_tokens_for_paths(self, paths: List[PathOrStr], source: Optional[str]) -> int:
        """
        Count the number of tokens for a set of source files in parallel.

        Args:
            source_config: The source configuration.
            dtype: The data type of the source tokens.
        """

        with ThreadPoolExecutor(max_workers=self.processes) as executor:
            futures = []
            for path in paths:
                futures.append(executor.submit(self._count_tokens_for_file, path))

            with Progress() as progress:
                results = []
                task = progress.add_task(
                    f"Counting available tokens for source: {source}", total=len(futures)
                )
                for future in as_completed(futures):
                    progress.update(task, advance=1)
                    results.append(future.result())

            return sum(results)

    def _count_tokens_for_file(self, path: PathOrStr) -> int:
        return self._bytes_to_tokens(get_file_size(path), self.dtype)

    def _bytes_to_tokens(self, num_bytes: int, dtype: NumpyDatasetDType) -> int:
        """
        Convert bytes to tokens based on the dtype.
        """
        npdtype = dtype.as_np_dtype()
        return num_bytes // npdtype(int(0)).itemsize

    def render_mixture_outcome_tables(self, results: List[SourceTokenDetails]) -> None:
        """
        Render tables enumerating the global and per-source mixture outcomes.
        """

        console = Console()

        source_rows = [item.for_table(self.max_tokens) for item in results]
        source_headers = source_rows[0].keys()

        source_table = Table(title="Outcome by source")
        for header in source_headers:
            source_table.add_column(header)

        for row in source_rows:
            source_table.add_row(*[row[header] for header in source_headers])

        console.print(source_table)

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
        console.print(global_table)
