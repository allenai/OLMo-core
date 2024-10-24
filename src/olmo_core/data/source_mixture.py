import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import chain
from pprint import pprint
from typing import Dict, List, Optional

import tabulate
from tqdm import tqdm

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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Disable some noisy loggers
for name in logging.Logger.manager.loggerDict.keys():
    if name in (
        "boto",
        "urllib3",
        "s3transfer",
        "boto3",
        "botocore",
        "aiobotocore",
        "nose",
    ):
        logging.getLogger(name).setLevel(logging.CRITICAL)


@dataclass
class SourceMixtureConfig(Config):
    source_name: str
    target_ratio: float
    paths: List[PathOrStr]
    # 1.0 will result in a maximum of 1 repitition of the source data per epoch
    max_repetition_ratio: float = 1.0
    max_source_fraction: float = 1.0

    def validate(self):
        if self.target_ratio:
            if not 0 <= self.target_ratio <= 1:
                raise OLMoConfigurationError("target_ratio must be in the range [0, 1]")
            if not 0 <= self.max_source_fraction <= 1:
                raise OLMoConfigurationError("max_source_fraction must be in the range [0, 1]")
            if self.max_source_fraction < self.target_ratio:
                raise OLMoConfigurationError("max_source_fraction must be >= target_ratio")

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
    population: int
    num_selected: int

    def for_table(self, max_tokens: int) -> Dict:
        return {
            "source_name": self.config.source_name,
            "source_population": f"{self.population:.2e}",
            "num_sampled": f"{self.num_selected:.2e}",
            "target_ratio": self.config.target_ratio,
            "max_repetion_ratio": self.config.max_repetition_ratio,
            "max_source_fraction": self.config.max_source_fraction,
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
    path_tokens: List[SourcePathTokens]


@dataclass
class SourceMixtureDataset:
    """
    A dataset consisting of a fractionalized mixture of data sources.
    """

    sources: List[SourceMixtureOutcome]

    def to_path_instance_index(self) -> Dict[str, int]:
        """
        Convert the dataset to a dictionary of paths and instance counts to retain.
        """
        outcomes = chain.from_iterable([outcome.path_tokens for outcome in self.sources])
        return {str(outcome.path): outcome.tokens for outcome in outcomes}


@dataclass
class SourceMixtureDatasetConfig(Config):
    """
    A configuration class for building a dataset from a fractionalized mixture of sources.
    """

    max_tokens: int
    source_configs: List[SourceMixtureConfig]
    sequence_length: int
    dtype: NumpyDatasetDType
    processes: int = 1
    seed: int = 42

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

        print("--------------------------------------------------------------------------------")
        print("Generating a source mixture from configurations:")
        for source_config in self.source_configs:
            pprint(source_config)

        # Count the number of tokens available for each source
        for source_config in self.source_configs:
            log.info(f"Counting tokens for source: {source_config.source_name}")
            available_tokens_by_source[source_config.source_name] = self._count_tokens_for_paths(
                paths=source_config.paths, source=source_config.source_name
            )

        tokens_details_by_source: List[SourceTokenDetails] = []

        # Calculate the number of tokens to include for each source
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
                        take_ratio=source.num_selected / source.population,
                    ),
                )
            )

        log.info("Outcome by source => ")
        print(
            tabulate.tabulate(
                [item.for_table(self.max_tokens) for item in tokens_details_by_source],
                headers="keys",
                tablefmt="pretty",
            ),
        )

        total_tokens = sum([item.population for item in tokens_details_by_source])
        selected_tokens = sum([item.num_selected for item in tokens_details_by_source])
        observed_global_ratio = selected_tokens / total_tokens

        log.info("Global outcome => ")
        print(
            tabulate.tabulate(
                [
                    {
                        "total_tokens": f"{total_tokens:.2e}",
                        "selected_tokens": f"{selected_tokens:.2e}",
                        "observed_global_ratio": f"{observed_global_ratio:.4}",
                    }
                ],
                tablefmt="pretty",
                headers="keys",
            ),
        )

        for outcome in completed:
            for item in outcome.path_tokens:
                log.info(f"Selected {item.tokens} tokens from {outcome.name} at {item.path}")

        return SourceMixtureDataset(completed)

    def get_paths_and_tokens_for_source(
        self, source_config: SourceMixtureConfig, take_ratio: float
    ) -> List[SourcePathTokens]:
        """
        Get the paths and resulting token count for a source.
        """
        # TODO: Handle repetition ratio by adding paths multiple times, max_repetition_ratio
        path_tokens = []
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

            return sum(
                [
                    future.result()
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"Counting tokens {'for ' + source if source else ''}",
                    )
                ]
            )

    def _count_tokens_for_file(self, path: PathOrStr) -> int:
        return self._bytes_to_tokens(get_file_size(path), self.dtype)

    def _bytes_to_tokens(self, num_bytes: int, dtype: NumpyDatasetDType) -> int:
        """
        Convert bytes to tokens based on the dtype.
        """
        npdtype = dtype.as_np_dtype()
        return num_bytes // npdtype(int(0)).itemsize
