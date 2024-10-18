import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import smart_open
import tabulate

from olmo_core.aliases import PathOrStr
from olmo_core.config import Config
from olmo_core.data import NumpyDatasetDType
from olmo_core.data.utils import load_array_slice, memmap_to_write
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import get_bytes_range, get_file_size


@dataclass
class SourceMixtureConfig(Config):
    source_name: str
    target_ratio: float
    paths: List[PathOrStr]
    # 1.0 will result in a maximum of 1 repitition of the source data per epoch
    max_repetion_ratio: float = 1.0
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

    source: SourceMixtureConfig
    source_population: int
    num_selected: int

    def for_table(self, max_tokens: int) -> Dict:
        return {
            "source_name": self.source.source_name,
            "source_population": f"{self.source_population:.2e}",
            "num_sampled": f"{self.num_selected:.2e}",
            "target_ratio": self.source.target_ratio,
            "max_repetion_ratio": self.source.max_repetion_ratio,
            "max_source_fraction": self.source.max_source_fraction,
            "observed_source_ratio": self.num_selected / self.source_population,
            "observed_global_ratio": self.num_selected / max_tokens,
        }


@dataclass
class SourceMixture:
    """
    A fractionalized mixture of source tokens.
    """

    source_name: str
    paths: List[str]


@dataclass
class SourceMixtureDataset:
    """
    A dataset consisting of a fractionalized mixture of data sources.
    """

    sources: List[SourceMixture]


@dataclass
class SourceMixtureDatasetConfig(Config):
    """
    A configuration class for building a dataset from a fractionalized mixture of sources.
    """

    max_tokens: int
    source_configs: List[SourceMixtureConfig]
    dtype: NumpyDatasetDType
    output_dir: PathOrStr
    processes: int = 1
    seed: int = 42

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

    def validate(self):
        if (total := sum([source.target_ratio for source in self.source_configs])) != 1.0:
            raise OLMoConfigurationError(f"target_ratios must sum to 1, got {total}")

    def build(self) -> SourceMixtureDataset:
        self.validate()
        random.seed(self.seed)
        available_tokens_by_source: Dict[str, int] = {}

        # Count the number of tokens available for each source
        for source_config in self.source_configs:
            available_tokens_by_source[source_config.source_name] = self._count_tokens_for_source(
                source_config, self.dtype
            )

        tokens_outcome_per_source: List[SourceTokenDetails] = []

        # Calculate the number of tokens to include for each source
        for source_config in self.source_configs:
            available_for_source = available_tokens_by_source[source_config.source_name]
            target_for_source = int(self.max_tokens * source_config.target_ratio)
            max_for_source = int(
                available_for_source
                * source_config.max_source_fraction
                * source_config.max_repetion_ratio
            )

            # Ensure that the available source tokens meet the target ratio requirement
            if not max_for_source >= target_for_source:
                raise OLMoConfigurationError(
                    f"Insufficient tokens for source: {source_config.source_name} @ target global ratio: {source_config.target_ratio} :: {max_for_source} < {target_for_source}"
                )

            tokens_outcome_per_source.append(
                SourceTokenDetails(
                    source=source_config,
                    source_population=available_for_source,
                    num_selected=target_for_source,
                )
            )

        completed = []
        for outcome in tokens_outcome_per_source:
            completed.append(self._handle_source_outcome(outcome))

        print("Mixing outcome by source:")
        print(
            tabulate.tabulate(
                [item.for_table(self.max_tokens) for item in tokens_outcome_per_source],
                headers="keys",
                tablefmt="pretty",
            ),
        )

        return SourceMixtureDataset(completed)

    def _handle_source_outcome(self, outcome: SourceTokenDetails) -> SourceMixture:
        """
        Write selected tokens for a source to a local file and return the path.
        """
        return SourceMixture(
            source_name=outcome.source.source_name,
            paths=self._write_tokens_for_source(self.dtype, outcome.num_selected, outcome.source),
        )

    def _write_tokens_for_source(
        self, dtype: NumpyDatasetDType, tokens_to_take: int, source_config: SourceMixtureConfig
    ) -> List[str]:
        """
        Stream selected tokens into a local file based on selection criteria.
        """
        # Shuffle the paths to avoid biasing our selection to sequential file paths
        paths = source_config.paths.copy()
        random.shuffle(paths)
        tokens_taken = 0
        written: List[str] = []

        # Make sure we have enough paths to accommodate repetitions
        for idx, path in enumerate(paths * math.ceil(source_config.max_repetion_ratio)):
            # Stop if we've taken enough tokens
            if tokens_taken >= tokens_to_take:
                break

            filename = f"{self.output_dir}/{idx:05}_{source_config.source_name}.npy"
            nda = load_array_slice(path, 0, tokens_to_take - tokens_taken, dtype.as_np_dtype())
            with memmap_to_write(
                path=Path(filename), shape=(len(nda),), dtype=dtype.as_np_dtype()
            ) as mm:
                mm[:] = nda

            written.append(filename)
            tokens_taken += tokens_to_take - tokens_taken

        return written

    def _count_tokens_for_source(
        self, source_config: SourceMixtureConfig, dtype: NumpyDatasetDType
    ) -> int:
        """
        Count the number of tokens for a set of source token files in parallel.

        Args:
            source_config: The source configuration.
            dtype: The data type of the source tokens.
        """

        def _count_tokens(path) -> int:
            size = get_file_size(path)
            tokens = self._bytes_to_tokens(size, dtype)
            return tokens

        with ThreadPoolExecutor(max_workers=self.processes) as executor:
            return sum(executor.map(_count_tokens, source_config.paths))

    def _bytes_to_tokens(self, num_bytes: int, dtype: NumpyDatasetDType) -> int:
        """
        Convert bytes to tokens based on the dtype.
        """
        npdtype = dtype.as_np_dtype()
        return num_bytes // npdtype(int(0)).itemsize

    def _tokens_to_bytes(self, num_tokens: int, dtype: NumpyDatasetDType) -> int:
        """
        Convert tokens to bytes based on the dtype.
        """

        npdtype = dtype.as_np_dtype()
        return num_tokens * npdtype(int(0)).itemsize
