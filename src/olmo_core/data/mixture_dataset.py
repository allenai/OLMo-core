import math
import multiprocessing as mp
import os
import random
import threading
from concurrent.futures import as_completed, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

import tabulate

from olmo_core.aliases import PathOrStr
from olmo_core.config import Config
from olmo_core.data import NumpyDatasetDType
from olmo_core.data.utils import load_array_slice, memmap_to_write
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import get_file_size


class ValueLock:
    def __init__(self):
        self._lock = threading.Lock()
        self._value = 0

    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    def add(self, value) -> int:
        with self._lock:
            self._value = self._value + value
            return self._value

    def value(self) -> int:
        with self._lock:
            return self._value


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

    source: SourceMixtureConfig
    source_population: int
    num_selected: int

    def for_table(self, max_tokens: int) -> Dict:
        return {
            "source_name": self.source.source_name,
            "source_population": f"{self.source_population:.2e}",
            "num_sampled": f"{self.num_selected:.2e}",
            "target_ratio": self.source.target_ratio,
            "max_repetion_ratio": self.source.max_repetition_ratio,
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
    dry_run: bool = False

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

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

        # Count the number of tokens available for each source
        for source_config in self.source_configs:
            print("Counting tokens for source: ", source_config.source_name)
            available_tokens_by_source[source_config.source_name] = self._count_tokens_for_paths(
                source_config.paths
            )

        tokens_outcome_per_source: List[SourceTokenDetails] = []

        # Calculate the number of tokens to include for each source
        for source_config in self.source_configs:
            available_for_source = available_tokens_by_source[source_config.source_name]
            target_for_source = int(self.max_tokens * source_config.target_ratio)
            max_for_source = int(
                available_for_source
                * source_config.max_source_fraction
                * source_config.max_repetition_ratio
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
        if not self.dry_run:
            for outcome in tokens_outcome_per_source:
                completed.append(self._handle_source_outcome(outcome))

        print(f"Mixing outcome by source: {'' if not self.dry_run else '(DRY RUN)'}")
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
        Write selected tokens into a local file based on selection criteria.
        """
        # Shuffle the paths to avoid biasing our selection to sequential file paths
        paths = source_config.paths.copy()
        random.shuffle(paths)
        written: List[str] = []
        taken = ValueLock()
        m = mp.Manager()
        write_lock = m.Lock()

        with ThreadPoolExecutor(max_workers=self.processes) as executor:
            print(f"Collecting {tokens_to_take:.2e} tokens for {source_config.source_name}")
            futures = []
            for idx, path in enumerate(paths * math.ceil(source_config.max_repetition_ratio)):
                futures.append(
                    executor.submit(
                        self._load_and_write_tokens,
                        idx,
                        path,
                        dtype,
                        tokens_to_take,
                        source_config.source_name,
                        taken,
                        write_lock,
                    )
                )

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Processing {source_config.source_name}",
            ):
                written.append(future.result())

        return [path for path in written if path is not None]

    def _load_and_write_tokens(
        self,
        index: int,
        path: PathOrStr,
        dtype: NumpyDatasetDType,
        tokens_to_take: int,
        source_name: str,
        taken: ValueLock,
        write_lock: threading.Lock,
    ) -> Optional[str]:
        """
        Load tokens from a source file and write them to a local file.
        """
        if taken.value() >= tokens_to_take:
            return None

        filename = f"{self.output_dir}/{index:05}_{source_name}.npy"
        print(f"Fetching {path} for {source_name}")
        nda = load_array_slice(path, 0, tokens_to_take, dtype.as_np_dtype())
        print(f"Fetched {len(nda):.2e} tokens for {source_name} {path}")

        # TODO: Why are we repeating files and or have empty arrays?
        with write_lock:
            nda = nda[: tokens_to_take - taken.value()]
            if len(nda) <= 0:
                print(f"Skipping {path} as it has no tokens left")
                return None
            with memmap_to_write(
                path=Path(filename), shape=(len(nda),), dtype=dtype.as_np_dtype()
            ) as mm:
                mm[:] = nda
                taken.add(len(nda))
                print(f"Wrote {len(nda):.2e} tokens to {filename}")

        return filename

    def _count_tokens_for_paths(self, paths: List[PathOrStr]) -> int:
        """
        Count the number of tokens for a set of source files in parallel.

        Args:
            source_config: The source configuration.
            dtype: The data type of the source tokens.
        """

        with ThreadPoolExecutor(max_workers=self.processes) as executor:
            return sum(executor.map(self._count_tokens_for_file, paths))

    def _count_tokens_for_file(self, path) -> int:
        return self._bytes_to_tokens(get_file_size(path), self.dtype)

    def _bytes_to_tokens(self, num_bytes: int, dtype: NumpyDatasetDType) -> int:
        """
        Convert bytes to tokens based on the dtype.
        """
        npdtype = dtype.as_np_dtype()
        return num_bytes // npdtype(int(0)).itemsize
