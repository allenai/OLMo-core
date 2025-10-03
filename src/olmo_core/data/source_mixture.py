import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Optional, Tuple

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


def round_down_to_multiple(x, *, n):
    return (x // n) * n


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
    max_source_fraction: float = 1.0
    """
    The maximum fraction of the source data to use. This can be used to pretend that a source
    has less data than it actually does, which can be useful for testing / experimentation.
    """

    def validate(self):
        if not self.paths:
            raise OLMoConfigurationError("paths must not be empty")
        if not 0 < self.target_ratio <= 1:
            raise OLMoConfigurationError("target_ratio must be > 0 and <= 1")
        if self.max_repetitions < 1:
            raise OLMoConfigurationError("max_repetitions must be >= 1")
        if not 0 < self.max_source_fraction <= 1:
            raise OLMoConfigurationError("max_source_fraction must be > 0 and <= 1")
        if self.max_source_fraction < 1 and self.max_repetitions > 1:
            raise NotImplementedError(
                "max_source_fraction < 1 with max_repetitions > 1 is not supported"
            )


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
        Convert the dataset to an indexed dictionary mapping (path, index) tuples to token counts.

        This method flattens all sources and their paths into a single dictionary where each
        key is a tuple of (path_string, enumeration_index) and the value is the number of
        tokens for that path.
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

    def build(
        self, *, npdtype: NumpyUIntTypes, sequence_length: int, eos_token_id: Optional[int] = None
    ) -> SourceMixtureDataset:
        self.validate()
        random.seed(self.seed)  # but we've already run seed_all()...

        log.info(f"Generating a mixture from {len(self.source_configs)} sources")

        # Calculate the number of tokens available and to include for each source
        tokens_details_by_source: List[SourceTokenDetails] = []
        for source_config in self.source_configs:
            # Count only usable tokens (from documents >= sequence_length) when eos_token_id is provided
            # Otherwise fall back to counting all tokens for backward compatibility
            if eos_token_id is not None:
                tokens_in_source = (
                    self._count_usable_tokens_for_source(
                        source_config=source_config,
                        npdtype=npdtype,
                        sequence_length=sequence_length,
                        eos_token_id=eos_token_id,
                    )
                    * source_config.max_source_fraction
                )
            else:
                tokens_in_source = (
                    self._count_tokens_for_source(
                        source_config=source_config,
                        npdtype=npdtype,
                    )
                    * source_config.max_source_fraction
                )
            needed_tokens = self.requested_tokens * source_config.target_ratio
            needed_for_source = round_down_to_multiple(needed_tokens, n=sequence_length)

            # Ensure that we have enough tokens in the source to meet the target ratio
            if tokens_in_source == 0:
                raise OLMoConfigurationError(
                    f"Source {source_config.source_name} has no usable tokens. "
                    f"All documents are shorter than sequence_length={sequence_length}."
                )
            elif tokens_in_source < needed_for_source:
                upsample_ratio = needed_for_source / tokens_in_source
                if upsample_ratio > source_config.max_repetitions:
                    raise OLMoConfigurationError(
                        f"Source {source_config.source_name} does not have enough tokens to meet the "
                        f"target ratio. Needs {needed_for_source}, has {tokens_in_source}, which "
                        f"requires an upsample ratio of {upsample_ratio:.4f} "
                        f"but max_repetitions is {source_config.max_repetitions}"
                    )
                else:
                    log.info(
                        f"Source {source_config.source_name} does not have enough tokens to meet the "
                        f"target ratio, upsampling by roughly {upsample_ratio:.4f}"
                    )

            tokens_details_by_source.append(
                SourceTokenDetails(
                    config=source_config,
                    population=int(tokens_in_source * source_config.max_repetitions),
                    num_selected=needed_for_source,
                )
            )

        tokens_per_path_per_source: Dict[str, List[SourcePathTokens]] = {}
        for source in tokens_details_by_source:
            tokens_per_path = self.get_tokens_per_path(
                source_config=source.config,
                token_details=source,
                npdtype=npdtype,
                sequence_length=sequence_length,
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
        full_instances_per_path = []
        remainders_per_path = []
        for tokens_per_path in all_tokens_per_path:
            full_instances, remainder = divmod(tokens_per_path, sequence_length)
            full_instances_per_path.append(full_instances)
            remainders_per_path.append(remainder)

        # we apply Hamilton's method for rounding (see https://mathematics-democracy-institute.org/apportionment/)
        # that is, we only round up the requested instances for the paths that have the largest remainders
        # there is a lot of literature on how any other allocation of increments is suboptimal
        # basically, the requested tokens that are closest to a full instance are rounded up first
        additional_instances_needed = requested_instances - sum(full_instances_per_path)
        if additional_instances_needed > 0:
            # Distribute additional instances evenly across all paths first (base),
            # then allocate any remaining instances (leftover) to paths with largest remainders
            #
            # TODO: does this actually work? Can the dataset simply read the remainders like this?
            # Seems we're pretending these full instances actually exist?
            num_paths = len(remainders_per_path)
            base, leftover = divmod(additional_instances_needed, num_paths)
            if base:
                full_instances_per_path = [n + base for n in full_instances_per_path]
            if leftover:
                largest_remainder_indicies = sorted(
                    range(num_paths), key=remainders_per_path.__getitem__, reverse=True
                )
                for idx in largest_remainder_indicies[:leftover]:
                    full_instances_per_path[idx] += 1

        final_tokens_per_path = [inst * sequence_length for inst in full_instances_per_path]

        i = 0
        completed: List[SourceMixtureOutcome] = []
        final_token_distribution: Dict[str, float] = {}
        for source_name, source_path_tokens in tokens_per_path_per_source.items():
            path_tokens = []
            for path_token in source_path_tokens:
                path_tokens.append(
                    SourcePathTokens(path=path_token.path, tokens=final_tokens_per_path[i])
                )
                i += 1

            completed.append(SourceMixtureOutcome(name=source_name, path_tokens=path_tokens))
            final_token_distribution[source_name] = sum(path.tokens for path in path_tokens)

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
        sequence_length: int,
    ) -> List[SourcePathTokens]:
        """
        Get the paths and resulting token count for a source.

        All token counts are rounded down to multiples of sequence_length to ensure
        that downstream NumpyFSLDataset can create fixed-length sequences.
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
                    raw_tokens = int(math.ceil(self._count_tokens_for_file(path, npdtype) * ratio))
                    # Round down to sequence_length multiple
                    tokens_to_keep = round_down_to_multiple(raw_tokens, n=sequence_length)
                    path_tokens.append(SourcePathTokens(path=path, tokens=tokens_to_keep))

            return path_tokens

        for path in source_config.paths:
            raw_tokens = int(math.ceil(self._count_tokens_for_file(path, npdtype) * take_ratio))
            # Round down to sequence_length multiple
            tokens_to_keep = round_down_to_multiple(raw_tokens, n=sequence_length)
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

    def _count_usable_tokens_for_source(
        self,
        source_config: SourceMixtureConfig,
        npdtype: NumpyUIntTypes,
        sequence_length: int,
        eos_token_id: int,
    ) -> int:
        """
        Count the total number of USABLE tokens for FSL datasets from a source, across all files.

        Only counts tokens from documents that are >= sequence_length, since shorter documents
        will be filtered out during instance creation.

        Args:
            source_config: The source configuration.
            npdtype: The data type of the source tokens.
            sequence_length: The target sequence length for FSL datasets.
            eos_token_id: The EOS token ID used to identify document boundaries.
        """
        log.info(
            f"Counting usable tokens (>= {sequence_length}) for source: {source_config.source_name}"
        )
        with ThreadPoolExecutor(max_workers=self.num_worker_threads) as executor:
            futures = []
            for path in source_config.paths:
                futures.append(
                    executor.submit(
                        self._count_usable_tokens_for_file,
                        path=path,
                        npdtype=npdtype,
                        sequence_length=sequence_length,
                        eos_token_id=eos_token_id,
                    )
                )

            with Progress(disable=self.quiet) as progress:
                results = []
                task = progress.add_task(
                    f"Counting usable tokens for source: {source_config.source_name}",
                    total=len(futures),
                )
                for future in as_completed(futures):
                    progress.update(task, advance=1)
                    results.append(future.result())

            return sum(results)

    def _count_tokens_for_file(self, path: PathOrStr, npdtype: NumpyUIntTypes) -> int:
        return bytes_to_tokens(get_file_size(path), dtype=npdtype)

    def _count_usable_tokens_for_file(
        self, path: PathOrStr, npdtype: NumpyUIntTypes, sequence_length: int, eos_token_id: int
    ) -> int:
        """
        Count usable tokens in a file (only from documents >= sequence_length).

        This ensures the token count matches what will actually be available after
        filtering in segment_documents_into_instances.

        Args:
            path: Path to the numpy file.
            npdtype: The data type of the source tokens.
            sequence_length: The target sequence length for FSL datasets.
            eos_token_id: The EOS token ID used to identify document boundaries.
        """
        from olmo_core.data.utils import iter_document_indices

        usable_tokens = 0

        try:
            for start_idx, end_idx in iter_document_indices(
                path, eos_token_id=eos_token_id, dtype=npdtype
            ):
                doc_length = end_idx - start_idx
                if doc_length >= sequence_length:
                    # Count tokens in multiples of sequence_length from this document
                    usable_tokens += (doc_length // sequence_length) * sequence_length
        except Exception:
            # If we can't iterate documents (no metadata), fall back to total count
            # This happens for files without document structure
            return self._count_tokens_for_file(path, npdtype)

        return usable_tokens

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
