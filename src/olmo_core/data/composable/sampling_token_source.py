import functools as ft
import hashlib
import logging
import typing
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from olmo_core.aliases import PathOrStr

from ..utils import get_rng
from .token_source import TokenRange, TokenSource, TokenSourceConfig
from .utils import as_ndarray

log = logging.getLogger(__name__)


@dataclass
class SamplingTokenSourceConfig(TokenSourceConfig):
    """
    A config for building a :class:`SamplingTokenSource`.
    """

    sources: List[TokenSourceConfig]
    max_tokens: int
    seed: Optional[int] = None

    def build(self, work_dir: PathOrStr) -> List["SamplingTokenSource"]:  # type: ignore[override]
        sources = [s for source in self.sources for s in source.build(work_dir=work_dir)]
        return [
            SamplingTokenSource(
                *sources,
                max_tokens=self.max_tokens,
                seed=self.seed,
                work_dir=work_dir,
            )
        ]


class SamplingTokenSource(TokenSource):
    """
    A token source that samples contiguous chunks of tokens from other token sources.
    This is useful for creating a smaller token source for testing or for building up
    mixes of sources.

    .. tip::
        Unlike :class:`SamplingDocumentSource`, this class doesn't take document boundaries
        into account when sampling, but is much faster to set up.

    :param sources: The sources to sample tokens from.
    :param max_tokens: The maximum number of tokens to sample.
    :param seed: A optional seed for sampling. If ``None``, the first ``N_s`` tokens are taken
      from each source where ``N_s`` is proportional to the size of the source.
    :param work_dir: A local working directory for caching preprocessing results.
    """

    Config = SamplingTokenSourceConfig

    def __init__(
        self,
        *sources: TokenSource,
        max_tokens: int,
        seed: Optional[int] = None,
        work_dir: PathOrStr,
    ):
        super().__init__(work_dir=work_dir)
        if not sources:
            raise ValueError("At least one source must be provided.")
        self._sources = sources
        assert max_tokens > 0
        self._max_tokens = max_tokens
        self._seed = seed

        # Determine how many tokens to sample from each source.
        total_tokens = sum(source.num_tokens for source in self.sources)
        max_tokens = min(max_tokens, total_tokens)
        source_sample_sizes: List[int] = []
        for source in sources:
            if max_tokens == total_tokens:
                source_sample_sizes.append(source.num_tokens)
            else:
                # We want `source.num_tokens / total_tokens ~= source_sample_size / max_tokens`.
                source_sample_sizes.append(int(max_tokens * (source.num_tokens / total_tokens)))

        # Determine sampling start/end offsets for each source.
        rng = None if seed is None else get_rng(seed)
        source_sampling_offsets: List[Tuple[int, int]] = []
        for source, source_sample_size in zip(sources, source_sample_sizes):
            if rng is None:
                source_sampling_offsets.append((0, source_sample_size))
            else:
                start_idx = rng.integers(0, source.num_tokens - source_sample_size)
                source_sampling_offsets.append((start_idx, start_idx + source_sample_size))
        self._source_sampling_offsets = tuple(source_sampling_offsets)

    @property
    def sources(self) -> Tuple[TokenSource, ...]:
        return self._sources

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @ft.cached_property
    def num_tokens(self) -> int:
        return sum((end_idx - start_idx) for (start_idx, end_idx) in self._source_sampling_offsets)

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            (
                f"class={self.__class__.__name__},"
                f"max_tokens={self.max_tokens},"
                f"seed={self.seed},"
            ).encode()
        )
        for source in self.sources:
            sha256_hash.update(f"source={source.fingerprint},".encode())
        return sha256_hash.hexdigest()

    def get_token_range(self, start_idx: int, end_idx: int) -> TokenRange:
        start_idx, end_idx = self.validate_indices(start_idx, end_idx)

        token_chunks: List[np.ndarray] = []
        mask_chunks: List[np.ndarray] = []
        source_start_offset = 0
        for source, (source_sample_start, source_sample_end) in zip(
            self.sources, self._source_sampling_offsets
        ):
            source_sample_size = source_sample_end - source_sample_start
            source_end_offset = source_start_offset + source_sample_size

            if source_start_offset <= start_idx < source_end_offset:
                token_rng = source.get_token_range(
                    start_idx - source_start_offset + source_sample_start,
                    min(end_idx - source_start_offset + source_sample_start, source_sample_end),
                )
                token_chunks.append(as_ndarray(token_rng["input_ids"]))
                if "label_mask" in token_rng:
                    mask_chunks.append(as_ndarray(token_rng["label_mask"]))

                if end_idx - source_start_offset + source_sample_start <= source_sample_end:
                    break
                else:
                    start_idx = source_end_offset

            source_start_offset = source_end_offset

        input_ids = np.concatenate(token_chunks)
        out: TokenRange = {"input_ids": typing.cast(Sequence[int], input_ids)}
        if mask_chunks:
            out["label_mask"] = typing.cast(Sequence[bool], np.concatenate(mask_chunks))
        return out
