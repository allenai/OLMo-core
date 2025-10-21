import dataclasses
import functools as ft
import hashlib
import logging
import typing
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError

from ..utils import get_rng
from .token_source import TokenRange, TokenSource, TokenSourceConfig
from .utils import SEED_NOT_SET, as_ndarray, resolve_seed

log = logging.getLogger(__name__)


@dataclass
class SamplingTokenSourceConfig(TokenSourceConfig):
    """
    A config for building a :class:`SamplingTokenSource`.
    """

    sources: List[TokenSourceConfig]
    max_tokens: Optional[int] = None
    factor: Optional[float] = None
    seed: Optional[int] = dataclasses.field(default_factory=lambda: resolve_seed(SEED_NOT_SET))
    label: Optional[str] = None

    def __post_init__(self):
        if (self.max_tokens is None) == (self.factor is None):
            raise OLMoConfigurationError("Exactly one of 'max_tokens' or 'factor' must be set.")

    def build(self, work_dir: PathOrStr) -> List["SamplingTokenSource"]:  # type: ignore[override]
        sources = [s for source in self.sources for s in source.build(work_dir=work_dir)]
        max_tokens = self.max_tokens
        if max_tokens is None:
            assert self.factor is not None
            max_tokens = int(self.factor * sum(source.num_tokens for source in sources))
        return [
            SamplingTokenSource(
                *sources,
                max_tokens=max_tokens,
                seed=self.seed,
                work_dir=work_dir,
                label=self.label,
            )
        ]


class SamplingTokenSource(TokenSource):
    """
    A token source that samples contiguous chunks of tokens from other token sources.
    This can be used to adjust the effective size of a source.

    .. tip::
        Unlike :class:`SamplingDocumentSource`, this class doesn't take document boundaries
        into account when sampling, but is much faster to set up.

    :param sources: The sources to sample tokens from.
    :param max_tokens: The maximum number of tokens to sample.
    :param seed: A optional seed for sampling. If ``None``, the first ``N_s`` tokens are taken
        from each source where ``N_s`` is proportional to the size of the source.
    """

    Config = SamplingTokenSourceConfig

    DISPLAY_ICON = "\uedec"

    def __init__(
        self,
        *sources: TokenSource,
        max_tokens: int,
        seed: Optional[int] = SEED_NOT_SET,
        work_dir: PathOrStr,
        label: Optional[str] = None,
    ):
        from .mixing_document_source import MixingDocumentSource
        from .mixing_token_source import MixingTokenSource

        if not sources:
            raise ValueError("At least one source must be provided.")
        assert max_tokens > 0

        super().__init__(work_dir=work_dir, label=label)

        unwound_sources: List[TokenSource] = []
        for source in sources:
            # Unwind any mixing sources so that we sample directly from each of their
            # sources in order to maintain the ratios.
            if isinstance(source, (MixingTokenSource, MixingDocumentSource)):
                unwound_sources.extend(source.sampled_sources)
            else:
                unwound_sources.append(source)

        # Determine how many tokens to sample from each source.
        total_tokens = sum(source.num_tokens for source in unwound_sources)
        source_sample_sizes: List[int] = []
        for source in unwound_sources:
            # We want `source.num_tokens / total_tokens ~= source_sample_size / max_tokens`,
            # so `source_sample_size = max_tokens * (source.num_tokens / total_tokens)`.
            source_sample_sizes.append(int(max_tokens * (source.num_tokens / total_tokens)))

        # Determine number of repetitions and sampling start/end offsets for each source.
        seed = resolve_seed(seed)
        rng = None if seed is None else get_rng(seed)
        final_sources: List[TokenSource] = []
        source_sampling_offsets: List[Tuple[int, int]] = []
        for source, source_sample_size in zip(unwound_sources, source_sample_sizes):
            n_repetitions = source_sample_size // source.num_tokens
            final_sources.extend([source] * n_repetitions)
            source_sampling_offsets.extend([(0, source.num_tokens)] * n_repetitions)

            remaining_sample_size = source_sample_size % source.num_tokens
            if remaining_sample_size > 0:
                if rng is None:
                    source_sampling_offsets.append((0, remaining_sample_size))
                else:
                    start_idx = rng.integers(0, source.num_tokens - remaining_sample_size)
                    source_sampling_offsets.append((start_idx, start_idx + remaining_sample_size))
                final_sources.append(source)

        self._og_sources = sources
        self._sources = tuple(final_sources)
        self._source_sampling_offsets = tuple(source_sampling_offsets)

    @property
    def sources(self) -> Tuple[TokenSource, ...]:
        return self._sources

    @property
    def source_sampling_offsets(self) -> Tuple[Tuple[int, int], ...]:
        return self._source_sampling_offsets

    @ft.cached_property
    def num_tokens(self) -> int:
        return sum((end_idx - start_idx) for (start_idx, end_idx) in self._source_sampling_offsets)

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update((f"class={self.__class__.__name__},").encode())
        for source, sampling_offsets in zip(self.sources, self.source_sampling_offsets):
            sha256_hash.update(f"source={source.fingerprint}{sampling_offsets},".encode())
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

    def children(self):
        return self._og_sources
