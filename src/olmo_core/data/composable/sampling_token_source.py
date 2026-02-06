import dataclasses
import functools as ft
import hashlib
import logging
import typing
from collections import deque
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
        from .sampling_document_source import SamplingDocumentSource
        from .sliced_token_source import SlicedTokenSource

        if not sources:
            raise ValueError("At least one source must be provided.")
        assert max_tokens > 0

        super().__init__(work_dir=work_dir, label=label)

        # Determine how many tokens to sample from each source.
        frontier = deque(sources)
        total_tokens = sum(source.num_tokens for source in sources)
        seed = resolve_seed(seed)
        rng = None if seed is None else get_rng(seed)
        final_sources: List[TokenSource] = []
        while frontier:
            source = frontier.popleft()

            # Unwind any mixing token sources into their sampling token sources,
            # and sampling token sources into their children so that we sample directly from each
            # the children in order to maintain the desired ratios.
            # However we'd never to handle sampling/mixing document sources differently.
            # For now we'll just disallow it.
            if isinstance(source, (SamplingDocumentSource, MixingDocumentSource)):
                raise NotImplementedError(
                    "Sampling/mixing document sources are not supported as children of SamplingTokenSource."
                )
            elif isinstance(source, MixingTokenSource):
                frontier.extend(source.sampled_sources)
            elif isinstance(source, SamplingTokenSource):
                frontier.extend(source.sources)
            else:
                # Determine how many tokens to sample from source such that while keeping the same
                # ratios between sources. For example, suppose source A makes up 75% of the
                # `total_tokens` available across all sources. Then we want the number of tokens
                # we sample from A to make up 75% of `max_tokens`. In other words, we want
                # `len(source) / total_tokens ~= source_sample_size / max_tokens`,
                # so `source_sample_size = max_tokens * (source.num_tokens / total_tokens)`.
                source_sample_size = int(max_tokens * (len(source) / total_tokens))

                # Determine number of repetitions and sampling start/end offsets for each source.
                n_repetitions = source_sample_size // source.num_tokens
                final_sources.extend([source] * n_repetitions)

                remaining_sample_size = source_sample_size % source.num_tokens
                if remaining_sample_size > 0:
                    start_idx = (
                        0
                        if rng is None
                        else rng.integers(0, source.num_tokens - remaining_sample_size)
                    )
                    end_idx = start_idx + remaining_sample_size
                    final_sources.append(
                        SlicedTokenSource(source, slice(start_idx, end_idx), work_dir=self.work_dir)
                    )

        self._og_sources = sources
        self._sources = tuple(final_sources)

    @property
    def sources(self) -> Tuple[TokenSource, ...]:
        return self._sources

    @ft.cached_property
    def num_tokens(self) -> int:
        return sum(source.num_tokens for source in self.sources)

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update((f"class={self.__class__.__name__},").encode())
        for source in self.sources:
            sha256_hash.update(f"source={source.fingerprint},".encode())
        return sha256_hash.hexdigest()

    def get_token_range(self, start_idx: int, end_idx: int) -> TokenRange:
        start_idx, end_idx = self.validate_indices(start_idx, end_idx)

        token_chunks: List[np.ndarray] = []
        mask_chunks: List[np.ndarray] = []
        source_start_offset = 0
        for source in self.sources:
            source_end_offset = source_start_offset + len(source)

            if source_start_offset <= start_idx < source_end_offset:
                token_rng = source.get_token_range(
                    start_idx - source_start_offset,
                    min(end_idx - source_start_offset, len(source)),
                )
                token_chunks.append(as_ndarray(token_rng["input_ids"]))
                if "label_mask" in token_rng:
                    mask_chunks.append(as_ndarray(token_rng["label_mask"]))

                if end_idx <= source_end_offset:
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
