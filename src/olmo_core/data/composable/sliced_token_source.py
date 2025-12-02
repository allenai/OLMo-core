import functools as ft
import hashlib
from typing import Optional

from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError

from .token_source import TokenRange, TokenSource


class SlicedTokenSource(TokenSource):
    """
    A token source that provides a slice of another token source.
    """

    def __init__(
        self,
        source: TokenSource,
        source_slice: slice,
        *,
        work_dir: PathOrStr,
        label: Optional[str] = None,
    ):
        if source_slice.step is not None and source_slice.step != 1:
            raise OLMoConfigurationError(
                f"'{self.__class__.__name__}' does not support slices with a step other than 1."
            )
        if source_slice.start is not None and source_slice.start < -source.num_tokens:
            raise OLMoConfigurationError(
                f"Slice start {source_slice.start} is out of bounds for source with "
                f"{source.num_tokens} tokens."
            )

        super().__init__(work_dir=work_dir, label=label)
        self._source = source
        self._slice = source_slice
        if self.num_tokens == 0:
            raise OLMoConfigurationError(
                f"{self.__class__.__name__} created with an empty slice ({source_slice}) from source "
                f"with {source.num_tokens:,d} tokens."
            )

    @property
    def source(self) -> TokenSource:
        return self._source

    @property
    def source_slice(self) -> slice:
        return self._slice

    @property
    def slice_start(self) -> int:
        if self.source_slice.start is None:
            return 0
        elif self.source_slice.start < 0:
            assert self.source_slice.start >= -self.source.num_tokens
            return self.source.num_tokens + self.source_slice.start
        else:
            return self.source_slice.start

    @property
    def slice_stop(self) -> int:
        if self.source_slice.stop is None:
            return self.source.num_tokens
        elif self.source_slice.stop < 0:
            return max(0, self.source.num_tokens + self.source_slice.stop)
        else:
            return self.source_slice.stop

    @property
    def slice_step(self) -> int:
        if self.source_slice.step is not None:
            return self.source_slice.step
        else:
            return 1

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            (
                f"class={self.__class__.__name__},"
                f"slice_start={self.slice_start},"
                f"slice_stop={self.slice_stop},"
                f"slice_step={self.slice_step},"
                f"source={self.source.fingerprint},"
            ).encode()
        )
        return sha256_hash.hexdigest()

    @property
    def num_tokens(self) -> int:
        if self.slice_step != 1:
            raise NotImplementedError(
                f"'{self.__class__.__name__}' does not support slices with a step other than 1."
            )
        if self.slice_start >= self.source.num_tokens:
            return 0
        return max(0, min(self.slice_stop, self.source.num_tokens) - self.slice_start)

    def get_token_range(self, start_idx: int, end_idx: int) -> TokenRange:
        if self.slice_step != 1:
            raise NotImplementedError(
                f"'{self.__class__.__name__}' does not support slices with a step other than 1."
            )
        start_idx, end_idx = self.validate_indices(start_idx, end_idx)
        return self.source.get_token_range(start_idx + self.slice_start, end_idx + self.slice_start)

    def children(self):
        return [self.source]
