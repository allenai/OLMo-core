import functools as ft
import hashlib
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, TypedDict

from typing_extensions import NotRequired

import olmo_core.distributed.utils as dist_utils
import olmo_core.io as io
from olmo_core.aliases import PathOrStr
from olmo_core.config import Config
from olmo_core.exceptions import OLMoConfigurationError


class TokenRange(TypedDict):
    """
    A token range is just a dictionary that should include ``input_ids`` of the range and optionally a
    corresponding ``label_mask``.
    """

    input_ids: Sequence[int]
    """The token IDs for the range."""
    label_mask: NotRequired[Sequence[bool]]
    """An optional mask indicating which tokens should contribute to the loss."""


class TokenSource(metaclass=ABCMeta):
    """
    An abstract base class for a source of tokens, usually consumed by an :class:`InstanceSource`.
    It essentially represents an array of tokens.

    At a minimum, a :class:`TokenSource` must implement the methods/properties (1) :meth:`num_tokens`,
    (2) :meth:`get_token_range`, and (3) :meth:`fingerprint`.
    """

    def __init__(self, *, work_dir: PathOrStr):
        if io.is_url(work_dir):
            raise OLMoConfigurationError(
                f"'work_dir' should be a local path, not a URL ('{work_dir}')."
            )
        self._work_dir = Path(io.normalize_path(work_dir)) / self.__class__.__name__
        self._fs_local_rank = dist_utils.get_fs_local_rank()
        self._rank = dist_utils.get_rank()

    @property
    def work_dir(self) -> Path:
        """
        A local working directly that can be used by the token source for caching files during
        preprocessing.
        """
        return self._work_dir

    @property
    def fs_local_rank(self) -> int:
        """
        The local rank of the current process with respect to filesystem access of the working
        directory.
        """
        return self._fs_local_rank

    @property
    def rank(self) -> int:
        """The global rank of the current process across the entire distributed job."""
        return self._rank

    @property
    @abstractmethod
    def fingerprint(self) -> str:
        """A unique, deterministic string representing the contents of the source."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_tokens(self) -> int:
        """The number of tokens available from this source."""
        raise NotImplementedError

    @abstractmethod
    def get_token_range(self, start_idx: int, count: int) -> TokenRange:
        """
        Get a range of ``count`` contiguous tokens starting from ``start_idx``, which
        can vary from ``0`` to ``num_tokens - 1``.

        Since a :class:`TokenSource` isn't necessarily aware of document boundaries (see :class:`DocumentSource`),
        the token range could start in the middle of a document and span multiple documents.
        It's up to the consumers of a token source (e.g. an :class:`InstanceSource`) to get ranges
        that make sense for their use case.
        """
        raise NotImplementedError


class DocumentSource(TokenSource):
    """
    An abstract base class for a particular type of :class:`TokenSource` that's aware of document
    boundaries. This class provides one additional method: :meth:`get_document_offsets()`.
    """

    @abstractmethod
    def get_document_offsets(self) -> Iterable[tuple[int, int]]:
        """Get the start (inclusive) and end (exclusive) token indices of each document."""
        raise NotImplementedError


class InMemoryTokenSource(TokenSource):
    """
    An in-memory implementation of a :class:`TokenSource`. Primarily meant for testing.
    """

    def __init__(
        self,
        *,
        tokens: Sequence[int],
        work_dir: PathOrStr,
        label_mask: Optional[Sequence[bool]] = None,
    ):
        super().__init__(work_dir=work_dir)
        self.tokens = tokens
        self.label_mask = label_mask
        if self.label_mask is not None:
            assert len(self.tokens) == len(self.label_mask)

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update((f"class={self.__class__.__name__},").encode())
        for idx in range(self.num_tokens):
            sha256_hash.update(f"token={self.tokens[idx]},".encode())
            if self.label_mask is not None:
                sha256_hash.update(f"mask={self.label_mask[idx]},".encode())
        return sha256_hash.hexdigest()

    @property
    def num_tokens(self) -> int:
        return len(self.tokens)

    def get_token_range(self, start_idx: int, count: int) -> TokenRange:
        assert count >= 0
        if start_idx < 0:
            start_idx = self.num_tokens + start_idx
        end_idx = start_idx + count
        assert 0 <= start_idx < self.num_tokens
        assert 0 < end_idx <= self.num_tokens
        out: TokenRange = {"input_ids": self.tokens[start_idx:end_idx]}
        if self.label_mask is not None:
            out["label_mask"] = self.label_mask[start_idx:end_idx]
        return out


@dataclass
class TokenSourceConfig(Config):
    """A base config class for configuring and building a :class:`TokenSource`."""

    @abstractmethod
    def build(self, work_dir: PathOrStr) -> TokenSource:
        """Build the token source."""
        raise NotImplementedError


@dataclass
class DocumentSourceConfig(TokenSourceConfig):
    """A base config class for configuring and building a :class:`DocumentSource`."""

    @abstractmethod
    def build(self, work_dir: PathOrStr) -> DocumentSource:
        """Build the document source."""
        raise NotImplementedError
