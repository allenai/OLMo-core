import functools as ft
import hashlib
import typing
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
from typing_extensions import NotRequired

from olmo_core.aliases import PathOrStr
from olmo_core.config import Config

from ..tokenizer import TokenizerConfig
from .source_abc import SourceABC
from .utils import SEED_NOT_SET, as_ndarray, resolve_seed

if TYPE_CHECKING:
    from .sampling_document_source import (
        SamplingDocumentSource,
        SamplingDocumentSourceConfig,
    )
    from .sampling_token_source import SamplingTokenSource, SamplingTokenSourceConfig
    from .sliced_token_source import SlicedTokenSource


class TokenRange(TypedDict):
    """
    A token range is just a dictionary that should include ``input_ids`` of the range and optionally a
    corresponding ``label_mask``.
    """

    input_ids: Sequence[int]
    """The token IDs for the range."""
    label_mask: NotRequired[Sequence[bool]]
    """An optional mask indicating which tokens should contribute to the loss."""


class TokenSource(SourceABC):
    """
    An abstract base class for a source of tokens, usually consumed by an :class:`InstanceSource`.
    It essentially represents an array of tokens.

    At a minimum, a :class:`TokenSource` must implement the methods/properties (1) :meth:`num_tokens`,
    (2) :meth:`get_token_range`, (3) :meth:`fingerprint`, and (4) :meth:`children`.
    """

    DISPLAY_ICON: ClassVar[str] = "\ueb7e"  # Nerd Font icon for visualizations

    def __len__(self) -> int:
        """The number of tokens available from this source, same as ``self.num_tokens``."""
        return self.num_tokens

    @abstractmethod
    def get_token_range(self, start_idx: int, end_idx: int) -> TokenRange:
        """
        Get a range of contiguous tokens starting from ``start_idx`` (0-based, inclusive) to ``end_idx`` (exclusive).

        Since a :class:`TokenSource` isn't necessarily aware of document boundaries (see :class:`DocumentSource`),
        the token range could start in the middle of a document and span multiple documents.
        It's up to the consumers of a token source (e.g. an :class:`InstanceSource`) to get ranges
        that make sense for their use case.
        """
        raise NotImplementedError

    def __getitem__(self, key: Union[int, slice]) -> TokenRange:
        """
        Get a range of tokens using either an integer index (for a singular token range) or a slice.
        """
        if isinstance(key, slice):
            start_idx = key.start if key.start is not None else 0
            end_idx = key.stop if key.stop is not None else self.num_tokens
            step = key.step if key.step is not None else 1
            token_rng = self.get_token_range(start_idx, end_idx)
            out: TokenRange = {"input_ids": token_rng["input_ids"][::step]}
            if "label_mask" in token_rng:
                out["label_mask"] = token_rng["label_mask"][::step]
            return out
        else:
            if key < 0:
                key = self.num_tokens + key
            return self.get_token_range(key, key + 1)

    def validate_indices(self, start_idx: int, end_idx: int) -> Tuple[int, int]:
        start_idx, end_idx = int(start_idx), int(end_idx)
        if start_idx < 0:
            start_idx = self.num_tokens + start_idx
        if end_idx < 0:
            end_idx = self.num_tokens + end_idx
        if end_idx == start_idx:
            raise ValueError(
                f"Invalid token range {start_idx=} → {end_idx=}, ranges cannot be empty."
            )
        if end_idx <= start_idx:
            raise ValueError(f"Invalid token range {start_idx=} → {end_idx=}.")
        if start_idx >= self.num_tokens or end_idx > self.num_tokens:
            raise IndexError(
                f"Token range {start_idx=} → {end_idx=} is out of bounds "
                f"for source {self} with {self.num_tokens:,d} tokens."
            )
        return start_idx, end_idx

    def __add__(self, other: "TokenSource") -> "ConcatenatedTokenSource":
        """
        Add two token sources together into a :class:`ConcatenatedTokenSource` or :class:`ConcatenatedDocumentSource`
        depending on the type of ``self`` and ``other``.
        """
        if isinstance(self, DocumentSource) and isinstance(other, DocumentSource):
            return ConcatenatedDocumentSource(self, other, work_dir=self.common_work_dir)
        elif isinstance(other, TokenSource):
            return ConcatenatedTokenSource(self, other, work_dir=self.common_work_dir)
        else:
            raise TypeError(f"Cannot add {type(self)} with {type(other)}.")

    def __mul__(self, factor: float) -> "SamplingTokenSource":
        """Re-size this source by a given factor by sampling tokens from it."""
        if isinstance(factor, (float, int)):
            return self.resize(factor)
        else:
            raise TypeError(f"Cannot multiply {type(self)} with {type(factor)}.")

    def sample(
        self,
        *,
        max_tokens: int,
        seed: Optional[int] = SEED_NOT_SET,
    ) -> "SamplingTokenSource":
        """
        Sample a contiguous chunk of tokens from this source.

        .. seealso::
            :meth:`resize()`

        :param max_tokens: The maximum number of tokens to sample.
        :param seed: A seed to use to randomize the sampling.
        """
        from .sampling_token_source import SamplingTokenSource

        return SamplingTokenSource(
            self,
            max_tokens=max_tokens,
            seed=seed,
            work_dir=self.common_work_dir,
        )

    def resize(self, factor: float, seed: Optional[int] = SEED_NOT_SET) -> "SamplingTokenSource":
        """
        Re-size this source by a given factor by sampling a contiguous chunk of tokens from it.

        .. seealso::
            :meth:`sample()`

        :param factor: The factor to resize the source by. For example, ``0.5`` will create a source
          with half the number of tokens, and ``2.0`` will create a source with twice the number of tokens.
        :param seed: A seed to use to randomize the sampling.
        """
        assert factor > 0
        return self.sample(max_tokens=int(self.num_tokens * factor), seed=seed)

    def split(self, ratio: float) -> Tuple["SlicedTokenSource", "SlicedTokenSource"]:
        """
        Split this source into two disjoint sources according to the given ratio.

        :param ratio: The ratio of the first split to original source. E.g., ``0.8`` means
          the first split will have 80% of the tokens and the second split will have 20%.
        """
        from .sliced_token_source import SlicedTokenSource

        assert 0 < ratio < 1
        split_idx = int(ratio * self.num_tokens)
        return (
            SlicedTokenSource(self, slice(0, split_idx), work_dir=self.common_work_dir),
            SlicedTokenSource(self, slice(split_idx, None), work_dir=self.common_work_dir),
        )


class InMemoryTokenSource(TokenSource):
    """
    An in-memory implementation of a :class:`TokenSource`. Primarily meant for testing.
    """

    DISPLAY_ICON = "\U000f035b"

    def __init__(
        self,
        tokens: Sequence[int],
        *,
        work_dir: PathOrStr,
        label_mask: Optional[Sequence[bool]] = None,
        label: Optional[str] = None,
    ):
        super().__init__(work_dir=work_dir, label=label)
        self._tokens = as_ndarray(tokens)
        self._label_mask = None if label_mask is None else as_ndarray(label_mask)
        if self._label_mask is not None:
            assert len(self._tokens) == len(self._label_mask)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._tokens})"

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update((f"class={self.__class__.__name__},tokens=").encode())
        sha256_hash.update(self._tokens.tobytes())
        if self._label_mask is not None:
            sha256_hash.update(b"mask=")
            sha256_hash.update(self._label_mask.tobytes())
        return sha256_hash.hexdigest()

    @property
    def num_tokens(self) -> int:
        return len(self._tokens)

    def get_token_range(self, start_idx: int, end_idx: int) -> TokenRange:
        start_idx, end_idx = self.validate_indices(start_idx, end_idx)
        out: TokenRange = {"input_ids": typing.cast(Sequence[int], self._tokens[start_idx:end_idx])}
        if self._label_mask is not None:
            out["label_mask"] = typing.cast(Sequence[bool], self._label_mask[start_idx:end_idx])
        return out

    def children(self):
        return []


class DocumentSource(TokenSource):
    """
    An abstract base class for a particular type of :class:`TokenSource` that's aware of document
    boundaries. This class has one additional abstract method: :meth:`get_document_offsets()`.
    """

    DISPLAY_ICON = "\uf15c"

    def sample_by_docs(
        self,
        *,
        max_tokens: int,
        seed: Optional[int] = SEED_NOT_SET,
    ) -> "SamplingDocumentSource":
        """
        Sample documents from this source.

        .. seealso::
            - :meth:`~TokenSource.sample()`
            - :meth:`resize_by_docs()`

        :param max_tokens: The maximum number of tokens to sample.
        :param seed: A seed to use to randomize the sampling.
        """
        from .sampling_document_source import SamplingDocumentSource

        return SamplingDocumentSource(
            self,
            max_tokens=max_tokens,
            seed=seed,
            work_dir=self.common_work_dir,
        )

    def resize_by_docs(
        self, factor: float, seed: Optional[int] = SEED_NOT_SET
    ) -> "SamplingDocumentSource":
        """
        Re-size this source by a given factor by sampling documents from it.

        .. seealso::
            - :meth:`~TokenSource.resize()`
            - :meth:`sample_by_docs()`

        :param factor: The factor to resize the source by. For example, ``0.5`` will create a source
          with half the number of tokens, and ``2.0`` will create a source with twice the number of tokens.
        :param seed: A seed to use to randomize the sampling.
        """
        assert factor > 0
        return self.sample_by_docs(max_tokens=int(self.num_tokens * factor), seed=seed)

    @abstractmethod
    def get_document_offsets(self) -> Iterable[tuple[int, int]]:
        """Get the start (inclusive) and end (exclusive) token indices of each document, in order."""
        raise NotImplementedError


class InMemoryDocumentSource(InMemoryTokenSource, DocumentSource):
    """
    An in-memory implementation of a :class:`DocumentSource`. Primarily meant for testing.
    """

    def __init__(
        self,
        tokens: Sequence[int],
        *,
        tokenizer: TokenizerConfig,
        work_dir: PathOrStr,
        label_mask: Optional[Sequence[bool]] = None,
        label: Optional[str] = None,
    ):
        super().__init__(tokens=tokens, work_dir=work_dir, label_mask=label_mask, label=label)
        self._tokenizer = tokenizer

    @property
    def eos_token_id(self) -> int:
        return self._tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._tokenizer.bos_token_id

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            (
                f"class={self.__class__.__name__},"
                f"eos_token_id={self.eos_token_id},"
                f"bos_token_id={self.bos_token_id},"
                f"tokens="
            ).encode()
        )
        sha256_hash.update(self._tokens.tobytes())
        if self._label_mask is not None:
            sha256_hash.update(b"mask=")
            sha256_hash.update(self._label_mask.tobytes())
        return sha256_hash.hexdigest()

    def children(self):
        return []

    def get_document_offsets(self) -> Iterable[tuple[int, int]]:
        if self.bos_token_id is None:
            doc_boundaries = (self._tokens == self.eos_token_id).nonzero()[0]
        else:
            doc_boundaries = np.logical_and(
                self._tokens[:-1] == self.eos_token_id, self._tokens[1:] == self.bos_token_id
            ).nonzero()[0]

        start_idx = 0
        for idx in doc_boundaries:
            end_idx = idx + 1
            yield start_idx, end_idx
            start_idx = end_idx

        # To avoid unexpected results, we ALWAYS treat the end of the source as the end of
        # a document, even if it doesn't end with an EOS token ID.
        if start_idx != self.num_tokens:
            yield start_idx, self.num_tokens


@dataclass
class TokenSourceConfig(Config):
    """A base config class for configuring and building a :class:`TokenSource`."""

    @abstractmethod
    def build(self, work_dir: PathOrStr) -> List[TokenSource]:
        """Build the token source."""
        raise NotImplementedError

    def __add__(self, other: "TokenSourceConfig") -> "TokenSourceConfig":
        """
        Add two token source config together into a :class:`ConcatenatedTokenSourceConfig`
        or :class:`ConcatenatedDocumentSourceConfig`
        depending on the type of ``self`` and ``other``.
        """
        if isinstance(self, DocumentSourceConfig) and isinstance(other, DocumentSourceConfig):
            return ConcatenatedDocumentSourceConfig(sources=[self, other])
        elif isinstance(other, TokenSourceConfig):
            return ConcatenatedTokenSourceConfig(sources=[self, other])
        else:
            raise TypeError(f"Cannot add {type(self)} with {type(other)}.")

    def __mul__(self, factor: float) -> "SamplingTokenSourceConfig":
        """Re-size this source by a given factor by sampling tokens from it."""
        if isinstance(factor, (float, int)):
            return self.resize(factor)
        else:
            raise TypeError(f"Cannot multiply {type(self)} with {type(factor)}.")

    def sample(
        self,
        *,
        max_tokens: int,
        seed: Optional[int] = SEED_NOT_SET,
    ) -> "SamplingTokenSourceConfig":
        """
        Sample a contiguous chunk of tokens from this source.

        :param max_tokens: The maximum number of tokens to sample.
        :param seed: A seed to use to randomize the sampling.
        """
        from .sampling_token_source import SamplingTokenSourceConfig

        return SamplingTokenSourceConfig(
            sources=[self],
            max_tokens=max_tokens,
            seed=resolve_seed(seed),
        )

    def resize(
        self, factor: float, seed: Optional[int] = SEED_NOT_SET
    ) -> "SamplingTokenSourceConfig":
        """
        Re-size this source by a given factor by sampling a contiguous chunk of tokens from it.

        :param factor: The factor to resize the source by. For example, ``0.5`` will create a source
          with half the number of tokens, and ``2.0`` will create a source with twice the number of tokens.
        :param seed: A seed to use to randomize the sampling.
        """
        from .sampling_token_source import SamplingTokenSourceConfig

        assert factor > 0
        return SamplingTokenSourceConfig(
            sources=[self],
            factor=factor,
            seed=resolve_seed(seed),
        )

    def split(self, ratio: float) -> Tuple["SplitTokenSourceConfig", "SplitTokenSourceConfig"]:
        """
        Split this source into two disjoint sources according to the given ratio.

        :param ratio: The ratio of the first split to original source. E.g., ``0.8`` means
          the first split will have 80% of the tokens and the second split will have 20%.
        """
        return SplitTokenSourceConfig(source=self, ratio=ratio, idx=0), SplitTokenSourceConfig(
            source=self, ratio=ratio, idx=1
        )


@dataclass
class SplitTokenSourceConfig(TokenSourceConfig):
    """A base config class for configuring and building a split :class:`TokenSource`."""

    source: TokenSourceConfig
    ratio: float
    idx: int

    def __post_init__(self):
        assert 0 < self.ratio < 1
        assert self.idx in (0, 1)

    def build(self, work_dir: PathOrStr) -> List["SlicedTokenSource"]:  # type: ignore[override]
        from .sliced_token_source import SlicedTokenSource

        sources = self.source.build(work_dir)
        source = (
            sources[0]
            if len(sources) == 1
            else ConcatenatedTokenSource(*sources, work_dir=work_dir)
        )
        split_idx = int(self.ratio * source.num_tokens)
        if self.idx == 0:
            return [SlicedTokenSource(source, slice(0, split_idx), work_dir=work_dir)]
        elif self.idx == 1:
            return [SlicedTokenSource(source, slice(split_idx, None), work_dir=work_dir)]
        else:
            raise ValueError(f"Invalid split index: {self.idx}")


@dataclass
class ConcatenatedTokenSourceConfig(TokenSourceConfig):
    """A base config class for configuring and building a :class:`ConcatenatedTokenSource`."""

    sources: List[TokenSourceConfig]
    label: Optional[str] = None

    def build(self, work_dir: PathOrStr) -> List["ConcatenatedTokenSource"]:  # type: ignore[override]
        sources = [
            source for source_config in self.sources for source in source_config.build(work_dir)
        ]
        return [
            ConcatenatedTokenSource(
                *sources,
                work_dir=work_dir,
                label=self.label,
            )
        ]


class ConcatenatedTokenSource(TokenSource):
    """
    A token source that can be created from concatenating multiple other token sources.
    """

    Config = ConcatenatedTokenSourceConfig

    DISPLAY_ICON = "\uf51e"

    def __init__(self, *sources: TokenSource, work_dir: PathOrStr, label: Optional[str] = None):
        super().__init__(work_dir=work_dir, label=label)
        unraveled_sources: List[TokenSource] = []
        for source in sources:
            if isinstance(source, ConcatenatedTokenSource):
                unraveled_sources.extend(source.sources)
            else:
                unraveled_sources.append(source)
        self._sources = tuple(unraveled_sources)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.sources}"

    @property
    def sources(self) -> Tuple[TokenSource, ...]:
        return self._sources

    def children(self):
        return self.sources

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update((f"class={self.__class__.__name__},").encode())
        for source in self.sources:
            sha256_hash.update(f"{source=},".encode())
        return sha256_hash.hexdigest()

    @property
    def num_tokens(self) -> int:
        return sum(source.num_tokens for source in self.sources)

    def get_token_range(self, start_idx: int, end_idx: int) -> TokenRange:
        start_idx, end_idx = self.validate_indices(start_idx, end_idx)

        token_chunks: List[np.ndarray] = []
        mask_chunks: List[np.ndarray] = []
        source_start_offset = 0
        for source in self.sources:
            source_size = source.num_tokens
            source_end_offset = source_start_offset + source_size

            if source_start_offset <= start_idx < source_end_offset:
                token_rng = source.get_token_range(
                    start_idx - source_start_offset, min(end_idx - source_start_offset, source_size)
                )
                token_chunks.append(as_ndarray(token_rng["input_ids"]))
                if "label_mask" in token_rng:
                    mask_chunks.append(as_ndarray(token_rng["label_mask"]))

                if end_idx - source_start_offset <= source_size:
                    break
                else:
                    start_idx = source_end_offset

            source_start_offset = source_end_offset
        else:
            raise IndexError(f"Failed to find tokens in range {start_idx=} → {end_idx=}.")

        input_ids = np.concatenate(token_chunks)
        out: TokenRange = {"input_ids": typing.cast(Sequence[int], input_ids)}
        if mask_chunks:
            out["label_mask"] = typing.cast(Sequence[bool], np.concatenate(mask_chunks))
        return out


@dataclass
class DocumentSourceConfig(TokenSourceConfig):
    """A base config class for configuring and building a :class:`DocumentSource`."""

    @abstractmethod
    def build(self, work_dir: PathOrStr) -> List[DocumentSource]:  # type: ignore[override]
        """Build the document source."""
        raise NotImplementedError

    def sample_by_docs(
        self,
        *,
        max_tokens: int,
        seed: Optional[int] = SEED_NOT_SET,
    ) -> "SamplingDocumentSourceConfig":
        """
        Sample documents from this source.

        .. seealso::
            - :meth:`~TokenSourceConfig.sample()`
            - :meth:`resize_by_docs()`

        :param max_tokens: The maximum number of tokens to sample.
        :param seed: A seed to use to randomize the sampling.
        """
        from .sampling_document_source import SamplingDocumentSourceConfig

        return SamplingDocumentSourceConfig(
            sources=[self],
            max_tokens=max_tokens,
            seed=resolve_seed(seed),
        )

    def resize_by_docs(
        self,
        factor: float,
        seed: Optional[int] = SEED_NOT_SET,
    ) -> "SamplingDocumentSourceConfig":
        """
        Re-size this source by a given factor by sampling documents from it.

        .. seealso::
            - :meth:`~TokenSourceConfig.resize()`
            - :meth:`sample_by_docs()`

        :param factor: The factor to resize the source by. For example, ``0.5`` will create a source
          with half the number of tokens, and ``2.0`` will create a source with twice the number of tokens.
        :param seed: A seed to use to randomize the sampling.
        """
        from .sampling_document_source import SamplingDocumentSourceConfig

        assert factor > 0
        return SamplingDocumentSourceConfig(
            sources=[self],
            factor=factor,
            seed=resolve_seed(seed),
        )


@dataclass
class ConcatenatedDocumentSourceConfig(DocumentSourceConfig):
    """A base config class for configuring and building a :class:`ConcatenatedDocumentSource`."""

    sources: List[DocumentSourceConfig]
    label: Optional[str] = None

    def build(self, work_dir: PathOrStr) -> List["ConcatenatedDocumentSource"]:  # type: ignore[override]
        sources = [
            source for source_config in self.sources for source in source_config.build(work_dir)
        ]
        return [
            ConcatenatedDocumentSource(
                *sources,
                work_dir=work_dir,
                label=self.label,
            )
        ]


class ConcatenatedDocumentSource(ConcatenatedTokenSource, DocumentSource):
    """
    A document source that can be created from concatenating multiple other document sources.
    """

    Config = ConcatenatedDocumentSourceConfig  # type: ignore[assignment]

    def __init__(self, *sources: DocumentSource, work_dir: PathOrStr, label: Optional[str] = None):
        super().__init__(*sources, work_dir=work_dir, label=label)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.sources}"

    @property
    def sources(self) -> Tuple[DocumentSource, ...]:
        return typing.cast(Tuple[DocumentSource, ...], self._sources)

    def get_document_offsets(self) -> Iterable[tuple[int, int]]:
        start_offset = 0
        for source in self.sources:
            source_size = source.num_tokens
            last_doc_end = 0
            for doc_start, doc_end in source.get_document_offsets():
                assert doc_start == last_doc_end  # API assumes consecutive documents
                yield doc_start + start_offset, doc_end + start_offset
                last_doc_end = doc_end

            # To avoid unexpected results, we ALWAYS treat the end of a source file as the end of
            # a document, even if it doesn't end with an EOS token ID. This *should* always be the case
            # anyway, but just to be careful.
            if last_doc_end != source_size:
                yield last_doc_end + start_offset, source_size + start_offset

            start_offset += source_size

    def children(self):
        return self.sources
