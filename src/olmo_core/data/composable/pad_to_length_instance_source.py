import functools as ft
import hashlib
import logging
import typing
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError

from ..tokenizer import TokenizerConfig
from ..types import NumpyDatasetDType
from .instance_source import Instance, InstanceSource, InstanceSourceConfig
from .numpy_document_source import NumpyDocumentSource
from .token_source import DocumentSource, DocumentSourceConfig
from .utils import as_ndarray, resolve_seed

log = logging.getLogger(__name__)


@dataclass
class PadToLengthInstanceSourceConfig(InstanceSourceConfig):
    """Config for :class:`PadToLengthInstanceSource`."""

    sources: List[DocumentSourceConfig]
    sequence_length: int
    tokenizer: TokenizerConfig
    max_sequence_length: Optional[int] = None
    label: Optional[str] = None

    @classmethod
    def from_npy(
        cls,
        *npy_paths: str,
        tokenizer: TokenizerConfig,
        sequence_length: int,
        max_sequence_length: Optional[int] = None,
        dtype: Optional[NumpyDatasetDType] = None,
        source_permutation_seed: Optional[int] = None,
        source_group_size: int = 1,
        label_mask_paths: Optional[List[str]] = None,
        expand_glob: Optional[bool] = None,
        label: Optional[str] = None,
    ) -> "PadToLengthInstanceSourceConfig":
        """
        Create a :class:`PadToLengthInstanceSourceConfig` from one or more tokenized ``.npy`` source files.
        """
        return cls(
            sources=[
                NumpyDocumentSource.Config(
                    source_paths=list(npy_paths),
                    tokenizer=tokenizer,
                    dtype=dtype,
                    source_permutation_seed=resolve_seed(source_permutation_seed),
                    source_group_size=source_group_size,
                    label_mask_paths=label_mask_paths,
                    expand_glob=expand_glob,
                )
            ],
            tokenizer=tokenizer,
            sequence_length=sequence_length,
            max_sequence_length=max_sequence_length,
            label=label,
        )

    def build(self, work_dir: PathOrStr) -> "PadToLengthInstanceSource":
        return PadToLengthInstanceSource(
            *[source for source_config in self.sources for source in source_config.build(work_dir)],
            sequence_length=self.sequence_length,
            max_sequence_length=self.max_sequence_length,
            work_dir=work_dir,
            tokenizer=self.tokenizer,
            label=self.label,
        )


class PadToLengthInstanceSource(InstanceSource):
    """
    An :class:`InstanceSource` that emits exactly ONE document per instance, right-padded to
    ``sequence_length`` with the tokenizer's pad token. Documents longer than ``sequence_length``
    are skipped (with a logged count).

    Use this instead of :class:`ConcatAndChunkInstanceSource` or :class:`PackingInstanceSource`
    when examples must not attend to each other -- e.g., SFT for attention variants that do not
    support intra-document masking (such as landmark attention), where packing would create a
    train/eval mismatch: at evaluation the model sees a single example in its window, so training
    instances should too. Because padding is appended *after* the document and attention is
    causal, the supervised tokens see exactly the same prefix they would at evaluation time.

    The emitted ``label_mask`` is ``False`` on padding and otherwise preserves the upstream
    ``label_mask`` (defaulting to ``True`` for document tokens when the upstream source has none).

    :param sources: Sources of documents.
    :param sequence_length: The length of each emitted instance; documents longer than this are skipped.
    :param tokenizer: The tokenizer configuration (provides the pad token ID).
    :param max_sequence_length: Must equal ``sequence_length`` if given (no sequence-length ramp).
    """

    Config = PadToLengthInstanceSourceConfig

    DISPLAY_ICON = ""

    def __init__(
        self,
        *sources: DocumentSource,
        sequence_length: int,
        work_dir: PathOrStr,
        tokenizer: TokenizerConfig,
        max_sequence_length: Optional[int] = None,
        label: Optional[str] = None,
    ):
        super().__init__(
            sequence_length=sequence_length,
            max_sequence_length=max_sequence_length,
            work_dir=work_dir,
            label=label,
        )
        if self.sequence_length != self.max_sequence_length:
            raise OLMoConfigurationError(
                f"'{self.__class__.__name__}' requires 'sequence_length' to be equal to 'max_sequence_length'."
            )
        self._sources = sources
        self._tokenizer = tokenizer

    @property
    def sources(self) -> Tuple[DocumentSource, ...]:
        return self._sources

    @property
    def pad_token_id(self) -> int:
        return self._tokenizer.pad_token_id

    @ft.cached_property
    def _document_index(self) -> Tuple[Tuple[int, int, int], ...]:
        """
        ``(source_idx, start, end)`` for every document that fits within ``sequence_length``,
        in source order.
        """
        index: List[Tuple[int, int, int]] = []
        n_skipped = 0
        for source_idx, source in enumerate(self.sources):
            for start, end in source.get_document_offsets():
                if end - start > self.sequence_length:
                    n_skipped += 1
                else:
                    index.append((source_idx, start, end))
        if n_skipped > 0:
            log.warning(
                f"{self}: skipped {n_skipped:,d} document(s) longer than "
                f"sequence_length={self.sequence_length:,d}."
            )
        if not index:
            raise OLMoConfigurationError(
                f"{self}: no documents fit within sequence_length={self.sequence_length:,d}."
            )
        return tuple(index)

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        # NOTE: 'sequence_length' affects which documents are skipped, so it's part of the fingerprint.
        sha256_hash.update(
            (
                f"class={self.__class__.__name__},"
                f"{self.sequence_length=},"
                f"pad_token_id={self.pad_token_id},"
            ).encode()
        )
        for source in self.sources:
            sha256_hash.update(f"source={source.fingerprint},".encode())
        return sha256_hash.hexdigest()

    def __len__(self) -> int:
        return len(self._document_index)

    def __getitem__(self, idx: int) -> Instance:
        idx = self.validate_index(idx)
        source_idx, start, end = self._document_index[idx]
        token_range = self.sources[source_idx].get_token_range(start, end)

        input_ids = as_ndarray(token_range["input_ids"]).astype(np.int_)
        if "label_mask" in token_range:
            label_mask = as_ndarray(token_range["label_mask"]).astype(np.bool_)
        else:
            label_mask = np.ones_like(input_ids, dtype=np.bool_)

        pad_shape = (0, self.sequence_length - input_ids.size)
        input_ids = np.pad(input_ids, pad_shape, constant_values=self.pad_token_id)
        label_mask = np.pad(label_mask, pad_shape, constant_values=False)

        return {
            "input_ids": typing.cast(Sequence[int], input_ids),
            "label_mask": typing.cast(Sequence[bool], label_mask),
        }

    def children(self):
        return self.sources
