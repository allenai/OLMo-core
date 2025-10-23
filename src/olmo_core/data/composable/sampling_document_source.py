import dataclasses
import functools as ft
import hashlib
import logging
import typing
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np

import olmo_core.distributed.utils as dist_utils
import olmo_core.io as io
from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError

from ..utils import get_rng, load_array_slice, write_array_to_disk
from .token_source import (
    ConcatenatedDocumentSource,
    DocumentSource,
    DocumentSourceConfig,
    TokenRange,
)
from .utils import SEED_NOT_SET, as_ndarray, resolve_seed

log = logging.getLogger(__name__)


@dataclass
class SamplingDocumentSourceConfig(DocumentSourceConfig):
    """
    A config for building a :class:`SamplingDocumentSource`.
    """

    sources: List[DocumentSourceConfig]
    max_tokens: Optional[int] = None
    factor: Optional[float] = None
    seed: Optional[int] = dataclasses.field(default_factory=lambda: resolve_seed(SEED_NOT_SET))
    label: Optional[str] = None

    def __post_init__(self):
        if (self.max_tokens is None) == (self.factor is None):
            raise OLMoConfigurationError("Exactly one of 'max_tokens' or 'factor' must be set.")

    def build(self, work_dir: PathOrStr) -> List["SamplingDocumentSource"]:  # type: ignore[override]
        sources = [s for source in self.sources for s in source.build(work_dir=work_dir)]
        max_tokens = self.max_tokens
        if max_tokens is None:
            assert self.factor is not None
            max_tokens = int(self.factor * sum(source.num_tokens for source in sources))
        return [
            SamplingDocumentSource(
                *sources,
                max_tokens=max_tokens,
                seed=self.seed,
                work_dir=work_dir,
                label=self.label,
            )
        ]


class SamplingDocumentSource(DocumentSource):
    """
    A document source that samples documents from other document sources.
    This can be used to adjust the effective size of a source.

    .. seealso::
        - :class:`SamplingTokenSource`
        - :class:`SamplingInstanceSource`

    :param sources: The sources to sample documents from.
    :param max_tokens: The maximum number of tokens to sample. The resulting source will have
        at most this many tokens, but potentially less because only whole documents are sampled.
    :param seed: A optional seed for sampling documents. If ``None``, no shuffling is done and
        the first documents are taken up to ``max_tokens``.
    """

    Config = SamplingDocumentSourceConfig

    DISPLAY_ICON = "\uedec"

    def __init__(
        self,
        *sources: DocumentSource,
        max_tokens: int,
        seed: Optional[int] = SEED_NOT_SET,
        work_dir: PathOrStr,
        label: Optional[str] = None,
    ):
        from .mixing_document_source import MixingDocumentSource

        assert max_tokens > 0
        if not sources:
            raise ValueError("At least one source must be provided.")

        super().__init__(work_dir=work_dir, label=label)

        unwound_sources: List[DocumentSource] = []
        for s in sources:
            # Unwind any mixing sources so that we sample directly from each of their
            # sources in order to maintain the ratios.
            if isinstance(s, MixingDocumentSource):
                unwound_sources.extend(s.sampled_sources)
            else:
                unwound_sources.append(s)

        source: DocumentSource
        if len(unwound_sources) > 1:
            source = ConcatenatedDocumentSource(*unwound_sources, work_dir=work_dir)
        else:
            source = unwound_sources[0]

        self._og_sources = sources
        self._source = source
        self._max_tokens = max_tokens
        self._seed = resolve_seed(seed)

        # Sample tokens from the source.
        log.info(f"Sampling documents from {self.source}...")
        self._sampled_document_offsets_path = self.work_dir / f"{self.fingerprint}-doc-indices.npy"
        self._sampled_cu_document_lens_path = self.work_dir / f"{self.fingerprint}-doc-lens.npy"
        if (
            not self._sampled_document_offsets_path.is_file()
            or not self._sampled_cu_document_lens_path.is_file()
        ) and self.fs_local_rank == 0:
            # Collect original document indices.
            document_offsets = np.fromiter(
                (idx for offsets in self.source.get_document_offsets() for idx in offsets),
                dtype=np.uint64,
            ).reshape(-1, 2)

            # Maybe shuffle OG doc indices.
            if self.seed is not None:
                rng = get_rng(self.seed)
                rng.shuffle(document_offsets, axis=0)

            # Find cumulative token counts, then repeat/truncate OG docs to get the target max number of tokens.
            document_lengths = document_offsets[:, 1] - document_offsets[:, 0]
            cu_document_lengths = np.cumsum(document_lengths, dtype=np.uint64)
            total_tokens = int(cu_document_lengths[-1])

            n_repetitions = max_tokens // total_tokens
            remaining_sample_size = max_tokens % total_tokens
            sampled_document_offsets = np.take(
                document_offsets,
                (cu_document_lengths <= remaining_sample_size).nonzero()[0],
                axis=0,
            )
            if n_repetitions > 0:
                sampled_document_offsets = np.concatenate(
                    [
                        np.tile(document_offsets, (n_repetitions, 1)),
                        sampled_document_offsets,
                    ]
                )

            if sampled_document_offsets.shape[0] == 0:
                raise RuntimeError(f"Unable to sample {self.max_tokens} tokens from {self.source}")

            # Now get the cumulative lengths of the sampled documents.
            sampled_document_lengths = (
                sampled_document_offsets[:, 1] - sampled_document_offsets[:, 0]
            )
            sampled_cu_document_lengths = np.concatenate(
                [
                    np.array([0], dtype=np.uint64),
                    np.cumsum(sampled_document_lengths, dtype=np.uint64),
                ]
            )

            # Write to disk.
            write_array_to_disk(
                sampled_document_offsets.reshape(-1), self._sampled_document_offsets_path
            )
            write_array_to_disk(sampled_cu_document_lengths, self._sampled_cu_document_lens_path)
        dist_utils.barrier()

    @property
    def source(self) -> DocumentSource:
        return self._source

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @ft.cached_property
    def num_docs(self) -> int:
        return io.get_file_size(self._sampled_document_offsets_path) // np.uint64(0).itemsize // 2

    @ft.cached_property
    def num_tokens(self) -> int:
        return int(
            load_array_slice(
                self._sampled_cu_document_lens_path, self.num_docs, self.num_docs + 1, np.uint64
            )[0]
        )

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            (
                f"class={self.__class__.__name__},"
                f"source={self.source.fingerprint},"
                f"max_tokens={self.max_tokens},"
                f"seed={self.seed},"
            ).encode()
        )
        return sha256_hash.hexdigest()

    def get_token_range(self, start_idx: int, end_idx: int) -> TokenRange:
        start_idx, end_idx = self.validate_indices(start_idx, end_idx)

        # NOTE: we need to map this range to ranges of tokens in the original source.
        # We do that by mapping this range to a range of documents, then getting the right range
        # of tokens from within each document.

        # Load cumulative document lengths for the sampled documents.
        cu_doc_lens = np.memmap(self._sampled_cu_document_lens_path, mode="r", dtype=np.uint64)

        # Get the document indices (with respect to the local sample) that encompasses the token range.
        doc_indices_in_sample = np.logical_and(
            (cu_doc_lens > start_idx)[1:], (cu_doc_lens[:-1] < end_idx)
        ).nonzero()[0]
        starting_doc, ending_doc = int(doc_indices_in_sample[0]), int(doc_indices_in_sample[-1])

        # Now load the corresponding offsets of the original documents.
        og_doc_offsets = load_array_slice(
            self._sampled_document_offsets_path, 2 * starting_doc, 2 * (ending_doc + 1), np.uint64
        ).reshape(-1, 2)

        # Finally, we iterate over the OG document offsets and load the corresponding ranges.
        document_input_ids: List[np.ndarray] = []
        document_label_masks: List[np.ndarray] = []
        tokens_remaining = end_idx - start_idx
        for doc_idx, (doc_start, doc_end) in enumerate(og_doc_offsets):
            if doc_idx == 0:
                doc_start += start_idx - int(cu_doc_lens[starting_doc])

            token_rng = self.source.get_token_range(
                int(doc_start), min(int(doc_end), doc_start + tokens_remaining)
            )
            document_input_ids.append(as_ndarray(token_rng["input_ids"]))
            if "label_mask" in token_rng:
                document_label_masks.append(as_ndarray(token_rng["label_mask"]))

            tokens_remaining -= document_input_ids[-1].size

        # Combine token IDs and maybe label masks for each document.
        input_ids = np.concatenate(document_input_ids)
        out: TokenRange = {"input_ids": typing.cast(Sequence[int], input_ids)}
        if document_label_masks:
            out["label_mask"] = typing.cast(Sequence[bool], np.concatenate(document_label_masks))
        return out

    def get_document_offsets(self) -> Iterable[tuple[int, int]]:
        cu_doc_lens = np.memmap(self._sampled_cu_document_lens_path, mode="r", dtype=np.uint64)
        start_offset = 0
        for cu_doc_len in cu_doc_lens[1:]:
            yield (start_offset, int(cu_doc_len))
            start_offset = int(cu_doc_len)

    def children(self):
        return self._og_sources
