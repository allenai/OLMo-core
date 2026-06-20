import functools as ft
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Tuple

from olmo_core.aliases import PathOrStr
from olmo_core.exceptions import OLMoConfigurationError

from .instance_source import Instance, InstanceSource, InstanceSourceConfig
from .token_source import DocumentSource, DocumentSourceConfig

log = logging.getLogger(__name__)


@dataclass
class LandmarkPackingInstanceSourceConfig(InstanceSourceConfig):
    """
    Config for :class:`LandmarkPackingInstanceSource`.

    :param source: The upstream *document* source (one document = one example, with its own
        ``label_mask``). Document boundaries come from :meth:`DocumentSource.get_document_offsets`.
    :param sequence_length: The length (in landmark-token space, i.e. *after* landmark insertion) of
        each emitted instance. Must be a multiple of ``block_size = mem_freq + 1``.
    :param mem_freq: Regular tokens between landmark tokens; block size is ``mem_freq + 1``.
    :param mem_id: The landmark (memory) token id.
    :param pad_id: Token id used to pad a document's last partial block up to a multiple of
        ``mem_freq``, and to fill the tail of a packed window. Padding never contributes to the loss.
    :param exclude_landmark_predictors: Also drop the loss term *at* each landmark position (see
        :class:`~olmo_core.data.composable.landmark_instance_source.LandmarkInstanceSource`).
    :param warn_drop_fraction: Log a warning if at least this fraction of documents are dropped for
        being longer than a single packed window (a signal to raise ``sequence_length``).
    """

    source: DocumentSourceConfig
    sequence_length: int
    mem_freq: int
    mem_id: int
    pad_id: int
    exclude_landmark_predictors: bool = False
    warn_drop_fraction: float = 0.01

    def build(self, work_dir: PathOrStr) -> "LandmarkPackingInstanceSource":
        from .token_source import ConcatenatedDocumentSource

        sources = self.source.build(work_dir)
        if len(sources) == 0:
            raise OLMoConfigurationError(
                "LandmarkPackingInstanceSource 'source' config built zero DocumentSources."
            )
        # A glob'd npy source builds one DocumentSource per file; concatenate them into a single
        # document stream (document boundaries are preserved across the concatenation).
        source = (
            sources[0]
            if len(sources) == 1
            else ConcatenatedDocumentSource(*sources, work_dir=work_dir)
        )
        return LandmarkPackingInstanceSource(
            source,
            sequence_length=self.sequence_length,
            mem_freq=self.mem_freq,
            mem_id=self.mem_id,
            pad_id=self.pad_id,
            exclude_landmark_predictors=self.exclude_landmark_predictors,
            warn_drop_fraction=self.warn_drop_fraction,
            work_dir=work_dir,
            label=self.label,
        )


class LandmarkPackingInstanceSource(InstanceSource):
    """
    Pack whole documents into fixed-length windows for **landmark attention with correct
    intra-document masking**.

    Unlike ``ConcatAndChunk -> LandmarkInstanceSource`` (which concatenates documents *before*
    inserting landmarks, so document boundaries land at arbitrary, non-block-aligned positions), this
    source inserts landmarks **per document**: each document's content is padded to a multiple of
    ``mem_freq`` and a landmark token is appended to every block, so the document occupies a whole
    number of landmark blocks. Documents are then greedily concatenated (next-fit) into windows of
    ``sequence_length`` tokens, and the tail of each window is padded. Because every document (and the
    tail pad) is a whole number of blocks, every document boundary is a multiple of ``block_size`` --
    exactly the alignment :class:`~olmo_core.nn.attention.LandmarkAttention` (and the fast/sparse
    variants) require for packed masking.

    Each emitted instance carries an explicit ``doc_lens`` (the per-document landmark-space lengths,
    summing to ``sequence_length``, with the tail pad as a final document), which the data loader
    forwards to the model for block-diagonal masking. Set ``generate_doc_lengths=False`` on the data
    loader -- the document boundaries come from here, not from EOS tokens.

    Landmark and pad positions are excluded from the loss (``label_mask`` is ``False`` there);
    content tokens keep the upstream ``label_mask`` (defaulting to ``True``).

    Documents that do not fit in a single empty window (their landmark length exceeds
    ``sequence_length``) are dropped; if at least ``warn_drop_fraction`` of documents are dropped a
    warning is logged, since that usually means ``sequence_length`` should be raised.

    .. note::
        The packing plan is built eagerly (one pass over the document offsets) and held in memory --
        ``O(#documents)`` integers -- which is appropriate for SFT-scale corpora.
    """

    Config = LandmarkPackingInstanceSourceConfig
    DISPLAY_ICON = ""

    def __init__(
        self,
        source: DocumentSource,
        *,
        sequence_length: int,
        mem_freq: int,
        mem_id: int,
        pad_id: int,
        work_dir: PathOrStr,
        exclude_landmark_predictors: bool = False,
        warn_drop_fraction: float = 0.01,
        label=None,
    ):
        if mem_freq < 1:
            raise OLMoConfigurationError(f"'mem_freq' must be >= 1 (got {mem_freq}).")
        block_size = mem_freq + 1
        if sequence_length % block_size != 0:
            raise OLMoConfigurationError(
                f"'sequence_length' ({sequence_length}) must be a multiple of the landmark block "
                f"size (mem_freq + 1 = {block_size})."
            )
        super().__init__(
            sequence_length=sequence_length,
            max_sequence_length=sequence_length,
            work_dir=work_dir,
            label=label,
        )
        self._source = source
        self.mem_freq = mem_freq
        self.mem_id = mem_id
        self.pad_id = pad_id
        self.block_size = block_size
        self.exclude_landmark_predictors = exclude_landmark_predictors
        self.warn_drop_fraction = warn_drop_fraction
        self._build_plan()

    @property
    def source(self) -> DocumentSource:
        return self._source

    def _landmark_len(self, content_len: int) -> int:
        """Landmark-space length of a document with ``content_len`` content tokens."""
        n_blocks = (content_len + self.mem_freq - 1) // self.mem_freq  # ceil, pad last block
        return n_blocks * self.block_size

    def _build_plan(self) -> None:
        """Greedy next-fit packing of documents into windows; record window -> document ranges."""
        offsets = [(int(s), int(e)) for s, e in self._source.get_document_offsets()]
        windows: List[List[int]] = []
        current: List[int] = []
        remaining = self.sequence_length
        dropped = 0
        for i, (s, e) in enumerate(offsets):
            content_len = e - s
            if content_len <= 0:
                continue
            lm_len = self._landmark_len(content_len)
            if lm_len > self.sequence_length:
                dropped += 1
                continue
            if lm_len > remaining:
                if current:
                    windows.append(current)
                current = []
                remaining = self.sequence_length
            current.append(i)
            remaining -= lm_len
        if current:
            windows.append(current)

        self._offsets = offsets
        self._windows = windows
        self._num_dropped = dropped
        n_docs = sum(1 for s, e in offsets if e - s > 0)
        if dropped:
            frac = dropped / max(n_docs, 1)
            msg = (
                f"LandmarkPackingInstanceSource dropped {dropped}/{n_docs} documents "
                f"({frac:.1%}) that exceeded sequence_length={self.sequence_length} "
                f"(in landmark-token space)."
            )
            if frac >= self.warn_drop_fraction:
                log.warning(msg + " Consider increasing sequence_length.")
            else:
                log.info(msg)

    @property
    def num_instances(self) -> int:
        return len(self._windows)

    @ft.cached_property
    def fingerprint(self) -> str:
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            (
                f"class={self.__class__.__name__},"
                f"{self.sequence_length=},"
                f"{self.mem_freq=},"
                f"{self.mem_id=},"
                f"{self.pad_id=},"
                f"{self.exclude_landmark_predictors=},"
                f"source={self._source.fingerprint},"
            ).encode()
        )
        return sha256_hash.hexdigest()

    def __len__(self) -> int:
        return self.num_instances

    def _emit_document(self, content: List[int], mask: List[bool]) -> Tuple[List[int], List[bool]]:
        """Pad content to a multiple of ``mem_freq`` and insert a landmark after each block."""
        pad = (-len(content)) % self.mem_freq
        if pad:
            content = content + [self.pad_id] * pad
            mask = mask + [False] * pad
        ids: List[int] = []
        new_mask: List[bool] = []
        for start in range(0, len(content), self.mem_freq):
            ids.extend(content[start : start + self.mem_freq])
            ids.append(self.mem_id)
            new_mask.extend(mask[start : start + self.mem_freq])
            new_mask.append(False)  # landmark excluded from loss
        if self.exclude_landmark_predictors:
            # Drop the loss term at each landmark position (predicts the next block's first token).
            for p in range(self.block_size - 1, len(new_mask) - 1, self.block_size):
                new_mask[p + 1] = False
        return ids, new_mask

    def __getitem__(self, idx: int) -> Instance:
        idx = self.validate_index(idx)
        input_ids: List[int] = []
        label_mask: List[bool] = []
        doc_lens: List[int] = []
        for doc_i in self._windows[idx]:
            s, e = self._offsets[doc_i]
            rng = self._source.get_token_range(s, e)
            content = list(rng["input_ids"])
            raw_mask = rng.get("label_mask")
            mask = list(raw_mask) if raw_mask is not None else [True] * len(content)
            ids, m = self._emit_document(content, mask)
            input_ids.extend(ids)
            label_mask.extend(m)
            doc_lens.append(len(ids))

        # Pad the tail of the window to ``sequence_length`` as a final (loss-free) document. The tail
        # is a whole number of blocks, and we keep the periodic ``is_mem`` invariant by placing a
        # landmark token at the end of each pad block (the pad document is isolated by the document
        # mask and contributes no loss, so its content is irrelevant -- this is just for tidiness).
        tail = self.sequence_length - len(input_ids)
        if tail < 0:
            raise RuntimeError(
                f"Packed window {idx} overflowed sequence_length ({len(input_ids)} > "
                f"{self.sequence_length}); this is a packing-plan bug."
            )
        if tail > 0:
            for _ in range(tail // self.block_size):
                input_ids.extend([self.pad_id] * self.mem_freq + [self.mem_id])
                label_mask.extend([False] * self.block_size)
            doc_lens.append(tail)

        return {"input_ids": input_ids, "label_mask": label_mask, "doc_lens": doc_lens}

    def children(self):
        return [self._source]
