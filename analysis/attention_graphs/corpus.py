"""
Corpus layout for attention-graph analysis.

A :class:`Corpus` describes how a set of documents is laid out as a single flat
sequence of tokens (the way a packed training batch looks to the model). It knows,
for every global token position, which document it belongs to, and it exposes a few
per-document conveniences (slices, landmark positions) that the mask builders in
:mod:`.masks` consume.

Everything downstream (mask construction, path counting) is expressed in terms of a
``Corpus``, so the same analysis works for uniform-length documents, variable-length
documents, or a single big document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class Corpus:
    """
    A flat token sequence partitioned into documents.

    :param doc_lengths: Number of tokens in each document, in sequence order.
    """

    doc_lengths: List[int]

    # Derived (filled in __post_init__).
    token_to_doc: np.ndarray = field(init=False)  # (T,) int, doc index per token
    doc_starts: np.ndarray = field(init=False)  # (N,) int, global start index of each doc
    doc_ends: np.ndarray = field(init=False)  # (N,) int, exclusive end index of each doc

    def __post_init__(self) -> None:
        self.doc_lengths = [int(x) for x in self.doc_lengths]
        starts = np.cumsum([0] + self.doc_lengths[:-1]).astype(np.int64)
        ends = np.cumsum(self.doc_lengths).astype(np.int64)
        self.doc_starts = starts
        self.doc_ends = ends
        tok2doc = np.zeros(int(ends[-1]) if len(ends) else 0, dtype=np.int64)
        for d, (s, e) in enumerate(zip(starts, ends)):
            tok2doc[s:e] = d
        self.token_to_doc = tok2doc

    # -- basic properties -------------------------------------------------

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    @property
    def num_tokens(self) -> int:
        return int(self.token_to_doc.shape[0])

    def doc_slice(self, d: int) -> slice:
        """Global-position ``slice`` covering document ``d``."""
        return slice(int(self.doc_starts[d]), int(self.doc_ends[d]))

    def doc_positions(self, d: int) -> np.ndarray:
        """Array of global token indices belonging to document ``d``."""
        return np.arange(int(self.doc_starts[d]), int(self.doc_ends[d]))

    def landmark_positions(self, per_doc: int = 1) -> np.ndarray:
        """
        Global indices of the ``per_doc`` landmark / summary tokens of every document.

        Landmarks are taken as the *last* ``per_doc`` tokens of each document (an
        idealization of a trailing summary/landmark token). ``per_doc=0`` yields an
        empty array.
        """
        if per_doc <= 0:
            return np.zeros(0, dtype=np.int64)
        out = []
        for d in range(self.num_docs):
            e = int(self.doc_ends[d])
            k = min(per_doc, self.doc_lengths[d])
            out.extend(range(e - k, e))
        return np.asarray(out, dtype=np.int64)

    def is_landmark_mask(self, per_doc: int = 1) -> np.ndarray:
        """Boolean ``(T,)`` marking which tokens are landmarks (see :meth:`landmark_positions`)."""
        m = np.zeros(self.num_tokens, dtype=bool)
        m[self.landmark_positions(per_doc)] = True
        return m

    # -- constructors -----------------------------------------------------

    @classmethod
    def uniform(cls, num_docs: int, doc_len: int) -> "Corpus":
        """A corpus of ``num_docs`` documents each exactly ``doc_len`` tokens long."""
        return cls([doc_len] * num_docs)

    @classmethod
    def random_lengths(
        cls,
        num_docs: int,
        mean_len: int,
        std: float = 0.0,
        min_len: int = 1,
        seed: Optional[int] = 0,
    ) -> "Corpus":
        """
        A corpus whose document lengths are drawn from a (clipped) normal distribution
        with the given ``mean_len`` and ``std``. ``std=0`` reproduces :meth:`uniform`.
        """
        rng = np.random.default_rng(seed)
        if std <= 0:
            lengths: Sequence[int] = [mean_len] * num_docs
        else:
            raw = rng.normal(mean_len, std, size=num_docs)
            lengths = np.clip(np.round(raw), min_len, None).astype(int).tolist()
        return cls(list(lengths))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Corpus(num_docs={self.num_docs}, num_tokens={self.num_tokens}, "
            f"doc_lengths={self.doc_lengths[:6]}{'...' if self.num_docs > 6 else ''})"
        )
