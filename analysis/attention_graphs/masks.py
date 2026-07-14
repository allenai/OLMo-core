"""
Attention-mask builders for graph analysis (idealized).

Every builder returns a boolean matrix ``M`` of shape ``(T, T)`` where::

    M[i, j] == True   <=>   query token i attends to key token j

i.e. token ``i`` can *read* information from token ``j`` in a single layer. All masks
here are **causal**: ``M[i, j]`` can only be True when ``j <= i``.

These are deliberately idealized versions of the attention variants used in the repo
(dense, document-chunked, landmark, hierarchical-dilated, sliding-window, random global
links). They capture the *connectivity structure* that matters for counting cross-document
paths, not the numerical softmax. Register new variants with :func:`register`.

Reading convention for path counting (see :mod:`.paths`): an edge ``i -> j`` means "i
attends to j". A length-``L`` walk ``a -> k1 -> k2 -> ... -> b`` therefore traces how
information at token ``b`` can reach the representation of token ``a`` after ``L`` layers.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from .corpus import Corpus

MaskBuilder = Callable[..., np.ndarray]

_REGISTRY: Dict[str, MaskBuilder] = {}


def register(name: str) -> Callable[[MaskBuilder], MaskBuilder]:
    """Decorator registering a mask builder under ``name``."""

    def deco(fn: MaskBuilder) -> MaskBuilder:
        _REGISTRY[name] = fn
        return fn

    return deco


def available() -> list:
    """Sorted list of registered attention-type names."""
    return sorted(_REGISTRY)


def build_mask(name: str, corpus: Corpus, **params) -> np.ndarray:
    """
    Build the ``(T, T)`` boolean attention mask for attention type ``name``.

    :param name: A registered attention type (see :func:`available`).
    :param corpus: The token/document layout.
    :param params: Type-specific keyword parameters (window, chunk size, ...).
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown attention type {name!r}; available: {available()}")
    return _REGISTRY[name](corpus, **params)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _causal(T: int) -> np.ndarray:
    """Lower-triangular boolean (including diagonal): ``[i, j] = j <= i``."""
    idx = np.arange(T)
    return idx[:, None] >= idx[None, :]


def _same_doc(corpus: Corpus) -> np.ndarray:
    """Boolean ``(T, T)``: query and key belong to the same document."""
    d = corpus.token_to_doc
    return d[:, None] == d[None, :]


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

@register("dense")
def dense(corpus: Corpus) -> np.ndarray:
    """Full causal attention: every token attends to all earlier tokens (and itself)."""
    return _causal(corpus.num_tokens)


@register("sliding_window")
def sliding_window(corpus: Corpus, window: int = 128) -> np.ndarray:
    """
    Causal sliding-window attention: token ``i`` attends to ``j`` iff ``0 <= i - j < window``.
    Purely positional — ignores document boundaries.
    """
    T = corpus.num_tokens
    idx = np.arange(T)
    delta = idx[:, None] - idx[None, :]
    return (delta >= 0) & (delta < window)


@register("doc_chunked")
def doc_chunked(corpus: Corpus) -> np.ndarray:
    """
    Document-chunked attention: causal attention **restricted to within a document**.
    There are *no* direct cross-document edges, so at a single layer the number of
    A<->B connections is exactly zero.
    """
    return _causal(corpus.num_tokens) & _same_doc(corpus)


@register("doc_chunked_landmark")
def doc_chunked_landmark(
    corpus: Corpus,
    landmarks_per_doc: int = 1,
    landmark_reads_own_doc: bool = True,
) -> np.ndarray:
    """
    Document-chunked attention plus per-document **landmark** (summary) tokens that
    bridge documents. This models the *sparse-landmark* connectivity of the repo
    (``SparseLandmarkAttention``): a token sees its own chunk plus only the landmark
    tokens of earlier chunks.

    (Note: the plain grouped-softmax ``landmark`` in the repo lets a token reach *all*
    earlier tokens, gated through the landmark — connectivity-identical to :func:`dense`,
    just reweighted — so it is not a distinct graph. The sparse variant below is the one
    with genuinely different connectivity.)

    Idealized semantics:

    - Every token attends causally within its own document (as in :func:`doc_chunked`).
    - Landmark tokens are the last ``landmarks_per_doc`` tokens of each document.
    - Any token additionally attends to the landmark tokens of all *earlier* documents.
    - If ``landmark_reads_own_doc``, a landmark token attends to every token in its own
      document (so it genuinely summarizes the chunk).

    The only cross-document edges are token -> earlier-landmark, so cross-document paths
    require at least two layers: ``tokenA -> landmark(B) -> tokenB``.
    """
    T = corpus.num_tokens
    M = _causal(T) & _same_doc(corpus)

    is_lm = corpus.is_landmark_mask(landmarks_per_doc)
    causal = _causal(T)

    # every token -> earlier-doc landmark (causal guarantees "earlier position";
    # different-doc guarantees it's a *previous* document since docs are contiguous)
    cross = causal & (~_same_doc(corpus)) & is_lm[None, :]
    M = M | cross

    if landmark_reads_own_doc:
        # landmark query attends to all tokens in its own doc (not only causal-prefix)
        lm_read = is_lm[:, None] & _same_doc(corpus)
        M = M | lm_read
    return M


@register("global_tokens")
def global_tokens(corpus: Corpus, num_global: int = 1) -> np.ndarray:
    """
    Document-chunked attention plus a few **global** tokens per document (attention-sink
    style). The first ``num_global`` tokens of each document are global: every later token
    attends to them, and they attend causally to everything. Cross-document bridging is
    2-hop through a global token.
    """
    T = corpus.num_tokens
    causal = _causal(T)
    M = causal & _same_doc(corpus)

    is_global = np.zeros(T, dtype=bool)
    for d in range(corpus.num_docs):
        s = int(corpus.doc_starts[d])
        k = min(num_global, corpus.doc_lengths[d])
        is_global[s : s + k] = True

    M = M | (causal & is_global[None, :])  # anyone -> earlier global
    M = M | (causal & is_global[:, None])  # global -> anyone earlier
    return M


@register("hierarchical_dilated")
def hierarchical_dilated(
    corpus: Corpus,
    base: int = 2,
    max_docs: Optional[int] = None,
) -> np.ndarray:
    """
    Hierarchical / chunk-granularity dilated attention: a superset of :func:`doc_chunked`.

    - Full causal attention within a document.
    - Cross-document, a document attends to earlier documents at exponentially dilated
      offsets ``1, base, base**2, ...`` (measured in document index). When such a pair is
      connected, *all* token pairs between the two documents are connected (chunk-granular).

    :param base: Dilation base (``2`` -> offsets 1, 2, 4, 8, ...).
    :param max_docs: If set, only connect to earlier docs within this offset horizon.
    """
    T = corpus.num_tokens
    M = _causal(T) & _same_doc(corpus)
    d = corpus.token_to_doc

    N = corpus.num_docs
    offsets = []
    o = 1
    while o < N:
        if max_docs is not None and o > max_docs:
            break
        offsets.append(o)
        o *= base

    if offsets:
        doc_delta = d[:, None] - d[None, :]  # >0 means key is an earlier document
        allow = np.isin(doc_delta, offsets)
        M = M | (allow & _causal(T))
    return M


@register("random_doc")
def random_doc(corpus: Corpus, keep_prob: float = 0.1, seed: int = 0) -> np.ndarray:
    """
    Document-chunked attention plus random sparse **document-level** links (BigBird-style
    fixed sparsity, matching the repo's ``random_doc`` pattern): each ordered document
    pair ``(p, q)`` with ``q < p`` is kept independently with probability ``keep_prob``,
    and when kept, *all* causal token pairs between the two documents are connected.
    """
    T = corpus.num_tokens
    N = corpus.num_docs
    M = _causal(T) & _same_doc(corpus)

    rng = np.random.default_rng(seed)
    keep = rng.random((N, N)) < keep_prob  # keep[p, q]: doc p attends earlier doc q
    d = corpus.token_to_doc
    edge = keep[d[:, None], d[None, :]] & (d[:, None] > d[None, :])
    M = M | (edge & _causal(T))
    return M
