"""
Counting information-flow paths between documents through an attention graph.

Given a single-layer attention mask ``M`` (see :mod:`.masks`) treated as the adjacency
matrix of a directed graph (edge ``i -> j`` iff token ``i`` attends to token ``j``), an
``L``-layer transformer that reuses the same connectivity at every layer lets information
flow along **length-``L`` walks**. The number of length-``L`` walks from token ``a`` to
token ``b`` is exactly ``(M ** L)[a, b]`` (standard adjacency-power counting).

So "how many direct paths connect document A to document B in an ``L``-layer model" is::

    paths(A, B; L) = sum over a in A, b in B of (M ** L)[a, b]

This module computes those counts, aggregates them to a document-by-document matrix,
and provides sweeps over layers / corpus size / attention type. It also offers a boolean
**reachability** variant (does *any* path exist) which avoids the combinatorial blow-up
of raw walk counts.

Worked example (from the request): one layer, docs A and B of 20 tokens each, full
attention -> ``paths(A, B; 1) = 20 * 20 = 400``. Two layers over a corpus of ``N`` such
docs -> each of the 400 A/B endpoint pairs has up to ``20 * N`` intermediate tokens, giving
up to ``400 * 20 * N`` walks. :func:`doc_pair_paths` reproduces these (with causality
applied, so the realized numbers are the causal fraction of those upper bounds).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .corpus import Corpus
from .masks import build_mask


def walk_count_matrix(M: np.ndarray, num_layers: int) -> np.ndarray:
    """
    ``(T, T)`` matrix whose ``[a, b]`` entry is the number of length-``num_layers`` walks
    from token ``a`` to token ``b`` in the attention graph ``M``.

    :param M: Boolean or 0/1 ``(T, T)`` adjacency (query attends key).
    :param num_layers: Walk length ``L`` (= number of transformer layers). ``L=1`` returns
        the mask itself (direct connections).

    Uses ``float64`` because walk counts grow exponentially with depth and overflow int64.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    A = M.astype(np.float64)
    P = A.copy()
    for _ in range(num_layers - 1):
        P = P @ A
    return P


def reachable_matrix(M: np.ndarray, num_layers: int) -> np.ndarray:
    """
    Boolean ``(T, T)`` matrix: ``[a, b]`` is True iff *some* walk of length ``<= num_layers``
    connects ``a`` to ``b``. Cheap alternative to :func:`walk_count_matrix` when you only
    care whether documents become connected (not by how many paths).
    """
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")
    A = M.astype(bool)
    reach = A.copy()
    frontier = A.copy()
    for _ in range(num_layers - 1):
        # one more hop: (frontier @ A) as boolean
        frontier = (frontier.astype(np.float32) @ A.astype(np.float32)) > 0
        reach |= frontier
    return reach


def _aggregate_docs(token_mat: np.ndarray, corpus: Corpus, reduce: str = "sum") -> np.ndarray:
    """
    Aggregate a ``(T, T)`` token-level matrix into an ``(N, N)`` document-level matrix by
    summing (or averaging) over the token block for each document pair.

    :param reduce: ``"sum"`` (total paths between the two docs) or ``"mean"`` (per
        token-pair average).
    """
    N = corpus.num_docs
    d = corpus.token_to_doc
    # sum token_mat over doc blocks via matrix multiply with a one-hot doc indicator
    onehot = np.zeros((corpus.num_tokens, N), dtype=np.float64)
    onehot[np.arange(corpus.num_tokens), d] = 1.0
    block_sum = onehot.T @ token_mat.astype(np.float64) @ onehot  # (N, N)
    if reduce == "sum":
        return block_sum
    if reduce == "mean":
        sizes = np.asarray(corpus.doc_lengths, dtype=np.float64)
        denom = np.outer(sizes, sizes)
        return block_sum / denom
    raise ValueError(f"unknown reduce {reduce!r}")


def doc_pair_paths(
    M: np.ndarray,
    corpus: Corpus,
    num_layers: int,
    reduce: str = "sum",
) -> np.ndarray:
    """
    ``(N, N)`` matrix of length-``num_layers`` path counts between every ordered pair of
    documents. Entry ``[p, q]`` = number of walks starting in document ``p`` and ending in
    document ``q`` (so with causal attention it is non-zero only when ``p`` is at or after
    ``q``).

    :param reduce: ``"sum"`` for total paths, ``"mean"`` for per-token-pair average.
    """
    W = walk_count_matrix(M, num_layers)
    return _aggregate_docs(W, corpus, reduce=reduce)


def doc_pair_reachable(M: np.ndarray, corpus: Corpus, num_layers: int) -> np.ndarray:
    """
    ``(N, N)`` boolean-ish matrix: ``[p, q]`` = fraction of token pairs ``(a in p, b in q)``
    that are connected by *some* walk of length ``<= num_layers``. ``1.0`` means the two
    documents are fully connected at that depth; ``0.0`` means not connected at all.
    """
    R = reachable_matrix(M, num_layers).astype(np.float64)
    return _aggregate_docs(R, corpus, reduce="mean")


def average_cross_doc_paths(
    M: np.ndarray,
    corpus: Corpus,
    num_layers: int,
    reduce: str = "sum",
    include_self: bool = False,
) -> float:
    """
    Scalar summary: mean of :func:`doc_pair_paths` over all ordered document pairs
    ``p != q`` (or all pairs if ``include_self``). This is "the average number of paths
    between two documents" for the given attention type and depth.
    """
    D = doc_pair_paths(M, corpus, num_layers, reduce=reduce)
    N = corpus.num_docs
    if include_self:
        mask = np.ones((N, N), dtype=bool)
    else:
        mask = ~np.eye(N, dtype=bool)
    vals = D[mask]
    return float(vals.mean()) if vals.size else 0.0


# --------------------------------------------------------------------------
# sweeps
# --------------------------------------------------------------------------

def sweep_layers(
    corpus: Corpus,
    attn_types: Dict[str, dict],
    layers: List[int],
    reduce: str = "sum",
    metric: str = "paths",
) -> Dict[str, np.ndarray]:
    """
    For each attention type, compute the average cross-document connectivity as a function
    of the number of layers.

    :param attn_types: Mapping ``display_name -> {"name": registered_type, **params}``.
    :param layers: List of layer counts to evaluate.
    :param metric: ``"paths"`` (avg path count, via :func:`average_cross_doc_paths`) or
        ``"reachable"`` (avg fraction of connected token pairs, via
        :func:`doc_pair_reachable`).

    :returns: Mapping ``display_name -> array`` aligned with ``layers``.
    """
    out: Dict[str, np.ndarray] = {}
    for label, spec in attn_types.items():
        spec = dict(spec)
        name = spec.pop("name")
        M = build_mask(name, corpus, **spec)
        vals = []
        for L in layers:
            if metric == "paths":
                vals.append(average_cross_doc_paths(M, corpus, L, reduce=reduce))
            elif metric == "reachable":
                R = doc_pair_reachable(M, corpus, L)
                off = R[~np.eye(corpus.num_docs, dtype=bool)]
                vals.append(float(off.mean()) if off.size else 0.0)
            else:
                raise ValueError(f"unknown metric {metric!r}")
        out[label] = np.asarray(vals, dtype=np.float64)
    return out


def sweep_corpus_size(
    doc_len: int,
    corpus_sizes: List[int],
    attn_types: Dict[str, dict],
    num_layers: int,
    reduce: str = "sum",
    seed: Optional[int] = 0,
) -> Dict[str, np.ndarray]:
    """
    For each attention type, compute the average cross-document path count as a function of
    the number of documents in the corpus (all documents of length ``doc_len``).

    :returns: Mapping ``display_name -> array`` aligned with ``corpus_sizes``.
    """
    out: Dict[str, np.ndarray] = {label: [] for label in attn_types}
    for N in corpus_sizes:
        corpus = Corpus.uniform(N, doc_len)
        for label, spec in attn_types.items():
            spec_c = dict(spec)
            name = spec_c.pop("name")
            M = build_mask(name, corpus, **spec_c)
            out[label].append(average_cross_doc_paths(M, corpus, num_layers, reduce=reduce))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}
