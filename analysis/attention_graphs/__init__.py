"""
Attention-graph analysis: count information-flow paths between documents.

Treat a transformer's attention mask as a directed graph over tokens (edge ``i -> j`` iff
token ``i`` attends to token ``j``). An ``L``-layer model lets information flow along
length-``L`` walks, so the number of "direct paths" connecting document A to document B is
the sum of length-``L`` walk counts over token pairs ``(a in A, b in B)`` — i.e. a block
sum of ``M ** L``. This package builds idealized masks for the repo's attention variants
(dense, document-chunked, landmark, hierarchical-dilated, sliding-window, random-doc,
global-token) and computes / visualizes those path counts across corpus sizes and depths.

Quick start::

    from attention_graphs import Corpus, build_mask, doc_pair_paths
    corpus = Corpus.uniform(num_docs=8, doc_len=20)
    M = build_mask("doc_chunked_landmark", corpus, landmarks_per_doc=1)
    D = doc_pair_paths(M, corpus, num_layers=2)   # (8, 8) path counts

See ``analysis/run_analysis.py`` for an end-to-end figure-generating example.
"""

from .corpus import Corpus
from .masks import available, build_mask, register
from .paths import (
    average_cross_doc_paths,
    doc_pair_paths,
    doc_pair_reachable,
    reachable_matrix,
    sweep_corpus_size,
    sweep_layers,
    walk_count_matrix,
)

__all__ = [
    "Corpus",
    "available",
    "build_mask",
    "register",
    "walk_count_matrix",
    "reachable_matrix",
    "doc_pair_paths",
    "doc_pair_reachable",
    "average_cross_doc_paths",
    "sweep_layers",
    "sweep_corpus_size",
]
