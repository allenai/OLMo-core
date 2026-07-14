#!/usr/bin/env python
"""
End-to-end demo of the attention-graph analysis.

Generates a set of figures under ``OLMo-core/visualizations/attention_graphs/``:

  1. mask_<type>.png            - the token-level attention mask for each variant
  2. docpaths_<type>.png        - document-by-document path counts (small corpus, L layers)
  3. sweep_layers.png           - avg cross-doc paths vs #layers, per attention type
  4. sweep_layers_reach.png     - avg fraction of connected doc pairs vs #layers
  5. sweep_corpus_size.png      - avg cross-doc paths vs #documents, per attention type

Run:  python analysis/run_analysis.py
(pure numpy + matplotlib; no GPU needed)
"""

from __future__ import annotations

import os
import sys

# allow "python analysis/run_analysis.py" from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attention_graphs import Corpus, build_mask, doc_pair_paths, sweep_corpus_size, sweep_layers  # noqa: E402
from attention_graphs import viz  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(REPO, "visualizations", "attention_graphs")

# The attention variants to compare. Each is {"name": registered_type, **params}.
ATTN_TYPES = {
    "dense": {"name": "dense"},
    "sliding_window(w=8)": {"name": "sliding_window", "window": 8},
    "doc_chunked": {"name": "doc_chunked"},
    "doc_chunked+landmark": {"name": "doc_chunked_landmark", "landmarks_per_doc": 1},
    "global_tokens(1)": {"name": "global_tokens", "num_global": 1},
    "hierarchical_dilated": {"name": "hierarchical_dilated", "base": 2},
    "random_doc(p=0.1)": {"name": "random_doc", "keep_prob": 0.1, "seed": 0},
}


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1 & 2: per-type mask + document path matrix on a small corpus
    # ------------------------------------------------------------------ #
    small = Corpus.uniform(num_docs=6, doc_len=10)  # 60 tokens
    for label, spec in ATTN_TYPES.items():
        spec_c = dict(spec)
        name = spec_c.pop("name")
        M = build_mask(name, small, **spec_c)
        slug = label.replace("(", "_").replace(")", "").replace("=", "").replace(" ", "")
        f1 = viz.plot_mask(M, small, title=f"{label}  (mask)", save=os.path.join(OUTDIR, f"mask_{slug}.png"))
        D = doc_pair_paths(M, small, num_layers=3)
        f2 = viz.plot_doc_path_matrix(
            D, title=f"{label}  ({small.num_docs} docs, L=3)  paths A->B",
            save=os.path.join(OUTDIR, f"docpaths_{slug}.png"),
        )
        del f1, f2

    # ------------------------------------------------------------------ #
    # 3 & 4: sweep over number of layers
    # ------------------------------------------------------------------ #
    corpus = Corpus.uniform(num_docs=8, doc_len=16)  # 128 tokens
    layers = [1, 2, 3, 4, 6, 8]
    paths_by_L = sweep_layers(corpus, ATTN_TYPES, layers, metric="paths")
    viz.plot_sweep(
        layers, paths_by_L, xlabel="# layers (walk length L)",
        ylabel="avg # paths between two docs", title="Cross-document paths vs depth",
        save=os.path.join(OUTDIR, "sweep_layers.png"),
    )
    reach_by_L = sweep_layers(corpus, ATTN_TYPES, layers, metric="reachable")
    viz.plot_sweep(
        layers, reach_by_L, xlabel="# layers (walk length L)",
        ylabel="avg fraction of connected token pairs", title="Cross-document reachability vs depth",
        logy=False, save=os.path.join(OUTDIR, "sweep_layers_reach.png"),
    )

    # ------------------------------------------------------------------ #
    # 5: sweep over corpus size (number of documents)
    # ------------------------------------------------------------------ #
    sizes = [2, 4, 8, 16, 32]
    paths_by_N = sweep_corpus_size(doc_len=12, corpus_sizes=sizes, attn_types=ATTN_TYPES, num_layers=3)
    viz.plot_sweep(
        sizes, paths_by_N, xlabel="# documents in corpus",
        ylabel="avg # paths between two docs (L=3)", title="Cross-document paths vs corpus size",
        save=os.path.join(OUTDIR, "sweep_corpus_size.png"),
    )

    print(f"wrote figures to {OUTDIR}")
    # print a small text summary
    print("\navg cross-doc paths (L=3, 8 docs x 16 tok):")
    for label, arr in paths_by_L.items():
        i = layers.index(3)
        print(f"  {label:28s} {arr[i]:.3g}")


if __name__ == "__main__":
    main()
