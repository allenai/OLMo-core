"""
Plotting helpers for attention-graph analysis.

All functions return a matplotlib ``Figure`` and optionally save it. They are pure numpy
+ matplotlib (no GPU), so they run anywhere. Per the repo convention, save rendered
figures under ``OLMo-core/visualizations/``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .corpus import Corpus


def _doc_boundary_lines(ax, corpus: Corpus, **kw) -> None:
    kw = {"color": "white", "lw": 0.6, "alpha": 0.5, **kw}
    for b in corpus.doc_ends[:-1]:
        ax.axhline(b - 0.5, **kw)
        ax.axvline(b - 0.5, **kw)


def plot_mask(
    M: np.ndarray,
    corpus: Corpus,
    title: str = "attention mask",
    save: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of a token-level attention mask (rows = query token ``i``, cols = key token
    ``j``; a cell is on iff ``i`` attends ``j``). Document boundaries are drawn as lines.
    """
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.imshow(M.astype(float), cmap="Greys", interpolation="nearest", aspect="equal")
    _doc_boundary_lines(ax, corpus, color="tab:red", alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel("key token  j  (attended-to)")
    ax.set_ylabel("query token  i  (attends)")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_doc_path_matrix(
    D: np.ndarray,
    title: str = "paths between documents",
    log: bool = True,
    save: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of an ``(N, N)`` document-by-document path-count (or reachability) matrix.

    :param log: Use a log color scale (recommended — path counts span many orders of
        magnitude). Zeros are shown as the floor color.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    data = D.astype(float)
    if log:
        floor = data[data > 0].min() if np.any(data > 0) else 1.0
        norm = LogNorm(vmin=floor, vmax=max(data.max(), floor * 10))
        im = ax.imshow(np.where(data > 0, data, np.nan), cmap="viridis", norm=norm, aspect="equal")
    else:
        im = ax.imshow(data, cmap="viridis", aspect="equal")
    ax.set_title(title)
    ax.set_xlabel("document q  (information source)")
    ax.set_ylabel("document p  (information sink)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="# paths" + (" (log)" if log else ""))
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def plot_sweep(
    x: List,
    series: Dict[str, np.ndarray],
    xlabel: str,
    ylabel: str,
    title: str,
    logy: bool = True,
    save: Optional[str] = None,
) -> plt.Figure:
    """
    Line plot of one array per attention type against a shared x-axis (layers or corpus
    size). Zeros are plotted at the axis floor so doc-chunked (no cross-doc paths) is visible.
    """
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for label, y in series.items():
        y = np.asarray(y, dtype=float)
        ax.plot(x, y, marker="o", ms=4, label=label)
    if logy:
        ax.set_yscale("symlog")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig
