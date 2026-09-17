"""FigureSpec -> (PNG bytes, RenderAudit).

Two rules make the spec trustworthy, and both are load-bearing:

1. **Nothing is automatic.** `set_xlim`/`set_xticks`/`set_xticklabels` are always called
   explicitly, so a matplotlib locator or formatter can never silently disagree with the
   spec. (Without this, a version upgrade changes what is drawn while the spec still claims
   the old truth, and the corpus becomes confidently wrong at scale.)

2. **`bbox_inches="tight"` is never used.** A tight bbox re-crops the canvas *after* the
   audit has measured artist positions against `fig.bbox`, which invalidates every pixel
   threshold and can clip an edge tick label that the spec still counts as labeled. We use
   `layout="constrained"` instead.

`axes.unicode_minus=False` is global: matplotlib otherwise renders negatives with U+2212,
which no model emits, so a verbatim tick-label answer of "-5" could never be matched.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .spec import FigureSpec  # noqa: E402


@dataclass
class RenderAudit:
    """What matplotlib actually drew, in figure-pixel space.

    Consulted only for geometric guards -- whether two tick labels overlap, whether a
    curve crossing is far enough from another curve to be visible. Never to derive an
    answer.
    """

    fig_w: float
    fig_h: float
    # panel index -> axis role -> list of (label text, (x0, y0, x1, y1)) in pixels
    tick_label_boxes: dict = field(default_factory=dict)
    # panel index -> series index -> (N, 2) pixel coordinates of the plotted samples
    series_px: dict = field(default_factory=dict)
    axes_bbox: dict = field(default_factory=dict)
    offset_text: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)

    def drawn_labeled_ticks(self, panel: int, role: str) -> int:
        return len(self.tick_label_boxes.get(panel, {}).get(role, []))

    def validate(self, spec: FigureSpec) -> bool:
        """Hard consistency gate: what was drawn must equal what the spec claims.

        A figure that fails is discarded rather than corrected, because a spec/render
        disagreement means we do not know which one the image shows.
        """
        self.problems.clear()
        for i, panel in enumerate(spec.panels):
            for role, axis in (("x", panel.x), ("y", panel.y)):
                drawn = [t for t, _ in self.tick_label_boxes.get(i, {}).get(role, [])]
                want = [s for s in axis.ticks.labels if s.strip()]
                if drawn != want:
                    self.problems.append(
                        f"panel{i}.{role}: drawn {drawn!r} != spec {want!r}"
                    )
                boxes = self.tick_label_boxes.get(i, {}).get(role, [])
                for (_, a), (_, b) in zip(boxes, boxes[1:]):
                    if _overlap_frac(a, b) > 0.15:
                        self.problems.append(f"panel{i}.{role}: tick labels overlap")
                        break
                for _, bb in boxes:
                    if bb[0] < -1 or bb[1] < -1 or bb[2] > self.fig_w + 1 or bb[3] > self.fig_h + 1:
                        self.problems.append(f"panel{i}.{role}: tick label outside canvas")
                        break
        return not self.problems


def _overlap_frac(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / area if area > 0 else 0.0


def render(spec: FigureSpec) -> tuple[bytes, RenderAudit]:
    rows, cols = spec.grid_shape
    with plt.style.context(spec.style.style_sheet):
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["font.size"] = spec.style.font_size
        fig, axes = plt.subplots(
            rows, cols, figsize=spec.style.figsize, dpi=spec.style.dpi,
            squeeze=False, layout="constrained",
        )
        for ax in axes.ravel():
            ax.set_visible(False)

        for panel in spec.panels:
            r, c = panel.grid_pos
            ax = axes[r - 1][c - 1]
            ax.set_visible(True)
            for s in panel.series:
                x, y = np.asarray(s.x), np.asarray(s.y)
                if s.kind == "line":
                    ax.plot(x, y, color=s.color, linestyle=s.linestyle,
                            marker=s.marker, label=s.label)
                elif s.kind == "scatter":
                    ax.scatter(x, y, color=s.color, marker=s.marker or "o", label=s.label)
                elif s.kind == "bar":
                    ax.bar(x, y, color=s.color, label=s.label)
            # Explicit, always -- see rule 1 in the module docstring.
            ax.set_xlim(*panel.x.lim)
            ax.set_ylim(*panel.y.lim)
            ax.set_xticks(list(panel.x.ticks.values))
            ax.set_xticklabels(list(panel.x.ticks.labels), rotation=panel.x.ticks.rotation)
            ax.set_yticks(list(panel.y.ticks.values))
            ax.set_yticklabels(list(panel.y.ticks.labels), rotation=panel.y.ticks.rotation)
            if panel.x.label:
                ax.set_xlabel(panel.x.label)
            if panel.y.label:
                ax.set_ylabel(panel.y.label)
            if panel.title:
                ax.set_title(panel.title)
            if panel.grid:
                ax.grid(True, alpha=0.3)
            if panel.has_legend and panel.legend_entries:
                if panel.legend_loc == "outside right":
                    # CharXiv's legend templates explicitly include legends drawn outside
                    # the axes; a corpus whose legends are always inside teaches "nothing
                    # inside the panel -> no legend", which shows up as NA over-declaration
                    # (t12 precision 74.5% after the first training run).
                    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                              fontsize=spec.style.font_size * 0.85)
                else:
                    ax.legend(loc=panel.legend_loc, fontsize=spec.style.font_size * 0.85)
        if spec.suptitle:
            fig.suptitle(spec.suptitle)

        fig.canvas.draw()
        audit = _audit(fig, axes, spec)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")  # NB: no bbox_inches -- see rule 2
        plt.close(fig)
    return buf.getvalue(), audit


def _audit(fig, axes, spec: FigureSpec) -> RenderAudit:
    renderer = fig.canvas.get_renderer()
    audit = RenderAudit(fig_w=fig.bbox.width, fig_h=fig.bbox.height)
    for i, panel in enumerate(spec.panels):
        r, c = panel.grid_pos
        ax = axes[r - 1][c - 1]
        audit.axes_bbox[i] = tuple(ax.get_window_extent(renderer).extents)
        audit.tick_label_boxes[i] = {}
        for role, getter in (("x", ax.get_xticklabels), ("y", ax.get_yticklabels)):
            boxes = []
            for t in getter():
                text = t.get_text()
                if not text.strip():
                    continue  # unlabeled tick: drawn, but carries no text
                boxes.append((text, tuple(t.get_window_extent(renderer).extents)))
            audit.tick_label_boxes[i][role] = boxes
        audit.offset_text[i] = {
            "x": ax.xaxis.get_offset_text().get_text(),
            "y": ax.yaxis.get_offset_text().get_text(),
        }
        audit.series_px[i] = {
            j: ax.transData.transform(np.column_stack([s.x, s.y]))
            for j, s in enumerate(panel.series)
        }
    return audit
