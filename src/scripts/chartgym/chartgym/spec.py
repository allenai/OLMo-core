"""Fully-explicit chart ground truth.

A ``FigureSpec`` is the complete description of a figure: every tick value, every tick
label string, every series sample, every label. The renderer sets all of it explicitly --
no auto-locators, no autoscaling -- so the spec *is* the truth and no answer ever has to be
recovered from pixels.

The one thing a spec cannot know is what matplotlib actually drew (a label can be clipped,
two labels can overlap, a formatter can invent an offset string). That is what
``audit.RenderAudit`` is for, and it is consulted only for *guards* -- never to derive an
answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass(frozen=True)
class TickSpec:
    values: tuple[float, ...]
    labels: tuple[str, ...]          # "" means the tick is drawn but not labeled
    rotation: float = 0.0

    @property
    def n_labeled(self) -> int:
        return sum(1 for s in self.labels if s.strip())

    def labeled_pairs(self) -> list[tuple[float, str]]:
        return [(v, s) for v, s in zip(self.values, self.labels) if s.strip()]


@dataclass(frozen=True)
class AxisSpec:
    role: str                        # "x" | "y"
    label: Optional[str]             # None => no axis label (a legitimate NA case)
    lim: tuple[float, float]
    ticks: TickSpec
    scale: str = "linear"
    numeric_ticks: bool = True       # False for categorical axes (an NA case for spacing)

    @property
    def tick_spacing(self) -> Optional[float]:
        """Constant spacing between consecutive labeled numeric ticks, else None.

        None is the ground truth for "not applicable": non-numeric ticks, fewer than two
        ticks, or non-constant spacing. CharXiv templates 8/9 have exactly this NA branch,
        and 17.9% / 16.9% of their instances take it.
        """
        if not self.numeric_ticks:
            return None
        vals = [v for v, s in self.ticks.labeled_pairs()]
        if len(vals) < 2:
            return None
        diffs = [round(b - a, 9) for a, b in zip(vals, vals[1:])]
        span = max(abs(v) for v in vals) or 1.0
        if max(diffs) - min(diffs) > 1e-6 * span:
            return None
        return diffs[0]


@dataclass(frozen=True)
class SeriesSpec:
    label: Optional[str]             # None => not in the legend
    kind: str                        # "line" | "scatter" | "bar"
    x: tuple[float, ...]
    y: tuple[float, ...]
    color: str
    linestyle: str = "-"
    marker: Optional[str] = None
    shape: str = "linear"            # generative family, the gold for trend questions
    n_peaks: int = 0                 # prominent local maxima, by construction


@dataclass(frozen=True)
class PanelSpec:
    grid_pos: tuple[int, int]        # 1-indexed (row, col)
    title: Optional[str]
    x: AxisSpec
    y: AxisSpec
    series: tuple[SeriesSpec, ...]
    has_legend: bool
    legend_loc: str = "best"
    grid: bool = False

    @property
    def legend_entries(self) -> tuple[str, ...]:
        if not self.has_legend:
            return ()
        return tuple(s.label for s in self.series if s.label)

    @property
    def is_line_plot(self) -> bool:
        return any(s.kind == "line" for s in self.series)


@dataclass(frozen=True)
class StyleSpec:
    style_sheet: str
    font_size: float
    dpi: int
    figsize: tuple[float, float]
    seed: int


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    seed: int
    grid_shape: tuple[int, int]
    panels: tuple[PanelSpec, ...]
    style: StyleSpec
    suptitle: Optional[str] = None
    difficulty: str = "medium"
    mpl_version: str = ""

    @property
    def n_panels(self) -> int:
        return len(self.panels)

    def total_labeled_ticks(self) -> int:
        """Labeled ticks summed over every axis of every panel.

        This is the statistic behind the model's single worst measured skill (29.91% at
        n=224, a systematic under-count with median signed error -6).
        """
        return sum(p.x.ticks.n_labeled + p.y.ticks.n_labeled for p in self.panels)
