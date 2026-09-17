"""Random FigureSpec generation, with degenerate cases at CharXiv's measured rates.

The NA_RATES below are not invented. They are the measured gold "Not Applicable" rates per
CharXiv descriptive template on the validation split (n=4000, 25.0% of the benchmark
overall) -- see `outputs/chartgym/stage0/RESULTS.md`. A generator that always produces a
titled, legended, numerically-ticked line chart would teach the model that every question
is answerable, which is the single most likely way this corpus makes the model *worse*:
templates 1,2,3,8,9,10,11,12,13 all have an NA branch, and the model already fails them
(template 11 NA recall is 3.8-14.1%).

So inapplicability is generated deliberately and at the right frequency. It is trained as a
capability -- "this chart has no legend" -- never as CharXiv's literal answer token.
"""
from __future__ import annotations

import numpy as np

from .spec import AxisSpec, FigureSpec, PanelSpec, SeriesSpec, StyleSpec, TickSpec

# Measured gold-NA rates, CharXiv descriptive validation.
NA_RATES = {
    "no_title": 0.59,          # t1
    "no_xlabel": 0.117,        # t2
    "no_ylabel": 0.137,        # t3
    "x_spacing_na": 0.179,     # t8  (categorical or non-constant ticks)
    "y_spacing_na": 0.169,     # t9
    "not_line_plot": 0.446,    # t10, t11
    "no_legend": 0.277,        # t12, t13 (mean of 25.3 / 29.7)
}

# Train-pool style sheets. The eval corpus draws from EVAL_STYLES instead, so a gain
# cannot come from memorising one look.
TRAIN_STYLES = ("default", "seaborn-v0_8", "seaborn-v0_8-whitegrid", "ggplot", "bmh",
                "classic", "tableau-colorblind10")
EVAL_STYLES = ("fivethirtyeight", "Solarize_Light2", "grayscale", "seaborn-v0_8-dark",
               "seaborn-v0_8-colorblind")

QUANTITIES = [
    ("Throughput", "GB/s"), ("Latency", "ms"), ("Accuracy", "%"), ("Training loss", None),
    ("Energy", "kWh"), ("Population", "millions"), ("Revenue", "USD"), ("Temperature", "C"),
    ("Error rate", "%"), ("Memory usage", "MB"), ("Bandwidth", "Mbps"), ("Yield", "kg/ha"),
    ("Signal power", "dBm"), ("Concentration", "mol/L"), ("Response time", "s"),
]
X_QUANTITIES = [
    ("Epoch", None), ("Time", "s"), ("Iteration", None), ("Year", None),
    ("Sample size", None), ("Frequency", "Hz"), ("Distance", "km"), ("Depth", "m"),
]
TITLES = [
    "Model comparison", "Ablation study", "Scaling behaviour", "Convergence analysis",
    "Sensitivity to noise", "Throughput versus load", "Regional breakdown",
    "Effect of batch size", "Baseline versus proposed", "Long-horizon evaluation",
]
# Legend label reading is the one confirmed *perception* deficit that prompting does not fix:
# template 13's accuracy on answerable items is 62.34%, and a premise cue moves it -4.5 (it
# only trades accuracy for over-declaring inapplicability). Short tokens like "Baseline" do
# not exercise that skill, so the pool spans short names, technical strings with digits and
# punctuation, and the long hyphenated forms real figures actually carry.
SERIES_NAMES_SHORT = [
    "Baseline", "Proposed", "Control", "Reference", "Ours", "Prior work", "Random", "Oracle",
]
SERIES_NAMES_TECHNICAL = [
    "ResNet-50", "ViT-B/16", "k=8", "alpha=0.3", "T=0.7", "lr 1e-4", "top-p 0.9",
    "n_layers=12", "batch 256", "seed 42", "fp16", "8-bit",
]
SERIES_NAMES_LONG = [
    "Ablation: no pretraining", "Fine-tuned (in-domain)", "Zero-shot transfer",
    "Self-supervised baseline", "Human annotator agreement", "Ensemble of 5 runs",
    "With data augmentation", "Without regularisation", "Curriculum schedule",
    "Frozen backbone, linear probe",
]
SERIES_NAMES = SERIES_NAMES_SHORT + SERIES_NAMES_TECHNICAL + SERIES_NAMES_LONG
CATEGORIES = [
    ["Q1", "Q2", "Q3", "Q4"], ["North", "South", "East", "West"],
    ["Low", "Medium", "High"], ["Alpha", "Beta", "Gamma", "Delta"],
    ["Train", "Val", "Test"], ["2019", "2020", "2021", "2022", "2023"],
]
SHAPES = ("linear", "exponential growth", "exponential decay", "logarithmic",
          "S-shaped", "oscillating", "rises then falls", "falls then rises")

DIFFICULTY = {
    "easy":   dict(series=(1, 2), panels=(1, 1), ticks=(4, 6),  noise=(0.0, 0.02),
                   rot=(0,), labels=SERIES_NAMES_SHORT),
    "medium": dict(series=(2, 4), panels=(1, 4), ticks=(5, 9),  noise=(0.02, 0.08),
                   rot=(0, 30), labels=SERIES_NAMES_SHORT + SERIES_NAMES_TECHNICAL),
    "hard":   dict(series=(4, 7), panels=(2, 9), ticks=(8, 14), noise=(0.08, 0.20),
                   rot=(0, 45, 90), labels=SERIES_NAMES),
}


def _nice_ticks(lo: float, hi: float, n: int, rng) -> tuple[list[float], list[str]]:
    """Evenly spaced ticks with readable labels, constant spacing by construction."""
    step = (hi - lo) / (n - 1)
    vals = [lo + i * step for i in range(n)]
    mag = max(abs(lo), abs(hi))
    if mag >= 100 and all(abs(v - round(v)) < 1e-9 for v in vals):
        labels = [f"{v:.0f}" for v in vals]
    elif mag >= 10:
        labels = [f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}" for v in vals]
    else:
        labels = [f"{v:.2f}".rstrip("0").rstrip(".") or "0" for v in vals]
    return vals, labels


def _curve(shape: str, x: np.ndarray, rng, noise: float) -> tuple[np.ndarray, int]:
    t = (x - x.min()) / (np.ptp(x) or 1.0)
    if shape == "linear":
        y, peaks = t * rng.uniform(0.6, 1.0) + rng.uniform(0, 0.2), 0
    elif shape == "exponential growth":
        y, peaks = np.expm1(3 * t) / np.expm1(3.0), 0
    elif shape == "exponential decay":
        y, peaks = np.exp(-3 * t), 0
    elif shape == "logarithmic":
        y, peaks = np.log1p(9 * t) / np.log(10.0), 0
    elif shape == "S-shaped":
        y, peaks = 1 / (1 + np.exp(-10 * (t - 0.5))), 0
    elif shape == "oscillating":
        k = rng.integers(2, 5)
        y, peaks = 0.5 + 0.4 * np.sin(2 * np.pi * k * t), int(k)
    elif shape == "rises then falls":
        y, peaks = 1 - 4 * (t - 0.5) ** 2, 1
    else:  # falls then rises
        y, peaks = 4 * (t - 0.5) ** 2, 0
    y = np.asarray(y, dtype=float)
    if noise:
        y = y + rng.normal(0, noise, size=y.shape)
    return y, peaks


def sample_figure(figure_id: str, seed: int, difficulty: str, *, eval_split: bool = False) -> FigureSpec:
    rng = np.random.default_rng(seed)
    d = DIFFICULTY[difficulty]
    styles = EVAL_STYLES if eval_split else TRAIN_STYLES

    n_panels = int(rng.integers(d["panels"][0], d["panels"][1] + 1))
    cols = 1 if n_panels == 1 else int(rng.integers(1, min(4, n_panels) + 1))
    rows = int(np.ceil(n_panels / cols))

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
               "#8c564b", "#17becf", "#e377c2"]
    panels = []
    for p in range(n_panels):
        r, c = divmod(p, cols)
        categorical = rng.random() < NA_RATES["x_spacing_na"]
        not_line = rng.random() < NA_RATES["not_line_plot"]
        kind = "bar" if categorical else ("scatter" if not_line else "line")
        n_series = 1 if kind == "bar" else int(rng.integers(d["series"][0], d["series"][1] + 1))

        if categorical:
            cats = list(rng.permutation(CATEGORIES[int(rng.integers(len(CATEGORIES)))]))
            xs = np.arange(len(cats), dtype=float)
            x_ticks = TickSpec(tuple(xs), tuple(cats), float(rng.choice(d["rot"])))
            x_lim = (-0.8, len(cats) - 0.2)
            x_axis_label = None if rng.random() < NA_RATES["no_xlabel"] else "Category"
            numeric_x = False
        else:
            xq, xu = X_QUANTITIES[int(rng.integers(len(X_QUANTITIES)))]
            lo = float(rng.choice([0, 1, 10, 100, 2000]))
            hi = lo + float(rng.choice([5, 10, 20, 50, 100]))
            nt = int(rng.integers(d["ticks"][0], d["ticks"][1] + 1))
            tv, tl = _nice_ticks(lo, hi, nt, rng)
            x_ticks = TickSpec(tuple(tv), tuple(tl), float(rng.choice(d["rot"])))
            x_lim = (lo, hi)
            xs = np.linspace(lo, hi, int(rng.integers(20, 60)))
            x_axis_label = None if rng.random() < NA_RATES["no_xlabel"] else (
                f"{xq} ({xu})" if xu else xq)
            numeric_x = True

        yq, yu = QUANTITIES[int(rng.integers(len(QUANTITIES)))]
        noise = float(rng.uniform(*d["noise"]))
        series, all_y = [], []
        # A distinct label pool per figure, so one legend does not mix registers.
        label_pool = list(rng.permutation(d["labels"]))
        show_legend = rng.random() >= NA_RATES["no_legend"] and n_series >= 1
        for s in range(n_series):
            shape = SHAPES[int(rng.integers(len(SHAPES)))]
            if kind == "bar":
                y = rng.uniform(0.2, 1.0, size=len(xs))
                peaks = 0
            else:
                y, peaks = _curve(shape, xs, rng, noise)
                y = y + s * rng.uniform(0.0, 0.25)
            all_y.append(y)
            series.append(SeriesSpec(
                label=label_pool[s % len(label_pool)] if show_legend else None,
                kind=kind, x=tuple(map(float, xs)), y=tuple(map(float, y)),
                color=palette[s % len(palette)],
                linestyle=["-", "--", "-.", ":"][s % 4] if kind == "line" else "-",
                marker=None if kind == "line" else "o",
                shape=shape if kind == "line" else "n/a", n_peaks=peaks,
            ))
        ymin = float(min(a.min() for a in all_y)); ymax = float(max(a.max() for a in all_y))
        pad = 0.08 * (ymax - ymin or 1.0)
        ylo, yhi = ymin - pad, ymax + pad
        nt = int(rng.integers(d["ticks"][0], d["ticks"][1] + 1))
        yv, yl = _nice_ticks(ylo, yhi, nt, rng)
        # y-spacing NA: blank out an interior tick label so spacing is no longer constant
        if rng.random() < NA_RATES["y_spacing_na"] and len(yl) > 3:
            yl = list(yl); yl[len(yl) // 2] = ""; yl = tuple(yl)
        y_axis_label = None if rng.random() < NA_RATES["no_ylabel"] else (
            f"{yq} ({yu})" if yu else yq)

        panels.append(PanelSpec(
            grid_pos=(r + 1, c + 1),
            title=None if rng.random() < NA_RATES["no_title"] else
                  TITLES[int(rng.integers(len(TITLES)))],
            x=AxisSpec("x", x_axis_label, x_lim, x_ticks, numeric_ticks=numeric_x),
            y=AxisSpec("y", y_axis_label, (ylo, yhi), TickSpec(tuple(yv), tuple(yl))),
            series=tuple(series), has_legend=show_legend,
            legend_loc=str(rng.choice(["upper right", "upper left", "lower right", "lower left"])),
            grid=bool(rng.random() < 0.4),
        ))

    return FigureSpec(
        figure_id=figure_id, seed=seed, grid_shape=(rows, cols), panels=tuple(panels),
        style=StyleSpec(
            style_sheet=str(rng.choice(styles)),
            font_size=float(rng.uniform(8, 13)),
            dpi=int(rng.choice([100, 120, 150])),
            figsize=(float(3.6 * cols + rng.uniform(0, 1.5)),
                     float(2.9 * rows + rng.uniform(0, 1.2))),
            seed=seed,
        ),
        suptitle=(TITLES[int(rng.integers(len(TITLES)))]
                  if n_panels > 1 and rng.random() > NA_RATES["no_title"] else None),
        difficulty=difficulty,
        mpl_version=__import__("matplotlib").__version__,
    )
