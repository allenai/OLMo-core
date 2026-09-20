"""Question families, grouped by the capability they train.

Two rules define this file, and both are enforced by tests rather than by care:

**The held-out primitive is panel layout.** No family may ask about the subplot grid --
not rows, not columns, not panel count, not "which panel is where". Multi-panel figures are
still rendered and still questioned; panels are referenced *by title* ("in the panel titled
'Ablation study'"), never by grid position. The readout for whether capability transfers
without training the benchmark's own question is CharXiv templates 18 + 19 (pooled 83.97%,
the only isolated candidate that had room to move; colorbar was disqualified at 91.43%).

**Phrasing is never borrowed from CharXiv.** A test asserts token-Jaccard < 0.6 and no
shared 6-gram against all 19 descriptive templates. Where a family happens to read the same
visual statistic, it asks for it in its own words and often in a different answer format.

Inapplicability is a first-class answer, phrased naturally ("this chart has no legend"),
never as CharXiv's literal token -- CharXiv's own prompt supplies that format at eval time,
so what has to transfer is the *detection*, not the wording.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

CAP_COUNT = "counting"
CAP_GEOM = "geometry"
CAP_TREND = "trend"
CAP_OCR = "ocr"
CAP_APPLIC = "applicability"

HELD_OUT_PRIMITIVE = "panel_layout"

_REG: dict[str, dict] = {}


def family(fid: str, capability: str, answer_type: str, held_out: bool = False):
    def deco(fn: Callable):
        _REG[fid] = dict(fn=fn, capability=capability, answer_type=answer_type,
                         held_out=held_out)
        return fn
    return deco


def _qa(fid, question, answer, *, tol=None, na=False, target=None):
    """One QA record. `answer` is the bare gold the eval scores; `target` is what training
    supervises -- for procedure-supervised families it is the enumeration/search trace
    ending in the answer, everywhere else it defaults to the bare answer.

    Why the split exists: prompting the model to enumerate lifted CharXiv t17 by +20
    (32.14 -> 52.23, chi2=21.75) but a global cue cost -2.0 benchmark-wide because it
    cannot be enumeration-for-aggregates and terse-for-read-offs at once. Per-question
    supervision can: trace targets go ONLY on aggregation/verification families, read-off
    families stay terse, so the discrimination itself is what gets trained.
    """
    m = _REG[fid]
    return dict(family=fid, capability=m["capability"], question=question,
                answer=str(answer), answer_type="na" if na else m["answer_type"],
                tol=tol, held_out=m["held_out"], is_na=na,
                target=str(target) if target is not None else str(answer))


def _panel_ref(spec, panel):
    """Reference a panel WITHOUT using its grid position (the held-out primitive)."""
    if spec.n_panels == 1:
        return ""
    if panel.title:
        return f"In the panel titled \"{panel.title}\", "
    return None  # untitled panel in a multi-panel figure: not referable without layout


def _tick_list(axis):
    return ", ".join(lbl for _, lbl in axis.ticks.labeled_pairs())


def _axis_enum_trace(panel):
    """Enumeration trace over one panel's two axes, ending in the total."""
    nx, ny = panel.x.ticks.n_labeled, panel.y.ticks.n_labeled
    return (f"Horizontal axis ticks: {_tick_list(panel.x)} ({nx}). "
            f"Vertical axis ticks: {_tick_list(panel.y)} ({ny}). "
            f"Total: {nx + ny}.")


# --------------------------------------------------------------------- counting

@family("cnt.ticks_total", CAP_COUNT, "int")
def _ticks_total(spec, audit, rng):
    """Total labelled ticks -- PANEL-scoped on multi-panel figures.

    The first release of this family asked about the whole figure. Trained at 25% mixture
    share, that flipped the checkpoint's CharXiv template-17 error from under-counting
    (median -6) to over-counting (median +13, 207 of 224 wrong answers high) and cost
    -28.57 on the template: CharXiv's question carries a subplot locator and is scoped to
    the referenced subplot, so the model had learned the right skill at the wrong scope.
    Scope is part of the answer. Figure-scope phrasing is now emitted only where the two
    scopes coincide -- single-panel figures.
    """
    out = []
    if spec.n_panels == 1:
        q = rng.choice([
            "Add up every tick mark that carries a written label on all axes of this figure. How many are there?",
            "Counting all axes together, how many tick marks have text written next to them?",
            "How many of the tick marks in this figure are annotated with a value or name? Count across all axes.",
        ])
        panel = spec.panels[0]
        out.append(_qa("cnt.ticks_total", q, spec.total_labeled_ticks(),
                       target=_axis_enum_trace(panel)))
        return out
    for panel in spec.panels:
        ref = _panel_ref(spec, panel)
        if ref is None:
            continue
        n = panel.x.ticks.n_labeled + panel.y.ticks.n_labeled
        q = rng.choice([
            f"{ref}add up the tick marks carrying a written label on both of its axes. How many are there?",
            f"{ref}counting its two axes together, how many tick marks have text written next to them?",
        ])
        out.append(_qa("cnt.ticks_total", q, n, target=_axis_enum_trace(panel)))
    return out


@family("cnt.ticks_axis", CAP_COUNT, "int")
def _ticks_axis(spec, audit, rng):
    out = []
    for panel in spec.panels:
        ref = _panel_ref(spec, panel)
        if ref is None:
            continue
        for role, axis, word in (("x", panel.x, "horizontal"), ("y", panel.y, "vertical")):
            q = rng.choice([
                f"{ref}how many labelled tick marks sit along the {word} axis?",
                f"{ref}count the tick marks with written values on the {word} axis.",
            ])
            n_ax = axis.ticks.n_labeled
            trace = f"The labelled ticks are {_tick_list(axis)} - {n_ax} in total."
            out.append(_qa("cnt.ticks_axis", q[0].upper() + q[1:] if not ref else q,
                           n_ax, target=trace))
    return out


@family("cnt.series", CAP_COUNT, "int")
def _series(spec, audit, rng):
    out = []
    for panel in spec.panels:
        ref = _panel_ref(spec, panel)
        if ref is None:
            continue
        # guard: series must be visually distinguishable
        if len({(s.color, s.linestyle) for s in panel.series}) != len(panel.series):
            continue
        q = rng.choice([
            f"{ref}how many separate data series are plotted?",
            f"{ref}count the distinct traces drawn on the axes.",
        ])
        out.append(_qa("cnt.series", q[0].upper() + q[1:] if not ref else q, len(panel.series)))
    return out


@family("cnt.legend_entries", CAP_COUNT, "int")
def _legend_entries(spec, audit, rng):
    out = []
    for panel in spec.panels:
        ref = _panel_ref(spec, panel)
        if ref is None:
            continue
        q = rng.choice([
            f"{ref}how many items are listed in the key?",
            f"{ref}count the entries shown in the legend box.",
        ])
        q = q[0].upper() + q[1:] if not ref else q
        if not panel.has_legend:
            # Halved, matching ocr.legend_names: v3's asymmetry (names halved, count at
            # full rate) tilted the count family's prior toward absence -- t12 fell -9.34
            # with 23/42 errors being false "no legend" declarations.
            if rng.random() < 0.5:
                out.append(_qa("cnt.legend_entries", q,
                               "There is no legend on this chart.", na=True,
                               target="Checking for a key inside and beside the axes: none "
                                      "is drawn. There is no legend on this chart."))
        else:
            names = ", ".join(panel.legend_entries)
            n_e = len(panel.legend_entries)
            out.append(_qa("cnt.legend_entries", q, n_e,
                           target=f"Checking for a key inside and beside the axes: one is "
                                  f"drawn, listing {names} - {n_e} entries."))
    return out


@family("cnt.peaks", CAP_COUNT, "int")
def _peaks(spec, audit, rng):
    out = []
    for panel in spec.panels:
        ref = _panel_ref(spec, panel)
        if ref is None:
            continue
        for s in panel.series:
            # only emit where the peak count is unambiguous by construction and the
            # curve is smooth enough that a human would agree
            if s.kind != "line" or s.shape not in ("oscillating", "rises then falls") :
                continue
            if s.label is None:
                continue
            q = (f"{ref}how many distinct high points does the \"{s.label}\" trace reach "
                 f"before it ends?")
            out.append(_qa("cnt.peaks", q[0].upper() + q[1:] if not ref else q, s.n_peaks))
    return out


# --------------------------------------------------------------------- dense OCR

@family("ocr.title", CAP_OCR, "verbatim")
def _title(spec, audit, rng):
    out = []
    for panel in spec.panels:
        ref = _panel_ref(spec, panel)
        if ref is None:
            continue
        q = rng.choice([
            f"{ref}write out the heading printed above the axes, exactly as it appears.",
            f"{ref}what wording is used for the heading of this chart?",
        ])
        q = q[0].upper() + q[1:] if not ref else q
        if panel.title:
            out.append(_qa("ocr.title", q, panel.title))
        else:
            out.append(_qa("ocr.title", q, "This chart has no heading written on it.",
                           na=True,
                           target="Checking above the axes for a heading: nothing is "
                                  "printed there. This chart has no heading written on it."))
    return out


@family("ocr.axis_label", CAP_OCR, "verbatim")
def _axis_label(spec, audit, rng):
    out = []
    for panel in spec.panels:
        ref = _panel_ref(spec, panel)
        if ref is None:
            continue
        for axis, word in ((panel.x, "horizontal"), (panel.y, "vertical")):
            q = rng.choice([
                f"{ref}transcribe the text naming the {word} axis, including any unit.",
                f"{ref}what wording labels the {word} axis?",
            ])
            q = q[0].upper() + q[1:] if not ref else q
            if axis.label:
                out.append(_qa("ocr.axis_label", q, axis.label))
            else:
                out.append(_qa("ocr.axis_label", q,
                               f"The {word} axis carries no written name.", na=True,
                               target=f"Checking along the {word} axis for a name: only "
                                      f"tick values are printed. The {word} axis carries "
                                      "no written name."))
    return out


@family("ocr.legend_names", CAP_OCR, "list")
def _legend_names(spec, audit, rng):
    out = []
    for panel in spec.panels:
        ref = _panel_ref(spec, panel)
        if ref is None:
            continue
        q = rng.choice([
            f"{ref}list every name that appears in the key, separated by commas.",
            f"{ref}write down the series names shown in the legend, separated by commas.",
        ])
        q = q[0].upper() + q[1:] if not ref else q
        if panel.has_legend and panel.legend_entries:
            names = ", ".join(panel.legend_entries)
            out.append(_qa("ocr.legend_names", q, names,
                           target=f"Checking for a key inside and beside the axes: one is "
                                  f"drawn. It lists: {names}"))
        elif rng.random() < 0.5:
            # NA recall on legend names was already 85-94% before any training; these
            # examples buy little and push the prior toward over-declaring absence
            # (t12 precision fell to 74.5%). Emit them at half rate.
            out.append(_qa("ocr.legend_names", q, "This chart has no key.", na=True,
                           target="Checking inside and beside the axes for a key: none is "
                                  "drawn. This chart has no key."))
    return out


@family("ocr.tick_extreme", CAP_OCR, "verbatim")
def _tick_extreme(spec, audit, rng):
    out = []
    for panel in spec.panels:
        ref = _panel_ref(spec, panel)
        if ref is None:
            continue
        for axis, word, end in ((panel.x, "horizontal", "furthest to the right"),
                                (panel.y, "vertical", "nearest the top")):
            pairs = axis.ticks.labeled_pairs()
            if len(pairs) < 2:
                continue
            label = max(pairs, key=lambda p: p[0])[1]
            q = (f"{ref}reading the {word} axis, what text is printed at the labelled tick "
                 f"{end}?")
            out.append(_qa("ocr.tick_extreme", q[0].upper() + q[1:] if not ref else q, label))
    return out


@family("ocr.tick_step", CAP_OCR, "float")
def _tick_step(spec, audit, rng):
    out = []
    for panel in spec.panels:
        ref = _panel_ref(spec, panel)
        if ref is None:
            continue
        for axis, word in ((panel.x, "horizontal"), (panel.y, "vertical")):
            q = (f"{ref}by how much does the value increase from one labelled tick to the "
                 f"next on the {word} axis?")
            q = q[0].upper() + q[1:] if not ref else q
            step = axis.tick_spacing
            if step is None:
                out.append(_qa("ocr.tick_step", q,
                               f"The {word} axis ticks are not evenly spaced numbers, "
                               "so no single step applies.", na=True))
            else:
                out.append(_qa("ocr.tick_step", q, f"{step:g}", tol=abs(step) * 0.02))
    return out


# --------------------------------------------------------------------- geometry

@family("geo.higher_at", CAP_GEOM, "cat")
def _higher_at(spec, audit, rng):
    out = []
    for panel in spec.panels:
        ref = _panel_ref(spec, panel)
        if ref is None or len(panel.series) < 2:
            continue
        named = [s for s in panel.series if s.label]
        if len(named) < 2:
            continue
        a, b = named[0], named[1]
        pairs = panel.x.ticks.labeled_pairs()
        if not pairs or not panel.x.numeric_ticks:
            continue
        xv, xl = pairs[len(pairs) // 2]
        ya = float(np.interp(xv, a.x, a.y)); yb = float(np.interp(xv, b.x, b.y))
        span = panel.y.lim[1] - panel.y.lim[0]
        if abs(ya - yb) < 0.08 * span:       # dead band: too close to call by eye
            continue
        q = (f"{ref}at the point where the horizontal axis reads {xl}, which of "
             f"\"{a.label}\" or \"{b.label}\" sits higher?")
        out.append(_qa("geo.higher_at", q[0].upper() + q[1:] if not ref else q,
                       a.label if ya > yb else b.label))
    return out


@family("geo.crosses", CAP_GEOM, "yesno")
def _crosses(spec, audit, rng):
    out = []
    for panel in spec.panels:
        ref = _panel_ref(spec, panel)
        if ref is None:
            continue
        q = rng.choice([
            f"{ref}does any plotted curve pass across another one anywhere in view?",
            f"{ref}do two of the drawn traces meet or cross at some point?",
        ])
        q = q[0].upper() + q[1:] if not ref else q
        if not panel.is_line_plot:
            out.append(_qa("geo.crosses", q,
                           "This chart does not draw any curves, so nothing can cross.",
                           na=True))
            continue
        named = [s for s in panel.series if s.kind == "line"]
        if len(named) < 2:
            out.append(_qa("geo.crosses", q,
                           "Only one curve is drawn, so there is nothing for it to cross.",
                           na=True))
            continue
        span = panel.y.lim[1] - panel.y.lim[0]
        crossed, ambiguous = False, False
        for i in range(len(named)):
            for j in range(i + 1, len(named)):
                grid = np.linspace(max(min(named[i].x), min(named[j].x)),
                                   min(max(named[i].x), max(named[j].x)), 400)
                diff = (np.interp(grid, named[i].x, named[i].y)
                        - np.interp(grid, named[j].x, named[j].y))
                if np.any(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0):
                    crossed = True
                elif np.min(np.abs(diff)) < 0.05 * span:
                    ambiguous = True   # tangent / near-touch: a human would disagree
        if ambiguous and not crossed:
            continue
        out.append(_qa("geo.crosses", q, "Yes" if crossed else "No"))
    return out


# --------------------------------------------------------------------- trend

@family("trd.shape", CAP_TREND, "cat")
def _shape(spec, audit, rng):
    out = []
    for panel in spec.panels:
        ref = _panel_ref(spec, panel)
        if ref is None:
            continue
        for s in panel.series:
            if s.kind != "line" or s.label is None:
                continue
            q = (f"{ref}describe in a few words how the \"{s.label}\" curve behaves from "
                 f"left to right.")
            out.append(_qa("trd.shape", q[0].upper() + q[1:] if not ref else q, s.shape))
    return out


@family("trd.direction", CAP_TREND, "cat")
def _direction(spec, audit, rng):
    out = []
    for panel in spec.panels:
        ref = _panel_ref(spec, panel)
        if ref is None:
            continue
        for s in panel.series:
            if s.kind != "line" or s.label is None:
                continue
            y = np.asarray(s.y)
            net = y[-1] - y[0]
            span = float(np.ptp(y)) or 1.0
            if abs(net) < 0.25 * span:
                continue          # dead band: no defensible overall direction
            q = (f"{ref}over the whole range shown, does \"{s.label}\" end up higher or "
                 f"lower than where it started?")
            out.append(_qa("trd.direction", q[0].upper() + q[1:] if not ref else q,
                           "higher" if net > 0 else "lower"))
    return out


# ------------------------------------------------- HELD-OUT PRIMITIVE: panel layout
# These are generated for the EVAL ONLY and never for training. They are the instrument
# that answers the overfitting question: if CharXiv templates 18/19 improve while nothing
# about grid geometry was ever trained, the gain is cross-capability transfer rather than
# the benchmark's own question being fitted. Keeping them in one block, all flagged
# held_out=True, is what makes `include_held_out=False` a complete guarantee rather than a
# hope -- `test_holdout` asserts the emitted training set contains none of them.

@family("pnl.grid_shape", CAP_COUNT, "phrase", held_out=True)
def _grid_shape(spec, audit, rng):
    rows, cols = spec.grid_shape
    q = rng.choice([
        "How are the panels of this figure arranged? Answer as rows by columns.",
        "Describe the arrangement of the plotting panels as a grid of rows and columns.",
    ])
    return [_qa("pnl.grid_shape", q, f"{rows} by {cols}")]


@family("pnl.count", CAP_COUNT, "int", held_out=True)
def _panel_count(spec, audit, rng):
    q = rng.choice([
        "How many separate plotting panels make up this figure?",
        "Count the individual plots drawn in this figure.",
    ])
    return [_qa("pnl.count", q, spec.n_panels)]


@family("pnl.position", CAP_GEOM, "phrase", held_out=True)
def _panel_position(spec, audit, rng):
    titled = [p for p in spec.panels if p.title]
    if spec.n_panels < 2 or not titled:
        return []
    panel = titled[int(rng.integers(len(titled)))]
    r, c = panel.grid_pos
    q = (f"Counting rows from the top and columns from the left, in which row and column "
         f"does the panel titled \"{panel.title}\" sit?")
    return [_qa("pnl.position", q, f"row {r}, column {c}")]


def emit_all(spec, audit, rng, *, include_held_out: bool) -> list[dict]:
    out = []
    for fid, meta in _REG.items():
        if meta["held_out"] and not include_held_out:
            continue
        try:
            out.extend(meta["fn"](spec, audit, rng))
        except Exception:  # a family that cannot handle this figure simply yields nothing
            continue
    return out


REGISTRY = _REG
