"""CEILING PROBE ONLY -- data that deliberately mirrors CharXiv's own question templates.

**This module must never be used to build a shipping corpus, and it is not a recipe.**

Everything else in ChartGym is written *away* from the benchmark: a test asserts no
generated question shares a 6-gram or reaches token-Jaccard 0.6 with any CharXiv template,
and one visual primitive (panel layout) is withheld entirely so CharXiv templates 18/19 act
as a transfer readout. This module does the opposite on purpose. It exists to answer one
question that the honest corpus structurally cannot:

    How much of CharXiv's remaining headroom is reachable by training at all?

The probe trains on the benchmark's exact question strings (copied verbatim from
``olmo_eval.common.image_qa.charxiv.DESCRIPTIVE_RESP_INST`` into
``metadata/charxiv_templates.json``), the benchmark's subplot-locator prefix, and the
benchmark's answer conventions including the literal "Not Applicable" token. A score
produced this way:

* is **not comparable** to any published CharXiv number, for this model or any other;
* is **not evidence of capability** -- it is an upper bound on score, not on skill;
* **destroys the transfer instrument**: templates 18/19 are trained here, so the
  panel-layout holdout that makes a v3-style result interpretable does not exist in a
  probe corpus.

Read the result only as a ceiling: the gap between it and the honest recipe is the
benchmark-fitting premium, and the gap between it and 100 is what no amount of training
reaches.

Templates 14 and 15 (colorbar) are **excluded**: ChartGym renders no colorbars, so every
probe instance would be an "absent" case. Training a check that only ever concludes absence
is the exact mechanism that broke v4 (CharXiv t12 lost 9 points to false "no legend"
declarations), and both colorbar templates are already at 90.8/94.3 -- roughly 0.6 points of
headroom between them. The ceiling this probe measures is therefore a slight *under*estimate,
by at most that much.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CAP = "charxiv_probe"
NA_TOKEN = "Not Applicable"

_TEMPLATES = json.loads(
    (Path(__file__).resolve().parents[1] / "metadata" / "charxiv_templates.json").read_text()
)

#: Templates this probe covers. 14/15 excluded -- see the module docstring.
PROBE_TEMPLATES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19)


def _prefix(spec, panel) -> str:
    """Mirror ``descriptive_query_helper``'s locator exactly."""
    if spec.n_panels == 1:
        return "For the current plot, "
    r, c = panel.grid_pos
    return f"For the subplot at row {r} and column {c}, "


def _question(qid: int, spec, panel) -> str:
    text = _TEMPLATES[str(qid)]
    if qid in (18, 19):
        return text  # the benchmark omits the locator for layout questions
    return text.format(_prefix(spec, panel))


def _labeled(axis):
    return axis.ticks.labeled_pairs()


def _trend_phrase(series) -> str | None:
    """CharXiv t16 wants a few words. Only emit where the spec makes one unambiguous."""
    return {
        "linear": "increases",
        "exponential growth": "increases",
        "exponential decay": "decreases",
        "logarithmic": "increases then stabilizes",
        "S-shaped": "increases then stabilizes",
        "rises then falls": "increases then decreases",
        "falls then rises": "decreases then increases",
    }.get(series.shape)


def emit_probe(spec, audit, rng) -> list[dict]:
    """One record per (template, panel) that the spec can answer exactly."""
    out: list[dict] = []

    def add(qid, panel, answer, atype="verbatim"):
        out.append(dict(
            family=f"charxiv.t{qid}", capability=CAP, question=_question(qid, spec, panel),
            answer=str(answer), answer_type=atype, tol=None, held_out=False,
            is_na=(str(answer) == NA_TOKEN), target=str(answer),
        ))

    for panel in spec.panels:
        # --- text read-offs -------------------------------------------------
        add(1, panel, panel.title or NA_TOKEN)
        add(2, panel, panel.x.label or NA_TOKEN)
        add(3, panel, panel.y.label or NA_TOKEN)

        xs, ys = _labeled(panel.x), _labeled(panel.y)
        if xs:
            add(4, panel, min(xs, key=lambda p: p[0])[1])
            add(5, panel, max(xs, key=lambda p: p[0])[1])
        if ys:
            add(6, panel, min(ys, key=lambda p: p[0])[1])
            add(7, panel, max(ys, key=lambda p: p[0])[1])

        # --- tick spacing (NA when non-numeric or non-constant) -------------
        for qid, axis in ((8, panel.x), (9, panel.y)):
            step = axis.tick_spacing
            add(qid, panel, NA_TOKEN if step is None else f"{step:g}",
                "na" if step is None else "float")

        # --- line-plot questions -------------------------------------------
        lines = [s for s in panel.series if s.kind == "line"]
        if not lines:
            add(10, panel, NA_TOKEN)
            add(11, panel, NA_TOKEN)
        else:
            add(10, panel, len(lines), "int")
            if len(lines) < 2:
                add(11, panel, "No", "yesno")
            else:
                span = panel.y.lim[1] - panel.y.lim[0]
                crossed = ambiguous = False
                for i in range(len(lines)):
                    for j in range(i + 1, len(lines)):
                        a, b = lines[i], lines[j]
                        grid = np.linspace(max(min(a.x), min(b.x)), min(max(a.x), max(b.x)), 400)
                        diff = np.interp(grid, a.x, a.y) - np.interp(grid, b.x, b.y)
                        if np.any(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0):
                            crossed = True
                        elif np.min(np.abs(diff)) < 0.05 * span:
                            ambiguous = True
                if crossed or not ambiguous:
                    add(11, panel, "Yes" if crossed else "No", "yesno")

        # --- legend ---------------------------------------------------------
        if panel.has_legend and panel.legend_entries:
            add(12, panel, len(panel.legend_entries), "int")
            add(13, panel, ", ".join(panel.legend_entries), "list")
        else:
            add(12, panel, NA_TOKEN)
            add(13, panel, NA_TOKEN)

        # --- trend (single unambiguous series only) --------------------------
        if len(lines) == 1:
            phrase = _trend_phrase(lines[0])
            if phrase:
                add(16, panel, phrase, "phrase")

        # --- total labelled ticks, scoped to the referenced panel ------------
        # Panel scope, not figure scope: v1 trained figure-scope totals and flipped the
        # checkpoint from under-counting (median -6) to over-counting (median +13),
        # costing 28.6 points on this template.
        add(17, panel, panel.x.ticks.n_labeled + panel.y.ticks.n_labeled, "int")

    # --- figure-level layout questions (no locator) --------------------------
    rows, cols = spec.grid_shape
    add(18, spec.panels[0], f"{rows} by {cols}", "phrase")
    add(19, spec.panels[0], spec.n_panels, "int")
    return out
