"""CPU-only tests for the ChartGym generator.

The two tests that matter most are `test_no_charxiv_phrasing` and `test_holdout_is_total`.
They are what make the design claims mechanical rather than a matter of care: the first
stops anyone from quietly borrowing CharXiv's wording, the second stops the held-out
primitive from leaking into training. Everything else guards correctness of the ground
truth, which is the one thing a synthetic corpus cannot be wrong about.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from chartgym.families import HELD_OUT_PRIMITIVE, REGISTRY, emit_all  # noqa: E402
from chartgym.render import render  # noqa: E402
from chartgym.sample import EVAL_STYLES, NA_RATES, TRAIN_STYLES, sample_figure  # noqa: E402
from chartgym.spec import AxisSpec, TickSpec  # noqa: E402

SEEDS = list(range(500, 540))


def _figures(n=24, eval_split=False):
    out = []
    for i in SEEDS[:n]:
        spec = sample_figure(f"t-{i}", i, ["easy", "medium", "hard"][i % 3],
                             eval_split=eval_split)
        png, audit = render(spec)
        if audit.validate(spec):
            out.append((spec, audit))
    return out


# ---------------------------------------------------------------- ground truth

def test_tick_total_matches_what_was_drawn():
    """The spec's tick count must equal the number of labels matplotlib actually drew.

    This is the corpus's single most important number (CharXiv template 17 is the model's
    worst skill at 29.91%), so a silent spec/render disagreement here would teach the wrong
    answer at scale.
    """
    for spec, audit in _figures():
        drawn = sum(audit.drawn_labeled_ticks(i, role)
                    for i in range(len(spec.panels)) for role in ("x", "y"))
        assert drawn == spec.total_labeled_ticks(), spec.figure_id


def test_tick_spacing_is_none_when_not_applicable():
    """`tick_spacing` must return None -- the NA gold -- for each degenerate case."""
    mk = lambda vals, labels, numeric=True: AxisSpec(  # noqa: E731
        "y", "L", (0, 10), TickSpec(tuple(vals), tuple(labels)), numeric_ticks=numeric)
    assert mk([0, 1, 2], ["0", "1", "2"]).tick_spacing == pytest.approx(1.0)
    assert mk([0, 1, 4], ["0", "1", "4"]).tick_spacing is None        # not constant
    assert mk([0, 1, 2], ["0", "", "2"]).tick_spacing == pytest.approx(2.0)  # blanked
    assert mk([0, 1], ["A", "B"], numeric=False).tick_spacing is None  # categorical
    assert mk([0], ["0"]).tick_spacing is None                         # single tick


def test_answers_are_never_empty():
    for spec, audit in _figures():
        for q in emit_all(spec, audit, np.random.default_rng(0), include_held_out=True):
            assert q["answer"].strip(), f"{q['family']} produced an empty answer"
            assert q["question"].strip().endswith(("?", ".")), q["question"]


# ---------------------------------------------------------------- the two design claims

# The 19 CharXiv descriptive templates, quoted here ONLY so the test can prove our phrasing
# does not resemble them.
CHARXIV_TEMPLATES = [
    "what is its title?",
    "what is the label of the x-axis?",
    "what is the label of the y-axis?",
    "what is the leftmost labeled tick on the x-axis?",
    "what is the rightmost labeled tick on the x-axis?",
    "what is the spatially lowest labeled tick on the y-axis?",
    "what is the spatially highest labeled tick on the y-axis?",
    "what is difference between consecutive numerical tick values on the x-axis?",
    "what is difference between consecutive numerical tick values on the y-axis?",
    "how many lines are there?",
    "do any lines intersect?",
    "how many discrete labels are there in the legend?",
    "what are the names of the labels in the legend?",
    "what is the difference between the maximum and minimum values of the tick labels on the continuous legend?",
    "what is the maximum value of the tick labels on the continuous legend?",
    "what is the general trend of data from left to right?",
    "what is the total number of explicitly labeled ticks across all axes?",
    "what is the layout of the subplots?",
    "what is the number of subplots?",
]
_STOP = {"the", "of", "is", "a", "an", "on", "in", "to", "are", "there", "what", "how", "this"}


def _tokens(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP}


def _ngrams(text, n=6):
    w = re.findall(r"[a-z]+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def test_no_charxiv_phrasing():
    """Our questions must not be paraphrases of CharXiv's own templates.

    Training on the benchmark's question shapes is precisely the benchmark fitting this
    corpus exists to avoid, and "we were careful" is not a guarantee. Jaccard < 0.6 and no
    shared 6-gram makes it one.
    """
    checked = 0
    for spec, audit in _figures():
        for q in emit_all(spec, audit, np.random.default_rng(1), include_held_out=True):
            qt, qg = _tokens(q["question"]), _ngrams(q["question"])
            for tmpl in CHARXIV_TEMPLATES:
                tt = _tokens(tmpl)
                j = len(qt & tt) / len(qt | tt) if (qt | tt) else 0.0
                assert j < 0.6, f"{q['family']} too close to CharXiv:\n  {q['question']}\n  {tmpl} (J={j:.2f})"
                assert not (qg & _ngrams(tmpl)), f"{q['family']} shares a 6-gram with:\n  {tmpl}"
            checked += 1
    assert checked > 100


def test_holdout_is_total():
    """Training emission must contain nothing about the held-out primitive.

    Panel layout is held out so that CharXiv templates 18/19 can act as a transfer readout.
    A leak would not just weaken the claim -- it would invalidate it, silently.
    """
    assert HELD_OUT_PRIMITIVE == "panel_layout"
    held = {fid for fid, m in REGISTRY.items() if m["held_out"]}
    assert held, "no held-out families registered"
    banned = re.compile(r"\b(row|rows|column|columns|grid|subplot|subplots|panels)\b", re.I)
    for spec, audit in _figures():
        train_qs = emit_all(spec, audit, np.random.default_rng(2), include_held_out=False)
        assert not ({q["family"] for q in train_qs} & held)
        for q in train_qs:
            assert not banned.search(q["question"]), \
                f"{q['family']} mentions the held-out primitive: {q['question']}"
        # ...and the pixels must still be there: the holdout removes the question, not the
        # figure. A regression that stopped rendering multi-panel figures would make the
        # transfer measurement meaningless while every test still passed.
    assert any(s.n_panels > 1 for s, _ in _figures()), "no multi-panel figures generated"


# ---------------------------------------------------------------- corpus properties

def test_na_cases_are_generated_at_the_measured_rate():
    """Inapplicable questions must actually occur.

    25.0% of CharXiv descriptive gold answers are "Not Applicable" and the model is worst
    exactly there (template 11 NA recall 3.8-14.1%). A corpus where every question is
    answerable would bias the model away from ever saying so -- the most likely way this
    program makes the model worse rather than better.
    """
    na = total = 0
    for spec, audit in _figures():
        for q in emit_all(spec, audit, np.random.default_rng(3), include_held_out=False):
            total += 1
            na += q["is_na"]
    assert total > 100
    assert 0.05 <= na / total <= 0.35, f"NA share {na / total:.3f} outside the intended band"


def test_train_and_eval_styles_are_disjoint():
    """A gain must not be attributable to memorising one look."""
    assert not set(TRAIN_STYLES) & set(EVAL_STYLES)
    train = {s.style.style_sheet for s, _ in _figures(eval_split=False)}
    ev = {s.style.style_sheet for s, _ in _figures(eval_split=True)}
    assert train <= set(TRAIN_STYLES) and ev <= set(EVAL_STYLES)
    assert not train & ev


def test_no_special_token_literals():
    """The v11 run died at step ~171 on exactly this class of leak."""
    bad = ("<im_start>", "<im_end>", "<im_patch>", "<|im_start|>", "<|im_end|>", "<image>",
           "<|endoftext|>")
    for spec, audit in _figures():
        for q in emit_all(spec, audit, np.random.default_rng(4), include_held_out=True):
            blob = q["question"] + q["answer"]
            for token in bad:
                assert token not in blob, f"{q['family']} leaked {token}"


def test_no_unicode_minus_in_gold():
    """A gold of U+2212 could never be matched by a model emitting ASCII '-'."""
    for spec, audit in _figures():
        for q in emit_all(spec, audit, np.random.default_rng(5), include_held_out=True):
            assert "−" not in q["answer"], q["family"]


def test_determinism():
    a = sample_figure("x", 77, "medium")
    b = sample_figure("x", 77, "medium")
    assert a == b
    assert render(a)[0] == render(b)[0]


def test_na_rates_are_the_measured_ones():
    """Guard against someone 'tidying' these into round numbers.

    They are measurements from CharXiv validation (see outputs/chartgym/stage0/RESULTS.md),
    not tuning knobs.
    """
    assert NA_RATES["not_line_plot"] == pytest.approx(0.446)
    assert NA_RATES["no_title"] == pytest.approx(0.59)
    assert NA_RATES["x_spacing_na"] == pytest.approx(0.179)


def test_tick_total_is_panel_scoped_on_multipanel_figures():
    """Scope is part of the answer.

    The figure-scoped version of this family flipped the trained checkpoint's CharXiv
    template-17 error from median -6 (under-count) to median +13 (over-count) and cost
    -28.57 on the template, because CharXiv's question is scoped to a referenced subplot.
    Figure-scope phrasing may only appear where it is unambiguous: single-panel figures.
    """
    seen_multi = seen_single = 0
    for spec, audit in _figures(n=30):
        qs = [q for q in emit_all(spec, audit, np.random.default_rng(7), include_held_out=False)
              if q["family"] == "cnt.ticks_total"]
        for q in qs:
            if spec.n_panels == 1:
                seen_single += 1
                assert int(q["answer"]) == spec.total_labeled_ticks()
            else:
                seen_multi += 1
                assert "figure" not in q["question"].lower(), q["question"]
                assert "panel titled" in q["question"], q["question"]
                per_panel = {p.x.ticks.n_labeled + p.y.ticks.n_labeled
                             for p in spec.panels if p.title}
                assert int(q["answer"]) in per_panel
                assert int(q["answer"]) < spec.total_labeled_ticks() or spec.n_panels == 1
    assert seen_single and seen_multi, "need both scopes exercised"
