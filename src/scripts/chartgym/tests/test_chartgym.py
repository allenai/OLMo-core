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


def test_procedure_traces_are_correct_and_selective():
    """Trace targets must (a) end in the bare answer, (b) exist ONLY on aggregation and
    NA-verification families, and (c) never leak onto read-off families.

    The selectivity is the entire point: a global enumeration prompt lifted CharXiv t17 by
    +20 but cost -2.0 benchmark-wide because it narrated simple look-ups too. Supervision
    is the instrument that can be enumeration-for-aggregates and terse-for-read-offs at
    once -- but only if the corpus actually is.
    """
    import re

    TRACE_FAMILIES = {"cnt.ticks_total", "cnt.ticks_axis", "cnt.legend_entries",
                      "ocr.legend_names"}
    seen_traced = set()
    for spec, audit in _figures(n=30):
        for q in emit_all(spec, audit, np.random.default_rng(9), include_held_out=False):
            if q["target"] == q["answer"]:
                continue
            seen_traced.add(q["family"])
            if q["is_na"]:
                # search trace: must end by expressing the NA conclusion
                assert q["answer"].rstrip(".") in q["target"], q
                assert q["target"].lower().startswith("checking"), q
            else:
                assert q["family"] in TRACE_FAMILIES, f"trace leaked onto {q['family']}"
                if q["answer_type"] == "list":
                    # verification trace on a list answer must end with the names verbatim
                    assert q["target"].endswith(q["answer"]), q
                else:
                    # enumeration/verification trace: the LAST number must be the answer
                    nums = re.findall(r"-?\d+(?:\.\d+)?", q["target"].replace(",", ""))
                    assert nums and float(nums[-1]) == float(q["answer"]), q
                # v4 symmetry requirement: any family that trains an absence check must
                # show the SAME check concluding presence. A check that only ever precedes
                # "absent" teaches that checking implies absence -- measured on CharXiv t12
                # as 23/42 errors being false "no legend" declarations.
                if q["family"] in ("cnt.legend_entries", "ocr.legend_names"):
                    assert q["target"].lower().startswith("checking"), q
    assert TRACE_FAMILIES <= seen_traced, f"missing traces for {TRACE_FAMILIES - seen_traced}"


# ---------------------------------------------------------------- ceiling probe

def test_probe_cannot_contaminate_an_honest_corpus():
    """The CharXiv-mirroring probe must be unreachable from the normal emission path.

    The probe exists to measure a ceiling, not to train a model. If a `charxiv.*` family
    could ever appear in a default build, every anti-benchmark-fitting guarantee in this
    package would be void and nothing downstream would notice.
    """
    for spec, audit in _figures(n=12):
        for split in (True, False):
            for q in emit_all(spec, audit, np.random.default_rng(11), include_held_out=split):
                assert not q["family"].startswith("charxiv."), q["family"]


def test_probe_actually_mirrors_the_benchmark():
    """...and conversely, the probe must really copy CharXiv, or it measures nothing.

    This is the mirror image of `test_no_charxiv_phrasing`: that test asserts the honest
    families share no 6-gram with the benchmark; this one asserts the probe families do.
    Together they make the distinction between the two corpora mechanical.
    """
    import json as _json

    from chartgym.families_charxiv_probe import PROBE_TEMPLATES, emit_probe

    _VERBATIM = _json.loads(
        (Path(__file__).resolve().parents[1] / "metadata" / "charxiv_templates.json").read_text()
    )
    seen_templates = set()
    for spec, audit in _figures(n=12):
        for q in emit_probe(spec, audit, np.random.default_rng(12)):
            tid = int(q["family"].split(".t")[1])
            seen_templates.add(tid)
            assert tid in PROBE_TEMPLATES
            assert q["target"] == q["answer"], "probe supervises the bare benchmark answer"
            # A probe question must overlap the benchmark wording it copies. Compare
            # against the VERBATIM templates (metadata/charxiv_templates.json), not the
            # abbreviated list used by test_no_charxiv_phrasing -- several of those are
            # under six words, so a 6-gram test against them is vacuous.
            tmpl = _VERBATIM[str(tid)]
            qt, tt = _tokens(q["question"]), _tokens(tmpl)
            jacc = len(qt & tt) / len(qt | tt) if (qt | tt) else 0.0
            assert (_ngrams(q["question"]) & _ngrams(tmpl)) or jacc >= 0.6, (
                f"t{tid} does not mirror its template (J={jacc:.2f})"
            )
    # 14/15 are colorbar-only and deliberately excluded; everything else must appear
    assert seen_templates >= set(PROBE_TEMPLATES) - {16}, sorted(seen_templates)
    assert not ({14, 15} & seen_templates)


def test_probe_answers_are_exact():
    from chartgym.families_charxiv_probe import emit_probe

    for spec, audit in _figures(n=12):
        by = {}
        for q in emit_probe(spec, audit, np.random.default_rng(13)):
            by.setdefault(int(q["family"].split(".t")[1]), []).append(q)
        rows, cols = spec.grid_shape
        assert by[18][0]["answer"] == f"{rows} by {cols}"
        assert by[19][0]["answer"] == str(spec.n_panels)
        # t17 is panel-scoped (figure scope cost 28.6 points in v1)
        per_panel = {p.x.ticks.n_labeled + p.y.ticks.n_labeled for p in spec.panels}
        for q in by[17]:
            assert int(q["answer"]) in per_panel


def test_probe_na_rates_match_measured_gold():
    """The probe must mirror CharXiv's ANSWER distribution, not just its phrasing.

    v5 inherited the generator's compounded absence rates and emitted 64.3% NA on template
    10 against the benchmark's 22.6%, and 35.0% vs 17.9% on template 8 -- the most likely
    cause of t8 losing 11.6 points. A probe whose NA frequency is wrong measures a ceiling
    for a benchmark that does not exist.
    """
    import collections

    from chartgym.families_charxiv_probe import GOLD_NA_RATE, emit_probe

    n = collections.Counter()
    na = collections.Counter()
    for spec, audit in _figures(n=30):
        for q in emit_probe(spec, audit, np.random.default_rng(21)):
            tid = int(q["family"].split(".t")[1])
            n[tid] += 1
            na[tid] += q["is_na"]
    for tid, gold in GOLD_NA_RATE.items():
        if n[tid] < 60:
            continue  # too few to test at this sample size
        got = na[tid] / n[tid]
        assert abs(got - gold) < 0.06, (
            f"t{tid} NA rate {got:.3f} is far from the measured gold {gold:.3f}"
        )
