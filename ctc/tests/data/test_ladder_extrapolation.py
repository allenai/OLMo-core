"""
Rungs beyond the calibrated table: extrapolation, ceilings, and the small-eval escape hatch.

The table stops at 32k; anything longer resolves through the least-squares fit over the task's own
rows. These tests pin the three properties that make that safe to rely on: the fit reproduces the
table (a calibrated rung never changes), the extrapolation reproduces the shipped ultra-long
ladders (the one external cross-check we have), and the guardrails -- ceilings, structural
constraints, the 500-example floor -- fail loudly instead of building mislabelled data.
"""

from __future__ import annotations

import pytest

from ctc.data import build, ladders
from ctc.format import registry
from ctc.format import rungs as rung_util
from ctc.tasks import load_all

load_all()


# --- the fit ------------------------------------------------------------------------------------


def test_table_rungs_are_exact_never_fitted():
    """A calibrated row is returned verbatim; the fit only serves labels the table lacks."""
    for task, ladder in ladders.LADDERS.items():
        for label, docs in ladder.items():
            assert ladders.docs_for_rung(task, label) == docs
            assert not ladders.is_extrapolated(task, label)


def test_extrapolation_matches_the_shipped_64k_ladder():
    """
    The one external cross-check: contradiction's shipped ``v2_clean`` 64k rung holds 1525
    documents, calibrated independently against the real tokenizer. The fit through the 2k-32k
    table must land within a percent of it -- much further off would mean the fit is measuring
    noise, not the corpus.
    """
    docs = ladders.docs_for_rung("contradiction", "64k")
    assert abs(docs - 1525) / 1525 < 0.01, docs
    assert ladders.is_extrapolated("contradiction", "64k")


def test_oolong_fit_is_the_identity():
    """oolong's rung values ARE token budgets, so the fit must degenerate to y = x exactly."""
    for label in ("64k", "256k", "1m", "10m"):
        assert ladders.docs_for_rung("oolong", label) == rung_util.parse_rung(label)


def test_ten_million_token_rungs_resolve_for_every_unbounded_task():
    """The headline requirement: any task without a ceiling resolves a 10m rung to a sane count."""
    for task in ladders.LADDERS:
        if task in ladders.CEILINGS:
            continue
        docs = ladders.docs_for_rung(task, "10m")
        assert docs > ladders.docs_for_rung(task, "32k"), task
        # The count must scale roughly linearly: ~320x the 32k count, within rounding slack.
        assert 200 < docs / ladders.docs_for_rung(task, "32k") < 400, task


def test_extrapolated_counts_grow_monotonically():
    for task in ("contradiction", "nq", "cycle", "mathmatch"):
        counts = [ladders.docs_for_rung(task, r) for r in ("64k", "128k", "256k", "512k", "1m")]
        assert counts == sorted(counts) and len(set(counts)) == len(counts), (task, counts)


def test_xabsence_extrapolation_stays_odd():
    """An xabsence example is 2P+k documents (k=3): an even count would silently build one pair
    below its label, which is exactly the class of off-by-one the table's own rows avoid."""
    for label in ("64k", "256k", "1m", "10m"):
        assert ladders.docs_for_rung("xabsence", label) % 2 == 1


# --- the guardrails -----------------------------------------------------------------------------


def test_ceiling_is_a_refusal_with_the_reason():
    with pytest.raises(ValueError, match="4,000 labeled HotpotQA units"):
        ladders.docs_for_rung("qdmatch_hpqa", "512k")
    with pytest.raises(ValueError, match="5,183 abstracts"):
        ladders.docs_for_rung("scifact", "2m")
    # At or below the ceiling still resolves.
    assert ladders.docs_for_rung("qdmatch_hpqa", "256k") > 0
    assert ladders.docs_for_rung("scifact", "1m") > 0


def test_unknown_task_is_still_a_keyerror():
    with pytest.raises(KeyError, match="no rung ladder"):
        ladders.docs_for_rung("nope", "2k")


def test_absurdly_small_rung_raises():
    with pytest.raises(ValueError, match="extrapolates to"):
        ladders.docs_for_rung("textgroups", "64")


# --- the build path -----------------------------------------------------------------------------


def test_build_eval_below_floor_requires_allow_small():
    spec = registry.get("mathmatch")
    with pytest.raises(ValueError, match="allow_small"):
        build.build_eval("mathmatch", spec, size=125, rungs=["2k"])


def test_build_eval_allow_small_flags_the_size():
    spec = registry.get("mathmatch")
    rungs, report = build.build_eval("mathmatch", spec, size=5, rungs=["2k"], allow_small=True)
    assert len(rungs["2k"]) == 5
    assert any("below the 500 floor" in note for note in report.notes)


def test_build_flags_extrapolated_rungs():
    """A rung past the table must be flagged in the report -- the flag is what tells a reader the
    x-coordinate is fitted, not measured."""
    spec = registry.get("mathmatch")
    _, train_report = build.build_train("mathmatch", spec, total=2, rungs=["64k"])
    assert any("extrapolated" in note for note in train_report.notes)
    rungs, eval_report = build.build_eval(
        "mathmatch", spec, size=5, rungs=["8k", "64k"], allow_small=True
    )
    assert any("64k" in note and "extrapolated" in note for note in eval_report.notes)
    # And the built example really is ~64k-rung sized, nested down to 8k.
    n64 = ladders.docs_for_rung("mathmatch", "64k")
    assert all(len(ex["documents"]) == n64 for ex in rungs["64k"])
    assert all(
        len(ex["documents"]) == ladders.docs_for_rung("mathmatch", "8k") for ex in rungs["8k"]
    )


def test_calibrated_build_carries_no_extrapolation_note():
    spec = registry.get("mathmatch")
    _, report = build.build_train("mathmatch", spec, total=2, rungs=["2k"])
    assert not any("extrapolated" in note for note in report.notes)


def test_shrink_ladder_nests_adjacent_rungs():
    """
    Three or more rungs must nest pairwise, not just against the longest. The parallel form of
    the shrink -- every rung independently derived from the canonical set -- satisfied the
    2-rung case and the docstring while failing the audit on every real (5-rung) ladder: 2k held
    distractors 4k had dropped. The chain is the fix, and this pins it.
    """
    from ctc.data import audit as audit_mod

    spec = registry.get("mathmatch")
    rungs, _ = build.build_eval(
        "mathmatch", spec, size=8, rungs=["2k", "4k", "8k"], allow_small=True
    )
    result = audit_mod.check_ladder_nesting(rungs, spec)
    assert not result.problems, result.problems
    for shorter, longer in (("2k", "4k"), ("4k", "8k")):
        for row_a, row_b in zip(rungs[shorter], rungs[longer]):
            assert {d["text"] for d in row_a["documents"]} <= {
                d["text"] for d in row_b["documents"]
            }
