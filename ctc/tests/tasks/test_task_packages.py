"""
The per-task contract.

These tests are deliberately generic: they run against every registered task, so a task added later
inherits the checks without anyone remembering to write them. Each one corresponds to a mistake
that has already been made once.
"""

from __future__ import annotations

import pytest

from ctc import tasks
from ctc.format import registry, rungs as rung_util


# Loaded at import, not in a fixture: the parametrize calls below run at collection time, which is
# before any fixture would have registered anything.
tasks.load_all()


def _specs():
    return [registry.get(n) for n in registry.names()]


def test_every_declared_module_loads():
    """strict=True by default: a task that silently vanishes gives a suite run one fewer row."""
    loaded = tasks.load_all()
    assert set(tasks.TASK_MODULES) <= set(loaded)
    assert tasks.import_errors() == {}


@pytest.mark.parametrize("spec", _specs(), ids=lambda s: s.name)
def test_rungs_are_valid_and_ascending(spec):
    assert spec.rungs, f"{spec.name} declares no ladder"
    assert list(spec.rungs) == rung_util.sort_rungs(spec.rungs)


@pytest.mark.parametrize("spec", _specs(), ids=lambda s: s.name)
def test_serializer_exists(spec):
    """A typo here would silently fall through to the default passage format."""
    from ctc.format import documents

    assert spec.serializer == "default" or spec.serializer in documents._SERIALIZERS


@pytest.mark.parametrize("spec", _specs(), ids=lambda s: s.name)
def test_instruction_is_declared(spec):
    """The instruction is hashed into the fingerprint; an empty one makes the guard vacuous."""
    assert spec.instruction.strip()


@pytest.mark.parametrize("spec", _specs(), ids=lambda s: s.name)
def test_primary_metric_is_produced_by_score(spec):
    """Named so a results table cannot switch between f1 and exact_match unnoticed."""
    produced = spec.score(None, [])
    assert spec.primary_metric in produced, (
        f"{spec.name}.primary_metric={spec.primary_metric!r} is not in {sorted(produced)}"
    )


@pytest.mark.parametrize("spec", _specs(), ids=lambda s: s.name)
def test_unparseable_scores_zero_not_crash(spec):
    """A None parse is a real, frequent case -- it must score, not raise."""
    assert spec.score(None, [])[spec.primary_metric] == 0.0


@pytest.mark.parametrize("spec", _specs(), ids=lambda s: s.name)
def test_score_reports_whether_it_parsed(spec):
    """Without this flag, a decoding regression is indistinguishable from a weaker model."""
    assert spec.score(None, [])["parsed"] == 0.0


# ── fingerprint derivation ──────────────────────────────────────────────────────────────────────

def test_fingerprint_is_derived_from_the_spec():
    spec = registry.get("contradiction")
    fp = spec.fingerprint(tokenizer="Qwen3.5-4B", doc_id_range=(1, 705))
    assert fp.task == "contradiction"
    assert fp.gold_index_base == 1
    assert fp.serializer == "contradiction"
    assert fp.item_separator == "\n\n"


def test_editing_an_instruction_changes_the_fingerprint():
    """This is the mechanism: a reworded prompt must invalidate old checkpoints."""
    spec = registry.get("contradiction")
    reworded = spec.__class__(**{**spec.__dict__, "instruction": spec.instruction + " Please."})
    assert reworded.fingerprint().prompt_hash != spec.fingerprint().prompt_hash


def test_a_spec_fingerprint_matches_itself():
    spec = registry.get("contradiction")
    spec.fingerprint().require_compatible_with(spec.fingerprint())


# ── contradiction specifics ─────────────────────────────────────────────────────────────────────

def test_contradiction_gold_is_one_based():
    """1-based here, 0-based for outlier/rerank/nq. The off-by-one read as a modelling result."""
    assert registry.get("contradiction").gold_index_base == 1


def test_contradiction_scoring_end_to_end():
    spec = registry.get("contradiction")
    parsed = spec.parse("1, 37], [6, 60]]")  # primed-bracket generation
    assert parsed == [[1, 37], [6, 60]]
    assert spec.score(parsed, [[1, 37], [6, 60]])["f1"] == 1.0


def test_contradiction_keeps_hallucinated_ids():
    """Dropping out-of-range ids would flatter a model that invents them."""
    spec = registry.get("contradiction")
    assert spec.score(spec.parse("[[1, 9999]]"), [[1, 4]])["precision"] == 0.0


def test_generate_reports_what_is_missing_rather_than_returning_nothing():
    from ctc.tasks.contradiction import generate

    with pytest.raises(NotImplementedError, match="generate_pubmed_contradiction_data"):
        list(generate.build("pubmed", rung="2k"))
