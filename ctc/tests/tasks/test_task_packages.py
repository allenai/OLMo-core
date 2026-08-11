"""
The per-task contract.

These tests are deliberately generic: they run against every registered task, so a task added later
inherits the checks without anyone remembering to write them. Each one corresponds to a mistake
that has already been made once.
"""

from __future__ import annotations

import json
from pathlib import Path

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
def test_stop_preset_exists(spec):
    """A typo would fall back to whatever the runner defaults to, silently."""
    from ctc.eval.stopping import STOP_PRESETS

    assert spec.stop in STOP_PRESETS, f"{spec.name}.stop={spec.stop!r} is not a known preset"


@pytest.mark.parametrize("spec", _specs(), ids=lambda s: s.name)
def test_set_answer_tasks_terminate_on_an_empty_answer(spec):
    """"[]" is a CORRECT answer -- there were no pairs -- and must not run to the budget.

    An earlier version of this suite asserted the opposite property (that set-answer tasks never
    newline-stop, on the theory that a newline could cut a wrapped list). That was wrong: these
    answers are single-line JSON, and enforcing it produced a "]]"-only stop rule, under which the
    empty answer "[]" contains no terminator at all. The model would ramble to max_new_tokens and
    parse_pairs would then return None -- recording a correct answer as a parse failure.
    """
    from ctc.eval.stopping import STOP_PRESETS, should_stop

    if not spec.answer_is_set:
        pytest.skip("not a set-answer task")
    cond = STOP_PRESETS[spec.stop]
    assert should_stop("[]\nand now some rambling", cond) is not None, (
        f"{spec.name} (stop={spec.stop!r}) has no terminator for an empty answer"
    )


@pytest.mark.parametrize("spec", _specs(), ids=lambda s: s.name)
def test_stop_rule_terminates_on_this_task_s_own_target(spec):
    """A rambling model must be truncated back to exactly the answer it was trained to emit.

    Uses each task's own build_target rather than a fixed string. An earlier version asserted the
    bracketed-pair shape for every answer_is_set task, which was wrong: retrieval's set answer is
    "[1], [2]" on one line and cot_retrieval's is "Relevant Document: [1]". The shapes differ per
    task, so the only meaningful generic check is against the task's own target.
    """
    import importlib

    from fixtures.generate_golden import PROMPT_EXAMPLES

    from ctc.eval.stopping import STOP_PRESETS, apply

    key = next((k for k in PROMPT_EXAMPLES if k == spec.name), None)
    if key is None:
        pytest.skip(f"no fixture example for {spec.name}")
    if not STOP_PRESETS[spec.stop].text_stops:
        # EOS-only by design: grouping and reorder emit long, legitimately multi-line answers, so
        # any text stop would cut a valid one short. Nothing can truncate a ramble here except the
        # model emitting EOS, which is the intended trade.
        pytest.skip(f"{spec.name} stops on EOS only; no text rule can apply")
    mod = importlib.import_module(f"ctc.tasks.{spec.name}.spec")
    build_target = getattr(mod, "build_target", None)
    if build_target is None:
        pytest.skip(f"{spec.name} declares no build_target")

    target = build_target(PROMPT_EXAMPLES[key])
    cond = STOP_PRESETS[spec.stop]
    generation = f"{target}\nand now some rambling"
    if cond.require_before is not None:
        # oolong's rule keys on the templated "answer:" line, which the bare target does not
        # contain -- the model emits it, the target does not. Prefix it so the generation looks
        # like what the model actually produces, rather than skipping the check.
        generation = f"{cond.require_before} {generation}"
    out = apply(generation, cond)
    if cond.require_before is not None:
        out = out.split(cond.require_before, 1)[1].strip()
    assert out.strip() == target.strip(), (
        f"{spec.name} (stop={spec.stop!r}) did not truncate back to its own target"
    )


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

# The pre-migration `force_unified` set inside build_prompt, verbatim. Restated here so that as the
# remaining 19 specs land, each one's `unified` flag is checked against what the old code actually
# did rather than against someone's recollection of it. matching_ngram and ruler are in the old set
# but are not canonical native tasks.
HISTORICAL_UNIFIED = {
    "contradiction", "qdmatch", "xabsence", "redundancy", "absence", "matching_ngram",
    "mathmatch", "strmatch", "cycle", "groups4", "textgroups", "reorder", "ruler",
}


@pytest.mark.parametrize("spec", _specs(), ids=lambda s: s.name)
def test_unified_flag_matches_the_pre_migration_behaviour(spec):
    """`unified` is not a preference -- it must match what build_prompt did for this task."""
    assert spec.unified == (spec.name in HISTORICAL_UNIFIED)


def test_fingerprint_records_query_position():
    """query_position really varies across runs (both/before/after are all in use) and changes the
    token stream substantially -- 'both' repeats the whole query block after the documents.

    Without it in the fingerprint, two checkpoints differing only here would compare as compatible.
    """
    spec = registry.get("contradiction")
    a = spec.fingerprint(query_position="before")
    b = spec.fingerprint(query_position="both")
    with pytest.raises(Exception):
        a.require_compatible_with(b)


def test_fingerprint_records_prompt_shape():
    spec = registry.get("contradiction")
    assert spec.fingerprint().prompt_shape == "unified"
    unified_fp = spec.fingerprint()
    classic_fp = spec.fingerprint(prompt_shape="classic")
    assert unified_fp.compare(classic_fp)


def test_invalid_query_position_is_rejected():
    with pytest.raises(ValueError, match="query_position"):
        registry.get("contradiction").fingerprint(query_position="middle")


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


def test_unsorted_gold_is_rejected_not_silently_unmatchable():
    """Predicted pairs are sorted and scoring is a set intersection, so [4, 1] can never match."""
    spec = registry.get("contradiction")
    with pytest.raises(ValueError, match="not sorted low-high"):
        spec.score([[1, 4]], [[4, 1]])


def test_unparseable_on_empty_gold_does_not_score_perfect():
    """The pre-migration scorer mapped None -> [] and returned 1.0 on all four metrics here.

    Contradiction has no empty-gold examples so it never fired, but redundancy/strmatch/mathmatch
    share that scorer and their instructions explicitly permit an empty answer.
    """
    spec = registry.get("contradiction")
    assert spec.score(None, [])["exact_match"] == 0.0
    assert spec.score([], [])["exact_match"] == 1.0  # a real empty answer can still be right


# ── prompt parity ───────────────────────────────────────────────────────────────────────────────
#
# The fixture was snapshotted from the pre-migration build_prompt BEFORE assemble.py was written,
# so these are an independent target, not a description of the port.

GOLDEN = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "golden_format.json").read_text()
)
#: Golden prompt cases whose task is registered. Grows automatically as specs land, so a newly
#: ported task is held to the pre-migration bytes without anyone adding a test for it.
_PROMPT_KEYS = [k for k in GOLDEN["prompts"] if k.split("|")[0] in registry.names()]


def test_prompt_parity_actually_covers_something():
    """Guards the filter above: if it silently matched nothing, every parity test would vanish."""
    covered = {k.split("|")[0] for k in _PROMPT_KEYS}
    assert covered, "no golden prompt cases matched any registered task"
    assert "contradiction" in covered


@pytest.mark.parametrize("key", sorted(_PROMPT_KEYS))
def test_prompt_matches_golden(key):
    """Byte-identical to what built every shard we have trained on, for every ported task."""
    from fixtures.generate_golden import PROMPT_EXAMPLES

    task, position, alpaca = key.split("|")
    got = registry.get(task).build_prompt(
        PROMPT_EXAMPLES[task],
        query_position=position,
        use_alpaca=(alpaca == "alpaca=True"),
    )
    assert got == GOLDEN["prompts"][key][0]


@pytest.mark.parametrize("key", sorted(_PROMPT_KEYS))
def test_target_matches_golden(key):
    """The training target, not just the prompt -- both halves of a shard must reproduce."""
    import importlib

    from fixtures.generate_golden import PROMPT_EXAMPLES

    task = key.split("|")[0]
    mod = importlib.import_module(f"ctc.tasks.{task}.spec")
    build_target = getattr(mod, "build_target", None)
    if build_target is None:
        pytest.skip(f"{task} declares no build_target yet")
    assert build_target(PROMPT_EXAMPLES[task]) == GOLDEN["prompts"][key][1]


def test_query_position_both_repeats_the_ask():
    """'both' duplicates the entire positioned block, which costs context at every rung."""
    from fixtures.generate_golden import PROMPT_EXAMPLES

    spec = registry.get("contradiction")
    after = spec.build_prompt(PROMPT_EXAMPLES["contradiction"], query_position="after")
    both = spec.build_prompt(PROMPT_EXAMPLES["contradiction"], query_position="both")
    assert len(both) > len(after)
    assert both.count("identify all pairs of claims") == 2


def test_unknown_query_position_raises():
    from fixtures.generate_golden import PROMPT_EXAMPLES

    with pytest.raises(ValueError, match="query_position"):
        registry.get("contradiction").build_prompt(
            PROMPT_EXAMPLES["contradiction"], query_position="middle"
        )


def test_generate_reports_what_is_missing_rather_than_returning_nothing():
    from ctc.tasks.contradiction import generate

    with pytest.raises(NotImplementedError, match="generate_pubmed_contradiction_data"):
        list(generate.build("pubmed", rung="2k"))
