"""
The train/eval builder: nesting, stability, and the contamination guards.

Each test here corresponds to a way the pre-migration pipeline could produce two files that look
correct and are not comparable.
"""

from __future__ import annotations

import random

import pytest

from ctc.data import build, ladders
from ctc.data.generators import base as _generators
from ctc.format import registry
from ctc.tasks import load_all

TASK = "textgroups"

#: Ladders a test can build with no corpus. The corpus-backed ones get the same coverage in
#: ``test_corpus_generators.py``, against fixture pools.
CORPUS_FREE = sorted(set(ladders.LADDERS) & set(_generators.corpus_free_names()))


@pytest.fixture(scope="module")
def spec():
    load_all()
    return registry.get(TASK)


@pytest.fixture(scope="module")
def canonical(spec):
    from ctc.data.generators import base as generators

    gen = generators.get(TASK)
    rng = random.Random(0)
    return [gen.build_example(rng, **gen.config(num_docs=40)) for _ in range(6)]


def gold_texts(example, spec):
    docs = example["documents"]
    return sorted(
        docs[i - spec.gold_index_base]["text"]
        for group in example["gold_doc_indices"]
        for i in group
    )


# ── shrink ──────────────────────────────────────────────────────────────────────────────────────


def test_shrink_keeps_every_gold_document(spec, canonical):
    smaller = build.shrink(canonical[0], 12, spec, random.Random(1))
    assert len(smaller["documents"]) == 12
    assert gold_texts(smaller, spec) == gold_texts(canonical[0], spec)


def test_shrink_remaps_gold_to_the_new_positions(spec, canonical):
    """The gold ids must point at the gold text after renumbering, not at whatever moved into it."""
    smaller = build.shrink(canonical[0], 15, spec, random.Random(1))
    docs = smaller["documents"]
    for group in smaller["gold_doc_indices"]:
        for i in group:
            assert docs[i - spec.gold_index_base]["text"] in gold_texts(canonical[0], spec)


def test_shrink_is_nested(spec, canonical):
    """A shorter rung's documents must be a subset of the longer one's, or the ladder compares
    different corpora and 'longer context' is confounded with 'different distractors'."""
    big = build.shrink(canonical[0], 30, spec, random.Random(1))
    small = build.shrink(big, 10, spec, random.Random(2))
    big_texts = [d["text"] for d in big["documents"]]
    small_texts = [d["text"] for d in small["documents"]]
    assert set(small_texts) <= set(big_texts)
    assert small_texts == [t for t in big_texts if t in set(small_texts)]  # order preserved


def test_shrink_shrinks_parallel_metadata(spec):
    """textgroups' _meta.counts is parallel to documents; a stale full-length copy silently
    mislabels every passage the CoT builder narrates."""
    from ctc.data.generators import base as generators

    tg_spec = registry.get("textgroups")
    gen = generators.get("textgroups")
    example = gen.build_example(random.Random(0), **gen.config(num_docs=14))
    smaller = build.shrink(example, 9, tg_spec, random.Random(1))
    counts = smaller["_meta"]["counts"]
    assert len(counts) == 9
    from ctc.tasks.textgroups.generate import count_feature

    feature, word = smaller["_meta"]["feature"], smaller["_meta"]["word"]
    assert [count_feature(d["text"], feature, word) for d in smaller["documents"]] == counts


def test_shrink_refuses_to_drop_gold(spec, canonical):
    with pytest.raises(ValueError, match="cannot hold"):
        build.shrink(canonical[0], 2, spec, random.Random(1))


# ── the eval ladder ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def small_ladder():
    """A three-rung ladder small enough to build in a test."""
    return {"2k": 8, "4k": 16, "8k": 30}


@pytest.fixture(scope="module")
def evalset(spec, small_ladder, request):
    original = ladders.LADDERS[TASK]
    ladders.LADDERS[TASK] = small_ladder
    request.addfinalizer(lambda: ladders.LADDERS.__setitem__(TASK, original))
    return build.build_eval(TASK, spec, size=500, seed=7)


def test_every_rung_grades_the_same_questions(spec, evalset):
    rungs, _ = evalset
    per_rung = [[gold_texts(ex, spec) for ex in rows] for rows in rungs.values()]
    assert all(rows == per_rung[0] for rows in per_rung), "gold drifted between rungs"


def test_rungs_are_row_aligned_and_full_size(evalset):
    rungs, report = evalset
    assert {label: len(rows) for label, rows in rungs.items()} == {label: 500 for label in rungs}
    assert report.total == 500 * len(rungs)


def test_document_counts_follow_the_ladder(evalset, small_ladder):
    rungs, _ = evalset
    for label, rows in rungs.items():
        assert {len(ex["documents"]) for ex in rows} == {small_ladder[label]}


def test_a_sub_500_eval_is_refused(spec):
    with pytest.raises(ValueError, match="below the suite floor"):
        build.build_eval(TASK, spec, size=100)


# ── stability and contamination ─────────────────────────────────────────────────────────────────


def test_the_eval_set_does_not_move_when_the_train_size_changes(spec, request):
    """
    The pre-migration drivers threaded one RNG through train then eval, so --num-train silently
    changed the eval set. Both files still looked fine, which is what made it dangerous.
    """
    original = ladders.LADDERS[TASK]
    ladders.LADDERS[TASK] = {"2k": 8, "4k": 16}
    request.addfinalizer(lambda: ladders.LADDERS.__setitem__(TASK, original))

    first, _ = build.build_eval(TASK, spec, size=500, seed=7)
    build.build_train(TASK, spec, total=40, seed=42)
    second, _ = build.build_eval(TASK, spec, size=500, seed=7)
    assert first == second


def test_train_rejects_examples_reusing_eval_gold(spec, request):
    """
    Note the seeds: passing the same number to both builds no longer collides, because streams are
    keyed by ``(seed, split, rung)``. So this forces the overlap directly -- take examples the train
    stream provably produces and declare them off-limits.
    """
    original = ladders.LADDERS[TASK]
    ladders.LADDERS[TASK] = {"2k": 8}
    request.addfinalizer(lambda: ladders.LADDERS.__setitem__(TASK, original))

    drawn, _ = build.build_train(TASK, spec, total=20, seed=42)
    _, report = build.build_train(TASK, spec, total=20, seed=42, eval_examples=drawn[:5])
    assert report.contaminated == 5, "the first five draws reuse held-out gold and must be rejected"

    _, clean = build.build_train(TASK, spec, total=20, seed=42)
    assert clean.contaminated == 0


def test_the_same_seed_no_longer_makes_train_and_eval_collide(spec, request):
    """A property of the keyed streams worth pinning: seed reuse across splits is now harmless,
    which is what makes seed 42 / seed 7 a convention rather than a correctness requirement."""
    original = ladders.LADDERS[TASK]
    ladders.LADDERS[TASK] = {"2k": 8}
    request.addfinalizer(lambda: ladders.LADDERS.__setitem__(TASK, original))

    rungs, _ = build.build_eval(TASK, spec, size=500, seed=42)
    _, report = build.build_train(TASK, spec, total=20, seed=42, eval_examples=rungs["2k"])
    assert report.contaminated == 0


def test_train_spreads_over_the_ladder(spec, request):
    original = ladders.LADDERS[TASK]
    ladders.LADDERS[TASK] = {"2k": 8, "4k": 16, "8k": 30}
    request.addfinalizer(lambda: ladders.LADDERS.__setitem__(TASK, original))

    examples, report = build.build_train(TASK, spec, total=30, seed=42)
    assert report.counts == {"2k": 10, "4k": 10, "8k": 10}
    assert sorted({len(ex["documents"]) for ex in examples}) == [8, 16, 30]


def test_a_parameter_space_too_small_fails_loudly(spec):
    """
    Silently returning near-duplicates would give a training set of the requested size and a
    fraction of the requested diversity. Driven with a stub rather than a real config, because
    every ported generator's space is large enough that provoking this for real would take minutes.
    """
    from ctc.data.generators.base import Generator

    one_example = {
        "documents": [{"text": "a"}, {"text": "b"}],
        "queries": ["q"],
        "answers": [],
        "gold_doc_indices": [[1, 2]],
        "source": "stub",
    }
    stub = Generator(
        name=TASK,
        task=TASK,
        source="stub",
        build_example=lambda rng, **kw: dict(one_example),
        defaults={"num_docs": 2},
    )
    report = build.BuildReport(task=TASK, split="train")
    with pytest.raises(RuntimeError, match="too small"):
        list(
            build._draw(
                stub,
                {"num_docs": 2},
                random.Random(0),
                count=5,
                seen=set(),
                forbidden_gold=set(),
                spec=spec,
                report=report,
            )
        )
    assert report.duplicates >= build.MAX_REJECTS_PER_EXAMPLE


@pytest.mark.parametrize("task", CORPUS_FREE)
def test_every_task_builds_at_every_rung_of_its_own_ladder(task):
    """
    The check that the pre-migration build spec never got. BUILD_MATRIX.md recorded a concrete
    command per task and its own header says nothing in it had been run: with the defaults it
    documents, mathmatch was infeasible at all five rungs and groups4 at the top three, because a
    fixed answer range cannot hold a ladder's worth of mutually-distant values. A ladder entry that
    no generator can satisfy is a rung that silently never gets built.


    Corpus-backed ladders get the same check in ``test_corpus_generators.py``, against fixture
    pools -- the property is identical, only the substrate has to be supplied.
    """
    from ctc.data.generators import base as generators

    load_all()
    gen = generators.get(task)
    for rung in ladders.rungs_for(task):
        config = gen.config()
        config[gen.scaling_param] = ladders.docs_for_rung(task, rung)
        example = gen.build_example(random.Random(0), **config)
        assert len(example["documents"]) == config[gen.scaling_param], f"{task}@{rung}"


def test_a_probe_firing_on_a_tiny_sample_is_advisory_not_fatal():
    """
    At n=10 a hit-rate probe moves 0.1 per example and MARGIN is ~one binomial SE, so a
    --allow-small-eval demo build was refused on noise -- which teaches everyone to pass --force.
    Below the sample floor a firing probe must still be REPORTED, but cannot fail the build; at or
    above the floor it stays exactly as strict as before.
    """
    from ctc.data.audit import MIN_PROBE_SAMPLES, run_probes

    spec = registry.get(TASK)

    # Synthesize a blatant length shortcut: gold is always the single longest document.
    def biased(i):
        docs = [{"text": "w " * (30 if j == 0 else 10)} for j in range(6)]
        return {
            "documents": docs,
            "queries": [],
            "answers": [["x"]],
            "gold_doc_indices": [1],
            "source": TASK,
        }

    tiny = [biased(i) for i in range(10)]
    results = {r.name: r for r in run_probes(TASK, tiny, spec)}
    fired = results["gold_length_bias"]
    assert fired.score > fired.chance + 0.2, "the fixture must actually trip the probe"
    assert not fired.failed, "below the floor it must be advisory"
    assert "ADVISORY" in fired.detail

    big = [biased(i) for i in range(MIN_PROBE_SAMPLES)]
    strict = {r.name: r for r in run_probes(TASK, big, spec)}
    assert strict["gold_length_bias"].failed, "at the floor the probe must still fail the build"
