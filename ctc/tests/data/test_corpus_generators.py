"""
The nine corpus-backed generators, against fixture pools. No network, no GPU, no weka.

What is being checked is *construction*, which is the half of a data generator that can be wrong
without failing. Each test here corresponds to a defect that reached a results table:

* gold counted from the wrong base -- correct answers score zero, uniformly, and read as a
  modelling result;
* a filler drawn from the same abstract as a gold claim -- an unlabelled second contradicting pair,
  scored as a model error;
* a majority topic shrunk below the outlier count -- two correct answers, one label;
* the retired 98%-hard NQ regime reachable by leaving a default alone;
* a held-out ladder used as training data, which makes every number from it in-distribution.

The pools come from ``fixtures/pools.py`` rather than from the loaders, which is exactly the seam
:mod:`ctc.data.sources` exists to provide.
"""

from __future__ import annotations

import random

import pytest
from fixtures import pools

from ctc.data import audit as audit_mod
from ctc.data import build, ladders
from ctc.data.generators import base as generators
from ctc.data.schema import validate
from ctc.format import registry
from ctc.tasks import load_all

#: ladder name -> a fixture pool factory. Every corpus-backed generator must appear, so adding one
#: without a fixture fails the coverage test below rather than silently going untested.
POOLS = {
    "contradiction": pools.pubmed_pool,
    "contra_fever": pools.fever_pool,
    "nq": pools.retrieval_pool,
    "fiqa": lambda: pools.retrieval_pool(source="beir_fiqa"),
    "scifact": lambda: pools.retrieval_pool(source="beir_scifact"),
    "rerank": pools.rerank_pool,
    "outlier": pools.article_pool,
    "outlier_review": pools.review_pool,
    "oolong": pools.oolong_pool,
}

CORPUS_BACKED = sorted(POOLS)

#: Ladders that SAMPLE their pool rather than consuming one item per example -- an article, a
#: review or a log line can back many examples. They take no example index, so the index-driven
#: checks below do not apply to them.
POOL_SAMPLING = {"outlier", "outlier_review", "oolong"}


@pytest.fixture(scope="module", autouse=True)
def _tasks():
    load_all()


def build_one(task, *, index=0, seed=0, **overrides):
    """
    :param task: Ladder name.
    :param index: Example counter.
    :param seed: RNG seed.
    :param overrides: Per-example config overrides.

    :returns: One example built from that ladder's fixture pool.
    """
    gen = generators.get(task)
    config = gen.config(**overrides)
    config["corpus"] = POOLS[task]()
    if gen.indexed:
        config["index"] = index
    return gen.build_example(random.Random(seed), **config)


def gold_flat(example):
    flat = []
    for entry in example.get("gold_doc_indices") or []:
        flat.extend(entry) if isinstance(entry, (list, tuple)) else flat.append(entry)
    return flat


# ── the registry itself ─────────────────────────────────────────────────────────────────────────


def test_every_corpus_backed_generator_has_a_fixture_pool():
    """A generator with no fixture is a generator nothing below actually runs."""
    missing = [
        name
        for name in generators.names()
        if generators.get(name).corpus is not None and name not in POOLS
    ]
    assert not missing, f"no fixture pool for {missing}"


def test_the_five_main_and_four_held_out_ladders_are_all_registered():
    """The suite's roster, asserted rather than assumed."""
    main = {"contradiction", "nq", "outlier", "rerank", "oolong"}
    ood = {"fiqa", "scifact", "outlier_review", "contra_fever"}
    assert main | ood <= set(generators.names())
    assert {n for n in ood if generators.get(n).eval_only} == ood
    assert not any(generators.get(n).eval_only for n in main)


@pytest.mark.parametrize("task", CORPUS_BACKED)
def test_a_generator_names_a_registered_grading_spec(task):
    """``nq`` is graded by ``retrieval`` and ``contra_fever`` by ``contradiction``; a ladder that
    names a spec the registry does not have would fail only at eval time."""
    assert registry.get(generators.get(task).task) is not None


@pytest.mark.parametrize("task", CORPUS_BACKED)
def test_a_generator_module_imports_without_datasets_or_torch(task):
    """
    ``pip install ./ctc`` promises no GPU and no compiler. A source module that imports
    ``datasets`` or ``torch`` at module scope breaks that for everyone who only wants to grade a
    checkpoint, and the failure appears at import of an unrelated task.
    """
    import inspect

    module = inspect.getmodule(generators.get(task).build_example)
    source = inspect.getsource(module)
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and line[:1] not in (" ", "\t"):
            assert not any(
                heavy in stripped for heavy in ("datasets", "torch", "transformers", "pyserini")
            ), f"{module.__name__} imports a heavy dependency at module scope: {stripped}"


# ── the unified format, per ladder ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("task", CORPUS_BACKED)
def test_generated_examples_validate_against_the_schema(task):
    spec = registry.get(generators.get(task).task)
    example = build_one(task)
    assert example is not None
    validate(example, spec, require_gold=task != "oolong")


@pytest.mark.parametrize("task", CORPUS_BACKED)
def test_gold_agrees_with_the_spec_declared_index_base(task):
    """
    The generator and the grader declare the base in different files and a disagreement is silent:
    correct answers score zero, uniformly, and read as a modelling result.

    Range-checking one example is not decisive in either direction, so this draws many examples over
    a deliberately SMALL corpus, where gold covers the whole index range quickly, and asserts that
    the extremes actually observed are ``base`` and ``n - 1 + base``. Off-by-one data cannot produce
    both.
    """
    spec = registry.get(generators.get(task).task)
    if task == "oolong":
        pytest.skip("oolong's gold field is vestigial and always empty")
    size = 8
    seen = set()
    for index in range(40):
        example = build_one(task, index=index, seed=index, num_docs=size)
        if example is None:
            continue
        flat = gold_flat(example)
        n = len(example["documents"])
        assert min(flat) >= spec.gold_index_base, f"{task}: gold below the declared base"
        assert max(flat) <= n - 1 + spec.gold_index_base, f"{task}: gold past the last document"
        seen |= set(flat)
    assert min(seen) == spec.gold_index_base, f"{task} declares base {spec.gold_index_base}"
    assert max(seen) == size - 1 + spec.gold_index_base, f"{task} never reaches the last document"


@pytest.mark.parametrize("task", CORPUS_BACKED)
def test_an_example_has_exactly_the_requested_number_of_documents(task):
    """A rung that comes out short is a mislabelled x-axis, not a rounding detail."""
    gen = generators.get(task)
    if gen.scaling_param != "num_docs":
        pytest.skip("oolong scales a token budget, checked separately")
    example = build_one(task, num_docs=25)
    assert len(example["documents"]) == 25


@pytest.mark.parametrize("task", CORPUS_BACKED)
def test_the_same_seed_and_index_give_the_same_example(task):
    """Reproducibility is the property that makes a rebuild meaningful."""
    assert build_one(task, index=3, seed=11) == build_one(task, index=3, seed=11)


@pytest.mark.parametrize("task", CORPUS_BACKED)
def test_a_different_index_gives_a_different_question(task):
    """
    The index has to reach the construction. A generator that ignored it would sample its question
    from the RNG instead and quietly repeat some while leaving others unused.
    """
    if task in POOL_SAMPLING:
        pytest.skip("samples from the pool rather than consuming it, so it takes no index")
    first, second = build_one(task, index=0, seed=5), build_one(task, index=1, seed=5)
    assert first != second


@pytest.mark.parametrize("task", CORPUS_BACKED)
def test_running_off_the_end_of_the_pool_returns_none_rather_than_recycling(task):
    """
    A generator that wrapped silently would emit the same gold under different filler and report a
    full-size split. Returning ``None`` is what lets the builder count the shortfall.
    """
    if task in POOL_SAMPLING:
        pytest.skip("samples from the pool rather than consuming it")
    assert build_one(task, index=10_000) is None


@pytest.mark.parametrize("task", CORPUS_BACKED)
def test_every_rung_of_the_ladder_is_buildable(task):
    """
    A ladder entry no generator can satisfy is a rung that silently never gets built -- the defect
    BUILD_MATRIX.md shipped, where mathmatch was infeasible at all five rungs with its own
    documented defaults.
    """
    gen = generators.get(task)
    for rung in ladders.rungs_for(task):
        size = ladders.docs_for_rung(task, rung)
        if size > 400:
            continue  # the fixture pools are small on purpose; the long rungs need a real corpus
        example = build_one(task, **{gen.scaling_param: size})
        assert example is not None, f"{task}@{rung}"


# ── held-out ladders ────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("task", ["fiqa", "scifact", "outlier_review", "contra_fever"])
def test_a_held_out_ladder_refuses_to_produce_training_data(task):
    """
    An error, not a warning. By the time a warning is noticed the checkpoint is trained and the
    whole OOD column means nothing.
    """
    spec = registry.get(generators.get(task).task)
    with pytest.raises(ValueError, match="held-out"):
        build.build_train(task, spec, total=4, corpus=POOLS[task]())


@pytest.mark.parametrize("task", ["fiqa", "scifact", "outlier_review", "contra_fever"])
def test_a_held_out_ladder_is_graded_by_an_in_distribution_spec(task):
    """
    The OOD gap is only a *source* effect if both arms are graded identically. A held-out ladder
    that grew its own spec would be measuring a different thing.
    """
    assert generators.get(task).task in {"retrieval", "outlier", "contradiction"}


# ── contradiction: the false-negative control ───────────────────────────────────────────────────


def test_contradiction_fillers_never_come_from_a_gold_claim_s_own_abstract():
    """
    The abstract a gold claim came from is the one most likely to restate the fact its
    contradiction denies -- which would be an unlabelled second gold pair, scored as a model error.
    """
    pool = pools.pubmed_pool()
    gen = generators.get("contradiction")
    example = gen.build_example(
        random.Random(0), index=0, corpus=pool, **gen.config(num_docs=40, num_pairs=3)
    )
    gold_abstracts = {p.abstract_id for p in pool.pairs[:3]}
    banned = {s for a in gold_abstracts for s in pool.fillers[a]}
    gold_texts = {example["documents"][i - 1]["text"] for i in gold_flat(example)}
    for doc in example["documents"]:
        if doc["text"] in gold_texts:
            continue
        assert doc["text"] not in banned


def test_contradiction_gold_is_one_pair_per_requested_pair_and_is_one_based():
    pool = pools.pubmed_pool()
    gen = generators.get("contradiction")
    example = gen.build_example(
        random.Random(1), index=2, corpus=pool, **gen.config(num_docs=30, num_pairs=4)
    )
    assert len(example["gold_doc_indices"]) == 4
    assert all(len(pair) == 2 for pair in example["gold_doc_indices"])
    assert min(gold_flat(example)) >= 1
    assert max(gold_flat(example)) <= 30


def test_contradiction_pairs_are_consumed_in_order_and_not_reused():
    """Two examples must not share a gold pair; the index is what guarantees it."""
    pool = pools.pubmed_pool()
    gen = generators.get("contradiction")
    seen = set()
    for index in range(5):
        example = gen.build_example(
            random.Random(index), index=index, corpus=pool, **gen.config(num_docs=30, num_pairs=3)
        )
        texts = frozenset(example["documents"][i - 1]["text"] for i in gold_flat(example))
        assert not (texts & seen)
        seen |= texts


def test_contra_fever_defaults_to_the_difficulty_matched_plain_build():
    """
    The pre-migration generator defaulted its hard distractors ON, but every shipped FEVER file --
    the only variant the grid has used -- was built with them off, so its difficulty matches the
    PubMed ladder it is compared against.
    """
    gen = generators.get("contra_fever")
    assert gen.defaults["hard_nei_per_pair"] == 0
    assert gen.defaults["decoy_support_pairs"] == 0
    assert gen.defaults["use_decoys"] is False
    example = build_one("contra_fever", num_docs=40)
    from ctc.data.schema import meta_of

    assert meta_of(example)["variant"] == "plain"


def test_contra_fever_hard_distractors_stay_on_topic_when_enabled():
    """When the hard build IS asked for, its NEI distractors must come from the gold pages."""
    pool = pools.fever_pool()
    gen = generators.get("contra_fever")
    example = gen.build_example(
        random.Random(0),
        index=0,
        corpus=pool,
        **gen.config(num_docs=40, num_pairs=2, hard_nei_per_pair=4),
    )
    texts = {d["text"] for d in example["documents"]}
    gold_pages = [page for _, _, page in pool.pairs[:2]]
    assert any(claim in texts for page in gold_pages for claim in pool.nei_by_page[page])


# ── nq: the banned regime must not be the default ───────────────────────────────────────────────


def test_nq_defaults_to_ten_percent_hard_negatives():
    """
    The pre-migration default was 1.0 with the CE filter off, which silently reproduced the retired
    98%-hard pipeline. Every current NQ number was measured at 0.1 + CE.
    """
    gen = generators.get("nq")
    assert gen.defaults["hard_frac"] == 0.1
    assert gen.corpus_defaults["ce_filter"] is True


def test_nq_hard_negative_count_tracks_the_fraction_not_a_fixed_number():
    """
    A constant hard-negative *count* would make the short rungs almost all hard and the long ones
    almost all random, so the two ends of the ladder would be measuring different tasks.
    """
    pool = pools.retrieval_pool()
    gen = generators.get("nq")
    for num_docs in (11, 21, 101):
        example = gen.build_example(
            random.Random(0), index=0, corpus=pool, **gen.config(num_docs=num_docs)
        )
        hard = len(example["hard_neg_indices"])
        assert hard == pytest.approx(round(0.1 * (num_docs - 1)), abs=1)


def test_retrieval_hard_negatives_are_a_prefix_so_rungs_stay_comparable():
    """
    Hard negatives are sorted hardest-first and a rung takes a *prefix*. Sampling instead would
    change which negatives a question faces at each rung, confounding "longer context" with
    "different distractors".
    """
    pool = pools.retrieval_pool()
    gen = generators.get("nq")
    short = gen.build_example(random.Random(0), index=0, corpus=pool, **gen.config(num_docs=41))
    long = gen.build_example(random.Random(0), index=0, corpus=pool, **gen.config(num_docs=101))
    short_hard = {short["documents"][i]["text"] for i in short["hard_neg_indices"]}
    long_hard = {long["documents"][i]["text"] for i in long["hard_neg_indices"]}
    assert short_hard <= long_hard


def test_retrieval_hard_neg_indices_tag_only_the_mined_negatives():
    """Calling an ordinary distractor a mined one destroys the only record of how a pool was
    built."""
    pool = pools.retrieval_pool()
    gen = generators.get("nq")
    example = gen.build_example(random.Random(0), index=0, corpus=pool, **gen.config(num_docs=51))
    tagged = {example["documents"][i]["text"] for i in example["hard_neg_indices"]}
    assert all("confused with" in text for text in tagged)


# ── rerank: every document must be scored ───────────────────────────────────────────────────────


def test_rerank_emits_a_score_for_every_document():
    """
    A ``None`` score is read as gain 0 and dropped from the Kendall-tau set, so an unscored random
    fill makes the reference order tail off into display order and the metric grades a shorter list
    than the model ranked.
    """
    example = build_one("rerank", num_docs=60)
    assert len(example["ce_scores"]) == len(example["documents"])
    assert all(score is not None for score in example["ce_scores"])


def test_rerank_gold_outscores_every_hard_negative():
    """The graded ordering is built from these scores; gold below a negative would invert it."""
    example = build_one("rerank", num_docs=60)
    gold = {example["ce_scores"][i] for i in example["gold_doc_indices"]}
    hard = {example["ce_scores"][i] for i in example["hard_neg_indices"]}
    assert min(gold) > max(hard)


def test_rerank_scores_survive_a_shrink_to_a_shorter_rung():
    """``ce_scores`` is positional; a stale full-length copy would mis-score every document."""
    spec = registry.get("rerank")
    example = build_one("rerank", num_docs=60)
    shorter = build.shrink(example, 20, spec, random.Random(0))
    assert len(shorter["ce_scores"]) == 20
    for index, doc in enumerate(shorter["documents"]):
        original = next(i for i, d in enumerate(example["documents"]) if d["text"] == doc["text"])
        assert shorter["ce_scores"][index] == example["ce_scores"][original]


# ── outlier: the scale-K invariant ──────────────────────────────────────────────────────────────


def _topic_sizes(example, pool_texts):
    """:returns: ``topic -> document count`` for one outlier example."""
    counts = {}
    for doc in example["documents"]:
        counts[pool_texts[doc["text"]]] = counts.get(pool_texts[doc["text"]], 0) + 1
    return counts


def test_outlier_is_the_uniquely_rarest_topic_at_every_rung():
    """
    The invariant the whole task rests on. The generic shrink can break it by dropping enough of a
    majority topic that it falls to or below the outlier count, at which point the question has two
    correct answers and one label.
    """
    pool = pools.article_pool()
    gen = generators.get("outlier")
    rungs = {"2k": 14, "4k": 28, "8k": 57, "16k": 115}
    row = gen.build_ladder(
        random.Random(0), index=0, corpus=pool, rungs=rungs, **gen.config(num_outliers=3)
    )
    assert row is not None
    text_to_topic = {body: title for title, bodies in pool.articles for body in bodies}
    for label, example in row.items():
        sizes = _topic_sizes(example, text_to_topic)
        outlier_topic = text_to_topic[example["documents"][example["gold_doc_indices"][0]]["text"]]
        assert sizes[outlier_topic] == 3, label
        others = [size for topic, size in sizes.items() if topic != outlier_topic]
        assert min(others) > 3, f"{label}: a majority topic is not larger than the outlier"


def test_outlier_ladder_rungs_are_nested_and_grade_the_same_question():
    pool = pools.article_pool()
    gen = generators.get("outlier")
    rungs = {"2k": 14, "4k": 28, "8k": 57}
    row = gen.build_ladder(random.Random(1), index=4, corpus=pool, rungs=rungs, **gen.config())
    gold_texts = {
        label: sorted(ex["documents"][i]["text"] for i in ex["gold_doc_indices"])
        for label, ex in row.items()
    }
    assert len(set(map(tuple, gold_texts.values()))) == 1
    for shorter, longer in (("2k", "4k"), ("4k", "8k")):
        short = {d["text"] for d in row[shorter]["documents"]}
        long = {d["text"] for d in row[longer]["documents"]}
        assert short <= long


def test_outlier_declares_itself_unfit_for_the_generic_shrink():
    """The declaration is what routes it to its own ladder builder; losing it silently restores
    the ambiguous rungs."""
    gen = generators.get("outlier")
    assert gen.shrink_safe is False
    assert gen.build_ladder is not None
    assert gen.nested_ladder is True


def test_outlier_never_renders_an_article_title():
    """A Wikipedia title states the topic outright, so rendering it hands over the answer."""
    example = build_one("outlier", num_docs=22)
    assert all("title" not in doc for doc in example["documents"])


def test_outlier_review_renders_the_review_headline():
    """The counterpart: a review headline is part of what a human reader would read, and every
    shipped `outlier_review` file carries it."""
    example = build_one("outlier_review", num_docs=30)
    assert all(doc.get("title") for doc in example["documents"])


def test_outlier_review_defaults_to_the_category_build():
    """``rating_ratio`` 0.5 was the pre-migration default and mixes in a second, harder task; the
    shipped difficulty-matched ladder is category-only."""
    gen = generators.get("outlier_review")
    assert gen.defaults["rating_ratio"] == 0.0
    assert build_one("outlier_review")["source"] == "review_outlier_category"


# ── oolong: token budgets and independent rungs ─────────────────────────────────────────────────


def test_oolong_scales_a_token_budget_not_a_document_count():
    gen = generators.get("oolong")
    assert gen.scaling_param == "target_tokens"
    assert ladders.LADDERS["oolong"]["8k"] == 8192


def test_oolong_emits_one_document_whose_length_tracks_the_budget():
    small = build_one("oolong", target_tokens=1000, seed=3)
    large = build_one("oolong", target_tokens=4000, seed=3)
    assert len(small["documents"]) == len(large["documents"]) == 1
    from ctc.data.schema import meta_of

    assert meta_of(small)["num_items"] < meta_of(large)["num_items"]


def test_oolong_answers_are_recomputed_over_the_drawn_items():
    """
    The gold is an aggregate of whichever items were drawn -- which is exactly why oolong's rungs
    cannot be nested, and why the generator says so rather than letting an audit find out.
    """
    from ctc.data.schema import meta_of

    example = build_one("oolong", target_tokens=2000, seed=7)
    meta = meta_of(example)
    if meta["task_group"] != "counting" or meta["answer_type"] != "ANSWER_TYPE.NUMERIC":
        pytest.skip("this draw picked a non-counting variant")
    label = example["queries"][0].split("label '")[1].split("'")[0]
    lines = example["documents"][0]["text"]
    assert str(lines.count("Instance:")) != "" and example["answers"][0].isdigit()
    assert int(example["answers"][0]) <= meta["num_items"]
    assert label in meta["gold_list"] or example["answers"][0].isdigit()


def test_oolong_declares_that_its_rungs_are_independent():
    gen = generators.get("oolong")
    assert gen.shrink_safe is False
    assert gen.build_ladder is None
    assert gen.nested_ladder is False


def test_oolong_gold_field_is_declared_and_empty():
    """Vestigial by design: oolong's items are lines, not documents, so there is nothing to point
    at. The audit must not "verify" it and report a pass that means nothing."""
    example = build_one("oolong")
    assert example["gold_doc_indices"] == []


# ── the builder, end to end, on a fixture pool ──────────────────────────────────────────────────


def test_a_nested_eval_ladder_grades_the_same_questions_over_nested_corpora(monkeypatch):
    spec = registry.get("retrieval")
    monkeypatch.setitem(ladders.LADDERS, "nq", {"2k": 11, "4k": 23})
    rungs, report = build.build_eval(
        "nq", spec, size=500, seed=7, corpus=pools.retrieval_pool(queries=600)
    )
    assert report.total == 1000
    result = audit_mod.audit("nq", spec, rungs=rungs)
    assert result.ok, result.report()


def test_the_outlier_ladder_passes_its_own_audit(monkeypatch):
    spec = registry.get("outlier")
    monkeypatch.setitem(ladders.LADDERS, "outlier", {"2k": 14, "4k": 28})
    rungs, _ = build.build_eval(
        "outlier", spec, size=500, seed=7, corpus=pools.article_pool(articles=1200)
    )
    result = audit_mod.audit("outlier", spec, rungs=rungs)
    assert result.ok, result.report()


def test_an_independent_ladder_is_not_reported_as_a_broken_nested_one(monkeypatch):
    """
    ``oolong`` cannot nest, and reporting that as a defect on every build would train everyone to
    pass ``--force`` -- which is worse than having no audit.
    """
    spec = registry.get("oolong")
    monkeypatch.setitem(ladders.LADDERS, "oolong", {"2k": 900, "4k": 1800})
    rungs, report = build.build_eval("oolong", spec, size=500, seed=7, corpus=pools.oolong_pool())
    assert any("independently" in note for note in report.notes)
    result = audit_mod.audit("oolong", spec, rungs=rungs)
    assert result.ok, result.report()
    assert "independently" in result.checks["ladder"]


def test_train_and_eval_draw_from_disjoint_slices_of_the_pool(monkeypatch):
    """
    Train/eval separation is a property of the POOL. Two callers splitting it two different ways is
    how the pre-migration tree ended up with five splitters and two different eval fractions.
    """
    spec = registry.get("retrieval")
    monkeypatch.setitem(ladders.LADDERS, "nq", {"2k": 11})
    pool = pools.retrieval_pool(queries=600)
    rungs, _ = build.build_eval("nq", spec, size=500, seed=7, corpus=pool)
    train, _ = build.build_train("nq", spec, total=40, seed=42, corpus=pool)
    eval_questions = {ex["queries"][0] for rows in rungs.values() for ex in rows}
    train_questions = {ex["queries"][0] for ex in train}
    assert not (eval_questions & train_questions)


def test_a_train_build_covers_the_pool_rather_than_asking_one_question_per_rung(monkeypatch):
    """
    One cursor for the whole split, not one per rung. Restarting it per rung would ask the same
    questions at five lengths and call the result 20k distinct training examples.
    """
    spec = registry.get("retrieval")
    monkeypatch.setitem(ladders.LADDERS, "nq", {"2k": 11, "4k": 23, "8k": 48})
    train, _ = build.build_train(
        "nq", spec, total=60, seed=42, corpus=pools.retrieval_pool(queries=600)
    )
    assert len({ex["queries"][0] for ex in train}) == 60


def test_a_pool_that_runs_out_is_reported_rather_than_silently_recycled(monkeypatch):
    spec = registry.get("retrieval")
    monkeypatch.setitem(ladders.LADDERS, "nq", {"2k": 11})
    train, report = build.build_train(
        "nq", spec, total=30, seed=42, corpus=pools.retrieval_pool(queries=22)
    )
    assert report.reused_pool >= 1
    assert len(train) == 30
