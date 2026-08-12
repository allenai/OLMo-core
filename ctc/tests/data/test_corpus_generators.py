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
from ctc.data.schema import gold_of, validate
from ctc.data.sources import hotpotqa as hotpotqa_source
from ctc.format import registry
from ctc.tasks import load_all

#: ladder name -> a fixture pool factory. Every corpus-backed generator must appear, so adding one
#: without a fixture fails the coverage test below rather than silently going untested.
POOLS = {
    "contradiction": pools.pubmed_pool,
    "redundancy": pools.redundancy_pool,
    "contra_fever": pools.fever_pool,
    "nq": pools.retrieval_pool,
    # 2 gold and a supply of only 8 hard negatives, both of which are corpus properties HotpotQA's
    # construction has to survive rather than fixture convenience.
    "hotpotqa": lambda: pools.retrieval_pool(source="hotpotqa", gold=2, hard=8),
    "fiqa": lambda: pools.retrieval_pool(source="beir_fiqa"),
    "scifact": lambda: pools.retrieval_pool(source="beir_scifact"),
    "rerank": pools.rerank_pool,
    "outlier": pools.article_pool,
    "outlier_review": pools.review_pool,
    "oolong": pools.oolong_pool,
    "absence": pools.book_pool,
    "xabsence": pools.paraphrase_pool,
    # reorder shares absence's Gutenberg books -- one corpus, one loader, one split-by-book -- and
    # reduces them to per-book passage streams. The books must be long: the 32k rung is 233
    # passages of ~100 words each.
    "reorder": pools.reorder_pool,
    "qdmatch_nq": pools.unit_pool,
    # Two gold per question, so k relevant queries give 2k pairs -- the multi-gold path.
    "qdmatch_hpqa": lambda: pools.unit_pool(gold=2, source="hotpotqa"),
    "grouping_labeled": pools.openalex_pool,
}

CORPUS_BACKED = sorted(POOLS)

#: Per-task overrides for the deliberately tiny corpora some checks below use. ``redundancy`` plants
#: K gold *and* H hard-negative PAIRS, so its defaults need 18 documents and cannot be built at 8 at
#: all -- the hard negatives are the task, not padding, so the generator refuses rather than
#: dropping them.
SMALL_CORPUS: dict = {
    "redundancy": {"num_pairs": 1, "num_hardneg": 1},
    # A HotpotQA bridge query brings TWO gold documents, so the default k=3 needs 6 document slots
    # and an 8-item example only has 4. The generator returns None rather than dropping a hop --
    # a one-gold "multi-hop" example would silently select the single-doc instruction wording.
    "qdmatch_hpqa": {"num_relevant": 1},
}

#: Ladders that SAMPLE their pool rather than consuming one item per example -- an article, a
#: review or a log line can back many examples. They take no example index, so the index-driven
#: checks below do not apply to them.
POOL_SAMPLING = {
    "outlier",
    "outlier_review",
    "oolong",
    "absence",
    "xabsence",
    # reorder draws a window of one book's prose; qdmatch consumes ~num_docs query units per
    # example rather than one.
    "reorder",
    "qdmatch_nq",
    "qdmatch_hpqa",
}

#: Ladders that take an example index for **stratification** rather than for pool consumption.
#: ``grouping_labeled`` derives its concept LEVEL from the index -- deliberately, since drawing it
#: from the RNG is what let the realised level mix drift with N (port record trap 14) -- but it
#: samples its papers, so it can neither run off the end of its pool nor recycle it.
STRATIFIED = {"grouping_labeled"}


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


def gold_flat(example, spec=None):
    """
    :param example: A unified-format example.
    :param spec: Its task spec, so the gold field is read off the task rather than assumed.
        ``reorder`` keeps its gold in ``gold_order`` and ``qdmatch`` in ``gold_pairs``; assuming
        ``gold_doc_indices`` silently reports "no gold" for both, which would turn every check
        below into a check of nothing.

    :returns: Its gold indices, flattened.
    """
    gold = gold_of(example, spec) if spec is not None else example.get("gold_doc_indices")
    flat = []
    for entry in gold or []:
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
        example = build_one(
            task, index=index, seed=index, num_docs=size, **SMALL_CORPUS.get(task, {})
        )
        if example is None:
            continue
        flat = gold_flat(example, spec)
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
    if task in POOL_SAMPLING | STRATIFIED:
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


# ── hotpotqa: two gold, and both have to survive everything ─────────────────────────────────────


def _hotpot_row(titles=("Alpha", "Beta", "Gamma", "Delta"), supporting=("Alpha", "Gamma")):
    """
    :param titles: Context paragraph titles, in context order.
    :param supporting: Titles ``supporting_facts`` names, once per supporting *sentence*.

    :returns: A raw HotpotQA ``distractor`` row, in the dataset's parallel-array shape.
    """
    return {
        "question": "Which of them came first?",
        "answer": "Alpha",
        "supporting_facts": {"title": list(supporting), "sent_id": [0] * len(supporting)},
        "context": {
            "title": list(titles),
            "sentences": [[f"{t} is a thing.", "It was described in a book."] for t in titles],
        },
    }


def test_hotpotqa_prepare_query_splits_gold_from_the_benchmark_s_own_distractors():
    """
    Where a multi-hop task goes wrong quietly: mixing one supporting paragraph into the distractors
    leaves a structurally valid example whose second hop is scored as a model error.
    """
    query = hotpotqa_source.prepare_query(_hotpot_row())
    assert [c.id for c in query.gold] == ["Alpha", "Gamma"]
    assert [c.id for c in query.hard] == ["Beta", "Delta"]
    assert query.answers == ("Alpha",)


def test_hotpotqa_filler_comes_from_the_same_population_as_gold():
    """
    Filler is the other rows' *supporting* paragraphs, never their retrieved distractors, and the
    reason is length. In HotpotQA the supporting paragraphs average 71.2 words and the
    TF-IDF-retrieved distractors 96.1, so a corpus filled with distractors makes gold "the short
    ones": on a 17-document build, naming the two shortest documents recovered 0.227 of gold
    against a 0.118 chance baseline. Drawing filler from gold's own population took that to 0.137.

    Nothing fails if this is reverted -- the generic ``gold_length_bias`` probe still passes,
    because "is a gold document the single longest or shortest" is too weak a statistic to see a
    25-word gap in a mean. Which is exactly why the rule is pinned here rather than left to the
    audit.
    """
    fillers = hotpotqa_source.filler_candidates(_hotpot_row())
    assert [c.id for c in fillers] == ["Alpha", "Gamma"]


def test_hotpotqa_filler_keeps_context_order_rather_than_set_order():
    """
    ``supporting_facts`` is de-duplicated through a set, and a set of strings iterates in an order
    that depends on ``PYTHONHASHSEED``. Emitting in that order would make the filler pool -- and so
    every file built from it -- differ between two processes given the same seed, which is the one
    guarantee :func:`ctc.data.build.build_train` is built on.
    """
    row = _hotpot_row(titles=("D", "C", "B", "A"), supporting=("A", "C"))
    assert [c.id for c in hotpotqa_source.filler_candidates(row)] == ["C", "A"]


def test_hotpotqa_prepare_query_does_not_render_the_title():
    """Every shipped HotpotQA document is bare paragraph text. The title is the entity the question
    is about, so rendering it hands over a hop; it survives only as the de-duplication id."""
    query = hotpotqa_source.prepare_query(_hotpot_row())
    assert query.gold[0].text == "Alpha is a thing. It was described in a book."


def test_hotpotqa_prepare_query_drops_a_row_whose_gold_is_missing_from_the_context():
    """Gold pointing at a paragraph the model is never shown is unanswerable, not hard."""
    assert hotpotqa_source.prepare_query(_hotpot_row(supporting=("Alpha", "Omega"))) is None


def test_hotpotqa_prepare_query_drops_a_row_that_is_not_two_hop():
    """``supporting_facts`` holds one entry per supporting *sentence*, so two entries can name one
    paragraph. That is a single-gold example, which selects the other instruction wording and stops
    being a multi-hop question."""
    assert hotpotqa_source.prepare_query(_hotpot_row(supporting=("Alpha", "Alpha"))) is None


def test_hotpotqa_emits_exactly_two_zero_based_gold_documents():
    """
    Both hops or neither. All 500 examples at all four shipped rungs carry exactly two gold
    indices, flat and 0-based; an example that lost one would still validate and still score, and
    would simply mark the model wrong for a hop it was never shown gold for.
    """
    example = build_one("hotpotqa", num_docs=17)
    gold = example["gold_doc_indices"]
    assert len(gold) == 2 and not isinstance(gold[0], list)
    assert min(gold) >= 0 and max(gold) <= 16


def test_hotpotqa_keeps_both_gold_documents_at_every_rung_of_the_ladder():
    """
    The nested ladder is derived by shrinking the longest rung, and the shrink is only correct if
    it keeps *both* hops. Losing one on the way down would leave the short rungs grading a
    different, single-hop question while every row count still lined up.

    Read through :func:`ctc.tasks._retrieval.flatten_gold` rather than off the raw field, because
    that is the scorer's view and the two can disagree: a shrink that re-shaped a flat ``[3, 17]``
    into ``[[3], [17]]`` leaves both hops present in the field and yet hands the grader only one.
    Indexing the raw field would still find two documents and pass.
    """
    from ctc.tasks._retrieval import flatten_gold

    spec = registry.get("retrieval")
    labels = ladders.rungs_for("hotpotqa")
    canonical = build_one("hotpotqa", num_docs=ladders.docs_for_rung("hotpotqa", labels[-1]))
    gold_texts = {canonical["documents"][i]["text"] for i in flatten_gold(canonical)}
    assert len(gold_texts) == 2
    for label in labels[:-1]:
        n_docs = ladders.docs_for_rung("hotpotqa", label)
        shorter = build.shrink(canonical, n_docs, spec, random.Random(0))
        assert len(shorter["documents"]) == n_docs
        assert {shorter["documents"][i]["text"] for i in flatten_gold(shorter)} == gold_texts


def test_hotpotqa_hard_negatives_run_out_rather_than_being_topped_up():
    """
    HotpotQA ships 8 distractors per question, so at the long rungs ``hard_frac=0.1`` asks for more
    than exist. The cap has to come from the supply: a generator that made up the difference would
    be mixing in a second corpus, which is the length mismatch this loader exists to avoid.
    """
    assert len(build_one("hotpotqa", num_docs=22)["hard_neg_indices"]) == 2  # round(0.1 * 20)
    long = build_one("hotpotqa", num_docs=288)
    assert len(long["hard_neg_indices"]) == 8  # the whole supply, not round(0.1 * 286)


def test_shrinking_a_multi_gold_example_keeps_its_gold_flat_and_scorable():
    """
    Regression: :func:`ctc.data.build.shrink` used to rewrite a flat ``[3, 17]`` into ``[[3], [17]]``.
    ``ctc.tasks._retrieval.flatten_gold`` reads a nested gold as one group per query and returns
    only the first, so every shrunk multi-gold rung -- ``hotpotqa`` and ``fiqa`` alike -- was graded
    on one gold document and marked wrong for the rest, while the longest rung was graded on all of
    them. Nothing failed: the row counts matched, the audit's own flattener handled both shapes,
    and the rungs simply disagreed for a reason unrelated to context length.
    """
    from ctc.tasks._retrieval import flatten_gold

    spec = registry.get("retrieval")
    example = build_one("hotpotqa", num_docs=40)
    shorter = build.shrink(example, 17, spec, random.Random(0))
    assert all(isinstance(g, int) for g in shorter["gold_doc_indices"])
    assert len(flatten_gold(shorter)) == 2


def test_hotpotqa_selects_the_multi_gold_instruction_and_names_both_ids():
    """
    The only in-distribution ladder that exercises the multi-gold path at all. The wording switches
    on ``has_multi_gold`` and is hashed into the format fingerprint, and the target must name both
    1-based ids -- ``nq``, ``fiqa`` and ``scifact`` never reach either branch.
    """
    from ctc.format.prompts import RETRIEVAL_INSTRUCTION_MULTI_DOC
    from ctc.tasks.retrieval import spec as retrieval_spec

    example = build_one("hotpotqa", num_docs=17)
    assert RETRIEVAL_INSTRUCTION_MULTI_DOC in retrieval_spec.build_prompt(example)
    target = retrieval_spec.build_target(example)
    assert target == ", ".join(f"[{g + 1}]" for g in sorted(example["gold_doc_indices"]))


def test_hotpotqa_defaults_match_the_shipped_build():
    """
    The staged pool is ``hotpotqa_train_k{11,24,50,100,205}_bridge_hn{1,2,5,10,20}_4000`` --
    bridge-only, hard negatives at exactly n/10. Both are pinned here so a later edit has to be
    deliberate; the suite-wide ban on the all-hard-negative regime rides on the second.
    """
    gen = generators.get("hotpotqa")
    assert gen.defaults["hard_frac"] == 0.1
    assert gen.corpus_defaults["question_type"] == "bridge"
    assert gen.corpus_defaults["ce_filter"] is True
    assert gen.eval_only is False


def test_hotpotqa_ladder_is_the_recalibrated_one_not_the_build_matrix_row():
    """
    The shipped rung files carry 17/36/72/144 documents. BUILD_MATRIX row 2's 11/24/50/100/205 is
    the ladder the 2026-07-19 FIX2 pass *measured at 0.64-0.69x of its labels* and replaced.
    """
    assert ladders.LADDERS["hotpotqa"] == {"2k": 17, "4k": 36, "8k": 72, "16k": 144, "32k": 288}


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


# ── absence: what is missing, and the ladder that cannot be nested ──────────────────────────────


def _absence_second_version(example):
    """:param example: An absence example. :returns: Its rendered Version B."""
    return example["queries"][0]


def test_absence_gold_names_the_removed_positions_zero_based():
    """
    The base flips here relative to the pair family: ``ctc.tasks._absence.score`` does
    ``{int(g) + 1 for g in gold}``, so a generator emitting 1-based indices would shift every id by
    one and read as a weak model rather than as a bug.
    """
    example = build_one("absence", num_docs=32)
    gold = example["gold_doc_indices"]
    assert min(gold) >= 0 and max(gold) <= 31
    validate(example, registry.get("absence"))


def test_absence_removes_exactly_the_gold_sentences_from_the_second_version():
    """
    The one property the whole task rests on. A sentence left in Version B while listed as gold is
    an unanswerable question; one dropped without being listed is an unlabelled correct answer.
    """
    example = build_one("absence", num_docs=32)
    second = _absence_second_version(example)
    gold = set(example["gold_doc_indices"])
    for i, doc in enumerate(example["documents"]):
        assert (doc["text"] not in second) == (i in gold)


def test_absence_answers_are_the_first_four_words_of_each_removed_sentence_in_order():
    """The snippet answer form the shipped Gutenberg files carry; the prefix-uniqueness filter is
    what makes it unambiguous, so it is worth pinning even while the spec grades ids."""
    from ctc.tasks.absence.sources.gutenberg import first_four

    example = build_one("absence", num_docs=32)
    docs = example["documents"]
    assert example["answers"] == [first_four(docs[i]["text"]) for i in example["gold_doc_indices"]]


def test_absence_target_renders_the_ids_one_based():
    from ctc.tasks.absence.spec import build_target

    example = build_one("absence", num_docs=32)
    expected = ", ".join(f"[{g + 1}]" for g in sorted(example["gold_doc_indices"]))
    assert build_target(example) == f"Missing: {expected}"


def test_absence_is_textdiff_reads_the_normalised_metadata_spelling():
    """
    Regression. ``make_example`` normalises metadata to ``_meta`` while the shipped pre-migration
    files use bare ``meta``; reading only the bare one made this ``False`` for every example built
    in this repo, so a textdiff example would be routed to the id scorer with nothing to show for
    it.
    """
    from ctc.tasks.absence.spec import is_textdiff

    assert is_textdiff(build_one("absence", num_docs=32))
    assert is_textdiff({"meta": {"format": "textdiff"}})
    assert not is_textdiff({"_meta": {"format": "ids"}})


def test_absence_skips_a_window_whose_sentences_share_a_four_word_opening():
    """Two sentences opening alike give an answer that matches both: one label, two correct
    answers. The window is refused rather than emitted."""
    from ctc.data.sources import gutenberg as gutenberg_source
    from ctc.tasks.absence.sources.gutenberg import build_example

    run = gutenberg_source.ProseRun(
        book="b0",
        sentences=tuple(f"Once upon a time there lived a person numbered {i}." for i in range(20)),
    )
    pool = gutenberg_source.BookPool(runs=(run,), provenance={})
    assert build_example(random.Random(0), corpus=pool, num_docs=10, num_removed=3) is None


def test_absence_rungs_are_generated_independently_and_the_build_says_so(monkeypatch):
    """
    Version B is *rendered text* in ``queries[0]``, so it is a function of the whole corpus: a
    generic shrink drops a distractor and leaves the query still listing it. The audit's nesting
    check keys a rung's identity on the gold text plus the query, so rebuilding the query per rung
    reads as "the rungs grade different questions" -- which is why there is no ``build_ladder``
    either, and why the ladder's rung-to-rung deltas carry eval-set resampling noise.
    """
    generator = generators.get("absence")
    assert not generator.shrink_safe and generator.build_ladder is None
    assert not generator.nested_ladder

    spec = registry.get("absence")
    monkeypatch.setitem(ladders.LADDERS, "absence", {"2k": 20, "4k": 30})
    rungs, report = build.build_eval(
        "absence", spec, size=500, seed=7, corpus=pools.book_pool(books=20)
    )
    assert {label: len(rows) for label, rows in rungs.items()} == {"2k": 500, "4k": 500}
    assert any("independently" in note for note in report.notes)


def test_absence_ladder_is_the_measured_one_not_build_matrix_row_18():
    """
    Row 18 charged each sentence once at ~20 tokens and landed on {90,180,360,720,1440}. An absence
    prompt carries the corpus TWICE -- numbered as Version A, then again inside Version B -- and the
    shipped n10/n50/n200 files measure 548/3117/14790 Qwen3 tokens, i.e. ~76 tok/document. The
    estimate overshoots by ~3.4x, so the staged ``n1440`` file is a ~109k-token file labelled 32k.
    """
    assert ladders.LADDERS["absence"] == {"2k": 32, "4k": 60, "8k": 114, "16k": 222, "32k": 438}


def test_absence_gold_is_not_findable_by_position_or_length():
    """The two shortcuts an absence task is most exposed to: a removed item betrayed by where it
    sat, or by being the longest or shortest thing in the corpus."""
    examples = [build_one("absence", seed=s, num_docs=32) for s in range(60)]
    results = {
        p.name: p for p in audit_mod.run_probes("absence", examples, registry.get("absence"))
    }
    assert not results["gold_position_bias"].failed, str(results["gold_position_bias"])
    assert not results["gold_length_bias"].failed, str(results["gold_length_bias"])


# ── xabsence: two corpora, and the claims with no twin ──────────────────────────────────────────


def _sides(example):
    return [doc["corpus"] for doc in example["documents"]]


def test_xabsence_documents_are_an_A_block_then_a_B_block_under_one_index():
    """The serializer renders ``[i] A: text`` positionally and the instruction says "the OTHER
    corpus"; interleaving the blocks would make the shared index meaningless."""
    example = build_one("xabsence", num_docs=59)
    sides = _sides(example)
    assert set(sides) == {"A", "B"}
    assert sides == sorted(sides)  # every A before every B
    assert len(example["documents"]) == 59
    validate(example, registry.get("xabsence"))


def test_xabsence_gold_is_zero_based_and_every_gold_document_is_genuinely_unmatched():
    """
    Gold is 0-based (``_absence.score`` adds one), and the label has to be true: a gold document
    whose twin is also in the context is not unmatched, and a matched document with no twin is an
    unlabelled correct answer.
    """
    pool = pools.paraphrase_pool()
    twins = {p.original: p.paraphrase for p in pool.pairs}
    twins.update({p.paraphrase: p.original for p in pool.pairs})
    for seed in range(20):
        example = build_one("xabsence", seed=seed, num_docs=59)
        gold = set(example["gold_doc_indices"])
        assert min(gold) >= 0 and max(gold) < len(example["documents"])
        texts = {doc["text"] for doc in example["documents"]}
        for i, doc in enumerate(example["documents"]):
            assert (twins[doc["text"]] in texts) == (i not in gold)


def test_xabsence_an_orphan_is_rendered_in_the_form_its_own_corpus_uses():
    """
    The leak that made this task stop being an all-pairs task. Inserting every orphan as an
    original left B-side orphans as the one non-paraphrase among paraphrases, and a trained 4B read
    it off that single document -- recall 0.98 on B-side orphans against 0.08 on A-side, ``set_f1``
    pinned at ~0.5 and FLAT in n from 39 to 669 documents.
    """
    pool = pools.paraphrase_pool()
    originals = {p.original for p in pool.pairs}
    paraphrases = {p.paraphrase for p in pool.pairs}
    seen = set()
    for seed in range(30):
        example = build_one("xabsence", seed=seed, num_docs=59)
        for i in example["gold_doc_indices"]:
            doc = example["documents"][i]
            seen.add(doc["corpus"])
            assert doc["text"] in (originals if doc["corpus"] == "A" else paraphrases)
    assert seen == {"A", "B"}, "orphans must land on both sides, or the probe above proves nothing"


def test_xabsence_ladder_is_nested_and_grades_the_same_orphans_at_every_rung():
    """
    Dropping whole matched pairs is the only safe resize: removing one half of a pair orphans its
    partner, which is a correct answer the label does not list. So the rungs nest by pair.
    """
    generator = generators.get("xabsence")
    assert not generator.shrink_safe and generator.build_ladder is not None
    row = generator.build_ladder(
        random.Random(0),
        index=0,
        corpus=pools.paraphrase_pool(),
        rungs={"2k": 59, "4k": 119, "8k": 243},
        num_docs=59,
    )
    gold_texts = None
    previous = None
    for label in ("2k", "4k", "8k"):
        example = row[label]
        texts = {doc["text"] for doc in example["documents"]}
        current = {example["documents"][i]["text"] for i in example["gold_doc_indices"]}
        if gold_texts is None:
            gold_texts, previous = current, texts
        else:
            assert current == gold_texts
            assert previous <= texts
            previous = texts


def test_xabsence_a_generic_shrink_would_invent_gold_which_is_why_it_is_refused():
    """
    Not a style preference. ``build.shrink`` drops random non-gold documents; each one strands its
    partner as a genuinely unmatched claim the label does not name, so the shorter rung would mark
    a correct answer wrong.
    """
    spec = registry.get("xabsence")
    example = build_one("xabsence", num_docs=119)
    shorter = build.shrink(example, 59, spec, random.Random(0))
    texts = {doc["text"] for doc in shorter["documents"]}
    pool = pools.paraphrase_pool()
    twins = {p.original: p.paraphrase for p in pool.pairs}
    twins.update({p.paraphrase: p.original for p in pool.pairs})
    labelled = set(shorter["gold_doc_indices"])
    stranded = [
        i
        for i, doc in enumerate(shorter["documents"])
        if twins[doc["text"]] not in texts and i not in labelled
    ]
    assert stranded, "if a random shrink no longer strands partners, revisit shrink_safe"


def test_xabsence_every_orphan_keeps_its_lexical_decoys_at_the_shortest_rung():
    """
    The decoys are what stop the orphan being the one document with no lexically close counterpart.
    They are placed at the head of the matched list precisely so a rung prefix keeps them; losing
    them at the short rungs would leave the ladder more solvable at 2k than at 32k, which is the
    opposite of the axis being measured.
    """
    generator = generators.get("xabsence")
    row = generator.build_ladder(
        random.Random(1),
        index=0,
        corpus=pools.paraphrase_pool(),
        rungs={"2k": 59, "8k": 243},
        num_docs=59,
    )
    short = {doc["text"] for doc in row["2k"]["documents"]}
    long = {doc["text"] for doc in row["8k"]["documents"]}
    assert short <= long
    assert row["2k"]["_meta"]["decoys_per_unmatched"] == 2


def test_xabsence_refuses_a_ladder_whose_shortest_rung_cannot_hold_the_decoys():
    generator = generators.get("xabsence")
    with pytest.raises(ValueError, match="lexical decoys"):
        generator.build_ladder(
            random.Random(0),
            index=0,
            corpus=pools.paraphrase_pool(),
            rungs={"tiny": 11, "2k": 59},
            num_docs=59,
        )


def test_xabsence_decoys_cut_the_lexical_shortcut():
    """
    Measured, not asserted in principle. On the shipped pre-migration files this heuristic recovers
    0.867 / 0.727 / 0.453 of the orphans at 19 / 39 / 99 documents against chance baselines of
    0.158 / 0.077 / 0.030 -- word overlap alone, with no comparison of meaning anywhere.
    """
    spec = registry.get("xabsence")
    with_decoys = [build_one("xabsence", seed=s, num_docs=59) for s in range(25)]
    without = [
        build_one("xabsence", seed=s, num_docs=59, decoys_per_unmatched=0) for s in range(25)
    ]
    fixed = audit_mod.unmatched_by_lexical_overlap(with_decoys, spec)
    raw = audit_mod.unmatched_by_lexical_overlap(without, spec)
    assert raw.failed, "the shortcut has to be present without decoys, or this proves nothing"
    assert fixed.score < raw.score
    assert not fixed.failed, str(fixed)


def test_xabsence_split_for_docs_meets_the_rung_exactly_and_refuses_an_all_orphan_corpus():
    """
    The rung label is a context length, so the document count is met exactly; an odd remainder
    becomes one more orphan, which the model is never told the size of anyway. A rung with no
    matched pair at all has no wrong answer left to give.
    """
    from ctc.tasks.xabsence.generate import split_for_docs

    assert split_for_docs(59, 3) == (28, 3)
    assert split_for_docs(58, 3) == (
        27,
        4,
    )  # the odd one out becomes an answer, not a lost document
    assert all(2 * p + k == n for n in range(9, 60) for p, k in [split_for_docs(n, 3)])
    with pytest.raises(ValueError, match="matched pair"):
        split_for_docs(4, 3)


def test_xabsence_ladder_is_measured_and_odd_so_2p_plus_k_lands_on_the_label():
    """
    BUILD_MATRIX row 22 estimated ~95 tok/pair and gave P18/P39/P81/P165/P333, i.e. 39/81/165/333/
    669 documents; the shipped p8/p18/p48 files measure 772/1394/3424 tokens at 19/39/99 documents,
    so the estimate overshoots by ~1.4x. Every value is odd because an example is ``2P + k``.
    """
    assert ladders.LADDERS["xabsence"] == {"2k": 59, "4k": 119, "8k": 243, "16k": 489, "32k": 981}
    assert all((n - 3) % 2 == 0 for n in ladders.LADDERS["xabsence"].values())


def test_xabsence_pool_refilters_a_pool_mined_at_a_looser_threshold(tmp_path):
    """
    The staged pre-migration pool was mined at ``--max-overlap 0.3``, and that is exactly the
    setting the lexical shortcut lives at. Re-applying the filter on read is what stops an old file
    being trusted as-is.
    """
    from ctc.data.sources import paraphrase

    path = tmp_path / "pool.jsonl"
    path.write_text(
        '{"claim": "aspirin reduced cardiac mortality in the cohort", '
        '"paraphrase": "aspirin reduced cardiac mortality in the group"}\n'
        '{"claim": "aspirin reduced cardiac mortality in the cohort two", '
        '"paraphrase": "lower deaths from heart disease followed treatment"}\n',
        encoding="utf-8",
    )
    loose = paraphrase.load_pool(pool_path=str(path), max_overlap=0.9)
    tight = paraphrase.load_pool(pool_path=str(path), max_overlap=0.22)
    assert len(loose) == 2
    assert len(tight) == 1
    assert tight.provenance["dropped_overlap"] == 1


def test_xabsence_exact_copy_mode_is_the_string_matchable_variant():
    """
    Kept as the no-LLM fallback, and the probe is what stops it being used by accident: a
    byte-identical twin makes "the document with no lexical counterpart" find every orphan.
    """
    from ctc.data.sources import paraphrase

    pool = paraphrase.ParaphrasePool(
        pairs=tuple(
            paraphrase.ParaphrasePair(
                f"Finding number {i} concerned outcome {i}.",
                f"Finding number {i} concerned outcome {i}.",
            )
            for i in range(120)
        ),
        provenance={"pool": "exact-copy"},
    )
    generator = generators.get("xabsence")
    examples = [
        generator.build_example(random.Random(s), corpus=pool, num_docs=59) for s in range(10)
    ]
    result = audit_mod.unmatched_by_lexical_overlap(examples, registry.get("xabsence"))
    assert result.score == 1.0 and result.failed
