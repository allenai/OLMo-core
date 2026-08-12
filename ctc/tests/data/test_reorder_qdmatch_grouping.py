"""
The three ladders whose gold is not a set of document ids: reorder, qdmatch and grouping_labeled.

They are grouped because they break the assumption every other generator satisfies -- that gold is
``gold_doc_indices``, a subset of positions, over documents that can be dropped. Here gold is a
permutation of *all* positions, an ordered pair list over a shared index, and a partition of *all*
positions respectively, and each of those shapes has its own way of being silently wrong:

* reorder's answer is a permutation, so a shuffle that is not uniform leaves display order
  correlated with source order and the task is solvable without reading anything;
* a qdmatch pair is ORDERED, so anything that sorts within a pair turns wrong answers into right
  ones -- including the shared nested-ladder shrink;
* grouping's gold covers every document, so the generic shrink has nothing to drop and the generic
  length probe is true by construction rather than by defect.

The corpus-agnostic checks (index base, determinism, ladder feasibility) live in
``test_corpus_generators.py`` with every other generator. What is here is what is specific to
these three.
"""

from __future__ import annotations

import random

import pytest
from fixtures import pools

from ctc.data import audit as audit_mod
from ctc.data import build, ladders
from ctc.data.generators import base as generators
from ctc.data.schema import gold_of, validate
from ctc.format import registry
from ctc.tasks import load_all


@pytest.fixture(scope="module", autouse=True)
def _tasks():
    load_all()


def _spec(ladder):
    return registry.get(generators.get(ladder).task)


# ── reorder ─────────────────────────────────────────────────────────────────────────────────────


def reorder_example(*, num_docs=14, seed=0, corpus=None):
    """
    :param num_docs: Passages per example.
    :param seed: RNG seed.
    :param corpus: An override pool.

    :returns: One reorder example from the fixture passage pool.
    """
    gen = generators.get("reorder")
    return gen.build_example(
        random.Random(seed), corpus=corpus or pools.reorder_pool(), num_docs=num_docs
    )


def test_reorder_gold_order_actually_restores_the_source_order():
    """
    The one property the whole task rests on, and it is a *direction* that is easy to invert:
    ``gold_order[i]`` is the display id of the passage that originally sat at source position
    ``i``, not the source position of display ``i``. Both are permutations of ``1..n``, both
    validate, and the inverted one scores a model that is exactly right as exactly wrong.
    """
    pool = pools.reorder_pool()
    example = reorder_example(corpus=pool)
    documents = example["documents"]
    restored = [documents[display - 1]["text"] for display in example["gold_order"]]
    book = next(b for b in pool.books if b.book == example["_meta"]["book"])
    start = example["_meta"]["start"]
    assert restored == list(book.passages[start : start + len(documents)])


def test_reorder_gold_order_is_a_permutation_of_one_to_n():
    example = reorder_example(num_docs=20)
    assert sorted(example["gold_order"]) == list(range(1, 21))


def test_reorder_writes_source_not_source_type():
    """
    The pre-migration files wrote the corpus tag under ``source_type`` and left ``source`` unset,
    so every ``source``-keyed consumer -- including the filler-provenance audit -- skipped reorder
    entirely. Settling that is part of this port.
    """
    example = reorder_example()
    assert example["source"] == "reorder_gutenberg"
    assert "source_type" not in example


def test_reorder_gold_lives_in_gold_order_not_gold_doc_indices():
    """A reorder example has no ``gold_doc_indices`` at all; the spec must say so, or the shared
    build layer validates against a field that is never there."""
    example = reorder_example()
    assert "gold_doc_indices" not in example
    assert gold_of(example, _spec("reorder")) == example["gold_order"]


def test_reorder_target_round_trips_through_the_spec_parser():
    """Generation and grading agree, checked end to end rather than by inspection."""
    spec = _spec("reorder")
    example = reorder_example(num_docs=12)
    parsed = spec.parse(example["answers"][0], n_docs=12)
    assert parsed == example["gold_order"]
    assert spec.score(parsed, example["gold_order"])["kendall_tau"] == 1.0


def test_reorder_passages_are_within_the_word_band():
    """
    A passage far off the target is both a rung-label error and a length cue. The floor is the
    target (a passage is closed only once it reaches it) and the ceiling is ``max_words``.
    """
    example = reorder_example(num_docs=20)
    lengths = example["_meta"]["passage_word_lens"]
    assert all(100 <= length <= 160 for length in lengths), lengths


def test_reorder_drops_the_trailing_partial_passage():
    """
    The anti-shortcut measure this construction turns on. Grouping sentences up to a word target
    leaves a short remainder at the end of every run; emit it and the shortest passage is
    systematically the LAST one in source order, pinning one position of the permutation for free.
    """
    from ctc.tasks.reorder.generate import passage_runs

    sentences = ["word " * 20 + "end." for _ in range(11)]  # 21 words each: 5 make a passage
    runs = passage_runs(sentences, target_words=100, max_words=160)
    assert [len(run) for run in runs] == [2]
    assert all(len(p.split()) >= 100 for run in runs for p in run)


def test_reorder_an_over_long_passage_breaks_its_run_rather_than_being_emitted():
    """A 400-word "sentence" is a scrape artifact; emitted, it is four times every other passage in
    the example, which is a length cue and a rung-label error at once."""
    from ctc.tasks.reorder.generate import passage_runs

    normal = ["word " * 20 + "end." for _ in range(10)]
    monster = ["huge " * 400 + "end."]
    runs = passage_runs(normal + monster + normal, target_words=100, max_words=160)
    assert len(runs) == 2
    assert all(len(p.split()) <= 160 for run in runs for p in run)


def test_reorder_shuffle_leaves_no_display_order_signal():
    """
    The audit probe, run over many examples. A biased or partial shuffle scores well above 0.5 and
    the task becomes answerable with ``[1, 2, 3, ...]``.
    """
    pool = pools.reorder_pool()
    examples = [
        example
        for seed in range(60)
        if (example := reorder_example(num_docs=12, seed=seed, corpus=pool)) is not None
    ]
    probe = audit_mod.reorder_display_order_leak(examples, _spec("reorder"))
    assert not probe.failed, probe


def test_reorder_shortest_passage_has_no_preferred_source_position():
    pool = pools.reorder_pool()
    examples = [
        example
        for seed in range(60)
        if (example := reorder_example(num_docs=12, seed=seed, corpus=pool)) is not None
    ]
    probe = audit_mod.reorder_length_position_bias(examples, _spec("reorder"))
    assert not probe.failed, probe


def test_reorder_records_how_many_run_seams_a_block_crosses():
    """
    A block may span a prose-run boundary -- confining it to one run caps the ladder at its 4k rung
    -- so the cost has to be measurable rather than assumed.
    """
    pool = pools.reorder_pool()
    examples = [reorder_example(num_docs=40, seed=seed, corpus=pool) for seed in range(20)]
    seams = [example["_meta"]["run_seams"] for example in examples if example]
    assert all(isinstance(count, int) and count >= 0 for count in seams)
    assert any(count == 0 for count in seams), "no block was seam-free; the fixture runs are short"


def test_reorder_declares_itself_unnestable():
    """
    Every passage is gold, so the generic shrink has nothing to drop and a shorter rung is a
    different permutation. Declaring otherwise would route reorder through ``shrink`` and produce a
    ladder whose rungs quietly grade different questions.
    """
    gen = generators.get("reorder")
    assert not gen.shrink_safe and gen.build_ladder is None and not gen.nested_ladder


def test_reorder_refuses_a_single_passage_example():
    with pytest.raises(ValueError, match="permutation"):
        reorder_example(num_docs=1)


def test_reorder_pool_splits_by_book():
    """Two blocks from one book share vocabulary, characters and narrator."""
    pool = pools.reorder_pool()
    train = {book.book for book in pool.for_split("train").books}
    evalset = {book.book for book in pool.for_split("eval").books}
    assert train and evalset and not (train & evalset)


# ── qdmatch ─────────────────────────────────────────────────────────────────────────────────────


def qdmatch_example(
    *, ladder="qdmatch_nq", num_docs=40, num_relevant=3, seed=0, corpus=None, **overrides
):
    """
    :param ladder: ``qdmatch_nq`` or ``qdmatch_hpqa``.
    :param num_docs: Total items, M + N.
    :param num_relevant: Relevant queries.
    :param seed: RNG seed.
    :param corpus: An override pool.
    :param overrides: Further build knobs.

    :returns: One qdmatch example from the fixture unit pool.
    """
    gen = generators.get(ladder)
    pool = corpus or (
        pools.unit_pool() if ladder == "qdmatch_nq" else pools.unit_pool(gold=2, source="hotpotqa")
    )
    return gen.build_example(
        random.Random(seed),
        corpus=pool,
        num_docs=num_docs,
        num_relevant=num_relevant,
        **overrides,
    )


def test_qdmatch_gold_pairs_are_query_then_document():
    """
    The pair is ORDERED: ``[query_id, document_id]``. A swapped pair is a different claim, and
    ``parse_qd_pairs`` deliberately does not sort, so a generator that emitted them the other way
    round would score every correct model answer as wrong.
    """
    example = qdmatch_example()
    documents = example["documents"]
    for query_id, doc_id in example["gold_pairs"]:
        assert documents[query_id - 1]["type"] == "query"
        assert documents[doc_id - 1]["type"] == "document"


def test_qdmatch_every_gold_document_answers_its_own_query():
    """The pairing is the label; a mis-tagged owner produces a valid-looking file that is wrong."""
    example = qdmatch_example()
    documents = example["documents"]
    for query_id, doc_id in example["gold_pairs"]:
        gadget = documents[query_id - 1]["text"].split()[-1].rstrip("?")
        assert f"Gadget {gadget} " in documents[doc_id - 1]["text"]


def test_qdmatch_separate_layout_keeps_every_query_id_below_every_document_id():
    """
    This is what makes the nested ladder safe. ``ctc.data.build.shrink`` sorts within a gold group
    when it derives a shorter rung; under the separate layout that sort is a no-op, and under an
    interleaved one it would swap pairs into different claims.
    """
    example = qdmatch_example(num_docs=60)
    queries = [i for i, item in enumerate(example["documents"], 1) if item["type"] == "query"]
    docs = [i for i, item in enumerate(example["documents"], 1) if item["type"] == "document"]
    assert max(queries) < min(docs)
    assert all(query_id < doc_id for query_id, doc_id in example["gold_pairs"])


def test_qdmatch_shrink_preserves_pair_order_and_pair_identity():
    """The nested ladder, end to end: a shorter rung must grade the same pairs."""
    spec = _spec("qdmatch_nq")
    example = qdmatch_example(num_docs=60)
    before = {
        (example["documents"][q - 1]["text"], example["documents"][d - 1]["text"])
        for q, d in example["gold_pairs"]
    }
    shorter = build.shrink(example, 30, spec, random.Random(1))
    after = {
        (shorter["documents"][q - 1]["text"], shorter["documents"][d - 1]["text"])
        for q, d in shorter["gold_pairs"]
    }
    assert before == after
    assert len(shorter["documents"]) == 30
    for query_id, doc_id in shorter["gold_pairs"]:
        assert shorter["documents"][query_id - 1]["type"] == "query"
        assert shorter["documents"][doc_id - 1]["type"] == "document"


def test_qdmatch_the_shuffled_layout_is_refused_with_its_reason():
    """A dropped ablation, refused rather than silently mis-shrunk. The message has to say why, or
    the next person re-enables it and the corruption is invisible."""
    with pytest.raises(ValueError, match="shrink"):
        qdmatch_example(num_docs=40, layout="shuffled")


def test_qdmatch_units_are_disjoint_so_no_unlabelled_pair_exists():
    """
    The correctness argument of the whole construction: a distractor query's document is never in
    the example and a distractor document's query never is either. If a unit were reused across
    roles, that example would hold a true pair nobody labelled, scored as a model error.
    """
    example = qdmatch_example(num_docs=60)
    documents = example["documents"]
    gold = {(q, d) for q, d in example["gold_pairs"]}
    queries = [(i, item["text"]) for i, item in enumerate(documents, 1) if item["type"] == "query"]
    docs = [(i, item["text"]) for i, item in enumerate(documents, 1) if item["type"] != "query"]
    for query_id, query in queries:
        gadget = query.split()[-1].rstrip("?")
        for doc_id, text in docs:
            if f"Gadget {gadget} " in text:
                assert (query_id, doc_id) in gold, "an unlabelled true pair reached the example"


def test_qdmatch_hpqa_gives_two_pairs_per_relevant_query():
    """``num_relevant`` counts QUERIES, not pairs -- the difference between the two ladders."""
    assert len(qdmatch_example(ladder="qdmatch_nq", num_relevant=3)["gold_pairs"]) == 3
    assert len(qdmatch_example(ladder="qdmatch_hpqa", num_relevant=3)["gold_pairs"]) == 6


def test_qdmatch_splits_items_evenly_between_queries_and_documents():
    """``num_docs`` is total ITEMS, because that is what the ladder and the shrink both measure."""
    example = qdmatch_example(num_docs=41)
    types = [item["type"] for item in example["documents"]]
    assert len(types) == 41
    assert types.count("query") == 20 and types.count("document") == 21


def test_qdmatch_does_not_emit_counts_that_a_shrink_would_falsify():
    """``num_queries``/``num_docs`` were top-level pre-migration fields; the nested ladder drops
    items in place, so a shrunk rung would carry the counts of the example it came from."""
    example = qdmatch_example()
    assert "num_queries" not in example and "num_docs" not in example
    assert example["_meta"]["num_relevant"] == 3


def test_qdmatch_refuses_a_config_that_cannot_hold_its_relevant_queries():
    with pytest.raises(ValueError, match="num_relevant"):
        qdmatch_example(num_docs=8, num_relevant=6)


def test_qdmatch_skips_a_draw_whose_gold_outgrows_the_document_slots():
    """Two-gold questions at k=3 need six document slots. A draw that cannot fit them is skipped,
    not truncated -- a one-gold "multi-hop" example silently selects the single-doc wording."""
    assert qdmatch_example(ladder="qdmatch_hpqa", num_docs=8, num_relevant=3) is None


def test_qdmatch_jsonl_units_reject_a_one_based_gold_file(tmp_path):
    """
    The retrieval family is 0-based. A 1-based file read here would pair every query with the
    document one place along, and the result would validate perfectly.
    """
    import json

    from ctc.tasks.qdmatch.generate import units_from_jsonl

    path = tmp_path / "retrieval.jsonl"
    path.write_text(
        json.dumps(
            {
                "documents": [{"text": "a"}, {"text": "b"}],
                "queries": ["q"],
                "gold_doc_indices": [2],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="0-based"):
        units_from_jsonl(str(path))


def test_qdmatch_obliq_is_refused_by_name():
    """ObliQ was dropped from the qdmatch roster on 2026-07-19 and re-entered the suite as a
    standalone retrieval row. The spec's ``sources`` said otherwise for months."""
    from ctc.tasks.qdmatch.generate import load_pool

    with pytest.raises(ValueError, match="ObliQ"):
        load_pool(source="obliq")
    assert "obliq" not in registry.get("qdmatch").sources


def test_qdmatch_gold_documents_have_no_preferred_position():
    pool = pools.unit_pool()
    examples = [qdmatch_example(num_docs=40, seed=seed, corpus=pool) for seed in range(40)]
    probe = audit_mod.qd_pair_position_bias([e for e in examples if e], _spec("qdmatch_nq"))
    assert not probe.failed, probe


# ── grouping_labeled ────────────────────────────────────────────────────────────────────────────


def grouping_example(*, num_docs=22, index=0, seed=0, corpus=None, **overrides):
    """
    :param num_docs: Abstracts per example.
    :param index: Example counter; fixes the concept level.
    :param seed: RNG seed.
    :param corpus: An override pool.
    :param overrides: Further build knobs.

    :returns: One grouping_labeled example from the fixture OpenAlex pool.
    """
    gen = generators.get("grouping_labeled")
    return gen.build_example(
        random.Random(seed),
        corpus=corpus or pools.openalex_pool().for_split("train"),
        index=index,
        num_docs=num_docs,
        **overrides,
    )


def test_grouping_gold_partitions_every_document_exactly_once():
    """The metric is pairwise over a partition; a document in two groups or in none is not a
    partition, and the pairwise score would still return a plausible number."""
    example = grouping_example(num_docs=22)
    placed = [i for cluster in example["gold_doc_indices"] for i in cluster]
    assert sorted(placed) == list(range(22))


def test_grouping_cluster_labels_stay_parallel_to_their_groups():
    """
    ``build_labeled_target`` zips ``gold_doc_indices[i]`` with ``cluster_labels[i]``. Pushing gold
    through ``remap_groups`` -- which sorts the OUTER list -- would give every group some other
    group's name while leaving the partition, and therefore the score, untouched.
    """
    pool = pools.openalex_pool().for_split("train")
    by_id = {}
    for paper in pool.papers:
        by_id[f"Paper {paper.id[1:]} on {paper.concepts[0]}"] = paper
    example = grouping_example(corpus=pool)
    level = example["level"]
    for cluster, label in zip(example["gold_doc_indices"], example["cluster_labels"]):
        for i in cluster:
            paper = by_id[example["documents"][i]["title"]]
            assert paper.concepts[level] == label


def test_grouping_answers_match_the_specs_own_target_builder():
    """
    ``answers[0]`` is built from the finished example by the task's own target builder rather than
    assembled inline, so the stored answer cannot drift from what training sees.
    """
    from ctc.tasks._grouping import build_labeled_target

    example = grouping_example()
    assert example["answers"][0] == build_labeled_target(example)


def test_grouping_target_round_trips_through_the_spec_parser():
    spec = _spec("grouping_labeled")
    example = grouping_example(num_docs=22)
    parsed = spec.parse(example["answers"][0], 22)
    assert spec.score(parsed, example["gold_doc_indices"])["pairwise_f1"] == 1.0


def test_grouping_level_is_a_function_of_the_index_not_of_the_rng():
    """
    The trap-14 fix. Drawing the level from the RNG and retrying on failure let the realised level
    mix follow each level's accept rate, which drifts with N -- so "more documents" and "finer
    grouping" moved together along the exact axis the ladder varies.
    """
    pool = pools.openalex_pool().for_split("train")
    levels = [
        grouping_example(index=index, seed=index * 7, corpus=pool)["level"] for index in range(12)
    ]
    assert levels == [0, 1, 2, 3] * 3


def test_grouping_level_subset_is_honoured():
    pool = pools.openalex_pool().for_split("train")
    levels = {
        grouping_example(index=index, corpus=pool, levels="0,2")["level"] for index in range(6)
    }
    assert levels == {0, 2}


def test_grouping_k_is_clamped_to_what_the_coarse_level_can_actually_build():
    """
    OpenAlex L0 has 19 top-level fields, a ceiling more data cannot raise. Asking for K beyond it
    made the pre-migration builder return ``None`` and the retry loop refill from a finer level.
    """
    pool = pools.openalex_pool().for_split("train")
    index = pool.level_index(0, min_per_value=1)
    assert len(index) == 19
    for num_docs in (22, 87, 175):
        k = grouping_example(index=0, num_docs=num_docs, corpus=pool)["k"]
        assert 2 <= k <= 19, (num_docs, k)


def test_grouping_min_k_by_capacity_is_a_lower_bound_not_an_upper_one():
    """The counter-intuitive half of the fix: coarse levels fail by having too FEW groups to hold
    the documents, not too many."""
    from ctc.tasks.grouping_labeled.generate import min_k_by_capacity

    assert min_k_by_capacity([10, 8, 6], 15) == 2
    assert min_k_by_capacity([10, 8, 6], 24) == 3
    assert min_k_by_capacity([10, 8, 6], 30) == 0


def test_grouping_partition_respects_every_pool_capacity():
    """The other half: overflow is redistributed rather than the whole example being rejected,
    which is what made the rejection rate -- and so the level mix -- pool-shape dependent."""
    from ctc.tasks.grouping_labeled.generate import partition_with_capacity

    caps = [3, 20, 4, 9]
    sizes = partition_with_capacity(30, caps, random.Random(0), min_per=1)
    assert sizes is not None and sum(sizes) == 30
    assert all(1 <= size <= cap for size, cap in zip(sizes, caps))
    assert partition_with_capacity(100, caps, random.Random(0)) is None


def test_grouping_splits_train_and_eval_by_publication_year():
    """
    A 2019 paper and a 2023 paper on one narrow concept are near-duplicates; a random split puts
    both sides of such a pair across the boundary and the eval partition is one the model has
    effectively seen.
    """
    pool = pools.openalex_pool()
    assert all(paper.year < pool.eval_year_min for paper in pool.for_split("train").papers)
    assert all(paper.year >= pool.eval_year_min for paper in pool.for_split("eval").papers)


def test_grouping_an_empty_split_is_an_error_not_an_empty_pool():
    """
    Silently building from the wrong side is exactly the contamination the year split exists to
    prevent, so an empty side is raised rather than returned.
    """
    from ctc.data.sources.openalex import OpenAlexPool

    pool = pools.openalex_pool()
    all_held_out = OpenAlexPool(papers=pool.papers, eval_year_min=1900)
    with pytest.raises(ValueError, match="empty"):
        all_held_out.for_split("train")


def test_grouping_declares_itself_unnestable():
    gen = generators.get("grouping_labeled")
    assert not gen.shrink_safe and gen.build_ladder is None and not gen.nested_ladder


def test_openalex_needs_its_pool_path_named():
    """There is no default corpus. A build against the wrong compact file is not detectable from
    its output, and the ~300 GB works snapshot is deliberately not fetched here."""
    from ctc.data.sources import openalex

    with pytest.raises(ValueError, match="compact OpenAlex pool"):
        openalex.load_pool()


# ── the three ladders together ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ladder", ["reorder", "qdmatch_nq", "qdmatch_hpqa", "grouping_labeled"])
def test_a_built_example_validates_against_its_spec(ladder):
    spec = _spec(ladder)
    if ladder == "reorder":
        example = reorder_example()
    elif ladder == "grouping_labeled":
        example = grouping_example()
    else:
        example = qdmatch_example(ladder=ladder)
    validate(example, spec)


@pytest.mark.parametrize("ladder", ["reorder", "qdmatch_nq", "qdmatch_hpqa", "grouping_labeled"])
def test_every_ladder_row_covers_the_specs_declared_rungs(ladder):
    """A rung the spec declares and the ladder does not is a KeyError at build time, long after
    the config was written -- the defect contradiction shipped."""
    spec = _spec(ladder)
    assert set(ladders.rungs_for(ladder)) == set(spec.rungs)


def test_reorders_longest_target_fits_its_decode_budget():
    """
    Reorder is the one task whose ANSWER grows with the rung, and the failure it causes is
    indistinguishable from a modelling result: ``parse_permutation`` requires an exact permutation
    of ``1..n``, so a target that does not fit the budget parses as ``None`` and scores
    ``kendall_tau`` 0.0 at *every* example of that rung.

    The 32k target measures ~4.5 Qwen3 tokens per id (median 1057 tokens at n=233), which is why
    the pre-migration 1024 was not enough. Five tokens per id is that measurement with headroom;
    the bound is asserted rather than the measurement, so the tokenizer is not a test dependency.
    """
    spec = _spec("reorder")
    longest = ladders.docs_for_rung("reorder", "32k")
    assert spec.max_new_tokens >= 5 * longest, (spec.max_new_tokens, longest)


def test_groupings_longest_target_fits_its_decode_budget():
    """
    Same failure mode, but driven by the concept level rather than by the rung alone: an L3 example
    at the 32k rung asks for near-singleton clusters, so its target is one labelled group per
    document. Measured at n=175 the target's median is 1392 tokens and its MAXIMUM is 2597, so a
    2048 budget truncates exactly the finest examples of the longest rung.
    """
    spec = _spec("grouping_labeled")
    longest = ladders.docs_for_rung("grouping_labeled", "32k")
    assert spec.max_new_tokens >= 15 * longest, (spec.max_new_tokens, longest)


@pytest.mark.parametrize("ladder", ["reorder", "grouping_labeled"])
def test_an_unnestable_ladder_says_so_in_its_build_report(ladder):
    """The report is the only place a reader learns that rung-to-rung deltas on these two carry
    eval-set resampling noise on top of the length effect."""
    spec = _spec(ladder)
    # 30 books so the eval tenth holds 3, i.e. enough distinct starting offsets for 500 examples.
    corpus = pools.reorder_pool(books=30) if ladder == "reorder" else pools.openalex_pool()
    _, report = build.build_eval(
        ladder, spec, size=500, rungs=["2k", "4k"], corpus=corpus, config=None
    )
    assert any("independently" in note for note in report.notes), report.notes
