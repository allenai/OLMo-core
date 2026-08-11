"""
Byte-parity between :mod:`ctc.format` and the pre-migration implementation.

``fixtures/golden_format.json`` was produced by running the old ``corpus_reasoning.lib`` code -- see
``fixtures/generate_golden.py``. It is the contract these modules must not break, because the same
functions both build training shards and grade checkpoints: a formatting change makes new data
incompatible with old checkpoints, and a parser change silently reprices published results.

A failure here means the port changed behaviour. Fix the port. Regenerating the fixture would
delete the evidence instead of the bug.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ctc.format import documents, metrics, parsing, prompts
from fixtures.generate_golden import DOC_CASES  # made importable by tests/conftest.py

GOLDEN = json.loads((Path(__file__).parents[1] / "fixtures" / "golden_format.json").read_text())


# ── Prompt constants ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(GOLDEN["prompt_constants"]))
def test_prompt_constants_are_unchanged(name):
    """Every template and instruction string is byte-identical to the pre-migration one."""
    if not hasattr(prompts, name):
        pytest.skip(f"{name} intentionally not ported (see documents.py)")
    assert getattr(prompts, name) == GOLDEN["prompt_constants"][name]


def test_no_prompt_constant_was_dropped_silently():
    """Only the two doc-formatting helpers moved out; every string constant stayed."""
    moved = {"format_doc", "format_doc_dict"}
    missing = {n for n in GOLDEN["prompt_constants"] if not hasattr(prompts, n)} - moved
    assert not missing, f"prompt constants lost in the port: {sorted(missing)}"


@pytest.mark.parametrize("top_k", sorted(GOLDEN["rerank_instruction"]))
def test_rerank_instruction(top_k):
    assert prompts.rerank_instruction(int(top_k)) == GOLDEN["rerank_instruction"][top_k]


# ── Document serialization ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", sorted(GOLDEN["documents"]))
def test_format_documents_matches_golden(key):
    """The if-chain became a dispatch table; the bytes it emits did not move."""
    task, titles = key.split("|")
    use_titles = titles == "use_titles=True"
    assert (
        documents.format_documents(DOC_CASES[task], task, use_titles=use_titles)
        == GOLDEN["documents"][key]
    )


def test_every_task_branch_is_covered():
    """The fixture must exercise every serializer, or the table could lose one unnoticed."""
    uncovered = set(documents._SERIALIZERS) - set(DOC_CASES)
    assert not uncovered, f"serializers with no golden case: {sorted(uncovered)}"


def test_items_are_blank_line_separated():
    """The chunked/landmark masks split on a blank line; single newlines put items in the wrong chunk."""
    docs = [{"text": "one"}, {"text": "two"}, {"text": "three"}]
    for task in list(documents._SERIALIZERS) + ["retrieval", "unknown"]:
        if task in ("qdmatch", "xabsence", "grouping", "grouping_labeled"):
            continue  # need extra keys; covered by the golden cases
        out = documents.format_documents(docs, task)
        assert out.count("\n\n") == 2, f"{task} did not separate 3 items with blank lines: {out!r}"


def test_reorder_collapses_internal_blank_lines():
    """A passage's own paragraph break would otherwise split it across two chunks."""
    out = documents.format_documents([{"text": "a\n\nb"}, {"text": "c"}], "reorder")
    assert out == "Passage [1]: a\nb\n\nPassage [2]: c"


def test_default_serializer_numbers_only_id_tasks():
    docs = [{"title": "T", "text": "x"}]
    assert "[1]" in documents.format_documents(docs, "retrieval")
    assert "[1]" not in documents.format_documents(docs, "qa")


# ── Parsers ─────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", sorted(GOLDEN["parse_doc_ids"]))
def test_parse_doc_ids(text):
    assert sorted(parsing.parse_doc_ids(text)) == GOLDEN["parse_doc_ids"][text]


def test_parse_doc_ids_accepts_a_primed_continuation():
    """The regression that scored niah 0.16 while msmarco scored 0.96 on the same ability."""
    assert parsing.parse_doc_ids("8]") == {8}


@pytest.mark.parametrize("key", sorted(GOLDEN["parse_outlier_ids"]))
def test_parse_outlier_ids(key):
    text, n = key.rsplit("|", 1)
    assert parsing.parse_outlier_ids(text, int(n)) == GOLDEN["parse_outlier_ids"][key]


def test_parse_outlier_ids_stays_strict_about_brackets():
    """Outlier targets emit a full 'Outliers: [1]' line, so a bare int is prose, not an answer.

    Relaxing this the way parse_doc_ids is relaxed would start matching the star rating in the
    preceding reasoning sentence.
    """
    assert parsing.parse_outlier_ids("Most are 5 star. Outliers: 3", 10) is None


@pytest.mark.parametrize("text", sorted(GOLDEN["parse_pairs"]))
def test_parse_pairs(text):
    assert parsing.parse_pairs(text) == GOLDEN["parse_pairs"][text]


def test_parse_pairs_recovers_a_dropped_primer():
    """Contradiction EM read ~0.60 against a true >0.9 because the first pair was dropped."""
    assert parsing.parse_pairs("1, 37], [6, 60], [35, 71]]") == [[1, 37], [6, 60], [35, 71]]


def test_parse_pairs_sorts_each_pair():
    """'1 contradicts 4' and '4 contradicts 1' are the same claim."""
    assert parsing.parse_pairs("[[4, 1]]") == [[1, 4]]


def test_parse_pairs_distinguishes_empty_from_unparseable():
    """'there are none' and 'produced nothing usable' score alike but mean opposite things."""
    assert parsing.parse_pairs("[]") == []
    assert parsing.parse_pairs("not a pair anywhere") is None


@pytest.mark.parametrize("text", sorted(GOLDEN["parse_qd_pairs"]))
def test_parse_qd_pairs(text):
    assert parsing.parse_qd_pairs(text) == GOLDEN["parse_qd_pairs"][text]


def test_qd_pairs_preserve_order():
    """(query_id, doc_id) over one shared index -- swapping them is a different claim."""
    assert parsing.parse_qd_pairs("[[8, 1]]") == [[8, 1]]
    assert parsing.parse_pairs("[[8, 1]]") == [[1, 8]]


@pytest.mark.parametrize("key", sorted(GOLDEN["pair_metrics"]))
def test_pair_metrics(key):
    pred, gold = key.split("|")
    got = metrics.pair_metrics(json.loads(pred), json.loads(gold))
    for k, want in GOLDEN["pair_metrics"][key].items():
        assert got[k] == pytest.approx(want), k


def test_pair_metrics_perfect_on_mutual_empty():
    assert metrics.pair_metrics([], [])["f1"] == 1.0


@pytest.mark.parametrize("text", sorted(GOLDEN["parse_cycles"]))
def test_parse_cycles(text):
    assert parsing.parse_cycles(text) == GOLDEN["parse_cycles"][text]


def test_cycles_are_normalized_to_sorted_sets():
    """A cycle is a set; the order the model walked it in carries no information."""
    assert parsing.parse_cycles("[[8, 3, 12]]") == [[3, 8, 12]]
    assert parsing.parse_cycles("[[3, 3, 8]]") == [[3, 8]]


def test_singleton_groups_are_dropped():
    """One item cannot form a cycle; admitting them lets a model score by listing every id."""
    assert parsing.parse_cycles("[[5]]") == []


def test_a_flat_list_is_read_as_one_cycle():
    assert parsing.parse_cycles("[3, 8, 12]") == [[3, 8, 12]]


@pytest.mark.parametrize("key", sorted(GOLDEN["cycle_metrics"]))
def test_cycle_metrics(key):
    pred, gold = key.split("|")
    got = metrics.cycle_metrics(json.loads(pred), json.loads(gold))
    for k, want in GOLDEN["cycle_metrics"][key].items():
        assert got[k] == pytest.approx(want), k


def test_partial_cycle_scores_zero_at_cycle_level_but_earns_claim_credit():
    """The two numbers answer different questions: found the right items vs grouped them right."""
    got = metrics.cycle_metrics([[3, 8]], [[3, 8, 12]])
    assert got["f1"] == 0.0
    assert got["claim_f1"] > 0.0


@pytest.mark.parametrize("key", sorted(GOLDEN["parse_partition"]))
def test_parse_partition(key):
    text, n = key.rsplit("|", 1)
    assert parsing.parse_partition(text, int(n)) == GOLDEN["parse_partition"][key]


def test_parse_partition_ignores_trailing_ramble():
    """The greedy-regex bug that roughly halved pairwise F1."""
    out = parsing.parse_partition('{"groups": [{"doc_ids": [1, 2]}]} then it kept talking {', 2)
    assert out == [[1, 2]]


def test_parse_partition_recovers_a_mid_array_start():
    """Chunked grouping @2k measured 0.44 without this recovery and 0.82 with it."""
    assert parsing.parse_partition('2, 3, 4]}, {"doc_ids": [1, 6]}]}', 6) == [[2, 3, 4], [1, 6]]


@pytest.mark.parametrize("key", sorted(GOLDEN["partition_to_labels"]))
def test_partition_to_labels(key):
    text, n = key.rsplit("|", 1)
    n = int(n)
    clusters = parsing.parse_partition(text, n) or []
    assert parsing.partition_to_labels(clusters, n) == GOLDEN["partition_to_labels"][key]


def test_partition_to_labels_gives_omitted_docs_their_own_cluster():
    assert parsing.partition_to_labels([[1, 2]], 4) == [0, 0, 1, 2]


@pytest.mark.parametrize("key", sorted(GOLDEN["parse_permutation"]))
def test_parse_permutation(key):
    text, n = key.rsplit("|", 1)
    assert parsing.parse_permutation(text, int(n)) == GOLDEN["parse_permutation"][key]


def test_parse_permutation_rejects_a_near_miss():
    assert parsing.parse_permutation("[1, 1, 2]", 3) is None


# ── Metrics ─────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", sorted(GOLDEN["normalize_answer"]))
def test_normalize_answer(text):
    assert metrics.normalize_answer(text) == GOLDEN["normalize_answer"][text]


@pytest.mark.parametrize("key", sorted(GOLDEN["qa_metrics"]))
def test_qa_metrics(key):
    pred, gold = key.rsplit("|", 1)
    want = GOLDEN["qa_metrics"][key]
    assert metrics.exact_match(pred, gold) == want["exact_match"]
    assert metrics.substring_match(pred, gold) == want["substring_match"]
    assert metrics.token_f1(pred, gold) == pytest.approx(want["token_f1"])


@pytest.mark.parametrize("key", sorted(GOLDEN["max_over_answers"]))
def test_max_over_answers(key):
    pred, answers = key.rsplit("|", 1)
    got = metrics.max_over_answers(metrics.token_f1, pred, json.loads(answers))
    assert got == pytest.approx(GOLDEN["max_over_answers"][key])


@pytest.mark.parametrize("key", sorted(GOLDEN["retrieval_metrics"]))
def test_retrieval_metrics(key):
    pred, gold = key.split("|")
    p, g = set(json.loads(pred)), set(json.loads(gold))
    want = GOLDEN["retrieval_metrics"][key]
    assert metrics.retrieval_exact_match(p, g) == want["exact_match"]
    assert metrics.retrieval_recall(p, g) == pytest.approx(want["recall"])
    assert metrics.retrieval_precision(p, g) == pytest.approx(want["precision"])
    assert metrics.retrieval_f1(p, g) == pytest.approx(want["f1"])


def test_empty_prediction_scores_zero_precision():
    """Abstaining must not be a way to score well."""
    assert metrics.retrieval_precision(set(), {1}) == 0.0


def test_aggregate_raises_on_a_missing_key():
    """Averaging over a subset would misreport eval_size."""
    with pytest.raises(KeyError):
        metrics.aggregate([{"f1": 1.0}, {}], ["f1"])
