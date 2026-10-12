"""
The two pair-family generators ported together: ``strmatch`` and ``redundancy``.

They are tested in one file on purpose. The pair family shares its answer format, parser, scorer and
gold convention (:mod:`ctc.tasks._pairs`), so the only things left that distinguish its members are
*what relation gold stands in* and *where the criterion is stated* -- and both are the kind of
property a tidy-up can erase without breaking anything visible. A redundancy example that grew a
``queries`` entry, or a strmatch example whose criterion stopped varying, would still validate,
still score, and quietly be the other task.

Each test below corresponds to a defect that reached data:

* the criterion collapsing into a fixed instruction (or the reverse);
* the bag-of-words shortcut that solved the shipped strmatch ladder outright (200/200 at 2k);
* the generic shrink dissolving pair-shaped decoys, which restores that shortcut at every rung
  below the longest;
* a filler drawn from an abstract that already contributed a planted sentence -- an unlabelled
  second redundant pair, scored as a model error;
* gold counted from the wrong base.
"""

from __future__ import annotations

import random
from itertools import combinations

import pytest
from fixtures import pools

from ctc.data import audit as audit_mod
from ctc.data import ladders
from ctc.data.generators import base as generators
from ctc.data.gold import check_indices
from ctc.data.sources import pubmed_redundancy
from ctc.format import registry
from ctc.tasks import load_all
from ctc.tasks.redundancy.spec import build_query as redundancy_query
from ctc.tasks.strmatch import generate as strmatch


@pytest.fixture(scope="module", autouse=True)
def _tasks():
    load_all()


def longest_shared_run(first, second):
    """
    :param first: One string's words.
    :param second: The other's.

    :returns: The length of the longest run of words appearing consecutively, in order, in both.
    """
    best = 0
    for i in range(len(first)):
        for j in range(len(second)):
            length = 0
            while (
                i + length < len(first)
                and j + length < len(second)
                and first[i + length] == second[j + length]
            ):
                length += 1
            best = max(best, length)
    return best


def matching_pairs(example, relation, threshold):
    """
    :param example: A strmatch example.
    :param relation: ``"substring"`` or ``"wordset"``.
    :param threshold: The example's threshold.

    :returns: Every 1-based pair meeting the criterion, found by brute force rather than by trusting
        the construction.
    """
    words = [doc["text"].split() for doc in example["documents"]]
    found = set()
    for a, b in combinations(range(len(words)), 2):
        if relation == "substring":
            hit = longest_shared_run(words[a], words[b]) >= threshold
        else:
            hit = len(set(words[a]) & set(words[b])) >= threshold
        if hit:
            found.add((a + 1, b + 1))
    return found


def redundancy_example(index=0, seed=0, **overrides):
    """
    :param index: Example counter.
    :param seed: RNG seed.
    :param overrides: Per-example config overrides.

    :returns: One redundancy example built from the fixture pool.
    """
    generator = generators.get("redundancy")
    config = generator.config(**overrides)
    config["corpus"] = pools.redundancy_pool()
    config["index"] = index
    return generator.build_example(random.Random(seed), **config)


# ── the distinction the two tasks exist to keep ─────────────────────────────────────────────────


def test_strmatch_carries_its_criterion_per_example_and_redundancy_does_not():
    """
    THE load-bearing difference between the two ported tasks.

    strmatch's threshold is a property of the *example*: it is written into ``queries[0]`` and the
    prompt is the instruction followed by that sentence. redundancy's criterion is fixed by the
    instruction and its ``queries`` is empty. Collapsing either direction produces data that
    validates and scores, and silently turns one task into the other.
    """
    strmatch_spec, redundancy_spec = registry.get("strmatch"), registry.get("redundancy")
    three = strmatch.build_example(random.Random(0), num_docs=40, span_len=3)
    four = strmatch.build_example(random.Random(0), num_docs=40, span_len=4)

    assert len(three["queries"]) == 1
    assert three["queries"] != four["queries"], "the criterion must vary with the threshold"
    assert "3 consecutive words" in three["queries"][0]
    assert "4 consecutive words" in four["queries"][0]
    # ...and it must reach the prompt, after the instruction rather than before it.
    prompt = strmatch_spec.build_prompt(three)
    assert three["queries"][0] in prompt
    assert prompt.index(strmatch_spec.instruction) < prompt.index(three["queries"][0])

    example = redundancy_example()
    assert (
        example["queries"] == []
    ), "redundancy's ask is the instruction; a query would be new data"
    assert redundancy_query(example) == redundancy_spec.instruction
    # The proof it is *fixed*: an example with a query planted in it still asks the same question,
    # and the planted text reaches neither the query builder nor the prompt.
    planted = dict(example, queries=["Find all pairs sharing at least 9 words."])
    assert redundancy_query(planted) == redundancy_spec.instruction
    assert "at least 9 words" not in redundancy_spec.build_prompt(planted)


def test_both_tasks_share_the_pair_family_contract():
    """Same gold shape, same base, same primary metric -- which is what makes them comparable."""
    for name in ("strmatch", "redundancy"):
        spec = registry.get(name)
        assert spec.gold_index_base == 1
        assert spec.primary_metric == "f1"
        assert spec.answer_is_set


# ── strmatch: exactness and the shortcut ────────────────────────────────────────────────────────


@pytest.mark.parametrize("relation,threshold", [("substring", 3), ("substring", 4), ("wordset", 3)])
def test_strmatch_plants_exactly_the_gold_pairs_and_no_others(relation, threshold):
    """
    Exactness is by construction, so nothing checks it at build time -- an O(N^2) verification per
    example would cost more than the build. Here it is brute-forced at a size where that is cheap.
    """
    kwargs = {"span_len": threshold} if relation == "substring" else {"share_words": threshold}
    if relation == "wordset":
        kwargs["num_scattered"] = 0
    for seed in range(6):
        example = strmatch.build_example(
            random.Random(seed), num_docs=44, num_pairs=3, relation=relation, **kwargs
        )
        gold = {tuple(pair) for pair in example["gold_doc_indices"]}
        assert len(gold) == 3
        assert matching_pairs(example, relation, threshold) == gold


def test_strmatch_scattered_decoys_share_words_without_sharing_a_run():
    """
    The whole mechanism: a decoy's bag-of-words overlap meets or beats gold's while its longest
    shared run is one word, so counting shared words no longer separates them.
    """
    example = strmatch.build_example(random.Random(3), num_docs=44, num_scattered=8)
    words = [doc["text"].split() for doc in example["documents"]]
    gold = {tuple(pair) for pair in example["gold_doc_indices"]}
    overlaps = {
        (a + 1, b + 1): len(set(words[a]) & set(words[b]))
        for a, b in combinations(range(len(words)), 2)
    }
    ties = [pair for pair, size in overlaps.items() if size >= 3 and pair not in gold]
    assert ties, "no non-gold pair reaches the gold overlap; the shortcut is wide open"
    for pair in ties:
        assert longest_shared_run(words[pair[0] - 1], words[pair[1] - 1]) < 3


def test_the_bag_of_words_shortcut_is_closed_and_the_pre_migration_setting_reopens_it():
    """
    Both halves matter. Without the second, a future ``num_scattered=0`` -- or a rewrite that drops
    the decoys as "redundant with the hard negatives" -- would leave this test green.
    """
    spec = registry.get("strmatch")
    fixed = [strmatch.build_example(random.Random(i), num_docs=44) for i in range(25)]
    premigration = [
        strmatch.build_example(random.Random(i), num_docs=44, num_scattered=0) for i in range(25)
    ]
    assert audit_mod.overlap_pair_is_gold(premigration, spec).score == 1.0
    assert audit_mod.overlap_pair_is_gold(fixed, spec).score <= 0.1
    assert not audit_mod.overlap_pair_is_gold(fixed, spec).failed


def test_wordset_refuses_scattered_decoys():
    """Under ``wordset`` a scattered decoy meets the criterion, so it would be unlabelled gold."""
    with pytest.raises(ValueError, match="wordset"):
        strmatch.build_example(random.Random(0), num_docs=44, relation="wordset", num_scattered=4)


def test_a_defaulted_decoy_count_shrinks_to_fit_but_an_explicit_one_does_not():
    """
    A tiny corpus degrades to fewer decoys and says so in ``_meta``; an explicit count that does not
    fit is an error, because silently building something smaller is how a defence disappears.
    """
    small = strmatch.build_example(random.Random(0), num_docs=9)
    assert len(small["documents"]) == 9
    assert small["_meta"]["num_scattered"] == 0
    with pytest.raises(ValueError, match="cannot hold"):
        strmatch.build_example(random.Random(0), num_docs=9, num_scattered=6)


# ── ladders: the shrink these two tasks cannot use ──────────────────────────────────────────────


@pytest.mark.parametrize("task", ["strmatch", "redundancy"])
def test_the_ladder_is_nested_and_grades_the_same_gold_at_every_rung(task):
    generator = generators.get(task)
    assert generator.build_ladder is not None, f"{task} must not use the generic shrink"
    assert not generator.shrink_safe
    spec = registry.get(task)
    sizes = {label: ladders.docs_for_rung(task, label) for label in ("2k", "4k", "8k")}
    config = generator.config()
    if generator.corpus is not None:
        config["corpus"] = pools.redundancy_pool()
    row = generator.build_ladder(random.Random(4), index=0, rungs=sizes, **config)

    labels = list(sizes)
    for label in labels:
        assert len(row[label]["documents"]) == sizes[label]
        check_indices(row[label]["gold_doc_indices"], sizes[label], base=spec.gold_index_base)
    for shorter, longer in zip(labels, labels[1:]):
        short = {doc["text"] for doc in row[shorter]["documents"]}
        long = {doc["text"] for doc in row[longer]["documents"]}
        assert short <= long, "a shorter rung holds documents the longer one does not"

    def gold_texts(example):
        docs = example["documents"]
        return sorted(
            tuple(sorted(docs[i - spec.gold_index_base]["text"] for i in pair))
            for pair in example["gold_doc_indices"]
        )

    assert len({tuple(gold_texts(row[label])) for label in labels}) == 1


def test_every_strmatch_rung_keeps_its_contiguity_decoys():
    """
    What the generic shrink destroyed. Measured on the first build attempt: shrinking the 32k rung
    down to 2k scored 0.935 on ``overlap_pair_is_gold`` against 0.005 for rungs built this way.
    """
    generator = generators.get("strmatch")
    sizes = {label: ladders.docs_for_rung("strmatch", label) for label in ("2k", "4k", "8k")}
    rows = [
        generator.build_ladder(random.Random(i), index=i, rungs=sizes, **generator.config())
        for i in range(20)
    ]
    spec = registry.get("strmatch")
    for label in sizes:
        examples = [row[label] for row in rows]
        assert all(
            ex["_meta"]["num_scattered"] >= 2 * ex["_meta"].get("num_hardneg", 0) for ex in examples
        )
        assert audit_mod.overlap_pair_is_gold(examples, spec).score <= 0.1, label


def test_every_redundancy_rung_keeps_all_of_its_hard_negatives():
    """
    The hard negatives ARE the task: without them "which two sentences are about one topic?" is a
    sufficient answer. A random shrink halves them, so the shorter rungs would get *easier* for a
    reason that is not context length.
    """
    generator = generators.get("redundancy")
    sizes = {label: ladders.docs_for_rung("redundancy", label) for label in ("2k", "4k", "8k")}
    config = dict(generator.config(), corpus=pools.redundancy_pool())
    row = generator.build_ladder(random.Random(1), index=2, rungs=sizes, **config)
    for label, example in row.items():
        assert len(example["hard_neg_indices"]) == 2 * config["num_hardneg"], label
        texts = {example["documents"][i - 1]["text"] for i in example["hard_neg_indices"]}
        assert len(texts) == 2 * config["num_hardneg"]


# ── redundancy: the false-negative control and the pool ─────────────────────────────────────────


def test_redundancy_fillers_never_come_from_an_abstract_that_planted_a_sentence():
    """
    A filler from a gold pair's own abstract may restate the fact the pair states -- an unlabelled
    second redundant pair, scored as a model error every time a model finds it. The control covers
    the hard-negative abstracts too, which the pre-migration generator also did.
    """
    pool = pools.redundancy_pool()
    generator = generators.get("redundancy")
    for index in range(5):
        example = generator.build_example(
            random.Random(index), **generator.config(num_docs=60), corpus=pool, index=index
        )
        planted = {
            *(p.claim for p in pool.pairs[index * 3 : index * 3 + 3]),
            *(p.paraphrase for p in pool.pairs[index * 3 : index * 3 + 3]),
            *(h.first for h in pool.hardnegs[index * 6 : index * 6 + 6]),
            *(h.second for h in pool.hardnegs[index * 6 : index * 6 + 6]),
        }
        used = {p.abstract_id for p in pool.pairs[index * 3 : index * 3 + 3]}
        used |= {h.abstract_id for h in pool.hardnegs[index * 6 : index * 6 + 6]}
        banned = {text for aid in used for text in pool.fillers.get(aid, ())}
        fillers = {doc["text"] for doc in example["documents"]} - planted
        assert not (fillers & banned)


def test_redundancy_hard_negative_indices_point_at_the_planted_pairs():
    """They are 1-based, like gold: ``shrink`` renumbers them with the spec's own base."""
    pool = pools.redundancy_pool()
    generator = generators.get("redundancy")
    example = generator.build_example(
        random.Random(0), **generator.config(num_docs=40), corpus=pool, index=0
    )
    planted = {h.first for h in pool.hardnegs[:6]} | {h.second for h in pool.hardnegs[:6]}
    assert min(example["hard_neg_indices"]) >= 1
    assert {example["documents"][i - 1]["text"] for i in example["hard_neg_indices"]} == planted
    gold_positions = {i for pair in example["gold_doc_indices"] for i in pair}
    assert not gold_positions & set(example["hard_neg_indices"])


def test_the_hard_negative_pool_takes_the_hardest_pair_each_abstract_offers():
    """
    The lever that narrows the lexical gap between gold (a paraphrase, high overlap) and the
    decoys. ``order="random"`` is the pre-migration behaviour and must stay reachable.
    """
    fillers = {
        "a": (
            "Serum ferritin increased by 30 percent in treated patients.",
            "Serum ferritin increased by 12 percent in treated patients.",
            "Enrollment was greater than expected across the four centres.",
        )
    }
    hardest = pubmed_redundancy.build_hardneg_pool(fillers, random.Random(0), order="overlap")
    assert len(hardest) == 1
    assert {hardest[0].first, hardest[0].second} == set(fillers["a"][:2])
    assert hardest[0].overlap > 0.5

    with pytest.raises(ValueError, match="order must be"):
        pubmed_redundancy.build_hardneg_pool(fillers, random.Random(0), order="closest")


def test_the_pool_splits_its_hard_negatives_as_well_as_its_gold():
    """
    A hard-negative pair is a question too. Sharing one between train and eval tells the model the
    answer for that pair, and every surface contamination check still passes.
    """
    pool = pools.redundancy_pool()
    train, evalset = pool.for_split("train"), pool.for_split("eval")
    assert not set(train.pairs) & set(evalset.pairs)
    assert not set(train.hardnegs) & set(evalset.hardnegs)
    assert len(train.hardnegs) + len(evalset.hardnegs) == len(pool.hardnegs)
    with pytest.raises(ValueError, match="split must be"):
        pool.for_split("holdout")
