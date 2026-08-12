"""
``oolong`` -- aggregate a question over a log of labelled items, one of the five in-distribution
ladders.

The odd one out of the suite in three ways, all consequences of one fact: **its items are lines
inside a single document, not separate documents.**

1. **The scaling axis is a token budget**, not a document count. Items are drawn until their
   cumulative token count reaches the target, so ``scaling_param`` is ``target_tokens`` and the
   rung table holds token budgets.
2. **The rungs are built independently.** The gold is *recomputed* over whichever items were drawn
   -- a different item set has a different most-frequent label -- so no shrink can preserve the
   answer, and the rungs cannot grade the same question. The generator declares this
   (``shrink_safe=False``) rather than letting an audit discover it, and the cost is that
   rung-to-rung deltas carry eval-set resampling noise the other four ladders do not have.
3. **Its chunk layout is line-based** (``spec.extra["chunk_by"] = "line"``), which is why the
   document-chunked converter needs an item regex for this task and no other.

.. warning::
   **The gold is only trustworthy because every variant is exactly recomputable.** The nine
   variants below are exactly those whose answer follows from per-item ``(date, user, label)`` with
   no inferred threshold; a draw where the argmax ties is *degenerate* and is retried rather than
   resolved arbitrarily, because a tie has two correct answers and one label.
"""

from __future__ import annotations

import random
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from ...data.generators.base import Generator
from ...data.schema import make_document, make_example
from ...data.sources import oolong as source

__all__ = ["VARIANTS", "build_variant", "build_example", "GENERATOR"]

#: ``(task group, answer type, builder key)``. Restricted to the exactly-recomputable variants --
#: counting, per-user and per-month aggregates over labels the items carry.
VARIANTS: Tuple[Tuple[str, str, str], ...] = (
    ("counting", "ANSWER_TYPE.LABEL", "count_mostfreq_label"),
    ("counting", "ANSWER_TYPE.NUMERIC", "count_numeric"),
    ("counting", "ANSWER_TYPE.COMPARISON", "count_comparison"),
    ("user", "ANSWER_TYPE.USER", "user_mostfreq_user"),
    ("user", "ANSWER_TYPE.LABEL", "user_mostfreq_label"),
    ("user", "ANSWER_TYPE.NUMERIC", "user_numeric"),
    ("user", "ANSWER_TYPE.COMPARISON", "user_comparison"),
    ("timeline", "ANSWER_TYPE.NUMERIC", "timeline_repr_n"),
    ("timeline", "ANSWER_TYPE.LABEL", "timeline_month_label"),
)

_COMPARISON = {1: "more common than", -1: "less common than", 0: "same frequency as"}

#: Tokens held back for the question, the chat template and the answer, so an example lands inside
#: its rung rather than a little over it.
QUESTION_RESERVE = 220

#: Fewer items than this and the aggregate questions stop being questions.
MIN_ITEMS = 4


def _compare(a: int, b: int) -> str:
    return _COMPARISON[(a > b) - (a < b)]


def _unique_argmax(counter: Counter) -> Optional[str]:
    """
    :param counter: Counts.

    :returns: The single most common key, or ``None`` on an empty or **tied** counter. A tie is not
        broken: it would give the example two correct answers and one label.
    """
    if not counter:
        return None
    top = counter.most_common(2)
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def build_variant(
    key: str, items: Sequence[source.Item], labels: Sequence[str], rng: random.Random
) -> Optional[Tuple[str, str]]:
    """
    Recompute one variant's gold over the drawn items.

    :param key: Builder key from :data:`VARIANTS`.
    :param items: The drawn items.
    :param labels: Canonical label order for this sub-dataset, so the rendered option list matches
        the benchmark's rather than our sort order.
    :param rng: Seeded RNG.

    :returns: ``(question, gold)``, or ``None`` when the draw is degenerate and the caller should
        retry.

    :raises ValueError: On an unknown variant key.
    """
    present = [label for label in labels if any(it.label == label for it in items)]
    if not present:
        return None

    if key == "count_mostfreq_label":
        gold = _unique_argmax(Counter(it.label for it in items))
        if gold is None:
            return None
        return (
            "In the above data, which of the labels is the most common? Give your final answer in "
            f"the form 'Label: answer' where answer is one of the labels: {', '.join(present)}.",
            gold,
        )

    if key == "count_numeric":
        label = rng.choice(present)
        return (
            f"In the above data, how many data points should be classified as label '{label}'? "
            "Give your final answer in the form 'Answer: number'.",
            str(sum(1 for it in items if it.label == label)),
        )

    if key == "count_comparison":
        if len(present) < 2:
            return None
        a, b = rng.sample(present, 2)
        return (
            f"In the above data, is label '{a}' more common, less common, or the same frequency "
            f"as label '{b}'? Give your final answer in the form 'Answer: {a} is [X] {b}', where "
            "[X] is 'more common than', 'less common than', or 'same frequency as'.",
            _compare(
                sum(1 for it in items if it.label == a), sum(1 for it in items if it.label == b)
            ),
        )

    if key == "user_mostfreq_user":
        gold = _unique_argmax(Counter(it.user for it in items))
        if gold is None:
            return None
        return (
            "In the above data, which user is represented most often? Give your final answer in "
            "the form 'User: [X]', where [X] is the user ID.",
            str(gold),
        )

    if key in ("user_mostfreq_label", "user_numeric", "user_comparison"):
        counts = Counter(it.user for it in items)
        # Prefer a user with at least two items, so the subset question is not trivially one row.
        candidates = [u for u, c in counts.items() if c >= 2] or list(counts)
        user = rng.choice(candidates)
        subset = [it for it in items if it.user == user]
        subset_labels = [label for label in labels if any(it.label == label for it in subset)]
        prefix = (
            "For the following question, only consider the subset of instances that are "
            f"associated with user IDs {user}. Among instances associated with these users, "
        )
        if key == "user_mostfreq_label":
            gold = _unique_argmax(Counter(it.label for it in subset))
            if gold is None:
                return None
            return (
                prefix
                + "which of the labels is the most common? Give your final answer in the form "
                f"'Label: answer' where answer is one of the labels: {', '.join(subset_labels)}.",
                gold,
            )
        if key == "user_numeric":
            label = rng.choice(subset_labels)
            return (
                prefix
                + f"how many data points should be classified as label '{label}'? Give your final "
                "answer in the form 'Answer: number'.",
                str(sum(1 for it in subset if it.label == label)),
            )
        if len(subset_labels) < 2:
            return None
        a, b = rng.sample(subset_labels, 2)
        return (
            prefix
            + f"is label '{a}' more common, less common, or the same frequency as label '{b}'? "
            f"Give your final answer in the form 'Answer: {a} is [X] {b}', where [X] is 'more "
            "common than', 'less common than', or 'same frequency as'.",
            _compare(
                sum(1 for it in subset if it.label == a), sum(1 for it in subset if it.label == b)
            ),
        )

    if key == "timeline_repr_n":
        n = rng.choice([1, 2])
        counts = Counter(it.date for it in items)
        plural = "time" if n == 1 else "times"
        return (
            f"In the above data, how many dates are represented exactly {n} {plural}? Give your "
            f"final answer in the form 'Answer: [X]', where [X] is the number of dates represented "
            f"exactly {n} {plural}.",
            str(sum(1 for _, c in counts.items() if c == n)),
        )

    if key == "timeline_month_label":
        months = sorted({it.month for it in items})
        if not months:
            return None
        month = rng.choice(months)
        name = datetime(2000, month, 1).strftime("%B")
        subset = [it for it in items if it.month == month]
        subset_labels = [label for label in labels if any(it.label == label for it in subset)]
        gold = _unique_argmax(Counter(it.label for it in subset))
        if gold is None:
            return None
        return (
            "For the following question, only consider the subset of instances that occur in "
            f"{name} of any year. Among instances occuring in {name}, which of the labels is the "
            "most common? Give your final answer in the form 'Label: answer' where answer is one "
            f"of the labels: {', '.join(subset_labels)}.",
            gold,
        )

    raise ValueError(f"unknown variant key {key!r}")


def build_example(
    rng: random.Random,
    *,
    corpus: source.OolongPool,
    target_tokens: int,
    min_tokens: int = 300,
    max_retries: int = 40,
) -> Optional[Dict]:
    """
    Build one OOLONG example at a sampled token budget.

    :param rng: Seeded RNG.
    :param corpus: The item pool. Drawn from, not consumed -- an item can appear in many examples --
        so this generator takes no example index.
    :param target_tokens: Upper bound of the per-example budget -- the rung. The actual budget is
        drawn uniformly in ``[min_tokens, target_tokens]``, so length is continuous within a rung
        rather than quantised to it.
    :param min_tokens: Lower bound of that draw.
    :param max_retries: Degenerate-draw retries before falling back to ``count_numeric``, whose
        answer is always defined.

    :returns: The example, or ``None`` when even the fallback was degenerate.
    """
    dataset = rng.choice(corpus.datasets)
    items = corpus.items[dataset]
    if len(items) < MIN_ITEMS:
        return None
    labels = corpus.labels[dataset]

    target = rng.randint(min(min_tokens, target_tokens), target_tokens)
    budget = max(min_tokens, target - corpus.preamble_tokens[dataset] - QUESTION_RESERVE)

    order = list(range(len(items)))
    rng.shuffle(order)
    chosen: List[int] = []
    used_tokens = 0
    for position in order:
        chosen.append(position)
        used_tokens += items[position].tokens + 1  # +1 for the joining newline
        if used_tokens >= budget:
            break
    for position in order[len(chosen) :]:
        if len(chosen) >= MIN_ITEMS:
            break
        chosen.append(position)
    if len(chosen) < MIN_ITEMS:
        return None

    drawn = [items[i] for i in chosen]
    built = None
    for _ in range(max_retries):
        group, answer_type, key = rng.choice(VARIANTS)
        result = build_variant(key, drawn, labels, rng)
        if result is not None:
            built = (group, answer_type, result)
            break
    if built is None:
        result = build_variant("count_numeric", drawn, labels, rng)
        if result is None:
            return None
        built = ("counting", "ANSWER_TYPE.NUMERIC", result)

    group, answer_type, (question, gold) = built
    preamble = corpus.preamble[dataset].format(N=len(drawn))
    text = preamble + "\n\n" + "\n".join(it.line for it in drawn)
    return make_example(
        documents=[make_document(text)],
        queries=[question],
        answers=[gold],
        source=f"oolong_{group}",
        # Declared and always empty: oolong's items are lines, not documents, so there is no
        # document to point at. The audit knows not to "verify" it.
        gold=[],
        meta={
            "answer_type": answer_type,
            "task_group": group,
            "gold_list": [gold],
            "context_len": corpus.preamble_tokens[dataset] + sum(it.tokens + 1 for it in drawn),
            "dataset": dataset,
            "num_items": len(drawn),
        },
    )


GENERATOR = Generator(
    name="oolong",
    task="oolong",
    source="oolong",
    build_example=build_example,
    defaults={"target_tokens": 8192, "min_tokens": 300, "max_retries": 40},
    scaling_param="target_tokens",
    shrink_safe=False,  # gold is recomputed over the drawn items; no shrink can preserve it
    corpus=source.load_pool,
    corpus_defaults={
        "hf_name": "oolongbench/oolong-synth",
        "max_context": 131_072,
        "tokenizer": "Qwen/Qwen3-0.6B",
        "seed": 42,
    },
    notes="line-based items in one document; rungs are TOKEN budgets and are built independently",
)
