"""
Shared machinery for the pair tasks.

Five tasks -- contradiction, redundancy, mathmatch, matching_ngram, strmatch -- ask the same
structural question: *which pairs of items in this corpus stand in relation R?* They differ only in
what R is (contradicting, redundant, arithmetically close, textually identical, string-similar) and
in how each item is rendered. The answer format, the parser, the scorer and the gold convention are
identical.

The pre-migration code already recognised this -- ``TASK_CFG`` routes redundancy, strmatch and
mathmatch to ``_eval_contradiction`` -- but the recognition was implicit, buried in a scorer lookup.
Making it explicit here means the family cannot drift apart one member at a time, which is the exact
failure that produced the duplicated-parser bugs recorded in :mod:`ctc.format.parsing`.

Lives at ``tasks/`` root rather than in any one task's directory, because it belongs to none of
them. Each task still gets its own package.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, Optional, Sequence

from ..format import assemble, metrics, parsing
from ..format.prompts import GENERIC_INSTRUCTION
from ..format.registry import TaskSpec

__all__ = ["check_gold", "parse", "score", "make_pair_spec"]


def check_gold(gold: Sequence[Sequence[int]]) -> None:
    """
    Assert the invariant the set comparison depends on.

    :param gold: Gold pairs.

    :raises ValueError: If a pair is not sorted low-high. Predicted pairs are sorted and scoring is
        a set intersection, so such a pair could never match and would silently cost recall on
        every example containing it.
    """
    for p in gold:
        if len(p) == 2 and p[0] > p[1]:
            raise ValueError(
                f"gold pair {list(p)} is not sorted low-high; it could never be matched, and would "
                "silently cost recall on every example that contains it"
            )


def parse(text: str, n_docs: Optional[int] = None):
    """
    Parse a pair answer.

    :param text: Raw model generation.
    :param n_docs: Corpus size. Unused -- pairs are deliberately not range-filtered, so a
        hallucinated id costs precision rather than vanishing.

    :returns: Sorted pairs, ``[]`` for an explicit empty answer, or ``None`` if nothing parsed.
    """
    return parsing.parse_pairs(text)


def score(parsed, gold: Sequence[Sequence[int]]) -> Dict[str, float]:
    """
    Score predicted pairs against gold.

    :param parsed: Output of :func:`parse`. ``None`` scores zero everywhere -- see the divergence
        note in :mod:`ctc.tasks.contradiction.spec`.
    :param gold: Gold pairs, 1-based, each sorted low-high.

    :returns: ``precision``, ``recall``, ``f1``, ``exact_match``, ``parsed``.

    :raises ValueError: If a gold pair is not sorted low-high.
    """
    check_gold(gold)
    if parsed is None:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "parsed": 0.0}
    return {**metrics.pair_metrics(parsed, gold), "parsed": 1.0}


def build_target(example: Dict) -> str:
    """
    :param example: A unified-format example whose ``gold_doc_indices`` holds the pairs directly,
        as 1-based item ids -- not document positions, despite sharing a field name with the
        0-based tasks.

    :returns: The pairs as JSON.
    """
    return json.dumps(example["gold_doc_indices"])


def make_pair_spec(
    *,
    name: str,
    instruction: str,
    serializer: str,
    description: str,
    rungs: tuple,
    query_builder: Callable[[Dict], str],
    max_new_tokens: int = 200,
    stop: str = "pairs",
    sources: tuple = (),
) -> TaskSpec:
    """
    Build the spec for one pair task.

    :param name: Task name.
    :param instruction: The task's instruction string, verbatim.
    :param serializer: Document serializer key.
    :param description: One line for ``--list-tasks``.
    :param rungs: The task's ladder.
    :param query_builder: ``(example) -> str``; the positioned ask. Separate per task because the
        criterion tasks append the example's own criterion and the others do not.
    :param max_new_tokens: Decode budget.
    :param stop: Stop preset.
    :param sources: Source corpora.

    :returns: The spec, not yet registered.
    """

    def build_prompt(example: Dict, **opts) -> str:
        return assemble.assemble(
            example,
            task=name,
            unified=True,
            header=GENERIC_INSTRUCTION,
            positioned=query_builder(example),
            **opts,
        )

    return TaskSpec(
        name=name,
        description=description,
        gold_index_base=1,
        instruction=instruction,
        serializer=serializer,
        unified=True,
        rungs=rungs,
        build_prompt=build_prompt,
        parse=parse,
        score=score,
        primary_metric="f1",
        max_new_tokens=max_new_tokens,
        stop=stop,
        answer_is_set=True,
        sources=sources,
    )
