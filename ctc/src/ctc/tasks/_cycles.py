"""
Shared machinery for the cycle family: cycle, groups4, textgroups.

These three ask *which **groups** of items satisfy relation R?* -- as opposed to the pair family's
*which pairs?*. The difference is not cosmetic: a group has variable size, so scoring compares sets
of sets rather than sets of pairs, and a partially-correct group scores zero at the cycle level
while still earning item-level credit. That is why these have their own scorer
(:func:`ctc.format.metrics.cycle_metrics`) rather than reusing the pair one, and why
``primary_metric`` is worth reading alongside ``claim_f1``.

The pre-migration ``TASK_CFG`` routed all three to ``_eval_cycle``, so the grouping was already
recognised; as with the pair family, it was implicit in a scorer lookup.

Gold is 1-based here, matching the rendered item numbers.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, Optional, Sequence

from ..format import assemble, metrics, parsing
from ..format.prompts import GENERIC_INSTRUCTION
from ..format.registry import TaskSpec

__all__ = ["parse", "score", "build_target", "make_cycle_spec"]


def parse(text: str, n_docs: Optional[int] = None):
    """
    Parse a cycle/group answer.

    :param text: Raw model generation.
    :param n_docs: Corpus size. Unused -- ids are not range-filtered, so a hallucinated id costs
        precision rather than silently disappearing.

    :returns: Cycles as sorted id lists, ``[]`` for an explicit empty answer, or ``None``.
    """
    return parsing.parse_cycles(text)


def score(parsed, gold: Sequence[Sequence[int]]) -> Dict[str, float]:
    """
    Score predicted groups against gold.

    :param parsed: Output of :func:`parse`. ``None`` scores zero on every metric, including
        ``claim_f1`` -- an unparseable generation found nothing, at any granularity.
    :param gold: Gold groups, 1-based.

    :returns: ``precision``, ``recall``, ``f1``, ``exact_match``, ``claim_f1``, ``parsed``.
    """
    if parsed is None:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "exact_match": 0.0,
            "claim_f1": 0.0,
            "parsed": 0.0,
        }
    return {**metrics.cycle_metrics(parsed, gold), "parsed": 1.0}


def build_target(example: Dict) -> str:
    """
    :param example: A unified-format example whose ``gold_doc_indices`` holds the groups directly,
        as 1-based item ids.

    :returns: The groups as JSON.
    """
    return json.dumps(example["gold_doc_indices"])


def make_cycle_spec(
    *,
    name: str,
    instruction: str,
    serializer: str,
    description: str,
    rungs: tuple,
    query_builder: Callable[[Dict], str],
    max_new_tokens: int = 200,
    stop: str = "newline",
    sources: tuple = (),
) -> TaskSpec:
    """
    Build the spec for one cycle-family task.

    :param name: Task name.
    :param instruction: The instruction string, verbatim.
    :param serializer: Document serializer key.
    :param description: One line for ``--list-tasks``.
    :param rungs: The task's ladder.
    :param query_builder: ``(example) -> str``. Separate per task because the ORDER differs --
        cycle puts the example's question before its instruction, the other two after.
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
