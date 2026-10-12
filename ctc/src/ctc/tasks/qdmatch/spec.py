"""
The ``qdmatch`` eval contract.

A single numbered list mixing M queries and N documents, each line tagged ``Query:`` or
``Document:``. A few query-document pairs are relevant. Name them.

The N-squared retrieval task: because queries and documents share one index, finding the answer
means comparing across the whole list rather than scanning for one target.

Two things differ from the other pair tasks and both matter:

* **Pairs are ORDERED.** A qdmatch pair is ``(query_id, document_id)``; swapping the two makes a
  different claim, so :func:`~ctc.format.parsing.parse_qd_pairs` does not sort. Feeding these to
  the contradiction parser -- which sorts -- would silently turn wrong answers into right ones.
* **Gold lives in ``gold_pairs``**, not ``gold_doc_indices``, and is already 1-based and ordered.
  Reading the usual field would find the wrong data or none at all.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ...format import assemble, metrics, parsing
from ...format.prompts import GENERIC_INSTRUCTION, QDMATCH_INSTRUCTION
from ...format.registry import TaskSpec

__all__ = ["SPEC", "build_query", "build_target", "parse", "score"]


def build_query(example: Dict) -> str:
    """
    :param example: A unified-format example. ``queries`` is unused -- the queries are already
        interleaved into the rendered list, so the instruction is the whole ask.

    :returns: The positioned ask.
    """
    return QDMATCH_INSTRUCTION


def build_prompt(example: Dict, **opts) -> str:
    """
    :param example: A unified-format example.
    :param opts: Assembly options. ``query_position`` is accepted and **overridden** -- see below.

    :returns: The prompt, with the instruction repeated before AND after the item list.
    """
    # qdmatch hardcodes query_position="both", ignoring the caller. The item list is long and
    # interleaves queries with documents, so the instruction is repeated immediately before the
    # response to remind the model of the task and output format. Honouring a caller's "after"
    # here would silently produce a prompt no qdmatch checkpoint was trained on.
    opts.pop("query_position", None)
    return assemble.assemble(
        example,
        task="qdmatch",
        unified=True,
        header=GENERIC_INSTRUCTION,
        positioned=build_query(example),
        query_position="both",
        **opts,
    )


def build_target(example: Dict) -> str:
    """
    :param example: A unified-format example carrying ``gold_pairs`` (1-based, ordered).

    :returns: The pairs as JSON, order preserved.
    """
    return json.dumps(example.get("gold_pairs", []))


def parse(text: str, n_docs: Optional[int] = None) -> Optional[List[List[int]]]:
    """
    :param text: Raw model generation.
    :param n_docs: Unused.

    :returns: Ordered pairs, ``[]`` for an explicit empty answer, or ``None``.
    """
    return parsing.parse_qd_pairs(text)


def score(parsed, gold) -> Dict[str, float]:
    """
    :param parsed: Output of :func:`parse`.
    :param gold: ``gold_pairs``, or the example carrying them. **Not** ``gold_doc_indices``.

    :returns: ``precision``, ``recall``, ``f1``, ``exact_match``, ``parsed``.
    """
    pairs = gold.get("gold_pairs", []) if isinstance(gold, dict) else gold
    if parsed is None:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "parsed": 0.0}
    # pair_metrics keys on tuple(p), so order is preserved -- exactly what this task needs.
    return {**metrics.pair_metrics(parsed, pairs), "parsed": 1.0}


SPEC = TaskSpec(
    name="qdmatch",
    description="Name every relevant (query, document) pair in one interleaved list.",
    gold_index_base=1,
    instruction=QDMATCH_INSTRUCTION,
    serializer="qdmatch",
    unified=True,
    honors_query_position=False,  # hardcoded to "both"
    rungs=("2k", "4k", "8k", "16k", "32k"),
    build_prompt=build_prompt,
    parse=parse,
    score=score,
    primary_metric="f1",
    max_new_tokens=200,
    stop="pairs",
    answer_is_set=True,
    # ObliQ is NOT a qdmatch corpus. It was dropped from this task's roster on 2026-07-19
    # (``BUILD_MATRIX.md`` rows 21a/21c) and re-entered the suite as a standalone in-context
    # *retrieval* row built by a different generator, with the shipped `qdmatch_*obliq*` pilot
    # JSONL explicitly marked do-not-use. Listing it here was the only in-repo statement of which
    # corpora this task builds from, and it said the opposite of the roster.
    sources=("nq", "hotpotqa"),
    extra={"gold_field": "gold_pairs"},
)
