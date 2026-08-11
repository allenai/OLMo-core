"""
The ``outlier`` eval contract.

The corpus is a set of items (product reviews, Wikipedia passages) that mostly share an attribute
-- a star rating, a product category -- and a few differ. Name the odd ones out.

.. note::
   **Legacy prompt path**, shared with ``grouping_labeled``: the documents are followed by the
   *raw* query string (no ``Question:`` prefix), with the instruction in the alpaca header. The
   path ignores ``query_position`` entirely, so the spec sets ``honors_query_position=False`` and
   the fingerprint pins the value rather than reporting a mismatch over an inert knob.

.. note::
   **Gold is 0-based**, converted to the 1-based ids the prompt renders.

The answer line is ``Outliers: [i], [j]``, preceded by one sentence naming the majority and
minority attributes. That sentence is *not* scored -- only the ids -- but it is why
:func:`~ctc.format.parsing.parse_outlier_ids` insists on bracketed ids: relaxing it the way
``parse_doc_ids`` is relaxed would start matching the star rating in that very sentence.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ...format import assemble, metrics, parsing
from ...format.prompts import OUTLIER_INSTRUCTION
from ...format.registry import TaskSpec

__all__ = ["SPEC", "build_query", "build_target", "parse", "score"]


def build_query(example: Dict) -> str:
    """
    :param example: A unified-format example.

    :returns: The raw query string -- no ``Question:`` prefix, and no instruction, which already
        sits in the header.
    """
    return example["queries"][0]


def build_prompt(example: Dict, **opts) -> str:
    """
    :param example: A unified-format example.
    :param opts: Assembly options. ``query_position`` is accepted and ignored, matching the legacy
        path.

    :returns: The prompt.
    """
    opts.pop("query_position", None)
    return assemble.assemble(
        example,
        task="outlier",
        unified=False,
        header=OUTLIER_INSTRUCTION,
        positioned=build_query(example),
        query_position="after",
        **opts,
    )


def build_target(example: Dict) -> str:
    """
    :param example: A unified-format example with 0-based ``gold_doc_indices``.

    :returns: The ``Outliers: [i], [j]`` answer line, ids 1-based and ascending.
    """
    ids = ", ".join(f"[{g + 1}]" for g in sorted(example["gold_doc_indices"]))
    return f"Outliers: {ids}"


def parse(text: str, n_docs: int) -> Optional[List[int]]:
    """
    :param text: Raw model generation.
    :param n_docs: Corpus size; ids outside ``1..n_docs`` are dropped as hallucinations.

    :returns: Ids in first-mention order, or ``None``.
    """
    return parsing.parse_outlier_ids(text, n_docs)


def score(parsed: Optional[List[int]], gold) -> Dict[str, float]:
    """
    :param parsed: Output of :func:`parse`.
    :param gold: 0-based gold indices, or the example carrying them.

    :returns: ``precision``, ``recall``, ``f1``, ``exact_match``, ``parsed``.
    """
    indices = gold["gold_doc_indices"] if isinstance(gold, dict) else gold
    gold_ids = {int(g) + 1 for g in indices}
    if parsed is None:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "parsed": 0.0}
    return {**metrics.set_metrics(set(parsed), gold_ids), "parsed": 1.0}


SPEC = TaskSpec(
    name="outlier",
    description="Name the items whose attribute differs from the majority.",
    gold_index_base=0,
    instruction=OUTLIER_INSTRUCTION,
    serializer="default",  # numbered, via TASKS_WITH_DOC_IDS
    unified=False,
    honors_query_position=False,  # legacy path hardcodes documents-then-query
    rungs=("2k", "4k", "8k", "16k", "32k"),
    build_prompt=build_prompt,
    parse=parse,
    score=score,
    primary_metric="f1",
    max_new_tokens=256,
    stop="newline",
    answer_is_set=False,
    sources=("amazon", "wiki"),
)
