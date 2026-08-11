"""
The ``redundancy`` eval contract.

The mirror image of contradiction: find every pair of claims stating the *same* fact, one a
paraphrase or restatement of the other. Same corpus shape, same answer format, same scorer, so it
shares :mod:`ctc.tasks._pairs` with the rest of the family.

Unlike contradiction, this task has always stopped at the end of its answer line rather than
decoding to the budget -- which is why it never showed contradiction's rambling behaviour, and part
of the evidence that contradiction's ``stop="eos"`` was an oversight rather than a decision.

.. note::
   The only redundancy eval file found during the port has **244 examples**, below the 500
   threshold. Numbers off it must be quoted with their size and error bar inline.
"""

from __future__ import annotations

from typing import Dict

from ...format.prompts import REDUNDANCY_INSTRUCTION
from .._pairs import build_target, make_pair_spec  # noqa: F401 (build_target re-exported)

__all__ = ["SPEC", "build_query", "build_target"]


def build_query(example: Dict) -> str:
    """
    :param example: A unified-format example. Its ``queries`` field is unused -- the instruction is
        the whole ask, which is why the task is unified.

    :returns: The positioned ask.
    """
    return REDUNDANCY_INSTRUCTION


SPEC = make_pair_spec(
    name="redundancy",
    description="Find every pair of claims that state the same fact (N^2 over the corpus).",
    instruction=REDUNDANCY_INSTRUCTION,
    serializer="redundancy",
    rungs=("2k", "4k", "8k", "16k", "32k"),
    query_builder=build_query,
    max_new_tokens=200,
    sources=("pubmed",),
)
