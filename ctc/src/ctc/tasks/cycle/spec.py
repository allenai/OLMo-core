"""
The ``cycle`` eval contract.

Claims assert strict comparisons ("A ranks strictly above B"). Find every set of claims whose edges
form a directed cycle -- an impossible loop. Cycles are variable-length, which is what puts this in
the cycle family rather than the pair family.

.. note::
   **The query comes BEFORE the instruction here**, uniquely among the group tasks (ruler is the
   only other task that does this). The per-example question states the comparison domain, and the
   instruction then fixes the output format. groups4 and textgroups put theirs the other way round.
   In the pre-migration code this was one line in a 17-branch chain and easy to miss; as a per-task
   function it is visible, and the golden fixture holds it in place.
"""

from __future__ import annotations

from typing import Dict

from ...format.prompts import CYCLE_INSTRUCTION
from .._cycles import build_target, make_cycle_spec  # noqa: F401 (build_target re-exported)

__all__ = ["SPEC", "build_query", "build_target"]


def build_query(example: Dict) -> str:
    """
    :param example: A unified-format example whose ``queries[0]`` states the task.

    :returns: The example's question followed by the format instruction -- in that order. Reversing
        it changes every shard this task has ever produced.
    """
    return f"{example['queries'][0]}\n\n{CYCLE_INSTRUCTION}"


SPEC = make_cycle_spec(
    name="cycle",
    description="Find every set of comparison claims forming a directed cycle.",
    instruction=CYCLE_INSTRUCTION,
    serializer="cycle",
    rungs=("2k", "4k", "8k", "16k", "32k"),
    query_builder=build_query,
    max_new_tokens=200,
    stop="pairs",
)
