"""
The ``groups4`` eval contract.

A list of arithmetic expressions; find every group of G whose values satisfy a closeness criterion
(all within X of each other). The criterion is per-example and lives in ``queries[0]``.

Scored by the shared cycle-family metric rather than the pair metric, because groups are
variable-length -- see :mod:`ctc.tasks._cycles`.

.. warning::
   **groups4 collapsed in Stage-3 2k validation** while 26 of 27 other full-attention tasks passed.
   A near-zero score here is a known prior, not necessarily a new regression, so check
   ``parse_rate`` and read generations before drawing a conclusion from it.
"""

from __future__ import annotations

from typing import Dict

from ...format.prompts import GROUPS4_INSTRUCTION
from .._cycles import build_target, make_cycle_spec  # noqa: F401 (build_target re-exported)

__all__ = ["SPEC", "build_query", "build_target"]


def build_query(example: Dict) -> str:
    """
    :param example: A unified-format example whose ``queries[0]`` states this example's criterion.

    :returns: The instruction followed by the criterion -- the opposite order from
        :mod:`ctc.tasks.cycle`.
    """
    return f"{GROUPS4_INSTRUCTION}\n\n{example['queries'][0]}"


SPEC = make_cycle_spec(
    name="groups4",
    description="Find every group of arithmetic expressions meeting a closeness criterion.",
    instruction=GROUPS4_INSTRUCTION,
    serializer="groups4",
    rungs=("2k", "4k", "8k", "16k", "32k"),
    query_builder=build_query,
    max_new_tokens=200,
    stop="pairs",
)
