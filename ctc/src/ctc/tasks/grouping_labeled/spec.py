"""
The ``grouping_labeled`` eval contract.

:mod:`ctc.tasks.grouping` plus a short name for each group. **The names are not scored** -- the
partition is, by exactly the same pairwise metric. The task exists to test whether being asked to
articulate the shared topic changes the partition the model produces, so its number is only
meaningful next to plain ``grouping``.


.. note::
   **This task takes a legacy prompt path.** The documents are followed by the *raw* query string
   -- no ``Question:`` prefix -- with the instruction in the alpaca header. The path also
   **ignores ``query_position`` entirely**: it hardcodes documents-then-query. That is why the spec
   sets ``honors_query_position=False``, which stops the fingerprint reporting a false mismatch
   over a knob that does nothing here.
"""

from __future__ import annotations

from typing import Dict

from ...format.prompts import GROUPING_LABELED_INSTRUCTION
from .._grouping import build_labeled_target, make_grouping_spec

__all__ = ["SPEC", "build_query", "build_target"]


def build_query(example: Dict) -> str:
    """
    :param example: A unified-format example whose ``queries[0]`` states the requested K.

    :returns: The raw query string -- no "Question:" prefix, and no instruction, which
        already sits in the header.
    """
    return example["queries"][0]


def build_target(example: Dict) -> str:
    """
    :param example: A unified-format example, optionally carrying ``cluster_labels``.

    :returns: The JSON grouping object with a ``label`` per group.
    """
    return build_labeled_target(example)


SPEC = make_grouping_spec(
    name="grouping_labeled",
    description="Partition abstracts into K categories, naming each group.",
    instruction=GROUPING_LABELED_INSTRUCTION,
    rungs=("2k", "4k", "8k", "16k", "32k"),
    query_builder=build_query,
    sources=("arxiv", "openalex"),
)
