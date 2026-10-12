"""
The ``textgroups`` eval contract.

The natural-language counterpart to :mod:`ctc.tasks.groups4`. Each passage carries a *textual*
feature value -- its noun count, or how often a common word appears -- and the model finds groups
whose feature values meet an aggregate target. The criterion, including which feature is meant, is
per-example and lives in ``queries[0]``.

Structurally identical to groups4: instruction then criterion, shared cycle-family scoring. It
differs in that the feature must be *read out of prose* rather than computed from an expression,
which is the point of having both.
"""

from __future__ import annotations

from typing import Dict

from ...format.prompts import TEXTGROUPS_INSTRUCTION
from .._cycles import build_target, make_cycle_spec  # noqa: F401 (build_target re-exported)

__all__ = ["SPEC", "build_query", "build_target"]


def build_query(example: Dict) -> str:
    """
    :param example: A unified-format example whose ``queries[0]`` names the feature and target.

    :returns: The instruction followed by the criterion.
    """
    return f"{TEXTGROUPS_INSTRUCTION}\n\n{example['queries'][0]}"


SPEC = make_cycle_spec(
    name="textgroups",
    description="Find every group of passages whose textual feature values meet a target.",
    instruction=TEXTGROUPS_INSTRUCTION,
    serializer="textgroups",
    rungs=("2k", "4k", "8k", "16k", "32k"),
    query_builder=build_query,
    max_new_tokens=200,
    stop="pairs",
)
