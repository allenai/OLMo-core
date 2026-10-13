"""
The ``xabsence`` eval contract (cross-corpus absence).

Two numbered corpora, A and B, under one shared index. Almost every claim has a paraphrase in the
*other* corpus; a few are unmatched. Find the unmatched ones.

Harder than :mod:`ctc.tasks.absence` in a specific way: absence compares an item against its own
copy, so the match is near-exact, whereas here the match is a paraphrase and must be recognised
semantically across the corpus boundary. That is why it is grouped with the N-squared tasks despite
producing the same flat id-set answer.

The answer anchors on ``Unmatched:`` rather than ``Missing:``; the shared parser accepts both.

.. note::
   Pre-2026-07-26 xabsence shards may be affected by the oolong ``--item-regex`` leak (a bare
   ``'||'`` matched every line). Check the build date of any shard before trusting a number.
"""

from __future__ import annotations

from typing import Dict

from ...format.prompts import (
    XABSENCE_INSTRUCTION,
    XABSENCE_ONESIDED_GUTENBERG_INSTRUCTION,
    XABSENCE_ONESIDED_INSTRUCTION,
)
from .._absence import make_absence_spec

__all__ = ["SPEC", "build_query", "build_target"]


def build_query(example: Dict) -> str:
    """
    The ask depends on the data, exactly as in the reference renderer: one-sided examples
    (``orphan_side == "A"``: every A item recurs verbatim in B except the orphans) get the one-sided
    instruction, worded "passages" for a prose (Gutenberg) corpus and "claims" otherwise; legacy
    two-sided paraphrase examples keep the original instruction.

    The suite's ``xabsence`` rows are one-sided Gutenberg exact copies. Rendering them with the
    two-sided *paraphrase* instruction -- what this function returned before -- described a task
    the context does not contain.

    :param example: A unified-format example. ``queries`` is unused -- both corpora are already in
        the rendered context, so the instruction is the whole ask.

    :returns: The positioned ask.
    """
    if example.get("orphan_side") == "A":
        n = example.get("num_unmatched", 3)
        if str(example.get("source", "")).endswith("gutenberg"):
            return XABSENCE_ONESIDED_GUTENBERG_INSTRUCTION.format(n=n)
        return XABSENCE_ONESIDED_INSTRUCTION.format(n=n)
    return XABSENCE_INSTRUCTION


def build_target(example: Dict) -> str:
    """
    :param example: A unified-format example with 0-based ``gold_doc_indices``.

    :returns: The ``Unmatched: [i], [j]`` answer line, with ids rendered 1-based.
    """
    from .._absence import build_target as _bt

    # the anchor the instruction asks for: one-sided data asks for "Missing:"
    return _bt(example, "Missing" if example.get("orphan_side") == "A" else "Unmatched")


SPEC = make_absence_spec(
    name="xabsence",
    description="Find claims with no paraphrase in the other corpus (cross-corpus absence).",
    instruction=XABSENCE_INSTRUCTION,
    serializer="xabsence",
    rungs=("2k", "4k", "8k", "16k", "32k"),
    query_builder=build_query,
    max_new_tokens=200,
    sources=("pubmed", "abstracts"),
)
