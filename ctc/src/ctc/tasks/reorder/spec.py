"""
The ``reorder`` eval contract.

Passages from one document are shown in random order under shuffled display ids; the model outputs
the permutation restoring the original order. Scored by Kendall's tau, so a nearly-right ordering
scores nearly right -- unlike the set tasks, where a near-miss scores zero.

Two things specific to this task:

* **The target comes from ``gold_order``**, a list of 1-indexed display ids already in source
  order. It is *not* derived from ``gold_doc_indices`` the way every other task's target is.
* **The serializer collapses each passage's internal blank lines to single newlines.** Gutenberg
  passages contain their own paragraph breaks, and without that collapse a passage would be split
  across two chunks under the chunked mask, leaking its tail into the unmasked region.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ...format import assemble, metrics, parsing
from ...format.prompts import REORDER_INSTRUCTION
from ...format.registry import TaskSpec

__all__ = ["SPEC", "build_query", "build_target", "parse", "score"]


def build_query(example: Dict) -> str:
    """
    :param example: A unified-format example. ``queries`` is unused -- restoring the order is the
        whole ask, which is why this task is unified.

    :returns: The positioned ask.
    """
    return REORDER_INSTRUCTION


def build_prompt(example: Dict, **opts) -> str:
    """
    :param example: A unified-format example.
    :param opts: Assembly options.

    :returns: The prompt.
    """
    from ...format.prompts import GENERIC_INSTRUCTION

    return assemble.assemble(
        example,
        task="reorder",
        unified=True,
        header=GENERIC_INSTRUCTION,
        positioned=build_query(example),
        **opts,
    )


def build_target(example: Dict) -> str:
    """
    :param example: A unified-format example carrying ``gold_order``.

    :returns: The permutation as a JSON array.
    """
    return json.dumps(example["gold_order"])


def parse(text: str, n_docs: Optional[int] = None) -> Optional[List[int]]:
    """
    :param text: Raw model generation.
    :param n_docs: Expected permutation length. Required here -- a permutation can only be
        recognised against a known length.

    :returns: The permutation, or ``None`` if the output is not an exact permutation of ``1..n``.
    """
    if n_docs is None:
        return None
    return parsing.parse_permutation(text, n_docs)


def score(parsed: Optional[List[int]], gold_order: List[int]) -> Dict[str, float]:
    """
    :param parsed: Output of :func:`parse`.
    :param gold_order: The true ordering, 1-indexed.

    :returns: ``kendall_tau``, ``pmr`` (perfect match rate), ``position_accuracy``, ``parsed``,
        plus ``spearman_rho`` when scipy is installed.

        Note ``kendall_tau`` ranges over ``[-1, 1]``, not ``[0, 1]`` like every other primary
        metric in the suite: a systematically reversed ordering scores ``-1``, and 0 means no
        better than chance. Averaging it alongside F1 scores without saying so would be misleading.
    """
    if parsed is None:
        return {
            "kendall_tau": 0.0,
            "pmr": 0.0,
            "position_accuracy": 0.0,
            "parsed": 0.0,
        }
    n = len(gold_order)
    correct = sum(1 for a, b in zip(parsed, gold_order) if a == b)
    out = {
        "kendall_tau": metrics.kendall_tau(parsed, gold_order),
        "pmr": float(list(parsed) == list(gold_order)),
        "position_accuracy": correct / n if n else 0.0,
        "parsed": 1.0,
    }
    out.update(metrics.ordering_extras(parsed, gold_order))
    return out


SPEC = TaskSpec(
    name="reorder",
    description="Restore the original order of shuffled passages (Kendall tau).",
    gold_index_base=1,  # gold_order carries 1-indexed display ids directly
    instruction=REORDER_INSTRUCTION,
    serializer="reorder",
    unified=True,
    rungs=("2k", "4k", "8k", "16k", "32k"),
    build_prompt=build_prompt,
    parse=parse,
    score=score,
    primary_metric="kendall_tau",
    # 2048, not the pre-migration 1024. Reorder is the one task whose ANSWER grows with the rung:
    # the target is a permutation of n ids, measured at ~4.5 Qwen3 tokens each, so the 32k rung's
    # target has a median length of 1057 tokens and does not fit in 1024. `parse_permutation`
    # requires an EXACT permutation of 1..n, so a truncated answer parses as None and scores
    # kendall_tau 0.0 -- for every example at that rung, reading as a long-context collapse rather
    # than as a decode budget. Sized against the measured 32k target; `grouping_labeled`, the other
    # task whose answer is a long list, already sets 2048 for the same reason.
    max_new_tokens=2048,
    stop="eos",
    answer_is_set=False,
    sources=("gutenberg",),
    # Declared, not defaulted. Without this the shared build/audit layer reads
    # `gold_doc_indices` -- which a reorder example does not have at all -- so `validate` rejects
    # every row for a missing gold field, and `gold_fingerprint` collapses every example onto one
    # digest, which would make the train/eval contamination guard report leakage everywhere.
    extra={"gold_field": "gold_order"},
)
