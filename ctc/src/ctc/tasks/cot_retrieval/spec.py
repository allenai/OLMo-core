"""
The ``cot_retrieval`` eval contract.

Retrieval with reasoning: the model explains why documents are relevant, then names their ids.
Scored identically to :mod:`ctc.tasks.retrieval` -- the reasoning is not graded, only the final ids
-- so the difference is entirely in the instruction and the decode budget (512 rather than 64,
since the reasoning has to fit).

.. warning::
   Despite the name, this task is **not** CoT in the ``cot_mode`` sense. ``cot_mode`` selected
   generated reasoning *targets* during data construction and was dropped in the port (no CTC-suite
   result ever used it). ``cot_retrieval`` is a distinct task whose instruction asks the model to
   reason; its shards are built with ``--cot-mode none`` like everything else. The two are
   unrelated, which the shared word makes easy to miss.

Multi-query is not supported, so only the single/multi-*gold* instruction split applies.
"""

from __future__ import annotations

from typing import Dict, Optional, Set

from ...format import assemble
from ...format.prompts import (
    COT_RETRIEVAL_INSTRUCTION_MULTI_DOC,
    COT_RETRIEVAL_INSTRUCTION_SINGLE,
)
from ...format.registry import TaskSpec
from .._retrieval import cot_retrieval_instruction, flatten_gold, has_multi_gold, questions_block
from ..retrieval.spec import parse, score  # noqa: F401 (scoring is identical to retrieval)

__all__ = ["SPEC", "build_query", "build_target", "parse", "score"]


def build_query(example: Dict) -> str:
    """
    :param example: A unified-format example.

    :returns: The positioned question(s).
    """
    return questions_block(example["queries"])


def build_prompt(example: Dict, **opts) -> str:
    """
    :param example: A unified-format example.
    :param opts: Assembly options.

    :returns: The prompt, in the classic shape.
    """
    return assemble.assemble(
        example,
        task="cot_retrieval",
        unified=False,
        header=cot_retrieval_instruction(example),
        positioned=build_query(example),
        **opts,
    )


def build_target(example: Dict) -> str:
    """
    :param example: A unified-format example with 0-based ``gold_doc_indices``.

    :returns: ``"Relevant Document: [1]"`` or ``"Relevant Documents: [1], [2]"``.

        Unlike :mod:`ctc.tasks.retrieval`, whose target is the bare ``[1]``, this task's target
        carries the answer-line prefix. The two are genuinely inconsistent in the pre-migration
        data; the port preserves that rather than tidying it, because tidying would invalidate
        every cot_retrieval checkpoint.
    """
    ids = ", ".join(f"[{g + 1}]" for g in sorted(flatten_gold(example)))
    label = "Relevant Documents" if has_multi_gold(example) else "Relevant Document"
    return f"{label}: {ids}"


SPEC = TaskSpec(
    name="cot_retrieval",
    description="Reason about relevance, then name the relevant document ids.",
    gold_index_base=0,
    instruction=COT_RETRIEVAL_INSTRUCTION_SINGLE,
    instruction_variants=(COT_RETRIEVAL_INSTRUCTION_MULTI_DOC,),
    serializer="default",
    unified=False,
    rungs=("2k", "4k", "8k", "16k", "32k"),
    build_prompt=build_prompt,
    parse=parse,
    score=score,
    primary_metric="f1",
    max_new_tokens=512,  # the reasoning has to fit before the answer line
    stop="newline",
    answer_is_set=True,
)
