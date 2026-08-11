"""
The ``retrieval`` eval contract.

Given a corpus and a question, name the document ids that answer it. The canonical target for six
different suite entries -- ``nq``, ``hotpotqa``, ``niah_contradiction``, ``beir_scifact``,
``beir_fiqa`` and ``msmarco`` all alias to this one task. They are *sources*, not separate tasks,
which is what :attr:`~ctc.format.registry.TaskSpec.sources` records.

The first classic-shape task in the port: the instruction lives in the alpaca header and the
question is positioned, rather than the instruction itself being positioned.

.. note::
   **The instruction depends on the example.** Single-gold, multi-gold and multi-query select three
   different wordings, so the same task produces three prompt shapes. All three are declared in
   ``instruction_variants`` and hashed into the fingerprint.

.. note::
   **Gold is 0-based**, converted to the 1-based ids the prompt renders. Same as the absence
   family, opposite to contradiction and the pair family.
"""

from __future__ import annotations

from typing import Dict, Optional, Set

from ...format import assemble, parsing
from ...format.prompts import (
    RETRIEVAL_INSTRUCTION_MULTI_DOC,
    RETRIEVAL_INSTRUCTION_MULTI_QUERY,
    RETRIEVAL_INSTRUCTION_SINGLE,
)
from ...format.registry import TaskSpec
from .._retrieval import (
    flatten_gold,
    is_multi_query,
    questions_block,
    retrieval_instruction,
    score_ids,
)

__all__ = ["SPEC", "build_query", "build_target", "parse", "score"]


def build_query(example: Dict) -> str:
    """
    :param example: A unified-format example.

    :returns: The positioned text -- here the example's own question(s), since the instruction sits
        in the header.
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
        task="retrieval",
        unified=False,
        header=retrieval_instruction(example),
        positioned=build_query(example),
        **opts,
    )


def build_target(example: Dict) -> str:
    """
    :param example: A unified-format example with 0-based ``gold_doc_indices``.

    :returns: The bracketed 1-based ids; multi-query examples are grouped ``Q1: ...; Q2: ...``.
    """
    gold = example["gold_doc_indices"]
    if is_multi_query(example):
        parts = []
        for i, per_query in enumerate(gold):
            ids = ", ".join(f"[{g + 1}]" for g in per_query)
            parts.append(f"Q{i + 1}: {ids}")
        return "; ".join(parts)
    return ", ".join(f"[{g + 1}]" for g in sorted(flatten_gold(example)))


def parse(text: str, n_docs: Optional[int] = None) -> Optional[Set[int]]:
    """
    :param text: Raw model generation.
    :param n_docs: Corpus size. Unused -- ids are not range-filtered here.

    :returns: The named ids, or ``None`` if none parsed. An empty set is never returned as a
        distinct answer: naming nothing is a failure to retrieve, not an assertion.
    """
    ids = parsing.parse_doc_ids(text)
    return ids if ids else None


def score(parsed, example_or_gold) -> Dict[str, float]:
    """
    :param parsed: Output of :func:`parse`.
    :param example_or_gold: The example, or its raw 0-based gold indices.

    :returns: ``exact_match``, ``recall``, ``precision``, ``f1``, ``parsed``.
    """
    example = (
        example_or_gold
        if isinstance(example_or_gold, dict)
        else {"gold_doc_indices": example_or_gold, "queries": [""]}
    )
    return score_ids(parsed, example)


SPEC = TaskSpec(
    name="retrieval",
    description="Name the document ids relevant to the question.",
    gold_index_base=0,
    instruction=RETRIEVAL_INSTRUCTION_SINGLE,
    instruction_variants=(RETRIEVAL_INSTRUCTION_MULTI_DOC, RETRIEVAL_INSTRUCTION_MULTI_QUERY),
    serializer="default",  # numbered, via TASKS_WITH_DOC_IDS
    unified=False,
    rungs=("2k", "4k", "8k", "16k", "32k"),
    build_prompt=build_prompt,
    parse=parse,
    score=score,
    primary_metric="f1",
    max_new_tokens=64,
    stop="newline",
    answer_is_set=True,
    sources=("nq", "hotpotqa", "niah_contradiction", "beir_scifact", "beir_fiqa", "msmarco"),
)
