"""
Shared machinery for the classic (non-unified) tasks: retrieval, cot_retrieval, qa.

These are the tasks that carry a **real per-example question**, so the prompt takes the classic
shape -- the task instruction in the alpaca header, the question positioned relative to the
documents -- rather than the unified shape used by contradiction and its relatives.

They also have an instruction that depends on the *example*, which nothing ported so far does:

============  ====================================================================
retrieval     single-gold, multi-gold, or multi-query -> three different strings
cot_retrieval single-gold or multi-gold -> two
qa            single-query or multi-query -> two
============  ====================================================================

Every variant is declared on the spec via ``instruction_variants`` and hashed into the format
fingerprint. Hashing only the primary string would let a variant be reworded without invalidating
any checkpoint, which is exactly the drift the fingerprint exists to catch.

**Gold is 0-based here**, like the absence family and unlike the pair family -- the scorer converts
with ``g + 1`` to reach the 1-based ids the prompt renders.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from ..format import assemble, metrics
from ..format.prompts import (
    COT_RETRIEVAL_INSTRUCTION_MULTI_DOC,
    COT_RETRIEVAL_INSTRUCTION_SINGLE,
    MULTI_QA_INSTRUCTION,
    QA_INSTRUCTION,
    RETRIEVAL_INSTRUCTION_MULTI_DOC,
    RETRIEVAL_INSTRUCTION_MULTI_QUERY,
    RETRIEVAL_INSTRUCTION_SINGLE,
)

__all__ = [
    "is_multi_query",
    "has_multi_gold",
    "questions_block",
    "retrieval_instruction",
    "cot_retrieval_instruction",
    "qa_instruction",
    "flatten_gold",
]


def is_multi_query(example: Dict) -> bool:
    """
    :param example: A unified-format example.

    :returns: Whether it carries more than one independent question.
    """
    return len(example["queries"]) > 1


def has_multi_gold(example: Dict) -> bool:
    """
    :param example: A unified-format example.

    :returns: Whether any query has more than one gold document. Selects between the single-doc and
        multi-doc instruction wordings, so it changes the prompt, not just the scoring.
    """
    gold = example["gold_doc_indices"]
    if not gold:
        return False
    if isinstance(gold[0], list):
        return any(len(g) > 1 for g in gold)
    return len(gold) > 1


def questions_block(queries: Sequence[str]) -> str:
    """
    :param queries: The example's questions.

    :returns: The positioned question text -- the classic shape's equivalent of the unified tasks'
        task query.
    """
    return assemble.build_questions_block(queries)


def retrieval_instruction(example: Dict) -> str:
    """
    :param example: A unified-format example.

    :returns: Whichever of the three retrieval instructions this example calls for.
    """
    if is_multi_query(example):
        return RETRIEVAL_INSTRUCTION_MULTI_QUERY
    return (
        RETRIEVAL_INSTRUCTION_MULTI_DOC if has_multi_gold(example) else RETRIEVAL_INSTRUCTION_SINGLE
    )


def cot_retrieval_instruction(example: Dict) -> str:
    """
    :param example: A unified-format example.

    :returns: The single- or multi-doc CoT retrieval instruction. Multi-*query* is not supported by
        this task, so it is not consulted.
    """
    return (
        COT_RETRIEVAL_INSTRUCTION_MULTI_DOC
        if has_multi_gold(example)
        else COT_RETRIEVAL_INSTRUCTION_SINGLE
    )


def qa_instruction(example: Dict) -> str:
    """
    :param example: A unified-format example.

    :returns: The single- or multi-query QA instruction.
    """
    return MULTI_QA_INSTRUCTION if is_multi_query(example) else QA_INSTRUCTION


def flatten_gold(example: Dict) -> List[int]:
    """
    Flatten single-query gold indices.

    :param example: A unified-format example. Single-query gold is stored either flat (``[0, 1]``)
        or wrapped in one list (``[[0, 1]]``), depending on the generator.

    :returns: The flat 0-based indices.
    """
    gold = example["gold_doc_indices"]
    if gold and isinstance(gold[0], list):
        return list(gold[0])
    return list(gold)


def score_ids(parsed, example: Dict) -> Dict[str, float]:
    """
    Score a predicted id set for a single-query retrieval example.

    :param parsed: Ids the model named, or ``None``.
    :param example: The example, whose ``gold_doc_indices`` are **0-based**.

    :returns: ``exact_match``, ``recall``, ``precision``, ``f1``, ``parsed``.
    """
    gold_ids = {g + 1 for g in flatten_gold(example)}
    if parsed is None:
        return {
            "exact_match": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "parsed": 0.0,
        }
    pred = set(parsed)
    return {
        "exact_match": float(metrics.retrieval_exact_match(pred, gold_ids)),
        "recall": metrics.retrieval_recall(pred, gold_ids),
        "precision": metrics.retrieval_precision(pred, gold_ids),
        "f1": metrics.retrieval_f1(pred, gold_ids),
        "parsed": 1.0,
    }
