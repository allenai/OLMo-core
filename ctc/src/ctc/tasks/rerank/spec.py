"""
The ``rerank`` eval contract (MS MARCO passage re-ranking).

Given a query and a candidate pool, order the documents from most to least relevant.

.. warning::
   **The old binary rerank format is DISABLED, and that is deliberate.** Data without per-document
   cross-encoder scores (``ce_scores``) yields a degenerate gold-first target and MRR-only metrics,
   so the pre-migration evaluator raises rather than scoring it -- "so stale numbers can't be read
   by accident". That guard is preserved here. Regenerate with
   ``generate_msmarco_trainhn_data.py`` for CE-graded data, which scores NDCG@10 and Kendall tau.

Scoring reflects **ordering quality**, not merely whether a gold document reached the top-k. MRR@10
alone would give full credit to a ranking that puts one gold first and then scrambles everything
after it, which for a re-ranking task is most of the answer.

**Gold is 0-based**, converted to the 1-based ids the prompt renders.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ...format import assemble
from ...format.prompts import RERANK_INSTRUCTION, rerank_instruction
from ...format.registry import TaskSpec
from .._retrieval import questions_block

__all__ = ["SPEC", "build_query", "build_target", "parse", "score", "require_ce_scores"]

#: NDCG cutoff, and the length of the ranked prefix that is scored.
TOP_K = 10


def require_ce_scores(example: Dict) -> None:
    """
    Refuse to score data that predates cross-encoder grading.

    :param example: A unified-format example.

    :raises NotImplementedError: When ``ce_scores`` is absent or all-``None``. The old format
        produces a degenerate gold-first target whose MRR-only numbers are not comparable with
        anything current, and silently scoring it is how a stale number gets read as a live one.
    """
    ce = example.get("ce_scores")
    if not (ce and any(s is not None for s in ce)):
        raise NotImplementedError(
            "DEPRECATED binary rerank data: no `ce_scores`. The old gold-first / MRR-only format "
            "is disabled so stale numbers cannot be read by accident. Regenerate with "
            "generate_msmarco_trainhn_data.py (CE-graded -> NDCG@10 + Kendall tau)."
        )


def build_query(example: Dict) -> str:
    """
    :param example: A unified-format example.

    :returns: The positioned question.
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
        task="rerank",
        unified=False,
        header=RERANK_INSTRUCTION,
        positioned=build_query(example),
        **opts,
    )


def build_target(example: Dict) -> str:
    """
    :param example: A unified-format example with 0-based ``gold_doc_indices``.

    :returns: The ``Ranking: [i], [j], ...`` answer line, ids 1-based.
    """
    ids = ", ".join(f"[{g + 1}]" for g in example["gold_doc_indices"])
    return f"Ranking: {ids}"


def parse(text: str, n_docs: Optional[int] = None) -> Optional[List[int]]:
    """
    Extract the ranked id list, **order preserved and duplicates dropped**.

    :param text: Raw model generation.
    :param n_docs: Corpus size; ids outside ``1..n_docs`` are dropped.

    :returns: The ranking, or ``None`` if nothing parsed.

    .. note::
       :func:`~ctc.format.parsing.parse_doc_ids` returns a *set* and is therefore wrong here --
       a ranking is an ordering, and set semantics would discard the entire answer.
    """
    import re

    ids = [int(x) for x in re.findall(r"\[?\s*(\d+)\s*\]", text)]
    if n_docs is not None:
        ids = [i for i in ids if 1 <= i <= n_docs]
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out or None


def score(parsed: Optional[List[int]], gold, k: int = TOP_K) -> Dict[str, float]:
    """
    :param parsed: Output of :func:`parse`.
    :param gold: 0-based gold indices, or the example carrying them.
    :param k: Cutoff for MRR and recall.

    :returns: ``mrr@10``, ``recall@10``, ``parsed``.
    """
    indices = gold["gold_doc_indices"] if isinstance(gold, dict) else gold
    gold_ids = {int(g) + 1 for g in indices}
    if not parsed:
        return {f"mrr@{k}": 0.0, f"recall@{k}": 0.0, "parsed": 0.0}

    prefix = parsed[:k]
    rr = 0.0
    for rank, doc in enumerate(prefix, start=1):
        if doc in gold_ids:
            rr = 1.0 / rank
            break
    recall = len(set(prefix) & gold_ids) / len(gold_ids) if gold_ids else 0.0
    return {f"mrr@{k}": rr, f"recall@{k}": recall, "parsed": 1.0}


SPEC = TaskSpec(
    name="rerank",
    description="Order candidate documents by relevance to the query (MRR@10).",
    gold_index_base=0,
    instruction=RERANK_INSTRUCTION,
    instruction_variants=(rerank_instruction(TOP_K),),
    serializer="default",  # numbered, via TASKS_WITH_DOC_IDS
    unified=False,
    rungs=("2k", "4k", "8k", "16k", "32k"),
    build_prompt=build_prompt,
    parse=parse,
    score=score,
    primary_metric="mrr@10",
    max_new_tokens=512,
    stop="newline",
    answer_is_set=False,
    sources=("msmarco",),
    extra={"top_k": TOP_K, "requires_ce_scores": True},
)
