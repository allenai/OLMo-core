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


def _ce_top_k(example: Dict, k: int) -> List[int]:
    """The ``k`` highest-CE document ids (1-based), CE=None (pooled-foreign docs) excluded."""
    ce = example.get("ce_scores") or []
    ranked = sorted((i for i, v in enumerate(ce) if v is not None),
                    key=lambda i: ce[i], reverse=True)
    return [i + 1 for i in ranked[:k]]


def _ce_pos_ref(example: Dict, k: int) -> List[int]:
    """The graded reference set: document ids (1-based) with ``CE > 0``, capped at the ``k``
    highest so the metric ceiling stays exactly 1.0 for a ``k``-slot answer.

    Measured on the token-accurate ladder (100 examples/rung, 2k and 32k): median 3 docs/example
    have CE > 0, p90 5, and there is essentially nothing in (-5, 0] -- relevance is bimodal. That
    is why the reference is the CE-POSITIVE set and not the CE-top-10: ranks 4-10 of the top-10
    are ordering noise among ~-11 CE docs that no model could reproduce, which put a ~0.5 luck
    ceiling on the top-10-recall variant this metric replaces (2026-08-14, measured before the
    first repriced number was ever produced)."""
    ce = example.get("ce_scores") or []
    pos = sorted((i for i, v in enumerate(ce) if v is not None and v > 0),
                 key=lambda i: ce[i], reverse=True)
    return [i + 1 for i in pos[:k]]


def build_target(example: Dict) -> str:
    """
    :param example: A unified-format example.

    :returns: The ``Ranking: [i], [j], ...`` answer line, 1-based -- the CE-ordered top-10 when
        ``ce_scores`` are present, which is both the SFT training target
        (``_rerank_reference_order``) and the answer :func:`score` grades for; the bare qrel gold
        otherwise.
    """
    ids = _ce_top_k(example, TOP_K) or [g + 1 for g in example["gold_doc_indices"]]
    return "Ranking: " + ", ".join(f"[{i}]" for i in ids)


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
    :param gold: The EXAMPLE (via ``extra["score_takes_example"]``), or bare 0-based gold
        indices from a legacy caller.
    :param k: Cutoff for every @k metric.

    :returns: ``ce_pos_recall`` (primary -- fraction of the CE-positive documents, median 3 /
        p90 5 per example, present in the model's first 10 ids), plus ``mrr@10``/``recall@10``
        against the single qrel gold, and ``parsed``. The qrel metrics saturate near 0.98 even
        for weak arms -- finding ONE known document is easy -- which is why they were demoted
        (prasann, 2026-08-14): the task is meant to measure tracking every genuinely relevant
        document.

    ``ce_pos_recall`` is always present (the registry contract requires the primary metric in
    every score() output); ``ce_ref_available`` says whether a CE reference set actually existed,
    so a legacy caller that passes bare indices produces a diagnosable 0/0 rather than a silent
    zero that reads as a model collapse.
    """
    example = gold if isinstance(gold, dict) else None
    indices = example["gold_doc_indices"] if example else gold
    gold_ids = {int(g) + 1 for g in indices}
    ce_ref = set(_ce_pos_ref(example, k)) if example else set()

    out: Dict[str, float] = {f"mrr@{k}": 0.0, f"recall@{k}": 0.0, "ce_pos_recall": 0.0,
                             "ce_ref_available": 1.0 if ce_ref else 0.0, "parsed": 0.0}
    if not parsed:
        return out

    prefix = parsed[:k]
    for rank, doc in enumerate(prefix, start=1):
        if doc in gold_ids:
            out[f"mrr@{k}"] = 1.0 / rank
            break
    out[f"recall@{k}"] = len(set(prefix) & gold_ids) / len(gold_ids) if gold_ids else 0.0
    if ce_ref:
        out["ce_pos_recall"] = len(set(prefix) & ce_ref) / len(ce_ref)
    out["parsed"] = 1.0
    return out


SPEC = TaskSpec(
    name="rerank",
    description="Order candidate documents by relevance; graded on CE-positive recall.",
    gold_index_base=0,
    instruction=RERANK_INSTRUCTION,
    instruction_variants=(rerank_instruction(TOP_K),),
    serializer="default",  # numbered, via TASKS_WITH_DOC_IDS
    unified=False,
    rungs=("2k", "4k", "8k", "16k", "32k"),
    build_prompt=build_prompt,
    parse=parse,
    score=score,
    primary_metric="ce_pos_recall",
    max_new_tokens=512,
    stop="newline",
    answer_is_set=False,
    sources=("msmarco",),
    extra={"top_k": TOP_K, "requires_ce_scores": True, "score_takes_example": True},
)
