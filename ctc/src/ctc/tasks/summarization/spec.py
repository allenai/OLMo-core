"""
The ``summarization`` eval contract (HELMET: ∞Bench Sum, Multi-LexSum, GovReport).

Write a faithful summary of the documents. Graded by ROUGE.

.. warning::
   **ROUGE requires ``pip install 'ctc[rouge]'``, and its absence is an ERROR here, not a
   fallback.** The pre-migration scorer caught the ImportError and silently wrote token-F1 under
   the keys ``rouge1_f``/``rougeL_f``. It did exactly that for the whole ``helmet_summ`` column of
   the CTC grid -- ``rouge_score`` was installed on sneetches but its dependency ``absl-py`` was
   not -- so every published "ROUGE" number there was really token-F1. The tell is
   ``rouge1_f == rougeL_f`` to full precision, which ROUGE-1 and ROUGE-L never are.

   Refusing to score is the correct behaviour: a metric that quietly becomes a different metric is
   worse than one that fails. :func:`score` therefore raises unless ``rouge_score`` imports, and
   ``score_token_f1`` exists as an explicit, differently-named opt-in for anyone who genuinely
   wants the cheap approximation.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from ...format import assemble, metrics
from ...format.prompts import SUMMARIZATION_INSTRUCTION
from ...format.registry import TaskSpec
from .._retrieval import questions_block

__all__ = ["SPEC", "build_query", "build_target", "parse", "score", "score_token_f1"]


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
        task="summarization",
        unified=False,
        header=SUMMARIZATION_INSTRUCTION,
        positioned=build_query(example),
        **opts,
    )


def build_target(example: Dict) -> str:
    """
    :param example: A unified-format example.

    :returns: The reference summary.
    """
    return str(example["answers"][0])


def parse(text: str, n_docs: Optional[int] = None) -> Optional[str]:
    """
    :param text: Raw model generation.
    :param n_docs: Unused.

    :returns: The generation, or ``None`` when it is empty. Summaries are free text; there is
        nothing to parse into.
    """
    return text if text.strip() else None


def score(parsed: Optional[str], gold: Sequence[str]) -> Dict[str, float]:
    """
    ROUGE against the reference summary.

    :param parsed: Output of :func:`parse`.
    :param gold: Reference summary/summaries.

    :returns: ``rouge1_f``, ``rougeL_f``, ``parsed``.

    :raises ImportError: If ``rouge_score`` is unavailable. Deliberate -- see the module warning.
        Silently substituting token-F1 under the ROUGE key is what corrupted the published
        helmet_summ column.
    """
    # Checked BEFORE the import: an unparseable generation scores zero regardless of which
    # library is installed, and requiring rouge to say so would make a missing dependency look
    # like a scoring failure.
    if parsed is None:
        return {"rouge1_f": 0.0, "rougeL_f": 0.0, "parsed": 0.0}

    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        raise ImportError(
            f"summarization needs rouge_score ({e}). Install with: pip install 'ctc[rouge]'. "
            "This is deliberately NOT falling back to token_f1: the pre-migration scorer did, "
            "under the SAME metric keys, and silently relabelled the entire helmet_summ column. "
            "If you genuinely want the approximation, call score_token_f1 explicitly."
        ) from None

    refs = list(gold) if not isinstance(gold, str) else [gold]
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    best = {"rouge1_f": 0.0, "rougeL_f": 0.0}
    for ref in refs:
        s = scorer.score(str(ref), parsed)
        best["rouge1_f"] = max(best["rouge1_f"], s["rouge1"].fmeasure)
        best["rougeL_f"] = max(best["rougeL_f"], s["rougeL"].fmeasure)
    return {**best, "parsed": 1.0}


def score_token_f1(parsed: Optional[str], gold: Sequence[str]) -> Dict[str, float]:
    """
    Token-F1 approximation, under **distinct keys** so it can never be mistaken for ROUGE.

    :param parsed: Output of :func:`parse`.
    :param gold: Reference summary/summaries.

    :returns: ``token_f1``, ``parsed``. Note the key: this is the whole point of the function.
    """
    if parsed is None:
        return {"token_f1": 0.0, "parsed": 0.0}
    refs = list(gold) if not isinstance(gold, str) else [gold]
    return {
        "token_f1": float(metrics.max_over_answers(metrics.token_f1, parsed, refs)),
        "parsed": 1.0,
    }


SPEC = TaskSpec(
    name="summarization",
    description="Write a faithful summary of the documents (ROUGE).",
    gold_index_base=0,
    instruction=SUMMARIZATION_INSTRUCTION,
    serializer="summarization",
    unified=False,
    rungs=("2k", "4k", "8k", "16k", "32k"),
    build_prompt=build_prompt,
    parse=parse,
    score=score,
    primary_metric="rouge1_f",
    max_new_tokens=1024,
    stop="eos",
    answer_is_set=False,
    sources=("helmet_summ", "govreport", "multi_lexsum", "infbench_sum"),
)
