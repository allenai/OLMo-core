"""
The ``qa`` eval contract.

Answer the question from the documents, in free text. The only ported task whose answer is prose
rather than a structure, which is why it is the only one scored by text overlap.

Scoring is deliberately **lenient**: the pre-migration ``compute_qa_metrics`` takes the maximum
over several extraction strategies -- the raw generation, the text after an ``Answer:`` prefix, the
text after ``</think>``, and that text's own ``Answer:`` line. A model that produces the right
answer in any of those positions gets credit. That is a defensible choice for free-text QA, but it
means a qa number is not comparable with the structured tasks' set-F1, and that ``exact_match``
here is far more forgiving than exact_match elsewhere in the suite.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from ...format import assemble, metrics
from ...format.prompts import MULTI_QA_INSTRUCTION, QA_INSTRUCTION
from ...format.registry import TaskSpec
from .._retrieval import qa_instruction, questions_block

__all__ = ["SPEC", "build_query", "build_target", "parse", "score", "extraction_candidates"]


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
        task="qa",
        unified=False,
        header=qa_instruction(example),
        positioned=build_query(example),
        **opts,
    )


def build_target(example: Dict) -> str:
    """
    :param example: A unified-format example.

    :returns: The bare answer text, **without** an ``Answer:`` prefix.

        The instruction asks for ``Answer: [answer]``, so a prefix would look right -- but the
        trained target is the bare answer, and the prompt's ``### Response:\\n`` is where the model
        begins. Adding the prefix would shift every qa target by two tokens against every existing
        checkpoint. (cot_retrieval, confusingly, *does* prefix its target. The two are inconsistent
        in the pre-migration data and the port preserves that rather than tidying it.)
    """
    answers = example["answers"]
    if len(answers) > 1:
        return ", ".join(str(a) for a in answers)
    return str(answers[0])


def extraction_candidates(text: str) -> List[str]:
    """
    Every span a QA answer might occupy.

    :param text: Raw model generation.

    :returns: The candidate spans, most-raw first. Scoring takes the best of these, which is what
        makes qa lenient -- see the module docstring.
    """
    out = [text]
    after_think = text.split("</think>", 1)[1] if "</think>" in text else None
    if after_think is not None:
        out.append(after_think)
        first_line = after_think.strip().split("\n")[0].strip()
        if first_line:
            out.append(first_line)
    for span in list(out):
        if "Answer:" in span:
            out.append(span.split("Answer:", 1)[1].strip().split("\n")[0].strip())
    return [s for s in out if s.strip()]


def parse(text: str, n_docs: Optional[int] = None) -> Optional[str]:
    """
    :param text: Raw model generation.
    :param n_docs: Unused.

    :returns: The generation itself. QA does not parse into a structure -- extraction happens
        inside :func:`score`, which needs the gold answers to pick the best span.
    """
    return text if text.strip() else None


def score(parsed: Optional[str], gold: Sequence[str]) -> Dict[str, float]:
    """
    :param parsed: Output of :func:`parse`.
    :param gold: Acceptable answers; many datasets supply several aliases.

    :returns: ``exact_match``, ``substring_exact_match``, ``f1``, ``parsed``.
    """
    if parsed is None:
        return {
            "exact_match": 0.0,
            "substring_exact_match": 0.0,
            "f1": 0.0,
            "parsed": 0.0,
        }
    answers = list(gold) if not isinstance(gold, str) else [gold]
    best = {"exact_match": 0.0, "substring_exact_match": 0.0, "f1": 0.0}
    for span in extraction_candidates(parsed):
        best["exact_match"] = max(
            best["exact_match"], float(metrics.max_over_answers(metrics.exact_match, span, answers))
        )
        best["substring_exact_match"] = max(
            best["substring_exact_match"],
            float(metrics.max_over_answers(metrics.substring_match, span, answers)),
        )
        best["f1"] = max(
            best["f1"], float(metrics.max_over_answers(metrics.token_f1, span, answers))
        )
    return {**best, "parsed": 1.0}


SPEC = TaskSpec(
    name="qa",
    description="Answer the question from the documents, in free text.",
    gold_index_base=0,
    instruction=QA_INSTRUCTION,
    instruction_variants=(MULTI_QA_INSTRUCTION,),
    serializer="default",  # unnumbered: qa is not in TASKS_WITH_DOC_IDS
    unified=False,
    rungs=("2k", "4k", "8k", "16k", "32k"),
    build_prompt=build_prompt,
    parse=parse,
    score=score,
    primary_metric="f1",
    max_new_tokens=64,
    stop="newline",
    answer_is_set=False,
    sources=("nq", "hotpotqa", "helmet_qa"),
)
