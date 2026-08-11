"""
Shared machinery for the absence family: absence and xabsence.

Both ask *what is NOT here?* -- the inverse of needle-in-a-haystack. ``absence`` shows an original
numbered corpus and a modified version, and asks which items were removed. ``xabsence`` shows two
corpora and asks which claims have no paraphrase in the other. The answer is a flat set of ids in
both cases, so they share a parser and a scorer.

.. warning::
   **Gold is 0-BASED here.** The pre-migration scorer does ``{int(g) + 1 for g in
   gold_doc_indices}`` -- the stored indices are document positions, and it converts them to the
   1-based ids the prompt actually renders. Contradiction and the whole pair family are 1-based, so
   this is one of the places the convention genuinely flips. Getting it wrong shifts every id by
   one and reads as a weak model rather than as a bug.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence, Set

from ..format import assemble, metrics, parsing
from ..format.prompts import GENERIC_INSTRUCTION
from ..format.registry import TaskSpec

__all__ = ["parse", "score", "build_target", "make_absence_spec"]


def parse(text: str, n_docs: int) -> Optional[Set[int]]:
    """
    Parse an absence answer.

    :param text: Raw model generation, anchored on ``Missing:`` or ``Unmatched:``.
    :param n_docs: Corpus size; ids outside ``1..n_docs`` are dropped.

    :returns: The ids, an empty set for an explicit "nothing is missing", or ``None``.
    """
    return parsing.parse_id_set(text, n_docs)


def score(parsed: Optional[Set[int]], gold: Sequence[int]) -> Dict[str, float]:
    """
    Score a predicted id set against gold.

    :param parsed: Output of :func:`parse`. ``None`` scores zero; an empty set does not, since
        "nothing is missing" is a real answer that can be right.
    :param gold: Gold indices, **0-based** -- converted here to the 1-based ids the model sees.

    :returns: ``precision``, ``recall``, ``f1``, ``exact_match``, ``parsed``.
    """
    gold_ids = {int(g) + 1 for g in gold}
    if parsed is None:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "parsed": 0.0}
    return {**metrics.set_metrics(set(parsed), gold_ids), "parsed": 1.0}


def build_target(example: Dict, anchor: str) -> str:
    """
    :param example: A unified-format example with 0-based ``gold_doc_indices``.
    :param anchor: ``"Missing"`` or ``"Unmatched"``, matching the task's instruction.

    :returns: The answer line, with ids rendered 1-based and bracketed.
    """
    ids = sorted(int(g) + 1 for g in example["gold_doc_indices"])
    return f"{anchor}: " + ", ".join(f"[{i}]" for i in ids)


def make_absence_spec(
    *,
    name: str,
    instruction: str,
    serializer: str,
    description: str,
    rungs: tuple,
    query_builder: Callable[[Dict], str],
    max_new_tokens: int = 200,
    stop: str = "newline",
    sources: tuple = (),
) -> TaskSpec:
    """
    Build the spec for one absence-family task.

    :param name: Task name.
    :param instruction: The instruction string, verbatim.
    :param serializer: Document serializer key.
    :param description: One line for ``--list-tasks``.
    :param rungs: The task's ladder.
    :param query_builder: ``(example) -> str``; the positioned ask.
    :param max_new_tokens: Decode budget.
    :param stop: Stop preset. ``newline`` is right here -- the answer is a single ``Missing: ...``
        line, and unlike the pair tasks there is no bracket-pair terminator to key on.
    :param sources: Source corpora.

    :returns: The spec, not yet registered.
    """

    def build_prompt(example: Dict, **opts) -> str:
        return assemble.assemble(
            example,
            task=name,
            unified=True,
            header=GENERIC_INSTRUCTION,
            positioned=query_builder(example),
            **opts,
        )

    return TaskSpec(
        name=name,
        description=description,
        gold_index_base=0,
        instruction=instruction,
        serializer=serializer,
        unified=True,
        rungs=rungs,
        build_prompt=build_prompt,
        parse=parse,
        score=score,
        primary_metric="f1",
        max_new_tokens=max_new_tokens,
        stop=stop,
        answer_is_set=False,  # a flat id set, not the bracketed-pair shape the "pairs" stop expects
        sources=sources,
    )
