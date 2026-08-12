"""
The ``absence`` eval contract (AbsenceBench).

The inverse of needle-in-a-haystack: the model sees a full numbered corpus, then a second version
with some items removed, and reports which are missing. Items may be claims, poem lines, numbers or
diff lines depending on the source domain, which is why the serializer renders them as neutral
numbered items rather than as "claims".

The second (modified) version is carried in ``queries[0]`` and is positioned *after* the
instruction, because the instruction refers to it as "the text below".

.. note::
   **A second answer format exists.** The Gutenberg text-diff variant
   (``meta.format == "textdiff"``) answers with the first four words of each removed sentence, as a
   JSON list of strings, rather than with ids. The pre-migration scorer branched on that field at
   the top of ``_eval_absence``. It is detected the same way here, and the snippet parser carries
   its own primed-bracket fix -- absence_gutenberg under chunked attention read ``parse_rate ~0.01``
   because the model dropped the leading ``[``, while the snippets it emitted were correct.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Set, Union

from ...format import metrics, parsing
from ...format.prompts import ABSENCE_INSTRUCTION
from .._absence import make_absence_spec
from .._absence import parse as parse_ids

__all__ = ["SPEC", "build_query", "build_target", "is_textdiff"]


def is_textdiff(example: Dict) -> bool:
    """
    :param example: A unified-format example.

    :returns: Whether this example uses the Gutenberg snippet answer format rather than ids.

    Both metadata spellings are accepted, and that is not tidiness: the shipped pre-migration files
    write bare ``meta`` while :func:`ctc.data.schema.make_example` normalises to ``_meta``, so
    reading only one spelling makes this return ``False`` for every example built in this repo --
    silently routing textdiff data to the id scorer.
    """
    meta = example.get("_meta") or example.get("meta") or {}
    return meta.get("format") == "textdiff"


def build_query(example: Dict) -> str:
    """
    :param example: A unified-format example whose ``queries[0]`` holds the modified corpus.

    :returns: The instruction followed by the modified version -- the instruction says "the text
        below", so the order is load-bearing.
    """
    return f"{ABSENCE_INSTRUCTION}\n\n{example['queries'][0]}"


def parse(text: str, n_docs: int) -> Union[Set[int], Sequence[str], None]:
    """
    Parse an absence answer, in whichever of the two formats applies.

    Both formats cannot be distinguished from the generation alone, so the id path is the default
    and the snippet path is reached via :func:`score` when the example declares it.

    :param text: Raw model generation.
    :param n_docs: Corpus size.

    :returns: Ids, snippets, or ``None``.
    """
    return parse_ids(text, n_docs)


def build_target(example: Dict) -> str:
    """
    :param example: A unified-format example.

    :returns: The ``Missing: [i], [j]`` answer line, with ids rendered 1-based.
    """
    from .._absence import build_target as _bt

    return _bt(example, "Missing")


def score_textdiff(parsed_snippets: Optional[Sequence[str]], gold_answers: Sequence[str]) -> Dict:
    """
    Score the Gutenberg snippet variant.

    :param parsed_snippets: Output of :func:`ctc.format.parsing.parse_snippet_list`.
    :param gold_answers: The example's ``answers``, i.e. the removed sentences' opening words.

    :returns: Set metrics over normalized snippets, plus ``parsed``.
    """
    gold = {parsing.normalize_snippet(s) for s in gold_answers}
    if parsed_snippets is None:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "parsed": 0.0}
    pred = {n for s in parsed_snippets if (n := parsing.normalize_snippet(s))}
    return {**metrics.set_metrics(pred, gold), "parsed": 1.0}


SPEC = make_absence_spec(
    name="absence",
    description="Identify which numbered items were removed from a modified copy of the corpus.",
    instruction=ABSENCE_INSTRUCTION,
    serializer="absence",
    rungs=("2k", "4k", "8k", "16k", "32k"),
    query_builder=build_query,
    max_new_tokens=200,
    sources=("pubmed", "gutenberg", "poetry", "numbers"),
)
