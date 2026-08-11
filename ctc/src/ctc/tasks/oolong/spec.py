"""
The ``oolong`` eval contract (Oolong, arXiv:2511.02817).

Aggregate reasoning over many labelled items: "how many are labelled X?", "which user posted
most?", "what is the date range?". The answer depends on *every* item, so it cannot be produced by
retrieving one span.

Three answer types, three scoring rules, chosen per example from ``_meta.answer_type``:

============  =========================================================================
NUMERIC       partial credit ``0.75 ** |gold - pred|`` -- being off by one beats being
              off by ten, which a strict exact-match would hide
list          set-overlap F1 over the semicolon/comma-split answer
everything     exact match after normalization
============  =========================================================================

Two task-specific quirks:

* The per-example ``question`` already carries the full instruction and answer-format spec, so the
  header is minimal and the context block is rendered verbatim with no per-document wrapper.
* Stopping keys on the templated ``answer:`` line rather than the first newline, because the
  preamble is legitimately multi-line -- see the ``oolong`` stop preset.

.. warning::
   **Rebuild any oolong shard built before 2026-07-26.** The ``--item-regex`` leak (a bare ``'||'``
   matched every line) affected shards built before that date.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from ...format import assemble
from ...format.prompts import OOLONG_INSTRUCTION
from ...format.registry import TaskSpec
from .._retrieval import questions_block

__all__ = ["SPEC", "build_query", "build_target", "parse", "score", "normalize"]


def normalize(s: str) -> str:
    """
    :param s: An answer fragment.

    :returns: Lowercased with brackets and quotes stripped -- the comparison form for the
        exact-match and set-overlap paths.
    """
    return re.sub(r"[\[\]'\"]", "", str(s)).strip().lower()


def build_query(example: Dict) -> str:
    """
    :param example: A unified-format example whose ``queries[0]`` carries the full question and its
        answer-format spec.

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
        task="oolong",
        unified=False,
        header=OOLONG_INSTRUCTION,
        positioned=build_query(example),
        **opts,
    )


def build_target(example: Dict) -> str:
    """
    :param example: A unified-format example.

    :returns: The answer text.
    """
    return str(example["answers"][0])


def parse(text: str, n_docs: Optional[int] = None) -> Optional[str]:
    """
    :param text: Raw model generation.
    :param n_docs: Unused.

    :returns: The text after the templated ``answer:`` marker when present, else the whole
        generation; ``None`` when empty.
    """
    if not text.strip():
        return None
    lowered = text.lower()
    if "answer:" in lowered:
        at = lowered.rindex("answer:") + len("answer:")
        return text[at:].strip()
    return text.strip()


def score(parsed: Optional[str], gold, answer_type: str = "") -> Dict[str, float]:
    """
    Score by the answer type this example declares.

    :param parsed: Output of :func:`parse`.
    :param gold: The gold answer(s), or the example carrying ``_meta.gold_list``/``answers``.
    :param answer_type: From ``_meta.answer_type``; ``"NUMERIC"`` selects partial credit. Ignored
        when ``gold`` is an example, which carries its own.

    :returns: ``score`` (the primary; partial credit for numerics) and ``exact_match``, plus
        ``parsed``.
    """
    if isinstance(gold, dict):
        meta = gold.get("_meta") or {}
        answer_type = meta.get("answer_type", answer_type)
        gold_list = meta.get("gold_list") or [gold["answers"][0]]
    else:
        gold_list = list(gold) if not isinstance(gold, str) else [gold]

    if parsed is None:
        return {"score": 0.0, "exact_match": 0.0, "parsed": 0.0}

    if "NUMERIC" in (answer_type or ""):
        nums = re.findall(r"-?\d+\.?\d*", parsed)
        try:
            err = abs(float(gold_list[0]) - float(nums[-1]))
            # Geometric decay, not a threshold: off-by-one must score better than off-by-ten, and
            # a hard exact-match would report both as total failure.
            return {"score": 0.75**err, "exact_match": float(err == 0), "parsed": 1.0}
        except (ValueError, IndexError):
            return {"score": 0.0, "exact_match": 0.0, "parsed": 1.0}

    if len(gold_list) > 1:
        pset = {normalize(x) for x in re.split(r"[;,]", parsed)}
        gset = {normalize(x) for x in gold_list}
        tp = len(pset & gset)
        p = tp / len(pset) if pset else 0.0
        r = tp / len(gset) if gset else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"score": f1, "exact_match": float(pset == gset), "parsed": 1.0}

    em = float(normalize(parsed) == normalize(gold_list[0]))
    return {"score": em, "exact_match": em, "parsed": 1.0}


SPEC = TaskSpec(
    name="oolong",
    description="Aggregate reasoning over many labelled items (partial credit).",
    gold_index_base=0,
    instruction=OOLONG_INSTRUCTION,
    serializer="oolong",
    unified=False,
    rungs=("2k", "4k", "8k", "16k", "32k"),
    build_prompt=build_prompt,
    parse=parse,
    score=score,
    primary_metric="score",
    max_new_tokens=256,
    stop="oolong",
    answer_is_set=False,
    sources=("oolong",),
    # The one task in the suite whose chunks are not documents. An oolong example is a single
    # context block of labelled lines, so document chunking would wrap the whole context in one
    # marker pair -- the training converter chunks by line, and eval must match or the model is
    # graded against a token stream it never saw.
    #
    # `item_regex` is the ESCAPED pattern, matching a literal `||` separator. A bare `'||'` is a
    # regex alternation of two empty strings, so it matches every line -- that is the leak that
    # wrapped oolong preambles as chunks in shards built before 2026-07-26.
    extra={"chunk_by": "line", "item_regex": r"\|\|"},
)
