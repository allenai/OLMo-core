"""
The ``contradiction`` eval contract.

Given a corpus of numbered claims, find every pair that cannot both be true. The answer is a set of
unordered pairs, scored by set-F1 over those pairs.

This is the suite's flagship N-squared task: the work is quadratic in corpus size, so it is the one
where a model that can only attend locally is expected to fall off first. It is also the task with
the largest history of grading bugs -- see :mod:`ctc.format.parsing` -- which is why the parse and
score functions here are references to shared, golden-tested implementations rather than local
copies.

Two facts that have each cost real debugging time:

* **Gold indices are 1-based here** and 0-based for outlier, rerank and nq. That lived in people's
  heads until it produced an off-by-one that read as a modelling result.
* **The rung label bounds corpus size, not prompt length.** These rung files fix the claim count
  (see :data:`CLAIMS_PER_RUNG`) and let per-claim length vary, so measured prompts at the 4k rung
  spanned 3,457-23,796 tokens. Sizing a decode budget from the label silently skipped 354/500
  examples and scored them 0 -- in both arms, which read as "no dense-vs-chunked gap".

.. warning::
   **``stop="pairs"`` is a deliberate change from the pre-migration evaluator.** There,
   contradiction ran under ``stop_rule="eos"``, whose ``should_stop`` returns ``False``
   unconditionally -- so nothing but EOS or the budget ended generation. No-cot checkpoints
   frequently never emit EOS, so they answered correctly and then rambled to the budget, and
   whatever the parser made of the ramble is what got scored. Stopping at the closing ``]]``
   terminates on the actual answer. This can move numbers relative to previously reported ones, in
   either direction, and is worth a back-to-back comparison on a known checkpoint before it is
   relied on.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence

from ...data.ladders import LADDERS
from ...format import assemble, metrics, parsing
from ...format.prompts import CONTRADICTION_INSTRUCTION, GENERIC_INSTRUCTION
from ...format.registry import TaskSpec

#: Corpus size per rung: claims, not tokens. See the module note on why this is not prompt length.
#:
#: These are the **re-calibrated** counts. The earlier ladder (77/167/346/705/1423) was fit against
#: a filler pool that turned out to be 92-99.6% FEVER/wiki -- one-line Wikipedia trivia at ~22.8
#: tokens per claim. Real PubMed claim sentences run ~43 tokens, so those counts overshoot every
#: token label by roughly 1.8x against the corrected pubmed-only pool: n=77 measures 3,413 tokens
#: rather than 2,048, and n=1423 measures 61,461 rather than 32,768. Refit over 25 examples per
#: rung on full rendered prefills gives ``tokens = 170 + 42.8 * n_docs`` (r^2 ~ 1.00 across n in
#: [77, 1423]), hence ``n = (target - 170) / 42.8``.
#:
#: The refit lands within a few documents of the ORIGINAL ladder (40/88/190/385/765), which was
#: calibrated on real PubMed and was right all along -- the intermediate "fix" only looked
#: necessary because the pool was contaminated.
#:
#: **Derived, not declared.** :data:`ctc.data.ladders.LADDERS` is what the builder actually reads,
#: so a literal here would be a second copy of a number that has already been wrong once -- and the
#: copy the tests pin, while the build used the other. Deriving makes drift impossible rather than
#: merely detectable.
CLAIMS_PER_RUNG: Dict[str, int] = dict(LADDERS["contradiction"])


def parse(text: str, n_docs: Optional[int] = None) -> Optional[List[List[int]]]:
    """
    Parse a contradiction answer into sorted claim-id pairs.

    :param text: Raw model generation.
    :param n_docs: Corpus size. Unused -- pairs are not range-filtered, so a hallucinated id counts
        against precision rather than vanishing. Silently dropping out-of-range ids would flatter a
        model that invents them.

    :returns: The pairs, ``[]`` for an explicit empty answer, or ``None`` when nothing parsed.
    """
    return parsing.parse_pairs(text)


def _check_gold(gold: Sequence[Sequence[int]]) -> None:
    """
    Assert the invariant the set comparison depends on.

    :param gold: Gold pairs.

    :raises ValueError: If a pair is not sorted low-high, which would make it unmatchable.
    """
    for p in gold:
        if len(p) == 2 and p[0] > p[1]:
            raise ValueError(
                f"gold pair {list(p)} is not sorted low-high. Predicted pairs are sorted, and "
                "scoring is a set intersection, so this pair could never be matched and would "
                "silently cost recall on every example that contains it."
            )


def score(
    parsed: Optional[Sequence[Sequence[int]]], gold: Sequence[Sequence[int]]
) -> Dict[str, float]:
    """
    Score predicted pairs against gold.

    .. warning::
       **This deliberately diverges from the pre-migration scorer.** ``_eval_contradiction`` mapped
       an unparseable generation to ``[]`` and scored it normally, so on an example with *no* gold
       pairs a garbage generation scored a perfect 1.0 on all four metrics. Contradiction data has
       no empty-gold examples (checked: 0 of 3,153 across three unified files), so this never fired
       there -- but the same scorer is shared by redundancy, strmatch and mathmatch, whose
       instructions explicitly permit an empty answer. Here ``None`` scores zero instead. Numbers
       from this scorer are therefore **not** guaranteed comparable with old ones on any task that
       has empty-gold examples.

    .. note::
       Gold pairs must be sorted low-high, because :func:`parse` sorts predicted pairs and the
       comparison is a set intersection -- a gold ``[4, 1]`` could never match a predicted
       ``[1, 4]``. The generators satisfy this (checked: 0 unsorted of 3,153) but never asserted
       it, so :func:`_check_gold` does.

    :param parsed: Output of :func:`parse`. ``None`` (unparseable) scores zero on every metric --
        it is not the same as ``[]``, which is a real answer and can be correct.
    :param gold: Gold pairs, 1-based, each sorted low-high.

    :returns: ``precision``, ``recall``, ``f1``, ``exact_match``, plus ``parsed`` as a 0/1 flag.
        Track ``parsed``: a drop in it means a decoding or truncation problem, and without it that
        is indistinguishable from the model getting worse.

    :raises ValueError: If a gold pair is not sorted low-high.
    """
    _check_gold(gold)
    if parsed is None:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "parsed": 0.0}
    return {**metrics.pair_metrics(parsed, gold), "parsed": 1.0}


def build_query(example: Dict) -> str:
    """
    The positioned ask.

    Contradiction has no per-example question, so the instruction *is* the ask and the example's
    ``queries`` field is unused. That is exactly why the task is ``unified``.

    :param example: A unified-format example.

    :returns: The text placed before/after/both relative to the claims.
    """
    return CONTRADICTION_INSTRUCTION


def build_target(example: Dict) -> str:
    """
    The training target: the gold pairs as JSON.

    :param example: A unified-format example. ``gold_doc_indices`` holds the pairs directly, as
        ``[[a, b], ...]`` of **1-based** claim ids -- not document positions, despite the field
        name it shares with the 0-based tasks.

    :returns: A JSON list of pairs, e.g. ``"[[1, 4], [3, 7]]"``.
    """
    return json.dumps(example["gold_doc_indices"])


def build_prompt(example: Dict, **opts) -> str:
    """
    Render one contradiction example into a prompt.

    :param example: A unified-format example with ``documents`` (the claims).
    :param opts: Assembly options -- ``query_position``, ``use_alpaca``, ``use_titles``,
        ``before_dummy``, ``after_dummy``. See :func:`ctc.format.assemble.assemble`.

    :returns: The prompt string.
    """
    return assemble.assemble(
        example,
        task=SPEC.name,
        unified=SPEC.unified,
        header=GENERIC_INSTRUCTION,
        positioned=build_query(example),
        **opts,
    )


SPEC = TaskSpec(
    name="contradiction",
    description="Find every pair of claims that cannot both be true (N^2 over the corpus).",
    gold_index_base=1,
    instruction=CONTRADICTION_INSTRUCTION,
    serializer="contradiction",
    # No per-example query -- "find every contradicting pair" IS the ask, so the instruction takes
    # the positioned slot and the alpaca header is the generic one.
    unified=True,
    rungs=("2k", "4k", "8k", "16k", "32k"),
    build_prompt=build_prompt,
    parse=parse,
    score=score,
    primary_metric="f1",
    max_new_tokens=512,
    stop="pairs",
    answer_is_set=True,
    sources=("pubmed", "fever", "wiki"),
    extra={"claims_per_rung": CLAIMS_PER_RUNG},
)
