"""
Rung -> document count, per task. The calibration table for the CTC suite.

A rung is a *token* budget; a generator takes a *document* count. This table is the bridge, and it
is per task because a contradiction claim is ~42 tokens while a BEIR SciFact abstract is ~365 -- the
same "8k" means 187 documents for one and 21 for the other.

**Getting this wrong is not a sizing nuisance, it is a mislabelled axis.** The pre-migration
contradiction ladder was fit against a filler pool that turned out to be 92-99.6% FEVER/wiki rather
than PubMed. Wikipedia trivia claims tokenize at ~22.8 tok/doc against PubMed's ~43, so re-running
the same document counts against the corrected pool overshot every rung by ~1.8x -- the file called
``rung_2048`` actually rendered to 3413 tokens, and ``32k`` to 61461. Every number plotted against
that axis was plotted against the wrong x.

So each row records how it was calibrated, and ``estimated`` rows are exactly the ones to re-measure
before quoting a length. The authority for un-ported tasks remains
``src/scripts/data/ctc_suite/BUILD_MATRIX.md`` in the pre-migration tree; rows land here as their
generators do. Contradiction's corrected row currently lives in its spec's
``extra["claims_per_rung"]`` and should move here when that generator is ported.
"""

from __future__ import annotations

from typing import Dict, List

from ..format import rungs as rung_util

__all__ = ["LADDERS", "docs_for_rung", "rungs_for", "max_rung"]

#: task -> {rung label: documents per example}. Synthetic tasks tokenize 1.5-3x higher per
#: character than natural text, which is why their per-document estimates are so much lower than
#: the retrieval tasks'.
LADDERS: Dict[str, Dict[str, int]] = {
    # ~25-30 tok/claim. BUILD_MATRIX recommends capping at n1000 and letting 32k run slightly short.
    "cycle": {"2k": 60, "4k": 130, "8k": 270, "16k": 550, "32k": 1100},
    # ~15-20 tok/expression -- the shortest documents in the suite, hence the largest counts.
    "groups4": {"2k": 100, "4k": 210, "8k": 440, "16k": 900, "32k": 1800},
    "mathmatch": {"2k": 48, "4k": 105, "8k": 220, "16k": 450, "32k": 900},
    # ~150 tok/passage: the feature must be spread densely over several sentences, so a textgroups
    # document is an order of magnitude longer than a groups4 one at the same task shape.
    "textgroups": {"2k": 11, "4k": 24, "8k": 50, "16k": 103, "32k": 210},
}

#: How each row was arrived at. ``estimated`` means an offline per-document token estimate, not a
#: measurement against the real tokenizer -- re-measure before quoting a context length.
CALIBRATION: Dict[str, str] = {
    "cycle": "estimated",
    "groups4": "estimated",
    "mathmatch": "estimated",
    "textgroups": "estimated",
}


def rungs_for(task: str) -> List[str]:
    """
    :param task: Task name.

    :returns: Its rung labels, ascending by context length.

    :raises KeyError: If the task has no ladder.
    """
    if task not in LADDERS:
        raise KeyError(
            f"no rung ladder for {task!r}; have {', '.join(sorted(LADDERS))}. Un-ported tasks keep "
            "their ladder in the pre-migration BUILD_MATRIX.md."
        )
    return rung_util.sort_rungs(LADDERS[task])


def docs_for_rung(task: str, rung: str) -> int:
    """
    :param task: Task name.
    :param rung: Rung label, in any accepted spelling (``"32k"``, ``"32768"``).

    :returns: Documents per example at that rung.

    :raises KeyError: If the task or rung is unknown.
    """
    ladder = {rung_util.normalize(k): v for k, v in LADDERS[task].items()}
    label = rung_util.normalize(rung)
    if label not in ladder:
        raise KeyError(f"{task} has no rung {label!r}; has {', '.join(rungs_for(task))}")
    return ladder[label]


def max_rung(task: str) -> str:
    """
    :param task: Task name.

    :returns: Its longest rung -- where a nested eval ladder is built before being shrunk down.
    """
    return rungs_for(task)[-1]
