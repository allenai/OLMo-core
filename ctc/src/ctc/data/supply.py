"""
Up-front supply check: refuse a rung the loaded pool provably cannot fill.

``ladders.CEILINGS`` refuses rungs whose impossibility is a property of the *task* (strmatch's
frozen vocabulary). But most bounds are properties of the *pool that was loaded* -- a bigger
pairs file lifts contradiction's, a wider filler pool lifts nq's -- so a static table would lie
the moment someone exports a larger pool. Without this check those builds fail at DRAW time, with
the 50-consecutive-rejections error, after minutes of work; the 10M sweep hit exactly that on
nine tasks.

Every bound here is deliberately **generous** (an upper bound on the documents one example could
ever hold, ignoring per-example exclusions like "fillers must avoid gold abstracts"). A refusal
is therefore always mathematically sound; passing the check does not guarantee the draw succeeds,
it only means the rung is not provably hopeless. The bound is keyed by the POOL CLASS NAME, not
the task, so one entry covers every ladder sharing a pool type and the module imports no task
code.

``OolongPool`` deliberately has NO bound: oolong draws items without replacement and, when the
pool runs out, **stops early** rather than failing -- an over-budget rung yields examples smaller
than their label, silently. That is a labeling problem the build report's extrapolation warning
covers, not a crash this check exists to prevent.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from . import ladders

__all__ = ["max_scaling_supply", "check"]


def _pubmed(pool: Any) -> Tuple[int, str]:
    total = sum(len(v) for v in pool.fillers.values())
    return total + 2 * len(pool.pairs), "filler sentences plus gold pairs"


def _retrieval(pool: Any) -> Tuple[int, str]:
    per_query = max((len(q.gold) + len(q.hard) + len(q.fill) for q in pool.queries), default=0)
    if pool.corpus:
        # Shared-corpus fillers: one example can hold the whole corpus plus its own query's
        # gold and mined negatives.
        return len(pool.corpus) + per_query, "corpus documents plus per-query candidates"
    return per_query, "pre-drawn documents on the best-supplied query"


def _book(pool: Any) -> Tuple[int, str]:
    return (
        max((len(r.sentences) for r in pool.runs), default=0),
        "sentences in the longest prose run",
    )


def _passage(pool: Any) -> Tuple[int, str]:
    return max((len(b.passages) for b in pool.books), default=0), "passages in the longest book"


def _paraphrase(pool: Any) -> Tuple[int, str]:
    return 2 * len(pool.pairs), "documents from the paraphrase-twin pairs"


def _unit(pool: Any) -> Tuple[int, str]:
    per_unit = 1 + max((len(u.gold) for u in pool.units), default=1)
    return per_unit * len(pool.units), "documents from the query units"


def _openalex(pool: Any) -> Tuple[int, str]:
    return len(pool.papers), "papers"


def _article(pool: Any) -> Tuple[int, str]:
    return sum(len(bodies) for _, bodies in pool.articles), "article passages"


def _review(pool: Any) -> Tuple[int, str]:
    return sum(len(v) for v in pool.by_category.values()), "reviews"


#: Pool class name -> (supply upper bound, what it counts). Keyed by class name rather than task
#: so one entry covers every ladder sharing the pool type, and the module imports no task code.
_BY_POOL: Dict[str, Callable[[Any], Tuple[int, str]]] = {
    "PubMedPool": _pubmed,
    "RedundancyPool": _pubmed,
    "RetrievalPool": _retrieval,
    "BookPool": _book,
    "PassagePool": _passage,
    "ParaphrasePool": _paraphrase,
    "UnitPool": _unit,
    "OpenAlexPool": _openalex,
    "ArticlePool": _article,
    "ReviewPool": _review,
}


def max_scaling_supply(pool: Any) -> Optional[Tuple[int, str]]:
    """
    :param pool: A loaded (and already split) corpus pool.

    :returns: ``(upper bound on the scaling axis one example can reach, what was counted)``, or
        ``None`` for a pool type with no registered bound.
    """
    fn = _BY_POOL.get(type(pool).__name__)
    return fn(pool) if fn is not None else None


def check(task: str, pool: Any, labels: Sequence[str]) -> None:
    """
    Refuse rungs the pool provably cannot fill, before any drawing starts.

    :param task: Ladder name, for :func:`ladders.docs_for_rung` and the message.
    :param pool: The split pool the build will draw from. ``None`` (synthetic) always passes.
    :param labels: The rung labels about to be built.

    :raises ValueError: Naming the first unfillable rung, what one example there needs, and what
        this pool can supply -- so the failure costs milliseconds and states its arithmetic,
        instead of surfacing as a rejection-limit error minutes into the draw.
    """
    if pool is None:
        return
    bound = max_scaling_supply(pool)
    if bound is None:
        return
    available, what = bound
    for label in labels:
        need = ladders.docs_for_rung(task, label)
        if need > available:
            raise ValueError(
                f"{task}: the {label} rung needs {need:,} per example, but the loaded pool "
                f"supplies at most {available:,} {what}. This bound is generous -- the draw "
                "cannot succeed. Drop the rung or export a larger pool "
                "(`ctc-data pool export`); the calibrated ladder tops out where the corpus does."
            )
