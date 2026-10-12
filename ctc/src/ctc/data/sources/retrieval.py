"""
The pool shape shared by every retrieval-flavoured ladder: ``nq``, ``hotpotqa``, ``fiqa``,
``scifact``, ``rerank``.

Five corpora, five mining strategies, one shape -- a query, its gold document(s), a hardest-first
list of *verified* hard negatives, and a corpus to draw ordinary distractors from. The
pre-migration tree wrote the tag/shuffle/enumerate idiom eight times across seven files and mined
hard negatives four different ways; only the mining genuinely differs, so only the mining is
per-corpus.

``hotpotqa`` is the one that does not mine at all: its candidates are the benchmark's own labelled
distractors, so its cross-encoder pass only *ranks* them (see :mod:`ctc.data.sources.hotpotqa`).
It is also the only one whose gold is reliably plural, which is why the multi-gold instruction and
scoring paths have a ladder exercising them.

.. warning::
   **A "hard negative" here is one that survived a relevance check.** BM25's top hits for a query
   are exactly the passages most likely to be *unlabeled positives*, and a sparsely-judged set like
   FiQA (~2.6 gold/query) is full of them. Every loader in this package therefore either
   cross-encoder-filters its candidates (:func:`margin_filter`) or documents why it does not.
   Skipping that step does not make the task harder, it makes the labels wrong: the model is
   penalised for retrieving a document that is relevant and simply unlabelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = ["Candidate", "QueryPool", "RetrievalPool", "margin_filter", "cross_encoder"]

#: Fraction of a pool's queries reserved for eval. One number, in one place: the pre-migration tree
#: had five train/eval splitters using 0.1 or 0.2 and ``int(round(x))`` or bare ``int(x)``.
EVAL_FRACTION = 0.1


@dataclass(frozen=True)
class Candidate:
    """
    One retrievable document.

    :param id: Corpus-stable id. Used for de-duplication across gold/hard/random, never emitted.
    :param text: The document body, already folded (BEIR folds ``title`` into ``text``).
    """

    id: str
    text: str


@dataclass(frozen=True)
class QueryPool:
    """
    Everything one example needs, before distractors are drawn.

    :param query: The question text.
    :param gold: Relevant document(s). At least one, or the example is dropped rather than emitted
        with empty gold -- retrieval F1 is undefined without it.
    :param hard: Verified hard negatives, **hardest first**. Order matters: a rung takes a prefix,
        so the same questions keep the same hardest negatives as the pool grows.
    :param answers: Answer strings, for the corpora that have them (NQ). Empty elsewhere; the
        retrieval task is graded on document ids, not answer text.
    :param fill: Ordinary distractors pre-drawn for *this* query. Set by loaders whose filler must
        itself be scored (``rerank``, where an unscored document silently drops out of the graded
        ordering). Empty means "draw from :attr:`RetrievalPool.corpus` at build time".
    :param scores: ``doc id -> cross-encoder relevance``, for the ladders graded on *ordering*
        (rerank). A missing id means "not scored", which the rerank grader treats as gain 0 and
        excludes from the Kendall-tau set -- not as irrelevant.
    """

    query: str
    gold: Tuple[Candidate, ...]
    hard: Tuple[Candidate, ...] = ()
    answers: Tuple[str, ...] = ()
    fill: Tuple[Candidate, ...] = ()
    scores: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalPool:
    """
    A whole corpus's worth of prepared queries plus the filler pool.

    :param queries: Prepared queries **in a fixed, already-shuffled order**. Example *i* of a build
        uses query *i*, so a build is resumable and shardable and no question is silently reused
        while another goes unused.
    :param corpus: Documents to draw ordinary (non-hard) distractors from. May be the same corpus
        the gold came from; loaders that sample a pool up front put it here.
    :param source: The ``source`` tag emitted on every example built from this pool.
    """

    queries: Tuple[QueryPool, ...]
    corpus: Tuple[Candidate, ...]
    source: str

    def __len__(self) -> int:
        return len(self.queries)

    def for_split(self, split: str) -> "RetrievalPool":
        """
        Slice off this split's queries.

        Disjoint **by query**, not by example: two examples built from one question are the same
        question however differently their distractors were drawn, so splitting anywhere else
        leaks. The corpus is shared deliberately -- a distractor is not a label.

        :param split: ``"train"`` or ``"eval"``.

        :returns: A pool holding only that split's queries.

        :raises ValueError: On an unknown split name.
        """
        cut = max(1, int(round(len(self.queries) * EVAL_FRACTION)))
        if split == "eval":
            kept = self.queries[:cut]
        elif split == "train":
            kept = self.queries[cut:]
        else:
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        return RetrievalPool(queries=kept, corpus=self.corpus, source=self.source)


def margin_filter(
    gold_scores: Sequence[float],
    candidates: Sequence[Candidate],
    candidate_scores: Sequence[float],
    margin: float,
) -> List[Candidate]:
    """
    Keep only candidates that score clearly *below* the gold -- SBERT's hard-negative recipe.

    A mined candidate scoring within ``margin`` of the gold is much more likely to be an unlabelled
    positive than a hard negative, and labelling it a negative is what turns a retrieval eval into
    a coin-flip on the sparsely-judged sets.

    :param gold_scores: Cross-encoder scores of this query's gold documents.
    :param candidates: Mined candidates, any order.
    :param candidate_scores: Their scores, parallel to ``candidates``.
    :param margin: Required gap below the best gold score. 3.0 is SBERT's value and the one every
        shipped ladder used.

    :returns: The survivors, **hardest (highest surviving score) first**.
    """
    if not gold_scores:
        # No gold score means no threshold. Treating every candidate as clean would silently admit
        # the unlabelled positives this filter exists to remove, so admit none.
        return []
    threshold = max(gold_scores) - margin
    kept = [(c, s) for c, s in zip(candidates, candidate_scores) if s < threshold]
    kept.sort(key=lambda pair: -pair[1])
    return [c for c, _ in kept]


def cross_encoder(model: Optional[str] = None, batch_size: int = 128):
    """
    Load the relevance scorer used by every hard-negative filter here.

    Lazily imported, and one implementation rather than the pre-migration tree's three near-copies
    -- one of which had the multi-GPU path and the other two did not.

    :param model: HF model id; defaults to ``cross-encoder/ms-marco-MiniLM-L-6-v2``, which is also
        the model MS MARCO's shipped score pickle came from, so all scores share one scale.
    :param batch_size: Pairs per forward pass, per GPU.

    :returns: An object with ``score(pairs) -> list[float]``.

    :raises ImportError: If the ``gen`` extra is not installed.
    """
    from .._scoring import CrossEncoderScorer

    return CrossEncoderScorer(model, batch_size=batch_size)
