"""
``msmarco._pool_scores``: the per-document relevance map rerank's graded ordering is built from.

The fixture rerank pool bypasses the loader entirely, so nothing else exercises this function --
and its worst historical failure was invisible to every other test: the gold-fallback local was
named ``fill``, shadowing the ``fill`` *parameter* (the random-fill candidates), so the
``score_all`` branch -- the default, and the one the module docstring calls load-bearing -- raised
``TypeError`` on the first query with an unscored fill document. That killed every real rerank
pool export while the fixture-backed tests stayed green.
"""

from __future__ import annotations

from ctc.data.sources.msmarco import _pool_scores
from ctc.data.sources.retrieval import Candidate


class _FakeScorer:
    def score(self, pairs):
        return [42.0] * len(pairs)


def test_score_all_scores_the_random_fill():
    gold = (Candidate("1", "the answer"),)
    hard = (Candidate("2", "nearly the answer"),)
    fill = (Candidate("3", "unrelated"), Candidate("4", "also unrelated"))
    score_map = {1: 10.0, 2: 5.0}
    scores = _pool_scores(gold, hard, fill, score_map, "q", _FakeScorer())
    assert scores == {"1": 10.0, "2": 5.0, "3": 42.0, "4": 42.0}


def test_an_unmined_gold_is_pinned_above_every_hard_negative():
    """A gold missing from the pickle is a coverage gap, not evidence of irrelevance."""
    gold = (Candidate("1", "the answer"),)
    hard = (Candidate("2", "nearly"),)
    scores = _pool_scores(gold, hard, (), {2: 5.0}, "q", None)
    assert scores["1"] > scores["2"]


def test_a_loader_shaped_query_fills_the_largest_rung_at_the_default_hard_frac():
    """
    The loader must size ``fill`` against the WORST case (``hard_frac=0``), because the builder
    takes only a ``hard_frac`` prefix of the mined negatives. The second real export sized it as
    ``max_docs - gold - len(hard)`` instead; with 93 mined negatives that left 156 fillers against
    a need of 224, so every query rejected at the 250-doc rung and the failure read as "the pool
    is too small" when it was mis-shaped.
    """
    import random

    from ctc.data.sources.retrieval import QueryPool
    from ctc.tasks.retrieval.generate import draw_pool

    max_docs = 250
    gold = (Candidate("g", "the answer"),)
    hard = tuple(Candidate(f"h{i}", f"near miss {i}") for i in range(93))
    fill = tuple(Candidate(f"f{i}", f"filler {i}") for i in range(max_docs - len(gold)))
    query = QueryPool(query="q", gold=gold, hard=hard, fill=fill)
    drawn = draw_pool(query, (), max_docs, 0.1, random.Random(0))
    assert drawn is not None
    assert len(drawn["gold"]) + len(drawn["hard"]) + len(drawn["random"]) == max_docs
