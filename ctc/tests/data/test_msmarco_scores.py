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
