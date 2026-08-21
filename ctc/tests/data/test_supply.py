"""
The up-front supply check: a rung the pool provably cannot fill refuses in milliseconds.

Before this, nine of the suite's tasks answered a 10M-rung request with the
50-consecutive-rejections error, minutes into the draw -- indistinguishable at a glance from a
parameter-space problem, and paid for again on every retry. The bound is generous by design
(an upper bound on what one example could ever hold), so every refusal is mathematically sound
and a pass is merely "not provably hopeless".
"""

from __future__ import annotations

import pytest
from fixtures import pools

from ctc.data import build, supply
from ctc.data.generators import base as generators
from ctc.format import registry
from ctc.tasks import load_all

#: Ladder -> a fixture pool. Every SEEDABLE pool family appears at least once, asserted below.
POOLS = {
    "contradiction": pools.pubmed_pool,
    "contra_fever": pools.fever_pool,
    "nq": pools.retrieval_pool,
    "rerank": pools.rerank_pool,
    "outlier": pools.article_pool,
    "outlier_review": pools.review_pool,
    "absence": pools.book_pool,
    "xabsence": pools.paraphrase_pool,
    "reorder": pools.reorder_pool,
    "qdmatch_nq": pools.unit_pool,
    "grouping_labeled": pools.openalex_pool,
}


@pytest.fixture(scope="module", autouse=True)
def _tasks():
    load_all()


def test_every_pool_family_has_a_registered_bound():
    covered = {type(POOLS[t]()).__name__ for t in POOLS}
    assert covered <= set(supply._BY_POOL), "a fixture pool family has no supply bound"
    missing = set(supply._BY_POOL) - covered - {"RedundancyPool", "PubMedPool"}
    # PubMedPool is covered via contradiction; RedundancyPool shares its bound and only exists on
    # the development branch's roster. OolongPool has no bound on purpose: it under-fills
    # instead of failing, so there is no crash for this check to prevent.
    assert not missing, f"registered bounds with no fixture exercising them: {missing}"


@pytest.mark.parametrize("task", sorted(POOLS))
def test_fixture_pools_pass_at_the_smallest_rung(task):
    supply.check(task, POOLS[task](), ["2k"])


@pytest.mark.parametrize("task", sorted(POOLS))
def test_an_unfillable_rung_is_refused_with_its_arithmetic(task):
    """The fixture pools are tiny, so a 10m request must refuse -- naming need and supply."""
    with pytest.raises(ValueError, match="supplies at most"):
        supply.check(task, POOLS[task](), ["10m"])


def test_the_refusal_happens_before_any_drawing(tmp_path):
    """Through the real entry point, and fast: build_eval must raise before the draw loop."""
    spec = registry.get(generators.get("contradiction").task)
    with pytest.raises(ValueError, match="supplies at most"):
        build.build_eval(
            "contradiction",
            spec,
            size=5,
            rungs=["10m"],
            allow_small=True,
            corpus=pools.pubmed_pool(),
        )


def test_a_synthetic_pool_is_never_refused():
    supply.check("textgroups", None, ["10m"])


def test_oolong_is_exempt_because_it_underfills_instead_of_failing():
    """An over-budget oolong rung yields smaller examples, not a crash; refusing it here would
    contradict the generator's own semantics (and the audit test that relies on them)."""
    supply.check("oolong", pools.oolong_pool(), ["10m"])
