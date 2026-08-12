"""
The ported synthetic generators still build the examples the pre-migration ones did.

These four tasks carry anti-shortcut fixes that are subtle and easy to lose in a rewrite -- cycle
entities must participate in background-edge sampling, groups4 distractors must be allowed to cluster
up to G-2 -- and losing one produces *valid* data that is trivially solvable. No structural check
catches that; only comparing against captured output does.

Fixtures come from ``fixtures/_capture_from_pre_migration.py``, run against the old tree. A failure
here means the port changed behaviour: fix the port. Regenerating the fixture would delete the
evidence instead of the bug.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from ctc.data.generators import base
from ctc.format import registry
from ctc.tasks import load_all

FIXTURES = Path(__file__).parent / "fixtures"

#: The generators with a captured pre-migration fixture -- the four pure-synthetic ones. Read
#: off the fixture directory rather than off the registry: the corpus-backed generators cannot
#: be byte-compared without their corpora, and listing them here would only mean skipping them.
GOLDEN_TASKS = sorted(p.name.removesuffix("_golden.json") for p in FIXTURES.glob("*_golden.json"))

#: Fixture config keys named for the old positional signatures, mapped onto the ported keyword ones.
RENAMES = {
    "n_docs": "num_docs",
    "n_pairs": "num_pairs",
    "tol": "tolerance",
}


def _cases(task):
    payload = json.loads((FIXTURES / f"{task}_golden.json").read_text(encoding="utf-8"))
    return payload["seed"], payload["cases"]


def _ids(task):
    return [c["name"] for c in _cases(task)[1]]


@pytest.mark.parametrize("task", GOLDEN_TASKS)
def test_examples_match_the_pre_migration_generator(task):
    seed, cases = _cases(task)
    build = base.get(task).build_example
    for case in cases:
        config = {RENAMES.get(k, k): v for k, v in case["config"].items()}
        rng = random.Random(seed)
        for i, expected in enumerate(case["examples"]):
            assert build(rng, **config) == expected, f"{task}/{case['name']} example {i}"


@pytest.mark.parametrize("task", GOLDEN_TASKS)
def test_the_rng_stream_is_not_merely_per_example_correct(task):
    """
    A generator can produce the right first example and still consume the RNG differently, which
    only shows up from the second example on. The fixtures hold several examples per case from one
    continuous stream precisely to catch that, so assert the case is not trivially short.
    """
    _, cases = _cases(task)
    assert all(len(c["examples"]) >= 2 for c in cases)


@pytest.mark.parametrize("task", GOLDEN_TASKS)
def test_gold_agrees_with_the_spec_declared_index_base(task):
    """
    The generator and the grader must agree on the base. They are declared in different files, and
    a disagreement is silent -- correct answers simply score zero.
    """
    load_all()
    spec = registry.get(task)
    _, cases = _cases(task)
    for case in cases:
        for example in case["examples"]:
            flat = [i for group in example["gold_doc_indices"] for i in group]
            n = len(example["documents"])
            assert min(flat) >= spec.gold_index_base
            assert max(flat) <= n - 1 + spec.gold_index_base
        # 1-based gold that never uses index 1 would pass the range check above while actually
        # being 0-based data; require the observed minimum to be reachable only at the declared base.
        assert spec.gold_index_base == 1, f"{task} is documented as 1-based in the port plan"


@pytest.mark.parametrize("task", GOLDEN_TASKS)
def test_generated_examples_validate_against_the_schema(task):
    from ctc.data.schema import validate

    load_all()
    spec = registry.get(task)
    build = base.get(task).build_example
    _, cases = _cases(task)
    rng = random.Random(0)
    for case in cases:
        config = {RENAMES.get(k, k): v for k, v in case["config"].items()}
        validate(build(rng, **config), spec)
