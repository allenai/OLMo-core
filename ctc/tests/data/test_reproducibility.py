"""
Data is exactly reproducible from its seed. Enforced two ways, because one is not enough.

**Behaviourally**: the same ``(seed, config)`` produces byte-identical JSONL, twice in a row and in
either order relative to other builds.

**Statically**: no generator may reach global randomness. This is the check that survives future
edits. A single ``random.shuffle(x)`` instead of ``rng.shuffle(x)``, or one ``np.random``, makes a
build depend on interpreter state -- and it *still passes* a same-process determinism test whenever
the global happens to have been seeded, so it will not show up until someone tries to rebuild the
data months later and gets different files. The pre-migration tree was clean on this (63 files, all
using seeded ``random.Random`` instances, zero global calls); this keeps it that way.
"""

from __future__ import annotations

import ast
import random
from pathlib import Path

import pytest

from ctc.data import build, ladders
from ctc.data.generators import base as generators
from ctc.format import registry
from ctc.tasks import load_all

#: Names on the ``random`` module that use the global RNG.
GLOBAL_RANDOM_CALLS = {
    "random",
    "randint",
    "randrange",
    "choice",
    "choices",
    "sample",
    "shuffle",
    "uniform",
    "seed",
    "getrandbits",
    "gauss",
    "normalvariate",
    "triangular",
    "betavariate",
}

_SRC = Path(__file__).parents[2] / "src"


def _module_file(target: str) -> Path:
    """
    :param target: A :data:`ctc.data.generators.base.GENERATORS` value.

    :returns: The file it names. Derived from the registry rather than from a task-name convention,
        so a generator that moves under ``sources/`` cannot silently drop out of these checks.
    """
    return _SRC / (target.partition(":")[0].replace(".", "/") + ".py")


#: Every module that constructs an example, plus the shared synthetic helpers and the assembly
#: modules the source generators delegate to -- all of them must be RNG-clean, not just the ones
#: named after a task.
GENERATOR_FILES = sorted(
    {_module_file(target) for target in generators.GENERATORS.values()}
    | {
        _SRC / "ctc" / "tasks" / "contradiction" / "generate.py",
        _SRC / "ctc" / "tasks" / "retrieval" / "generate.py",
        _SRC / "ctc" / "tasks" / "outlier" / "generate.py",
    }
)

#: Tasks a test can build end to end with no network, cache or HF download.
OFFLINE_TASKS = generators.corpus_free_names()


@pytest.mark.parametrize("path", GENERATOR_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_no_generator_touches_global_randomness(path):
    """
    ``random.shuffle(...)`` and ``np.random.*`` are banned; only methods on a passed-in
    ``random.Random`` instance are allowed.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        target = node.func.value
        name = getattr(target, "id", None) or getattr(getattr(target, "value", None), "id", None)
        if name in {"random", "np", "numpy"} and node.func.attr in GLOBAL_RANDOM_CALLS:
            offenders.append(f"line {node.lineno}: {name}.{node.func.attr}(...)")
        if name in {"np", "numpy"} and node.func.attr == "seed":
            offenders.append(f"line {node.lineno}: numpy seeding")
    assert not offenders, f"{path.name} uses global randomness: {offenders}"


@pytest.mark.parametrize("path", GENERATOR_FILES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_generators_do_not_import_numpy(path):
    """numpy's global RNG is separate from ``random``'s and is not seeded by anything here."""
    source = path.read_text(encoding="utf-8")
    assert "import numpy" not in source and "from numpy" not in source


@pytest.mark.parametrize("task", OFFLINE_TASKS)
def test_the_same_seed_gives_the_same_bytes(task, tmp_path):
    """End to end, through the real writer, comparing files rather than objects."""
    from ctc.data.io import save_jsonl

    load_all()
    spec = registry.get(task)
    original = ladders.LADDERS[task]
    ladders.LADDERS[task] = {"2k": max(8, min(original.values()) // 8)}
    try:
        paths = []
        for i in (0, 1):
            examples, _ = build.build_train(task, spec, total=12, seed=99)
            path = tmp_path / f"{task}_{i}.jsonl"
            save_jsonl(path, examples)
            paths.append(path)
        assert paths[0].read_bytes() == paths[1].read_bytes()
    finally:
        ladders.LADDERS[task] = original


@pytest.mark.parametrize("task", OFFLINE_TASKS)
def test_a_different_seed_gives_different_data(task):
    """The seed has to actually reach the construction -- a generator that ignored it would pass
    every determinism check above while producing one dataset forever."""
    load_all()
    spec = registry.get(task)
    original = ladders.LADDERS[task]
    ladders.LADDERS[task] = {"2k": max(8, min(original.values()) // 8)}
    try:
        first, _ = build.build_train(task, spec, total=6, seed=1)
        second, _ = build.build_train(task, spec, total=6, seed=2)
        assert first != second
    finally:
        ladders.LADDERS[task] = original


@pytest.mark.parametrize("task", OFFLINE_TASKS)
def test_an_example_does_not_depend_on_how_many_came_before_it(task):
    """
    Each example must be a pure function of its RNG state, so a build can be resumed, sharded or
    parallelised without changing the data.
    """
    gen = generators.get(task)
    config = gen.config()
    config[gen.scaling_param] = 12

    rng = random.Random(5)
    sequential = [gen.build_example(rng, **config) for _ in range(3)]
    assert gen.build_example(random.Random(5), **config) == sequential[0]
