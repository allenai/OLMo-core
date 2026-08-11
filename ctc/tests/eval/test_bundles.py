"""
The eval bundle table and how ``--tasks`` / ``--rungs`` resolve against it.

These are cheap assertions about a lookup table, and they are worth having because the table is the
one place the rung-to-file mapping exists. When it lived in three drivers, the copies drifted, and a
drifted copy grades a checkpoint against the wrong file while reporting a perfectly plausible score.
"""

from __future__ import annotations

import argparse

import pytest

from ctc.eval import bundles, cli


def test_every_task_has_ascending_rungs():
    for name, entry in bundles.BUNDLE.items():
        labels = entry.labels
        assert labels, f"{name} has no rungs"
        assert len(set(labels)) == len(labels), f"{name} repeats a rung label"
        numeric = [int(label.rstrip("k")) for label in labels]
        assert numeric == sorted(numeric), f"{name} rungs are not ascending: {labels}"


def test_every_bundle_task_names_a_registered_spec():
    import ctc.tasks
    from ctc.format import registry

    ctc.tasks.load_all()
    for name, entry in bundles.BUNDLE.items():
        assert entry.spec in registry.names(), f"{name} names unregistered spec {entry.spec!r}"


def test_ood_tasks_reuse_in_distribution_specs():
    """The OOD ladders are different *sources* graded by the same contract, not new tasks.

    If one ever grows its own spec, the OOD-vs-in-distribution gap stops being a source effect and
    becomes partly a grading difference -- which is exactly the confound these ladders exist to
    avoid.
    """
    assert bundles.get("contra_fever").spec == bundles.get("contradiction").spec
    assert bundles.get("outlier_review").spec == bundles.get("outlier").spec
    assert bundles.get("fiqa").spec == bundles.get("nq").spec
    assert bundles.get("scifact").spec == bundles.get("nq").spec


def test_groups_partition_the_bundle():
    assert set(bundles.GROUPS["main"]) | set(bundles.GROUPS["ood"]) == set(bundles.BUNDLE)
    assert not set(bundles.GROUPS["main"]) & set(bundles.GROUPS["ood"])
    assert len(bundles.GROUPS["main"]) == 5
    assert len(bundles.GROUPS["ood"]) == 4


def test_names_accepts_groups_and_comma_lists():
    assert bundles.names("main") == list(bundles.GROUPS["main"])
    assert bundles.names("nq,outlier") == ["nq", "outlier"]


def test_names_rejects_an_unknown_task():
    with pytest.raises(KeyError, match="unknown task"):
        bundles.names("nq,not_a_task")


def test_resolve_all_returns_every_rung_under_the_root(tmp_path):
    resolved = bundles.resolve("contradiction", "all", root=str(tmp_path))
    assert [label for label, _ in resolved] == ["2k", "8k", "16k", "32k"]
    for _, path in resolved:
        assert str(path).startswith(str(tmp_path))


def test_resolve_rejects_a_rung_the_task_does_not_have():
    """rerank has no 32k rung. Silently skipping it would leave a hole in a results table that
    reads as 'the model scored nothing there'."""
    with pytest.raises(KeyError, match="no rung"):
        bundles.resolve("rerank", "32k")


def test_bundle_root_prefers_explicit_then_env(monkeypatch):
    monkeypatch.setenv(bundles.ROOT_ENV, "/env/bundle")
    assert str(bundles.bundle_root("/explicit")) == "/explicit"
    assert str(bundles.bundle_root()) == "/env/bundle"
    monkeypatch.delenv(bundles.ROOT_ENV)
    assert str(bundles.bundle_root()) == bundles.DEFAULT_ROOT


def _args(**kwargs) -> argparse.Namespace:
    base = dict(tasks="main", rungs="all", bundle="/bundle")
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_plan_covers_every_main_rung():
    plan = cli.plan(_args())
    assert len(plan) == sum(len(bundles.get(t).labels) for t in bundles.GROUPS["main"])
    assert {item["task"] for item in plan} == set(bundles.GROUPS["main"])


def test_plan_skips_tasks_lacking_an_explicitly_requested_rung(capsys):
    """A task without the requested rung is skipped with a message, not an error -- otherwise
    ``--rungs 32k --tasks all`` could never run, since rerank and fiqa stop short."""
    plan = cli.plan(_args(tasks="all", rungs="32k"))
    tasks = {item["task"] for item in plan}
    assert "rerank" not in tasks and "fiqa" not in tasks
    assert "contradiction" in tasks
    assert "skipping rerank" in capsys.readouterr().err


def test_plan_rejects_an_unknown_task():
    with pytest.raises(SystemExit, match="unknown task"):
        cli.plan(_args(tasks="nope"))


def test_plan_carries_the_grading_spec_not_just_the_ladder_name():
    """The OOD ladders are graded by a different name than they are reported under, and the result
    file has to record both or contra_fever and contradiction collide in one row."""
    item = next(i for i in cli.plan(_args(tasks="contra_fever")) if i["rung"] == "2k")
    assert item["task"] == "contra_fever"
    assert item["spec"] == "contradiction"
