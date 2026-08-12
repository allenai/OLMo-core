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


def test_every_ladder_records_a_counted_eval_size():
    """Counted on weka, not inferred from the filename -- several disagree. Recorded so a sub-500
    ladder is flagged before the run rather than found in the results."""
    for name, entry in bundles.BUNDLE.items():
        assert entry.eval_size > 0, f"{name} has no recorded eval_size"


def test_only_scifact_is_below_the_eval_size_floor():
    small = {n for n, e in bundles.BUNDLE.items() if e.small_eval_warning}
    assert small == {"scifact"}
    assert "±0.02" in bundles.get("scifact").small_eval_warning


def test_filename_row_counts_are_not_trusted():
    """``nq_validation_k20_600.jsonl`` holds 500 rows and ``..._fever_plain_n100_k3.jsonl`` holds
    599. Reading the count off the name would mis-flag both."""
    assert bundles.get("nq").eval_size == 500
    assert bundles.get("contra_fever").eval_size == 599


def test_named_bundles_resolve_and_a_path_still_works():
    assert bundles.get_bundle("v2").name == "v2"
    assert bundles.get_bundle(None).name == bundles.DEFAULT_BUNDLE
    ad_hoc = bundles.get_bundle("/staged/copy")
    assert ad_hoc.root == "/staged/copy"
    assert ad_hoc.kind == "reliable"


def test_the_two_bundles_disagree_on_the_same_rung_label():
    """The whole reason bundles are named. contradiction's 64k is n=1602 in v2 and n=1525 in
    v2_clean, because the clean rebuild recalibrated against a PubMed-only filler pool. A number
    quoted without its bundle is not comparable to another."""
    v2 = dict(bundles.resolve("contradiction", "xlong", root="v2"))
    clean = dict(bundles.resolve("contradiction", "xlong", root="v2_clean"))
    assert "n1602" in v2["64k"].name
    assert "n1525" in clean["64k"].name
    assert v2["64k"].name != clean["64k"].name


def test_xlong_rungs_are_opt_in():
    """One 256k rung is hours per task, so `--rungs all` must not start one."""
    labels = [label for label, _ in bundles.resolve("contradiction", "all", root="v2")]
    assert labels == ["2k", "8k", "16k", "32k"]
    assert not set(labels) & set(bundles.XLONG_RUNGS)


def test_rungs_xlong_selects_only_the_ultra_long_ladder():
    labels = [label for label, _ in bundles.resolve("contradiction", "xlong", root="v2")]
    assert labels == list(bundles.XLONG_RUNGS)


def test_xlong_on_a_task_without_it_raises_rather_than_returning_nothing():
    """An empty result would read as "ran and found nothing" in a results table."""
    with pytest.raises(KeyError, match="no ultra-long rungs"):
        bundles.resolve("fiqa", "xlong", root="v2")


def test_an_xlong_rung_absent_from_a_bundle_names_the_bundle():
    """nq has xlong rungs in v2 but not in v2_clean; the error has to say which bundle it looked in
    or the reader assumes the rung does not exist at all."""
    bundles.resolve("nq", "64k", root="v2")
    with pytest.raises(KeyError, match="v2_clean"):
        bundles.resolve("nq", "64k", root="v2_clean")


def test_the_fast_bundle_supplies_its_own_ladders():
    """Its filenames encode the construction (``_tail10`` / ``_mux``), not the corpus size, so
    inheriting the base ladder would resolve to files that do not exist."""
    fast = bundles.get_bundle("fast")
    assert fast.kind == "fast"
    assert fast.declares_own_ladder("contradiction")
    assert not bundles.get_bundle("v2").declares_own_ladder("contradiction")
    assert "tail10" in fast.rungs_for("contradiction")["8k"]
    assert "mux" in fast.rungs_for("nq")["8k"]


def test_fast_all_is_filtered_to_the_rungs_it_actually_has():
    """The fast bundle starts at 8k. Returning the base ladder's 2k would name a missing file."""
    labels = [label for label, _ in bundles.resolve("contradiction", "all", root="fast")]
    assert labels == ["8k", "16k", "32k"]
    assert "2k" in bundles.get("contradiction").labels


def test_fast_reaches_1m_but_not_2m():
    labels = [label for label, _ in bundles.resolve("contradiction", "xlong", root="fast")]
    assert labels == ["64k", "128k", "256k", "512k", "1M"]


def test_a_task_the_fast_bundle_cannot_construct_says_so():
    """The fast bundle covers the five in-distribution ladders; the held-out four have no
    shared-corpus construction. Failing with the reason beats resolving to a filename nobody ever
    wrote."""
    with pytest.raises(KeyError, match="no fiqa"):
        bundles.resolve("fiqa", "all", root="fast")


def test_the_fast_bundle_covers_every_in_distribution_ladder():
    """outlier included: the planted construction replaced the prefix+tail one that could not be
    built above the small rungs."""
    fast = bundles.get_bundle("fast")
    assert set(fast.ladders) == set(bundles.GROUPS["main"])
    assert "planted" in fast.rungs_for("outlier")["1M"]


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
