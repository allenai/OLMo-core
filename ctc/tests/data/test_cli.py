"""
``ctc-data``: the one command a user actually types.

The command surface is the deliverable -- "generate a task's train or eval data with one short
command" -- so it gets tested like one: the corpus loader is monkeypatched to return a fixture
pool, and everything else runs for real, including the audit that decides whether anything is
written.
"""

from __future__ import annotations

import json

import pytest
from fixtures import pools

from ctc.data import cli, ladders
from ctc.data.generators import base as generators


@pytest.fixture(autouse=True)
def _small_ladders(monkeypatch):
    """Two short rungs per task, so a CLI test is seconds rather than minutes."""
    monkeypatch.setitem(ladders.LADDERS, "cycle", {"2k": 20, "4k": 40})
    monkeypatch.setitem(ladders.LADDERS, "nq", {"2k": 11, "4k": 23})
    monkeypatch.setitem(ladders.LADDERS, "fiqa", {"2k": 6, "4k": 12})
    monkeypatch.setitem(ladders.LADDERS, "contra_fever", {"2k": 8, "4k": 16})


@pytest.fixture
def fake_corpora(monkeypatch):
    """
    Swap each generator's corpus loader for a fixture pool, so no test reaches the network.

    Patched on the generator rather than on the loader module because a :class:`Generator` holds
    the callable, not its name -- which is the same reason the loader is a field at all.
    """
    import dataclasses

    for module, attribute, source in (
        ("ctc.tasks.retrieval.sources.nq", "GENERATOR", "nq"),
        ("ctc.tasks.retrieval.sources.beir", "FIQA", "beir_fiqa"),
    ):
        original = getattr(__import__(module, fromlist=[attribute]), attribute)
        monkeypatch.setattr(
            f"{module}.{attribute}",
            dataclasses.replace(
                original,
                corpus=lambda source=source, **kw: pools.retrieval_pool(queries=600, source=source),
            ),
        )
    fever = __import__("ctc.tasks.contradiction.sources.fever", fromlist=["GENERATOR"])
    monkeypatch.setattr(
        "ctc.tasks.contradiction.sources.fever.GENERATOR",
        dataclasses.replace(
            fever.GENERATOR, corpus=lambda **kw: pools.fever_pool(pairs=16000, fillers=2000)
        ),
    )


def test_list_names_every_ported_generator(capsys):
    assert cli.main(["list"]) == 0
    printed = capsys.readouterr().out
    for name in generators.names():
        assert name in printed


def test_list_flags_the_held_out_ladders(capsys):
    """A user must not have to read the source to learn a ladder is eval-only -- and must not be
    told a suite row is, now that fiqa/scifact/outlier_review train in-domain."""
    cli.main(["list"])
    printed = capsys.readouterr().out
    for line in printed.splitlines():
        stripped = line.strip()
        if stripped.startswith("contra_fever"):
            assert "HELD OUT" in line
        elif stripped.startswith(("fiqa", "scifact", "outlier_review")):
            assert "HELD OUT" not in line


def test_build_writes_a_train_file_and_one_file_per_rung(tmp_path):
    assert cli.main(["build", "--task", "cycle", "--train", "20", "--out", str(tmp_path)]) == 0
    root = tmp_path / "cycle"
    assert (root / "train.jsonl").exists()
    assert sorted(p.name for p in root.glob("eval_*.jsonl")) == ["eval_2k.jsonl", "eval_4k.jsonl"]
    rows = [json.loads(line) for line in (root / "eval_2k.jsonl").read_text().splitlines()]
    assert len(rows) == 500
    assert all(len(r["documents"]) == 20 for r in rows)


def test_split_eval_writes_no_training_data(tmp_path):
    assert cli.main(["build", "--task", "cycle", "--split", "eval", "--out", str(tmp_path)]) == 0
    root = tmp_path / "cycle"
    assert not (root / "train.jsonl").exists()
    assert (root / "eval_2k.jsonl").exists()


def test_split_train_writes_no_eval_ladder(tmp_path):
    assert (
        cli.main(
            [
                "build",
                "--task",
                "cycle",
                "--split",
                "train",
                "--train",
                "20",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )
    root = tmp_path / "cycle"
    assert (root / "train.jsonl").exists()
    assert not list(root.glob("eval_*.jsonl"))


def test_a_corpus_backed_task_builds_through_the_cli(tmp_path, fake_corpora):
    assert cli.main(["build", "--task", "nq", "--train", "20", "--out", str(tmp_path)]) == 0
    rows = [
        json.loads(line) for line in (tmp_path / "nq" / "eval_2k.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 500
    assert all(r["source"] == "nq" and r["gold_doc_indices"] for r in rows)


def test_a_held_out_ladder_builds_eval_only_without_being_asked(tmp_path, fake_corpora, capsys):
    """``--split both`` on a held-out ladder must not quietly build training data."""
    assert cli.main(["build", "--task", "contra_fever", "--out", str(tmp_path)]) == 0
    assert "held out" in capsys.readouterr().out
    assert not (tmp_path / "contra_fever" / "train.jsonl").exists()


def test_asking_a_held_out_ladder_for_training_data_is_an_error(tmp_path, fake_corpora):
    assert cli.main(["build", "--task", "contra_fever", "--split", "train", "--out", str(tmp_path)]) == 1


def test_an_override_reaches_the_generator(tmp_path):
    cli.main(
        ["build", "--task", "cycle", "--train", "20", "-C", "num_cycles=2", "--out", str(tmp_path)]
    )
    rows = [
        json.loads(line) for line in (tmp_path / "cycle" / "eval_2k.jsonl").read_text().splitlines()
    ]
    assert all(len(r["gold_doc_indices"]) == 2 for r in rows)


def test_a_mistyped_override_is_rejected_rather_than_ignored(tmp_path):
    """
    Silently ignoring it would build data at the default size and call it what was asked for --
    which is how a whole sweep ends up being one configuration measured five times.
    """
    with pytest.raises(SystemExit, match="no parameter"):
        cli.main(["build", "--task", "cycle", "-C", "num_cyclez=2", "--out", str(tmp_path)])


def test_an_override_without_a_value_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        cli.main(["build", "--task", "cycle", "-C", "num_cycles", "--out", str(tmp_path)])


def test_audit_reruns_over_written_files(tmp_path):
    cli.main(["build", "--task", "cycle", "--train", "20", "--out", str(tmp_path)])
    assert cli.main(["audit", "--task", "cycle", "--dir", str(tmp_path)]) == 0


def test_audit_on_an_empty_directory_says_so_rather_than_passing(tmp_path):
    assert cli.main(["audit", "--task", "cycle", "--dir", str(tmp_path)]) == 1


def test_the_task_names_are_the_ones_ctc_eval_takes():
    """
    ``ctc-data build --task nq`` and ``ctc-eval --task nq`` must mean the same ladder, and be
    graded by the same spec. Two naming schemes for one suite is how a results table ends up with
    rows nobody can match to a build.

    Asserted over the names the two sides share rather than over either side whole: the eval bundle
    may list a ladder whose generator is not ported yet, and a generator may exist for a ladder the
    shipped bundle has no files for. Neither is a naming disagreement.
    """
    from ctc.eval import bundles

    shared = set(bundles.BUNDLE) & set(generators.names())
    assert {"contradiction", "nq", "outlier", "rerank", "oolong"} <= shared
    assert {"fiqa", "scifact", "outlier_review", "contra_fever"} <= shared
    for name in sorted(shared):
        assert bundles.BUNDLE[name].spec == generators.get(name).task, name
        if name == "contra_fever":
            assert generators.get(name).eval_only, name
        else:
            # The eval bundle's "ood" group is a GRADING convention for the 5-task mixed models;
            # since 2026-08-20 it no longer implies the generator refuses training data.
            assert not generators.get(name).eval_only, name
