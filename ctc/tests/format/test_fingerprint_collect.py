"""
Collecting fingerprints from shard directories.

One function backs both the training callback and ``ctc-fingerprint collect``, so a checkpoint
stamped during training and one stamped afterwards record the same thing. These tests are the
reason that function lives here rather than in ``ctc.train``: they need no olmo-core, so they run
everywhere, and a guard whose tests only run on some machines is a guard that will quietly rot.
"""

from __future__ import annotations

import pytest

from ctc.format.fingerprint import (
    FingerprintSet,
    FormatFingerprint,
    collect_fingerprints,
    conflicting_formats,
    hash_prompt,
)


def make(task="contradiction", **overrides) -> FormatFingerprint:
    base = dict(
        task=task,
        prompt_hash=hash_prompt(f"{task.upper()} INSTRUCTION"),
        serializer=task,
        item_separator="\n\n",
        gold_index_base=1,
        chunk_layout="wrap_documents",
    )
    base.update(overrides)
    return FormatFingerprint(**base)


def shards(tmp_path, name, *fps):
    d = tmp_path / name
    FingerprintSet(list(fps) or [make(name)]).write(d)
    return d


# ── collection ──────────────────────────────────────────────────────────────────────────────────


def test_a_mix_is_gathered_from_its_shard_dirs(tmp_path):
    got, skipped = collect_fingerprints(
        [shards(tmp_path, "contradiction"), shards(tmp_path, "outlier")]
    )
    assert got.tasks == ["contradiction", "outlier"]
    assert skipped == []


def test_an_unfingerprinted_shard_dir_is_refused_by_default(tmp_path):
    """A partial record is worse than none: eval then calls a trained task out-of-distribution."""
    with pytest.raises(FileNotFoundError, match="nope"):
        collect_fingerprints([shards(tmp_path, "contradiction"), tmp_path / "nope"])


def test_skipping_is_possible_but_reported_back(tmp_path):
    got, skipped = collect_fingerprints(
        [shards(tmp_path, "contradiction"), tmp_path / "nope"], allow_missing=True
    )
    assert got.tasks == ["contradiction"]
    assert [str(tmp_path / "nope")] == skipped


def test_collecting_nothing_raises_rather_than_writing_an_empty_record(tmp_path):
    with pytest.raises(ValueError, match="no formats found"):
        collect_fingerprints([])


def test_extra_formats_are_unioned_in(tmp_path):
    got, _ = collect_fingerprints([shards(tmp_path, "contradiction")], extra=[make("oolong")])
    assert set(got.tasks) == {"contradiction", "oolong"}


def test_the_same_format_from_two_dirs_is_recorded_once(tmp_path):
    fp = make("contradiction")
    got, _ = collect_fingerprints([shards(tmp_path, "a", fp), shards(tmp_path, "b", fp)])
    assert len(got.formats) == 1


def test_two_different_formats_for_one_task_are_both_kept(tmp_path):
    got, _ = collect_fingerprints(
        [
            shards(tmp_path, "a", make(query_position="before")),
            shards(tmp_path, "b", make(query_position="after")),
        ]
    )
    assert len(got.for_task("contradiction")) == 2


# ── drift detection ─────────────────────────────────────────────────────────────────────────────


def test_conflicting_formats_names_the_differing_field(tmp_path):
    conflicts = conflicting_formats(
        FingerprintSet([make(query_position="before"), make(query_position="after")])
    )
    assert conflicts == {"contradiction": ["query_position"]}


def test_a_consistent_mix_reports_no_conflict():
    assert conflicting_formats(FingerprintSet([make("a"), make("b")])) == {}
