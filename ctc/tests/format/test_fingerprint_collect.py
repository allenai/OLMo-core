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


# ── data paths, and what happens when corpora are mixed ─────────────────────────────────────────
#
# The failure this guards is quiet. Deduplication used to be on the whole record, so two shard
# directories that produced an IDENTICAL format -- contradiction from PubMed and from FEVER, or one
# task spread over per-rung directories -- collapsed into one entry and the second directory's path
# vanished. The checkpoint then named one corpus out of however many it actually read, and nothing
# anywhere said otherwise.


def test_the_directory_a_format_came_from_is_recorded(tmp_path):
    got, _ = collect_fingerprints([shards(tmp_path, "contradiction")])
    assert got.formats[0].data_paths == (str((tmp_path / "contradiction").resolve()),)


def test_one_task_from_two_corpora_keeps_both_paths(tmp_path):
    """The mixed-corpus case: same format, two sources, one entry, both paths."""
    fp = make("contradiction")
    got, _ = collect_fingerprints([shards(tmp_path, "pubmed", fp), shards(tmp_path, "fever", fp)])
    assert len(got.formats) == 1
    assert got.formats[0].data_paths == (
        str((tmp_path / "pubmed").resolve()),
        str((tmp_path / "fever").resolve()),
    )


def test_a_five_task_mix_records_a_path_per_task(tmp_path):
    names = ["contradiction", "outlier", "oolong", "retrieval", "nq"]
    got, _ = collect_fingerprints([shards(tmp_path, n) for n in names])
    assert got.tasks == names
    assert all(len(fp.data_paths) == 1 for fp in got.formats)


def test_paths_from_two_formats_of_one_task_stay_separate(tmp_path):
    """A curriculum trains one task two ways; each layout keeps its own corpus."""
    got, _ = collect_fingerprints(
        [
            shards(tmp_path, "before", make(query_position="before")),
            shards(tmp_path, "after", make(query_position="after")),
        ]
    )
    entries = got.for_task("contradiction")
    assert len(entries) == 2
    assert {e.data_paths[0] for e in entries} == {
        str((tmp_path / "before").resolve()),
        str((tmp_path / "after").resolve()),
    }


def test_the_same_directory_twice_records_one_path(tmp_path):
    d = shards(tmp_path, "contradiction")
    got, _ = collect_fingerprints([d, d])
    assert got.formats[0].data_paths == (str(d.resolve()),)


def test_a_path_a_shard_dir_recorded_itself_is_preserved(tmp_path):
    """Tokenize time can record the SOURCE jsonl; collection adds the shard dir, not replaces it."""
    fp = make("contradiction").with_data_paths("/corpora/pubmed_claims.jsonl")
    got, _ = collect_fingerprints([shards(tmp_path, "contradiction", fp)])
    assert got.formats[0].data_paths == (
        "/corpora/pubmed_claims.jsonl",
        str((tmp_path / "contradiction").resolve()),
    )


def test_recording_paths_can_be_turned_off(tmp_path):
    got, _ = collect_fingerprints([shards(tmp_path, "contradiction")], record_source_paths=False)
    assert got.formats[0].data_paths == ()


# ── paths are provenance: recorded, never compared ──────────────────────────────────────────────


def test_the_same_data_staged_at_two_paths_is_compatible():
    """weka, node-local /data and an S3 mirror are the same corpus. Comparing paths fails them all."""
    weka = make().with_data_paths("/weka/oe-training/contra")
    local = make().with_data_paths("/data/prasann/contra")
    assert weka.compare(local) == []
    local.require_compatible_with(weka)


def test_same_format_as_ignores_paths_while_equality_does_not():
    a = make().with_data_paths("/a")
    b = make().with_data_paths("/b")
    assert a.same_format_as(b)
    assert a != b


def test_merging_two_different_formats_is_refused():
    """Merging them would manufacture one record that matches neither."""
    with pytest.raises(ValueError, match="different formats"):
        make(gold_index_base=1).merged_with(make(gold_index_base=0))


def test_paths_survive_a_round_trip_as_a_tuple(tmp_path):
    """A list here would silently fail equality against a freshly derived fingerprint."""
    FingerprintSet([make().with_data_paths("/a", "/b")]).write(tmp_path)
    got = FingerprintSet.read(tmp_path).formats[0]
    assert got.data_paths == ("/a", "/b")
    assert isinstance(got.data_paths, tuple)
