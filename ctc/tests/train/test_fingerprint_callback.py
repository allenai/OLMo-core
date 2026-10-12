"""
The training-time write point.

Only the olmo-core adapter is tested here -- collection and drift detection live in
``ctc/tests/format/test_fingerprint_collect.py``, which needs no olmo-core and therefore runs
everywhere. What is left is the part that can only fail against a real trainer: when the record is
resolved, who writes it, where it goes, and whether eval can read back what training wrote.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

pytest.importorskip("ctc.train", reason="needs olmo-core and its dependencies")

from ctc.format.fingerprint import (  # noqa: E402
    FINGERPRINT_FILENAME,
    FingerprintSet,
    FormatFingerprint,
    hash_prompt,
)
from ctc.train import FormatFingerprintCallback  # noqa: E402


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
    return str(d)


class FakeTrainer:
    """Stands in for the trainer's file writer, which may target a remote URL."""

    def __init__(self):
        self.written = {}

    def write_file(self, name, contents, dir=None):
        self.written[(str(dir), name)] = contents
        return f"{dir}/{name}"


def attached(cb):
    cb.trainer = FakeTrainer()
    return cb


# ── failing early ───────────────────────────────────────────────────────────────────────────────


def test_a_bad_configuration_fails_at_step_zero(tmp_path):
    """Not at checkpoint one, hours later, with the GPU time already spent."""
    cb = attached(FormatFingerprintCallback(collect_from=[str(tmp_path / "unfingerprinted")]))
    with pytest.raises(FileNotFoundError, match="unfingerprinted"):
        cb.pre_train()


def test_attaching_the_callback_with_nothing_configured_is_an_error(tmp_path):
    """Otherwise the guard looks on and writes nothing."""
    with pytest.raises(ValueError, match="no formats found"):
        attached(FormatFingerprintCallback()).pre_train()


# ── the write ───────────────────────────────────────────────────────────────────────────────────


def test_every_checkpoint_gets_the_record(tmp_path):
    cb = attached(FormatFingerprintCallback(collect_from=[shards(tmp_path, "contradiction")]))
    cb.pre_train()
    cb.post_checkpoint_saved("s3://bucket/run/step100")
    cb.post_checkpoint_saved("s3://bucket/run/step200")
    assert set(cb.trainer.written) == {
        ("s3://bucket/run/step100", FINGERPRINT_FILENAME),
        ("s3://bucket/run/step200", FINGERPRINT_FILENAME),
    }


def test_what_training_writes_is_what_eval_reads_back(tmp_path):
    """The round trip is the whole contract; a writer eval cannot parse is worse than none."""
    cb = attached(
        FormatFingerprintCallback(
            collect_from=[shards(tmp_path, "contradiction"), shards(tmp_path, "outlier")]
        )
    )
    cb.pre_train()
    cb.post_checkpoint_saved("/ckpt/step100")

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / FINGERPRINT_FILENAME).write_text(
        cb.trainer.written[("/ckpt/step100", FINGERPRINT_FILENAME)]
    )
    assert FingerprintSet.read(ckpt) == cb._resolved


def test_the_write_goes_through_the_trainer_not_the_filesystem(tmp_path):
    """Checkpoints routinely live on S3 or weka; a local open() would silently drop the record."""
    cb = attached(FormatFingerprintCallback(collect_from=[shards(tmp_path, "contradiction")]))
    cb.pre_train()
    cb.post_checkpoint_saved("s3://bucket/run/step100")
    payload = cb.trainer.written[("s3://bucket/run/step100", FINGERPRINT_FILENAME)]
    assert json.loads(payload)["formats"][0]["task"] == "contradiction"


def test_only_rank_zero_writes(tmp_path, monkeypatch):
    monkeypatch.setattr("ctc.train.fingerprint.get_rank", lambda: 3)
    cb = attached(FormatFingerprintCallback(collect_from=[shards(tmp_path, "contradiction")]))
    cb.pre_train()
    cb.post_checkpoint_saved("/ckpt/step100")
    assert cb.trainer.written == {}


def test_a_save_before_pre_train_still_records(tmp_path):
    """Resume paths do not always replay pre_train; a checkpoint must never go out unstamped."""
    cb = attached(FormatFingerprintCallback(collect_from=[shards(tmp_path, "contradiction")]))
    cb.post_checkpoint_saved("/ckpt/step0")
    assert ("/ckpt/step0", FINGERPRINT_FILENAME) in cb.trainer.written


# ── what the checkpoint says it was trained on ──────────────────────────────────────────────────


def test_the_checkpoint_records_the_shard_dirs_it_was_trained_on(tmp_path):
    cb = attached(
        FormatFingerprintCallback(
            collect_from=[shards(tmp_path, "contradiction"), shards(tmp_path, "outlier")]
        )
    )
    cb.pre_train()
    recorded = {p for fp in cb._resolved.formats for p in fp.data_paths}
    assert recorded == {
        str((tmp_path / "contradiction").resolve()),
        str((tmp_path / "outlier").resolve()),
    }


def test_a_task_fed_by_two_corpora_keeps_both(tmp_path):
    """Deduplication is on format, not on the record -- otherwise one corpus disappears."""
    for name in ("pubmed", "fever"):
        FingerprintSet([make("contradiction")]).write(tmp_path / name)
    cb = attached(
        FormatFingerprintCallback(collect_from=[str(tmp_path / "pubmed"), str(tmp_path / "fever")])
    )
    cb.pre_train()
    assert len(cb._resolved.formats) == 1
    assert len(cb._resolved.formats[0].data_paths) == 2


def test_recording_paths_can_be_turned_off(tmp_path):
    cb = attached(
        FormatFingerprintCallback(
            collect_from=[shards(tmp_path, "contradiction")], record_data_paths=False
        )
    )
    cb.pre_train()
    assert cb._resolved.formats[0].data_paths == ()


class LoaderTrainer(FakeTrainer):
    """A trainer whose data loader reports the shard files it will actually read."""

    def __init__(self, paths):
        super().__init__()
        self.data_loader = type("DL", (), {"dataset": type("DS", (), {"paths": paths})()})()


def test_a_recorded_path_the_loader_never_reads_is_flagged(tmp_path, caplog):
    """
    The record is what a future eval trusts, so a record describing a different build than the one
    running is worth saying out loud -- while it is still cheap to fix.
    """
    cb = FormatFingerprintCallback(collect_from=[shards(tmp_path, "contradiction")])
    cb.trainer = LoaderTrainer(["/data/some/other/build/shard-000.npy"])
    with caplog.at_level(logging.WARNING):
        cb.pre_train()
    assert "match nothing the data loader is reading" in caplog.text


def test_loader_paths_under_a_recorded_directory_are_not_flagged(tmp_path, caplog):
    d = shards(tmp_path, "contradiction")
    cb = FormatFingerprintCallback(collect_from=[d])
    cb.trainer = LoaderTrainer([f"{Path(d).resolve()}/shard-000.npy"])
    with caplog.at_level(logging.WARNING):
        cb.pre_train()
    assert "match nothing" not in caplog.text
