import json
import logging
from pathlib import Path

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.train.callbacks.checkpoint_ready_notifier import (
    CheckpointReadyNotifierCallback,
)


def make_checkpoint(root: Path, step: int, *, ephemeral: bool = False) -> Path:
    checkpoint = root / f"step{step}"
    (checkpoint / "train").mkdir(parents=True)
    (checkpoint / "model_and_optim").mkdir()
    (checkpoint / "train/rank0.pt").write_bytes(b"trainer state")
    (checkpoint / "model_and_optim/.metadata").write_bytes(b"dcp metadata")
    (checkpoint / ".metadata.json").write_text(
        json.dumps({"ephemeral": ephemeral, "version": "test"}), encoding="utf-8"
    )
    return checkpoint


def make_callback(tmp_path: Path, **kwargs) -> CheckpointReadyNotifierCallback:
    return CheckpointReadyNotifierCallback(
        inbox_dir=str(tmp_path / "control/inbox"),
        run_id="pilot-run",
        lineage_id="pilot-lineage",
        **kwargs,
    )


def event_path(tmp_path: Path, step: int) -> Path:
    return tmp_path / "control/inbox/pilot-run" / f"step-{step:012d}.ready.json"


def test_publishes_finalized_checkpoint_event(tmp_path: Path, monkeypatch) -> None:
    checkpoint = make_checkpoint(tmp_path / "checkpoints", 42)
    monkeypatch.setenv("BEAKER_EXPERIMENT_ID", "experiment-id")
    monkeypatch.setenv("BEAKER_TASK_ID", "task-id")
    monkeypatch.setenv("BEAKER_JOB_ID", "job-id")

    make_callback(tmp_path).post_checkpoint_saved(checkpoint)

    event = json.loads(event_path(tmp_path, 42).read_text(encoding="utf-8"))
    assert event["schema_version"] == 1
    assert event["event"] == "checkpoint_ready"
    assert event["run_id"] == "pilot-run"
    assert event["lineage_id"] == "pilot-lineage"
    assert event["step"] == 42
    assert event["checkpoint_path"] == str(checkpoint.resolve())
    assert event["checkpoint_format"] == "olmo_core"
    assert len(event["checkpoint_metadata_sha256"]) == 64
    assert event["beaker_experiment_id"] == "experiment-id"
    assert event["beaker_task_id"] == "task-id"
    assert event["beaker_job_id"] == "job-id"


def test_duplicate_event_is_idempotent(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path / "checkpoints", 7)
    callback = make_callback(tmp_path)

    callback.post_checkpoint_saved(checkpoint)
    path = event_path(tmp_path, 7)
    original_payload = path.read_bytes()
    original_mtime = path.stat().st_mtime_ns
    callback.post_checkpoint_saved(checkpoint)

    assert path.read_bytes() == original_payload
    assert path.stat().st_mtime_ns == original_mtime


def test_skips_ephemeral_checkpoint_by_default(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path / "checkpoints", 8, ephemeral=True)

    make_callback(tmp_path).post_checkpoint_saved(checkpoint)

    assert not event_path(tmp_path, 8).exists()


def test_can_include_ephemeral_checkpoint(tmp_path: Path) -> None:
    checkpoint = make_checkpoint(tmp_path / "checkpoints", 9, ephemeral=True)

    make_callback(tmp_path, include_ephemeral=True).post_checkpoint_saved(checkpoint)

    assert event_path(tmp_path, 9).exists()


def test_nonzero_rank_does_nothing(tmp_path: Path, monkeypatch) -> None:
    checkpoint = make_checkpoint(tmp_path / "checkpoints", 10)
    monkeypatch.setattr("olmo_core.train.callbacks.checkpoint_ready_notifier.get_rank", lambda: 1)

    make_callback(tmp_path).post_checkpoint_saved(checkpoint)

    assert not event_path(tmp_path, 10).exists()


def test_incomplete_checkpoint_warns_without_failing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    checkpoint = tmp_path / "checkpoints/step11"
    checkpoint.mkdir(parents=True)

    with caplog.at_level(logging.WARNING):
        make_callback(tmp_path).post_checkpoint_saved(checkpoint)

    assert not event_path(tmp_path, 11).exists()
    assert "Could not publish checkpoint-ready event" in caplog.text


def test_unavailable_inbox_warns_without_failing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    checkpoint = make_checkpoint(tmp_path / "checkpoints", 12)
    inbox_file = tmp_path / "not-a-directory"
    inbox_file.write_text("occupied", encoding="utf-8")
    callback = CheckpointReadyNotifierCallback(
        inbox_dir=str(inbox_file),
        run_id="pilot-run",
        lineage_id="pilot-lineage",
    )

    with caplog.at_level(logging.WARNING):
        callback.post_checkpoint_saved(checkpoint)

    assert "Could not publish checkpoint-ready event" in caplog.text


@pytest.mark.parametrize("field", ["run_id", "lineage_id"])
def test_rejects_unsafe_ids(tmp_path: Path, field: str) -> None:
    kwargs = {
        "inbox_dir": str(tmp_path),
        "run_id": "safe-run",
        "lineage_id": "safe-lineage",
    }
    kwargs[field] = "../unsafe"

    with pytest.raises(OLMoConfigurationError, match=field):
        CheckpointReadyNotifierCallback(**kwargs)


def test_rejects_relative_inbox() -> None:
    with pytest.raises(OLMoConfigurationError, match="absolute path"):
        CheckpointReadyNotifierCallback(
            inbox_dir="relative/inbox",
            run_id="safe-run",
            lineage_id="safe-lineage",
        )


def test_conflicting_existing_event_warns_and_is_preserved(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    checkpoint = make_checkpoint(tmp_path / "checkpoints", 13)
    path = event_path(tmp_path, 13)
    path.parent.mkdir(parents=True)
    original = b'{"schema_version": 1, "run_id": "someone-else"}\n'
    path.write_bytes(original)

    with caplog.at_level(logging.WARNING):
        make_callback(tmp_path).post_checkpoint_saved(checkpoint)

    assert path.read_bytes() == original
    assert "conflicting checkpoint-ready event" in caplog.text
