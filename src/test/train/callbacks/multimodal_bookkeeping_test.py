"""
Tests for the bookkeeping callbacks in :mod:`olmo_core.train.callbacks.multimodal`: the
overwriting metric saver, the same-step metric restore, the W&B auto-resume subclass and the
overwrite-capable file writer they share. The shared callbacks they subclass are unchanged.
"""

import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from olmo_core.train.callbacks.multimodal import (
    MultimodalMetricSaverCallback,
    MultimodalWandBCallback,
    RestoreMetricsCallback,
    write_file_overwrite,
)
from olmo_core.train.callbacks.wandb import WANDB_API_KEY_ENV_VAR
from olmo_core.train.checkpoint import Checkpointer
from olmo_core.train.trainer import Trainer


@pytest.fixture
def metrics_trainer(tmp_path, monkeypatch):
    monkeypatch.setattr("olmo_core.train.callbacks.metric_saver.get_rank", lambda: 0)
    trainer = SimpleNamespace(
        save_folder=tmp_path / "checkpoints",
        checkpointer=Checkpointer(work_dir=tmp_path / "work", save_overwrite=False),
    )
    trainer.write_file = partial(Trainer.write_file, trainer)
    return trainer


def test_write_file_overwrite_replaces_local_files_but_not_checkpoint_policy(metrics_trainer):
    first = write_file_overwrite(metrics_trainer, "notes/marker.json", '{"a": 1}')
    second = write_file_overwrite(metrics_trainer, "notes/marker.json", '{"a": 2}')
    assert Path(first) == Path(second) == metrics_trainer.save_folder / "notes/marker.json"
    assert json.loads(Path(second).read_text()) == {"a": 2}
    # The trainer's own writer still refuses to overwrite.
    metrics_trainer.write_file("step10/marker", "checkpoint")
    with pytest.raises(FileExistsError):
        metrics_trainer.write_file("step10/marker", "replacement")
    assert metrics_trainer.checkpointer.save_overwrite is False


def test_write_file_overwrite_uploads_remote_files_with_overwrite(metrics_trainer, monkeypatch):
    uploads = []
    monkeypatch.setattr(
        "olmo_core.train.callbacks.multimodal.upload",
        lambda source, target, save_overwrite=False: uploads.append(
            (Path(source).read_bytes(), target, save_overwrite)
        ),
    )
    target = write_file_overwrite(
        metrics_trainer, "metrics_step5.json", '{"x": 1}', dir="s3://bucket/run"
    )
    assert target == "s3://bucket/run/metrics_step5.json"
    assert uploads == [(b'{"x": 1}', "s3://bucket/run/metrics_step5.json", True)]


@pytest.mark.parametrize("step", [0, 100])
def test_same_step_fragments_update_one_snapshot(metrics_trainer, step):
    callback = MultimodalMetricSaverCallback(save_interval=100)
    callback.trainer = metrics_trainer
    callback.log_metrics(step, {"train/CE loss": 3.0, "eval/caption/CE loss": 4.0})
    callback.log_metrics(step, {"eval/caption/CE loss": 2.0, "eval/caption/CE gap": 0.1})
    snapshot = json.loads((metrics_trainer.save_folder / f"metrics_step{step}.json").read_text())
    assert snapshot == {
        "train/CE loss": 3.0,
        "eval/caption/CE loss": 2.0,
        "eval/caption/CE gap": 0.1,
    }
    callback.close()
    assert json.loads((metrics_trainer.save_folder / "metrics.json").read_text()) == snapshot


def test_resumed_metrics_replace_snapshots_without_overwriting_checkpoints(metrics_trainer):
    old = MultimodalMetricSaverCallback(save_interval=100)
    old.trainer = metrics_trainer
    old.log_metrics(100, {"eval/caption/CE loss": 4.0})
    old.close()
    metrics_trainer.write_file("step100/marker", "checkpoint")

    resumed = MultimodalMetricSaverCallback(save_interval=100)
    resumed.trainer = metrics_trainer
    resumed.log_metrics(100, {"eval/caption/CE loss": 2.0})
    resumed.log_metrics(100, {"eval/caption/CE gap": 0.2})
    resumed.close()
    expected = {"eval/caption/CE loss": 2.0, "eval/caption/CE gap": 0.2}
    for filename in ("metrics_step100.json", "metrics.json"):
        assert json.loads((metrics_trainer.save_folder / filename).read_text()) == expected
    with pytest.raises(FileExistsError):
        metrics_trainer.write_file("step100/marker", "replacement")
    assert (metrics_trainer.save_folder / "step100/marker").read_text() == "checkpoint"


@pytest.mark.parametrize("scheme", [None, "s3", "gs"])
def test_resume_restores_same_step_metrics_before_evaluation(metrics_trainer, monkeypatch, scheme):
    old = MultimodalMetricSaverCallback(save_interval=100)
    old.trainer = metrics_trainer
    old.log_metrics(100, {"train/CE loss": 3.0, "eval/caption/CE loss": 4.0})
    snapshot = metrics_trainer.save_folder / "metrics_step100.json"
    if scheme is not None:
        metrics_trainer.save_folder = f"{scheme}://bucket/checkpoints"
        exists = Mock(return_value=True)
        resource = Mock(return_value=snapshot)
        monkeypatch.setattr("olmo_core.train.callbacks.multimodal.file_exists", exists)
        monkeypatch.setattr("olmo_core.train.callbacks.multimodal.resource_path", resource)
    writes = []
    monkeypatch.setattr(
        "olmo_core.train.callbacks.multimodal.write_file_overwrite",
        lambda trainer, name, contents, dir=None: writes.append((name, json.loads(contents))),
    )
    metrics_trainer.global_step = 100
    metrics_trainer.checkpoint_loaded = True
    resumed = MultimodalMetricSaverCallback(save_interval=100)
    resumed.trainer = metrics_trainer
    metrics_trainer.callbacks = {"metrics": resumed}
    restore = RestoreMetricsCallback(metrics_callback="metrics")
    restore.trainer = metrics_trainer
    restore.pre_train()
    if scheme is not None:
        exists.assert_called_once_with(f"{scheme}://bucket/checkpoints/metrics_step100.json")
        resource.assert_called_once_with(f"{scheme}://bucket/checkpoints", "metrics_step100.json")
    resumed.log_metrics(100, {"eval/caption/CE loss": 2.0})
    assert resumed.metrics == {"train/CE loss": 3.0, "eval/caption/CE loss": 2.0}
    assert writes[-1] == ("metrics_step100.json", resumed.metrics)


def test_restore_metrics_is_a_no_op_without_a_loaded_checkpoint(metrics_trainer):
    metrics_trainer.global_step = 100
    metrics_trainer.checkpoint_loaded = False
    saver = MultimodalMetricSaverCallback(save_interval=100)
    saver.trainer = metrics_trainer
    metrics_trainer.callbacks = {"metrics": saver}
    restore = RestoreMetricsCallback(metrics_callback="metrics")
    restore.trainer = metrics_trainer
    restore.pre_train()
    assert not saver.metrics


class _MockWandB:
    def __init__(self, run_id: str = "run-123"):
        self.run = SimpleNamespace(id=run_id, path=f"entity/project/{run_id}")
        self.init_calls = []

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run


def _wandb_callback(tmp_path, wandb, *, step: int = 25) -> MultimodalWandBCallback:
    callback = MultimodalWandBCallback(
        name="bridge", project="vision-alignment", entity="test-entity", auto_resume=True
    )
    callback._wandb = wandb
    callback.trainer = SimpleNamespace(work_dir=tmp_path, global_step=step, checkpoint_loaded=False)
    return callback


def test_wandb_run_id_and_checkpoint_step_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv(WANDB_API_KEY_ENV_VAR, "test-key")
    callback = _wandb_callback(tmp_path, _MockWandB(), step=25)
    callback.pre_train()
    assert callback.state_dict() == {
        "run_id": "run-123",
        "step": 25,
        "name": "bridge",
        "project": "vision-alignment",
        "entity": "test-entity",
    }


def test_wandb_auto_resume_continues_existing_run(tmp_path, monkeypatch):
    monkeypatch.setenv(WANDB_API_KEY_ENV_VAR, "test-key")
    wandb = _MockWandB()
    callback = _wandb_callback(tmp_path, wandb, step=25)
    callback.load_state_dict(
        {
            "run_id": "run-123",
            "step": 20,
            "name": "bridge",
            "project": "vision-alignment",
            "entity": "test-entity",
        }
    )
    callback.pre_train()
    init = wandb.init_calls[0]
    assert init["id"] == "run-123"
    assert init["resume"] == "allow"
    assert init["allow_val_change"] is True


def test_wandb_auto_resume_rejects_different_run_identity(tmp_path, monkeypatch):
    monkeypatch.setenv(WANDB_API_KEY_ENV_VAR, "test-key")
    wandb = _MockWandB(run_id="new-run")
    callback = _wandb_callback(tmp_path, wandb)
    callback.load_state_dict(
        {
            "run_id": "old-run",
            "step": 20,
            "name": "different-name",
            "project": "vision-alignment",
            "entity": "test-entity",
        }
    )
    callback.pre_train()
    assert "id" not in wandb.init_calls[0]
    assert callback.run_id == "new-run"


def test_wandb_without_auto_resume_starts_a_fresh_run(tmp_path, monkeypatch):
    monkeypatch.setenv(WANDB_API_KEY_ENV_VAR, "test-key")
    wandb = _MockWandB()
    callback = _wandb_callback(tmp_path, wandb)
    callback.auto_resume = False
    callback.load_state_dict({"run_id": "old-run", "step": 20})
    callback.pre_train()
    assert "id" not in wandb.init_calls[0] and "resume" not in wandb.init_calls[0]
