import json
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from olmo_core.train.callbacks.metric_saver import MetricSaverCallback
from olmo_core.train.callbacks.restore_metrics import RestoreMetricsCallback
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


@pytest.mark.parametrize("step", [0, 100])
def test_same_step_training_and_evaluation_fragments_update_one_snapshot(metrics_trainer, step):
    callback = MetricSaverCallback(save_interval=100)
    callback.trainer = metrics_trainer
    callback.log_metrics(step, {"train/CE loss": 3.0, "eval/caption/CE loss": 4.0})
    callback.log_metrics(step, {"eval/caption/CE loss": 2.0, "eval/caption/CE gap": 0.1})
    snapshot = json.loads((metrics_trainer.save_folder / f"metrics_step{step}.json").read_text())
    assert snapshot == {
        "train/CE loss": 3.0,
        "eval/caption/CE loss": 2.0,
        "eval/caption/CE gap": 0.1,
    }
    assert metrics_trainer.checkpointer.save_overwrite is False
    callback.close()
    assert json.loads((metrics_trainer.save_folder / "metrics.json").read_text()) == snapshot


def test_resumed_metrics_replace_existing_snapshots_without_overwriting_checkpoints(
    metrics_trainer,
):
    old = MetricSaverCallback(save_interval=100)
    old.trainer = metrics_trainer
    old.log_metrics(100, {"eval/caption/CE loss": 4.0})
    old.close()
    metrics_trainer.write_file("step100/marker", "checkpoint")

    resumed = MetricSaverCallback(save_interval=100)
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
    assert metrics_trainer.checkpointer.save_overwrite is False


@pytest.mark.parametrize("scheme", [None, "s3", "gs"])
def test_resume_restores_same_step_metrics_before_evaluation(metrics_trainer, monkeypatch, scheme):
    old = MetricSaverCallback(save_interval=100)
    old.trainer = metrics_trainer
    old.log_metrics(100, {"train/CE loss": 3.0, "eval/caption/CE loss": 4.0})
    snapshot = metrics_trainer.save_folder / "metrics_step100.json"
    if scheme is not None:
        metrics_trainer.save_folder = f"{scheme}://bucket/checkpoints"
        exists = Mock(return_value=True)
        resource = Mock(return_value=snapshot)
        monkeypatch.setattr("olmo_core.train.callbacks.restore_metrics.file_exists", exists)
        monkeypatch.setattr("olmo_core.train.callbacks.restore_metrics.resource_path", resource)
    metrics_trainer.write_file = Mock(return_value="metrics_step100.json")
    metrics_trainer.global_step = 100
    metrics_trainer.checkpoint_loaded = True
    resumed = MetricSaverCallback(save_interval=100)
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
    assert json.loads(metrics_trainer.write_file.call_args.args[1]) == resumed.metrics
    assert metrics_trainer.write_file.call_args.kwargs == {"save_overwrite": True}


@pytest.mark.parametrize(
    "rank,loaded,exists", [(1, True, True), (0, False, True), (0, True, False)]
)
def test_resume_skips_unavailable_metrics(monkeypatch, rank, loaded, exists):
    exists_mock = Mock(return_value=exists)
    resource = Mock()
    monkeypatch.setattr("olmo_core.train.callbacks.restore_metrics.get_rank", lambda: rank)
    monkeypatch.setattr("olmo_core.train.callbacks.restore_metrics.file_exists", exists_mock)
    monkeypatch.setattr("olmo_core.train.callbacks.restore_metrics.resource_path", resource)
    saver = Mock(step_metrics_fname="metrics_step{step}.json")
    restore = RestoreMetricsCallback(metrics_callback="metrics")
    restore.trainer = SimpleNamespace(
        save_folder="s3://bucket/checkpoints",
        global_step=100,
        checkpoint_loaded=loaded,
        callbacks={"metrics": saver},
    )
    restore.pre_train()
    assert exists_mock.call_count == int(rank == 0 and loaded)
    resource.assert_not_called()
    saver.log_metrics.assert_not_called()


@pytest.mark.parametrize("global_overwrite", [False, True])
def test_checkpointer_per_file_overwrite_does_not_change_default(tmp_path, global_overwrite):
    checkpointer = Checkpointer(work_dir=tmp_path / "work", save_overwrite=global_overwrite)
    path = checkpointer.write_file(tmp_path, "metrics.json", "first")
    with pytest.raises(FileExistsError):
        checkpointer.write_file(tmp_path, "metrics.json", "blocked", save_overwrite=False)
    checkpointer.write_file(tmp_path, "metrics.json", "updated", save_overwrite=True)
    assert path.read_text() == "updated"
    assert checkpointer.save_overwrite is global_overwrite
    if global_overwrite:
        checkpointer.write_file(tmp_path, "metrics.json", "default")
        assert path.read_text() == "default"
    else:
        with pytest.raises(FileExistsError):
            checkpointer.write_file(tmp_path, "metrics.json", "default")
        assert path.read_text() == "updated"


@pytest.mark.parametrize("overwrite", [None, False, True])
def test_trainer_forwards_per_file_overwrite_without_changing_checkpointer(overwrite):
    trainer = SimpleNamespace(save_folder="s3://bucket/checkpoints", checkpointer=Mock())
    Trainer.write_file(trainer, "metrics.json", "{}", save_overwrite=overwrite)
    trainer.checkpointer.write_file.assert_called_once_with(
        trainer.save_folder, "metrics.json", "{}", save_overwrite=overwrite
    )


@pytest.mark.parametrize("overwrite", [None, False, True])
def test_remote_write_honors_per_file_overwrite(tmp_path, monkeypatch, overwrite):
    upload = Mock()
    monkeypatch.setattr("olmo_core.train.checkpoint.upload", upload)
    checkpointer = Checkpointer(work_dir=tmp_path, save_overwrite=False)
    target = checkpointer.write_file(
        "s3://bucket/checkpoints", "metrics.json", "{}", save_overwrite=overwrite
    )
    assert target == "s3://bucket/checkpoints/metrics.json"
    assert upload.call_args.kwargs["save_overwrite"] is (overwrite is True)
    assert checkpointer.save_overwrite is False
