import json
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from olmo_core.train.callbacks.metric_saver import MetricSaverCallback
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
