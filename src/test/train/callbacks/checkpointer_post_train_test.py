from types import SimpleNamespace
from unittest.mock import Mock

from olmo_core.train.callbacks.checkpointer import CheckpointerCallback
from olmo_core.train.checkpoint import Checkpointer


def _callback(tmp_path, step: int) -> CheckpointerCallback:
    callback = CheckpointerCallback()
    checkpointer = Mock(spec=Checkpointer)
    checkpointer.checkpoint_dirname.side_effect = Checkpointer.checkpoint_dirname
    checkpointer.dir_is_checkpoint.side_effect = lambda d: (tmp_path / f"step{step}").is_dir()
    callback.trainer = SimpleNamespace(  # type: ignore[assignment]
        save_folder=str(tmp_path), global_step=step, checkpointer=checkpointer
    )
    callback._save_checkpoint = Mock(return_value=str(tmp_path / f"step{step}"))  # type: ignore
    callback._await_last_checkpoint = Mock()  # type: ignore[method-assign]
    return callback


def test_post_train_skips_final_save_when_that_checkpoint_already_exists(tmp_path):
    # A restart that resumed from its own final checkpoint must not try to re-save it.
    (tmp_path / "step100").mkdir()
    callback = _callback(tmp_path, step=100)
    callback.post_train()
    callback._save_checkpoint.assert_not_called()  # type: ignore[attr-defined]
    assert callback._latest_checkpoint_step == 100


def test_post_train_saves_final_checkpoint_when_missing(tmp_path):
    callback = _callback(tmp_path, step=100)
    callback.post_train()
    callback._save_checkpoint.assert_called_once_with(save_async=False)  # type: ignore[attr-defined]
