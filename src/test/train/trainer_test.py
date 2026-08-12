from collections import OrderedDict, defaultdict
from threading import Event
from unittest.mock import Mock, patch

from olmo_core.io import clear_directory
from olmo_core.train.callbacks.checkpointer import CheckpointerCallback
from olmo_core.train.trainer import Trainer


def test_checkpoint_save_skips_checkpoint_independent_bookkeeping():
    trainer = Trainer.__new__(Trainer)
    trainer.bookkeeping_soft_timeout = 30
    trainer._multi_thread_pool = None
    trainer._single_thread_pool = None
    trainer._bookkeeping_queue = defaultdict(OrderedDict)
    trainer._checkpoint_independent_bookkeeping_ops = set()
    trainer._error = None
    trainer.global_step = 2
    trainer.save_folder = "checkpoints"
    trainer.checkpointer = Mock()
    trainer.checkpointer.checkpoint_dirname.return_value = "step2"
    trainer.train_module = Mock()
    trainer.callbacks = {}
    trainer._metrics = {}
    trainer._metrics_reduce_type = {}

    cleanup_started = Event()
    cleanup_finished = Event()
    release_cleanup = Event()

    def cleanup():
        cleanup_started.set()
        release_cleanup.wait()
        cleanup_finished.set()

    try:
        trainer.run_bookkeeping_op(
            cleanup,
            distributed=False,
            checkpoint_dependent=False,
        )
        assert cleanup_started.wait(timeout=5)

        checkpoint_future = Mock()
        trainer.checkpointer.save_async.return_value = checkpoint_future
        with patch.object(trainer, "state_dict", return_value={}):
            path, future = trainer.save_checkpoint_async()

        assert str(path) == "checkpoints/step2"
        assert future is checkpoint_future
        trainer.checkpointer.save_async.assert_called_once()
        assert not cleanup_finished.is_set()

        release_cleanup.set()
        trainer._join_bookkeeping_ops()
        assert cleanup_finished.is_set()
    finally:
        release_cleanup.set()
        if trainer._multi_thread_pool is not None:
            trainer._multi_thread_pool.shutdown(wait=True)


def test_checkpoint_removal_does_not_block_checkpoint_saves(tmp_path):
    checkpoint_path = tmp_path / "step1"
    checkpoint_path.mkdir()

    callback = Mock()
    callback.trainer = Mock()
    callback.trainer.checkpointer.METADATA_FNAME = "metadata.json"

    CheckpointerCallback._remove_checkpoint(callback, str(checkpoint_path))

    callback.trainer.run_bookkeeping_op.assert_called_once_with(
        clear_directory,
        str(checkpoint_path),
        op_name=f"clear_directory {checkpoint_path}",
        distributed=False,
        checkpoint_dependent=False,
    )
