import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist

from olmo_core.testing import BACKENDS, run_distributed_test
from olmo_core.train import ReduceType
from olmo_core.train.callbacks import CheckpointerCallback
from olmo_core.train.trainer import Trainer
from olmo_core.train.utils import reduce_metrics


class ThreadRecordingCallback:
    def __init__(self):
        self.post_save_thread_id = None

    def post_checkpoint_saved(self, path):
        del path
        self.post_save_thread_id = threading.get_ident()


def make_minimal_trainer(write_future: Future[None]):
    trainer = object.__new__(Trainer)
    trainer.global_step = 10
    trainer.save_folder = "/tmp/checkpoints"
    trainer.train_module = MagicMock()
    trainer.checkpointer = MagicMock()
    trainer.checkpointer.checkpoint_dirname.return_value = "step10"
    trainer.checkpointer.save_async.return_value = write_future
    trainer._async_checkpoint_completion_lock = threading.Lock()
    trainer._async_checkpoint_completions = {}
    trainer._completed_async_checkpoint_durations = {}
    trainer._completed_async_checkpoint_errors = {}
    trainer._pending_async_checkpoint_paths = set()
    setattr(trainer, "_log_metrics", MagicMock())
    setattr(trainer, "_join_bookkeeping_ops", MagicMock())
    setattr(trainer, "state_dict", MagicMock(return_value={}))
    return trainer


def test_async_checkpoint_writer_only_publishes_completion():
    writer_future: Future[None] = Future()
    trainer = make_minimal_trainer(writer_future)
    callback = ThreadRecordingCallback()
    trainer._iter_callbacks = MagicMock(return_value=iter([callback]))
    main_thread_id = threading.get_ident()

    path, completion_future = trainer.save_checkpoint_async()
    with ThreadPoolExecutor(max_workers=1) as executor:
        writer_thread_id = executor.submit(threading.get_ident).result()
        executor.submit(writer_future.set_result, None).result()

    assert callback.post_save_thread_id is None
    assert str(path) in trainer._async_checkpoint_completions

    completion = completion_future.result()
    assert completion.step == 10
    assert callback.post_save_thread_id == main_thread_id
    assert callback.post_save_thread_id != writer_thread_id
    assert str(path) in trainer._completed_async_checkpoint_durations
    assert completion_future.result() is completion


def test_async_checkpoint_writer_publishes_failure_without_finalization():
    writer_future: Future[None] = Future()
    trainer = make_minimal_trainer(writer_future)
    trainer._iter_callbacks = MagicMock(return_value=iter([]))
    _, completion_future = trainer.save_checkpoint_async()

    error = RuntimeError("checkpoint write failed")
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(writer_future.set_exception, error).result()

    with pytest.raises(RuntimeError, match="checkpoint write failed"):
        completion_future.result()
    completion = completion_future.completion()
    assert completion.error is error
    assert not trainer._async_checkpoint_completions
    assert str(completion.path) in trainer._completed_async_checkpoint_errors


def test_failed_async_checkpoint_skips_metadata_poll():
    writer_future: Future[None] = Future()
    trainer = make_minimal_trainer(writer_future)
    trainer._iter_callbacks = MagicMock(return_value=iter([]))
    path, completion_future = trainer.save_checkpoint_async()
    callback = CheckpointerCallback(save_async=True)
    callback.trainer = trainer
    callback._future = completion_future
    callback._latest_checkpoint_path = str(path)

    writer_future.set_exception(RuntimeError("checkpoint write failed"))
    callback._await_last_checkpoint()

    trainer.checkpointer.dir_is_checkpoint.assert_not_called()
    assert callback._future is None
    assert str(path) in trainer._completed_async_checkpoint_errors


def test_metric_snapshot_is_atomic_with_concurrent_recording():
    trainer = object.__new__(Trainer)
    trainer.global_step = 10
    trainer._metrics = {}
    trainer._metrics_reduce_type = {}
    trainer._metrics_lock = threading.RLock()
    started = threading.Event()

    def record_from_worker():
        started.set()
        trainer.record_metric("worker_metric", 3.0, reduce_type=ReduceType.max)

    with ThreadPoolExecutor(max_workers=1) as executor:
        with trainer._metrics_lock:
            future = executor.submit(record_from_worker)
            assert started.wait(timeout=1)
            assert not future.done()
        future.result()

    snapshot = trainer._take_metrics_snapshot()
    assert snapshot is not None
    metrics, reduce_types = snapshot
    assert metrics[10]["worker_metric"].item() == 3.0
    assert reduce_types["worker_metric"] == ReduceType.max
    assert trainer._metrics == {}

    trainer.record_metric("next_metric", torch.tensor(4.0), reduce_type=ReduceType.sum)
    assert "next_metric" not in metrics[10]
    next_metric = trainer.get_metric("next_metric")
    assert next_metric is not None
    assert next_metric.item() == 4.0


def test_checkpoint_gc_waits_for_global_async_finalization():
    callback = CheckpointerCallback(
        save_interval=None,
        ephemeral_save_interval=None,
        fixed_steps=None,
    )
    trainer = MagicMock()
    trainer.global_step = 10
    trainer.async_checkpoint_finalization_pending_for.return_value = True
    callback.trainer = trainer
    callback._checkpoints_to_remove = ["/checkpoints/step10"]
    remove_checkpoint = MagicMock()
    setattr(callback, "_remove_checkpoint", remove_checkpoint)

    callback.post_train_batch()
    remove_checkpoint.assert_not_called()
    assert callback._checkpoints_to_remove == ["/checkpoints/step10"]

    trainer.async_checkpoint_finalization_pending_for.return_value = False
    callback.post_train_batch()
    remove_checkpoint.assert_called_once_with("/checkpoints/step10")
    assert callback._checkpoints_to_remove == []


def test_save_async_false_does_not_enter_async_checkpoint_path():
    callback = CheckpointerCallback(save_async=False, save_interval=None)
    trainer = MagicMock()
    trainer.global_step = 10
    trainer.save_checkpoint.return_value = "/checkpoints/step10"
    callback.trainer = trainer

    path = callback._save_checkpoint()

    assert path == "/checkpoints/step10"
    trainer.save_checkpoint.assert_called_once_with(ephemeral=False)
    trainer.save_checkpoint_async.assert_not_called()
    trainer.finalize_async_checkpoint.assert_not_called()
    assert callback._future is None


def run_reconcile_async_checkpoint_durations():
    trainer = object.__new__(Trainer)
    trainer._bookkeeping_pg = None
    rank = dist.get_rank()

    incomplete = trainer._reconcile_async_checkpoint_durations(
        {"step10": (1.0 if rank == 0 else None, None)}
    )
    assert incomplete.completed_durations == {}
    assert incomplete.failures == {}

    local_duration = 1.0 + rank
    completed = trainer._reconcile_async_checkpoint_durations({"step10": (local_duration, None)})
    assert completed.completed_durations == {"step10": local_duration}
    assert completed.failures == {}


def run_report_async_checkpoint_duration_at_shared_metric_boundary():
    trainer = object.__new__(Trainer)
    trainer._bookkeeping_pg = None
    trainer.async_bookkeeping = False
    trainer.global_step = 20
    trainer._async_checkpoint_completion_lock = threading.Lock()
    trainer._pending_async_checkpoint_paths = {"step10"}
    trainer._completed_async_checkpoint_durations = {}
    trainer._completed_async_checkpoint_errors = {}
    trainer._async_checkpoint_reconciliation_future = None
    trainer._metrics_lock = threading.RLock()
    trainer._metrics = OrderedDict()
    trainer._metrics_reduce_type = {}
    rank = dist.get_rank()

    # Rank 0 finishing early must not create a rank-local, step-keyed metric.
    if rank == 0:
        trainer._completed_async_checkpoint_durations["step10"] = 1.0
    trainer._update_async_checkpoint_metrics()
    assert trainer._metrics == {}
    assert trainer._pending_async_checkpoint_paths == {"step10"}

    # At a later shared metric boundary both ranks have completed. They now file their local
    # durations under the same current step, and the normal MAX reduction reports the slowest.
    trainer._completed_async_checkpoint_durations["step10"] = 1.0 + rank
    trainer._update_async_checkpoint_metrics()
    snapshot = trainer._take_metrics_snapshot()
    assert snapshot is not None
    metrics, reduce_types = snapshot
    assert list(metrics) == [20]
    assert reduce_types["checkpoint/save_async_duration_s"] == ReduceType.max
    reduced = reduce_metrics(metrics, reduce_types, torch.device("cpu"))
    assert reduced[20]["checkpoint/save_async_duration_s"] == 2.0
    assert not trainer._pending_async_checkpoint_paths


def run_propagate_async_checkpoint_failure_at_shared_metric_boundary():
    trainer = object.__new__(Trainer)
    trainer._bookkeeping_pg = None
    trainer.async_bookkeeping = False
    trainer.global_step = 20
    trainer._async_checkpoint_completion_lock = threading.Lock()
    trainer._pending_async_checkpoint_paths = {"step10"}
    trainer._completed_async_checkpoint_durations = {}
    trainer._completed_async_checkpoint_errors = {}
    trainer._async_checkpoint_reconciliation_future = None
    trainer._metrics_lock = threading.RLock()
    trainer._metrics = OrderedDict()
    trainer._metrics_reduce_type = {}
    if dist.get_rank() == 0:
        trainer._completed_async_checkpoint_errors["step10"] = RuntimeError("disk failed")

    with pytest.raises(RuntimeError, match="disk failed"):
        trainer._update_async_checkpoint_metrics()
    assert not trainer._pending_async_checkpoint_paths


@pytest.mark.parametrize("backend", BACKENDS)
def test_reconcile_async_checkpoint_durations(backend):
    run_distributed_test(run_reconcile_async_checkpoint_durations, backend=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_report_async_checkpoint_duration_at_shared_metric_boundary(backend):
    run_distributed_test(
        run_report_async_checkpoint_duration_at_shared_metric_boundary, backend=backend
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_propagate_async_checkpoint_failure_at_shared_metric_boundary(backend):
    run_distributed_test(
        run_propagate_async_checkpoint_failure_at_shared_metric_boundary, backend=backend
    )
