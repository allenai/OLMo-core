from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event

import pytest
import torch

from olmo_core.train.trainer import Trainer, _metric_value_to_tensor


def _bookkeeping_trainer() -> Trainer:
    trainer = object.__new__(Trainer)
    trainer.bookkeeping_soft_timeout = 30
    trainer._bookkeeping_queue = defaultdict(OrderedDict)
    trainer._multi_thread_pool = ThreadPoolExecutor(max_workers=1)
    trainer._error = None
    return trainer


def test_metric_value_to_tensor_accepts_large_python_int():
    value = 10**20
    tensor = _metric_value_to_tensor(value)

    assert tensor.dtype == torch.float64
    assert tensor.item() == float(value)


def test_canceling_bookkeeping_op_can_remove_itself_from_queue():
    trainer = _bookkeeping_trainer()

    op_name = "test_op"
    op_id = "existing"
    future: Future[None] = Future()
    trainer._bookkeeping_queue[op_name][op_id] = future
    future.add_done_callback(lambda _: trainer._bookkeeping_queue[op_name].pop(op_id, None))

    try:
        trainer.run_bookkeeping_op(
            lambda: None,
            op_name=op_name,
            allow_multiple=False,
            distributed=False,
        )
        trainer._join_bookkeeping_ops(timeout=1)
        assert trainer._error is None
        assert not trainer._bookkeeping_queue[op_name]
    finally:
        trainer.multi_thread_pool.shutdown(wait=True)


@pytest.mark.parametrize("failure_source", ["op", "callback"])
def test_join_bookkeeping_ops_propagates_async_failures(failure_source: str):
    trainer = _bookkeeping_trainer()
    op_started = Event()
    release_op = Event()
    failure = ValueError(f"{failure_source} failed")

    def op() -> None:
        op_started.set()
        assert release_op.wait(timeout=1)
        if failure_source == "op":
            raise failure

    def callback(_: None) -> None:
        if failure_source == "callback":
            raise failure

    try:
        trainer.run_bookkeeping_op(op, cb=callback, distributed=False)
        assert op_started.wait(timeout=1)
        release_op.set()

        with pytest.raises(RuntimeError, match="An error occurred") as exc_info:
            trainer._join_bookkeeping_ops(timeout=1)

        assert exc_info.value.__cause__ is failure
    finally:
        release_op.set()
        trainer.multi_thread_pool.shutdown(wait=True)


def test_join_bookkeeping_ops_propagates_error_from_queued_future():
    trainer = _bookkeeping_trainer()
    failure = ValueError("queued future failed")
    future: Future[None] = Future()
    future.set_exception(failure)
    trainer._bookkeeping_queue["failed"]["op"] = future

    try:
        with pytest.raises(RuntimeError, match="An error occurred") as exc_info:
            trainer._join_bookkeeping_ops(timeout=0)

        assert exc_info.value.__cause__ is failure
        assert trainer._error is failure
    finally:
        trainer.multi_thread_pool.shutdown(wait=True)


def test_join_bookkeeping_ops_propagates_preexisting_error_after_self_removal():
    trainer = _bookkeeping_trainer()
    failure = ValueError("failed before join")

    def fail() -> None:
        raise failure

    try:
        trainer.run_bookkeeping_op(fail, distributed=False)
        trainer.multi_thread_pool.shutdown(wait=True)

        assert not trainer._bookkeeping_queue[fail.__qualname__]
        assert trainer._error is failure
        with pytest.raises(RuntimeError, match="An error occurred") as exc_info:
            trainer._join_bookkeeping_ops(timeout=0)

        assert exc_info.value.__cause__ is failure
    finally:
        trainer.multi_thread_pool.shutdown(wait=True)


def test_join_bookkeeping_ops_does_not_block_past_timeout():
    trainer = _bookkeeping_trainer()
    future: Future[None] = Future()
    trainer._bookkeeping_queue["pending"]["op"] = future

    try:
        trainer._join_bookkeeping_ops(timeout=0)
        assert not future.done()
        assert trainer._bookkeeping_queue["pending"]["op"] is future
    finally:
        future.cancel()
        trainer.multi_thread_pool.shutdown(wait=True)
