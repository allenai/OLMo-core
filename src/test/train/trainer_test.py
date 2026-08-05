from collections import OrderedDict, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from olmo_core.train.trainer import Trainer, _metric_value_to_tensor


def test_metric_value_to_tensor_accepts_large_python_int():
    value = 10**20
    tensor = _metric_value_to_tensor(value)

    assert tensor.dtype == torch.float64
    assert tensor.item() == float(value)


def test_canceling_bookkeeping_op_can_remove_itself_from_queue():
    trainer = object.__new__(Trainer)
    trainer.bookkeeping_soft_timeout = 30
    trainer._bookkeeping_queue = defaultdict(OrderedDict)
    trainer._multi_thread_pool = ThreadPoolExecutor(max_workers=1)
    trainer._error = None

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
