"""
Integration coverage for asynchronous trainer checkpointing.

This uses a real distributed checkpoint save and load while injecting deterministic per-rank
completion delays around the real DCP future. Run it with:

.. code-block:: bash

    pytest -v src/integration_tests/test_async_checkpointing.py
"""

import os
import random
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.distributed as dist

from olmo_core.distributed.utils import barrier, get_rank, get_world_size
from olmo_core.testing import run_distributed_test
from olmo_core.train.checkpoint import Checkpointer
from olmo_core.train.train_module import BasicTrainModule
from olmo_core.train.trainer import Trainer
from olmo_core.train.utils import reduce_metrics


def _delay_future(source: Future[None], delay: float, executor: ThreadPoolExecutor) -> Future[None]:
    delayed: Future[None] = Future()

    def propagate(source_future: Future[None]) -> None:
        def complete() -> None:
            try:
                source_future.result()
                time.sleep(delay)
            except BaseException as exc:
                delayed.set_exception(exc)
            else:
                delayed.set_result(None)

        executor.submit(complete)

    source.add_done_callback(propagate)
    return delayed


def _build_minimal_trainer(
    save_folder: Path,
    work_dir: Path,
    train_module: BasicTrainModule,
    checkpointer: Checkpointer,
) -> Trainer:
    trainer = object.__new__(Trainer)
    trainer.global_step = 0
    trainer.save_folder = str(save_folder)
    trainer.train_module = train_module
    trainer.checkpointer = checkpointer
    trainer.async_bookkeeping = False
    trainer._bookkeeping_pg = None
    trainer._async_checkpoint_completion_lock = threading.Lock()
    trainer._async_checkpoint_completions = {}
    trainer._completed_async_checkpoint_durations = {}
    trainer._completed_async_checkpoint_errors = {}
    trainer._pending_async_checkpoint_paths = set()
    trainer._async_checkpoint_reconciliation_future = None
    trainer._metrics_lock = threading.RLock()
    trainer._metrics = OrderedDict()
    trainer._metrics_reduce_type = {}
    setattr(trainer, "_log_metrics", MagicMock())
    setattr(trainer, "_join_bookkeeping_ops", MagicMock())
    setattr(trainer, "_iter_callbacks", MagicMock(side_effect=lambda: iter([])))
    setattr(
        trainer,
        "state_dict",
        MagicMock(side_effect=lambda: {"rank": get_rank(), "step": trainer.global_step}),
    )
    trainer.work_dir = work_dir
    return trainer


def run_async_checkpointing_integration(base_dir: Path) -> None:
    os.environ["OLMO_SHARED_FS"] = "1"
    rank = get_rank()
    world_size = get_world_size()
    torch.manual_seed(7)
    work_dir = base_dir / f"work-rank{rank}"
    work_dir.mkdir(parents=True, exist_ok=True)

    model = torch.nn.Linear(8, 8)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_module = BasicTrainModule(model, optim, 8)

    checkpointer = Checkpointer(work_dir=work_dir, process_group=dist.new_group())
    trainer = _build_minimal_trainer(base_dir, work_dir, train_module, checkpointer)
    original_save_async = checkpointer.save_async

    with ThreadPoolExecutor(max_workers=1) as delay_executor:
        round_delay = 0.0

        def delayed_save_async(*args, **kwargs):
            return _delay_future(original_save_async(*args, **kwargs), round_delay, delay_executor)

        setattr(checkpointer, "save_async", delayed_save_async)

        for round_idx in range(4):
            step = (round_idx + 1) * 10
            trainer.global_step = step

            # Change model and optimizer state before every checkpoint so every reload assertion
            # validates a distinct real DCP payload.
            model(torch.randn(2, 8)).sum().backward()
            optim.step()
            optim.zero_grad(set_to_none=True)
            expected_model_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            expected_optim_state = optim.state_dict()

            # Every rank derives the same delay vector. Seeded jitter varies timings while rotating
            # the slow rank guarantees that completion order changes across rounds.
            delay_rng = random.Random(20260814 + round_idx)
            delays = [0.10 + delay_rng.uniform(0.0, 0.05) for _ in range(world_size)]
            slow_rank = round_idx % world_size
            delays[slow_rank] += 0.60
            round_delay = delays[rank]

            path, completion_future = trainer.save_checkpoint_async()

            # Simulate useful training-side work while the real checkpoint is in flight. The
            # controlled slow rank must still be pending when this work finishes.
            work_deadline = time.perf_counter() + 0.25
            training_work = 0
            while time.perf_counter() < work_deadline:
                training_work += sum(i * i for i in range(200))
            assert training_work > 0
            if rank == slow_rank:
                assert not completion_future.done(), "checkpoint save unexpectedly blocked training"

            completion_future.result(timeout=30)

            # Reconcile at a shared metrics boundary. Every rank records under the same step, and
            # MAX must report the deliberately slowest completion for this round.
            trainer._update_async_checkpoint_metrics()
            snapshot = trainer._take_metrics_snapshot()
            assert snapshot is not None
            metrics, reduce_types = snapshot
            reduced = reduce_metrics(metrics, reduce_types, torch.device("cpu"))
            duration = reduced[step]["checkpoint/save_async_duration_s"]
            assert duration >= max(delays) - 0.05

            barrier()
            checkpoint_dir = base_dir / Checkpointer.checkpoint_dirname(step)
            reloaded_model = torch.nn.Linear(8, 8)
            reloaded_optim = torch.optim.AdamW(reloaded_model.parameters(), lr=1e-3)
            reloaded_module = BasicTrainModule(reloaded_model, reloaded_optim, 8)
            train_state = checkpointer.load(checkpoint_dir, reloaded_module)
            assert train_state == {"rank": rank, "step": step}
            torch.testing.assert_close(reloaded_model.state_dict(), expected_model_state)
            torch.testing.assert_close(reloaded_optim.state_dict(), expected_optim_state)


def test_async_checkpointing_integration(tmp_path: Path) -> None:
    run_distributed_test(
        run_async_checkpointing_integration,
        func_args=(tmp_path / "checkpoint-integration",),
        start_method="spawn",
    )
