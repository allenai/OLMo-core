"""
This module implements a highly efficient, yet flexible, language model trainer.

Features
--------

- Async checkpointing (optional) with local or remote checkpoint directories.
- Supports any type of parallel strategy.
- Async metric logging, with support for custom metrics, even those that need to be reduced across
  ranks.
- Flexible callback system for extending/modifying the training loop behavior.
- A powerful set of built-in callbacks.

Overview
--------

Call :func:`prepare_training_environment()` at the top of your training script,
then construct your trainer using a :class:`TrainerConfig`. Finally, call :meth:`Trainer.fit()`
and cleanup at the end of your script by calling :func:`teardown_training_environment()`.
For example::

    if __name__ == "__main__":
        prepare_training_environment()
        try:
            # Build model, optimizer, dataset...

            # Build trainer.
            trainer = trainer_config.build(model, optim, dataset)

            # Run the trainer.
            trainer.fit()
        finally:
            teardown_training_environment()

See the `train a language model <examples/train.html>`_ example for a complete, run-able demonstration.

API Reference
-------------
"""

import logging
from datetime import timedelta
from typing import Optional

import torch.distributed as dist
import torch.multiprocessing as mp

from ..distributed.utils import init_distributed, is_distributed
from ..io import add_cached_path_clients
from ..utils import LogFilterType, prepare_cli_environment, seed_all
from .common import Duration, DurationUnit, LoadStrategy, ReduceType
from .config import TrainerConfig
from .trainer import Trainer

__all__ = [
    "prepare_training_environment",
    "teardown_training_environment",
    "TrainerConfig",
    "Trainer",
    "LoadStrategy",
    "Duration",
    "DurationUnit",
    "ReduceType",
]


log = logging.getLogger(__name__)


def prepare_training_environment(
    *,
    seed: Optional[int] = None,
    backend: Optional[str] = "cpu:gloo,cuda:nccl",
    timeout: timedelta = timedelta(minutes=10),
    log_filter_type: Optional[LogFilterType] = None,
):
    """
    Prepare the environment for training, including setting up the distributed process group
    for distributed training.

    .. tip::
        Internally this calls:

        - :func:`~olmo_core.distributed.utils.init_distributed()`, which also calls :func:`torch.cuda.set_device()`
          for backends that support CUDA.
        - :func:`~olmo_core.utils.prepare_cli_environment()`

        So there's no need to call those separately.

    .. important::
        This should be invoked at the very start of your training script, such as at the beginning
        of the ``if __name__ == "__main__": ...`` block.

    :param seed: The seed to initialize RNG states with.
    :param backend: The distributed backend to use, if any. Set to ``None`` for non-distributed training.
        When using NCCL, ideally you should also include a CPU-only backend (the default) like GLOO,
        which allows the trainer to run async checkpointing and bookkeeping collectives on the CPU
        backend without blocking training operations.
    :param timeout: The timeout for initializing the distributed process group.
    :param log_filter_type: Determines which ranks are allowed to emit log messages below the
        ``WARNING`` level. You can also configure this through the env var ``LOG_FILTER_TYPE``.
        If neither are set, this defaults to "rank0_only".

        .. note::
            All ranks will always emit messages at the ``WARNING`` level or higher.
    """
    # Setting the mp start method to "spawn" avoids some data loader segfaults on LUMI.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")

    # Initialize process group.
    if backend is not None:
        init_distributed(backend=backend, timeout=timeout)

    # Configure logging, warning filters, exception hooks, and other CLI settings.
    prepare_cli_environment(log_filter_type=log_filter_type)

    # Add custom cached-path clients.
    add_cached_path_clients()

    # Init RNG states.
    if seed is not None:
        seed_all(seed)

    if is_distributed():
        log.info(f"Using distributed backend {dist.get_backend()}")


def teardown_training_environment():
    """
    To be run at the end of training. Tears down all distributed process groups.
    """
    if is_distributed():
        dist.destroy_process_group()
