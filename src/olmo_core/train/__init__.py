from datetime import timedelta
from typing import Optional

import torch.distributed as dist
import torch.multiprocessing as mp

from ..distributed.utils import init_distributed, is_distributed
from ..io import add_cached_path_clients
from ..utils import LogFilterType, prepare_cli_environment, seed_all
from .config import TrainerConfig
from .trainer import Trainer

__all__ = [
    "prepare_training_environment",
    "teardown_training_environment",
    "TrainerConfig",
    "Trainer",
]


def prepare_training_environment(
    *,
    seed: int = 0,
    backend: Optional[str] = "nccl",
    timeout: timedelta = timedelta(minutes=30),
    log_filter_type: Optional[LogFilterType] = None,
):
    """
    Prepare the environment for training, including setting up the distributed process group
    for distributed training.

    .. tip::
        Internally this calls :func:`~olmo_core.utils.prepare_cli_environment()`, so there's no
        need to call that separately.

    .. important::
        This should be invoked at the very start of your training script, such as at the beginning
        of the ``if __name__ == "__main__": ...`` block.

    :param seed: The seed to initialize RNG states with.
    :param backend: The distributed backend to use, if any. Set to ``None`` for non-distributed training.
    :param timeout: The timeout for initializing the distributed process group.
    :param log_filter_type: Which ranks emit INFO and below messages. You can also configure this
        through the env var ``LOG_FILTER_TYPE``. If neither are set, this defaults to "rank0_only".
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
    seed_all(seed)


def teardown_training_environment():
    """
    To be run at the end of training. Tears down the distributed process group.
    """
    if is_distributed():
        dist.destroy_process_group()
