from datetime import timedelta
from typing import Optional

import torch.multiprocessing as mp

from ..distributed.utils import init_distributed
from ..io import add_cached_path_clients
from ..utils import prepare_cli_environment
from .config import TrainerConfig
from .trainer import Trainer

__all__ = ["prepare_training_environment", "TrainerConfig", "Trainer"]


def prepare_training_environment(
    backend: Optional[str] = "nccl", timeout: timedelta = timedelta(minutes=30)
):
    """
    Prepare the environment for training, including setting up the distributed process group
    for distributed training.

    .. important::
        This should be invoked at the very start of your training script, such as at the beginning
        of the ``if __name__ == "__main__": ...`` block.

    :param backend: The distributed backend to use, if any. Set to ``None`` for non-distributed training.
    :param timeout: The timeout for initializing the distributed process group.
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
    prepare_cli_environment()

    # Add custom cached-path clients.
    add_cached_path_clients()
