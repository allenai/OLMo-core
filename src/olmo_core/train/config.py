from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..aliases import PathOrStr
from ..config import Config
from ..data.iterable_dataset import IterableDataset
from ..utils import get_default_device
from .callbacks import Callback
from .checkpoint import Checkpointer
from .trainer import Trainer
from .utils import Duration, DurationUnit


@dataclass
class TrainerConfig(Config):
    """
    A configuration class for easily building :class:`~olmo_core.train.trainer.Trainer` instances.
    """

    work_dir: PathOrStr
    save_folder: str
    train_sequence_length: int
    global_batch_size: int
    microbatch_size: int

    device: torch.device = field(default_factory=get_default_device)
    checkpointer: Checkpointer = Checkpointer()
    max_duration: Duration = Duration(value=1, unit=DurationUnit.epochs)
    metrics_log_interval: int = 1
    callbacks: List[Callback] = field(default_factory=list)
    fused_loss: bool = False
    z_loss_multiplier: Optional[float] = None
    autocast_precision: Optional[torch.dtype] = None

    def build(
        self, model: nn.Module, optim: Optimizer, train_loader: DataLoader[IterableDataset]
    ) -> Trainer:
        """
        Build the corresponding trainer.
        """
        kwargs = self.as_dict(recurse=False)
        return Trainer(model=model, optim=optim, train_loader=train_loader, **kwargs)

    def with_callback(self, callback: Callback) -> "TrainerConfig":
        """
        Add another callback.
        """
        self.callbacks.append(callback)
        return self
