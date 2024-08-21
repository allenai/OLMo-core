from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer

from ..aliases import PathOrStr
from ..config import Config
from ..data import DataCollator, IterableDataset
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
    dp_process_group: Optional[dist.ProcessGroup] = None
    data_loader_workers: int = 0
    data_loader_prefetch_factor: Optional[int] = None

    def build(
        self, model: nn.Module, optim: Optimizer, dataset: IterableDataset, collator: DataCollator
    ) -> Trainer:
        """
        Build the corresponding trainer.
        """
        kwargs = self.as_dict(recurse=False)
        return Trainer(model=model, optim=optim, dataset=dataset, collator=collator, **kwargs)

    def with_callback(self, callback: Callback) -> "TrainerConfig":
        """
        Add another callback.
        """
        self.callbacks.append(callback)
        return self
