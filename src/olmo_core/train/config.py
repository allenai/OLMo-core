from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer

from ..aliases import PathOrStr
from ..config import Config
from ..data import DataCollator, MemMapDataset
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
    global_batch_size: int
    microbatch_size: int

    device: torch.device = field(default_factory=get_default_device)
    save_overwrite: bool = False
    checkpointer_pg: Optional[dist.ProcessGroup] = None
    max_duration: Duration = Duration(value=1, unit=DurationUnit.epochs)
    metrics_log_interval: int = 1
    callbacks: List[Callback] = field(default_factory=list)
    fused_loss: bool = False
    z_loss_multiplier: Optional[float] = None
    autocast_precision: Optional[torch.dtype] = None
    dp_process_group: Optional[dist.ProcessGroup] = None
    data_seed: int = 0
    data_loader_workers: int = 0
    data_loader_prefetch_factor: Optional[int] = None

    def build(self, model: nn.Module, optim: Optimizer, dataset: MemMapDataset) -> Trainer:
        """
        Build the corresponding trainer.

        :param model: The model to train.
        :param optim: The optimizer to use.
        :param dataset: The dataset to train on.
        """
        kwargs = self.as_dict(recurse=False)

        checkpointer = Checkpointer(
            save_overwrite=kwargs.pop("save_overwrite"), process_group=kwargs.pop("checkpointer_pg")
        )

        collator = DataCollator(pad_token_id=dataset.pad_token_id)

        return Trainer(
            model=model,
            optim=optim,
            dataset=dataset,
            collator=collator,
            checkpointer=checkpointer,
            train_sequence_length=dataset.sequence_length,
            **kwargs,
        )

    def with_callback(self, callback: Callback) -> "TrainerConfig":
        """
        Add another callback.

        :param callback: The callback to add.
        """
        self.callbacks.append(callback)
        return self
