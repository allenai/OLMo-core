from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer

from ..config import Config, DType
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

    work_dir: str
    save_folder: str
    global_batch_size: int
    microbatch_size: int

    device: Optional[str] = None
    save_overwrite: bool = False
    max_duration: Duration = field(
        default_factory=lambda: Duration(value=1, unit=DurationUnit.epochs)
    )
    metrics_collect_interval: int = 5
    callbacks: List[Callback] = field(default_factory=list)
    fused_loss: bool = False
    z_loss_multiplier: Optional[float] = None
    autocast_precision: Optional[DType] = None
    data_seed: int = 0
    data_loader_workers: int = 0
    data_loader_prefetch_factor: Optional[int] = None

    def with_callback(self, callback: Callback) -> "TrainerConfig":
        """
        Add another callback.

        :param callback: The callback to add.
        """
        self.callbacks.append(callback)
        return self

    def build(
        self,
        model: nn.Module,
        optim: Optimizer,
        dataset: MemMapDataset,
        dp_process_group: Optional[dist.ProcessGroup] = None,
        checkpointer_pg: Optional[dist.ProcessGroup] = None,
    ) -> Trainer:
        """
        Build the corresponding trainer.

        :param model: The model to train.
        :param optim: The optimizer to use.
        :param dataset: The dataset to train on.
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)

        checkpointer = Checkpointer(
            save_overwrite=kwargs.pop("save_overwrite"),
            process_group=checkpointer_pg,
        )
        device = kwargs.pop("device", None)
        autocast_precision: Optional[DType] = kwargs.pop("autocast_precision", None)
        collator = DataCollator(pad_token_id=dataset.pad_token_id)

        return Trainer(
            model=model,
            optim=optim,
            dataset=dataset,
            collator=collator,
            checkpointer=checkpointer,
            train_sequence_length=dataset.sequence_length,
            autocast_precision=None if autocast_precision is None else autocast_precision.as_pt(),
            device=torch.device(device) if device is not None else get_default_device(),
            dp_process_group=dp_process_group,
            **kwargs,
        )
