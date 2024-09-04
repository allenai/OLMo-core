import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer

from ..config import Config, DType
from ..data import DataCollator, MemMapDataset
from ..exceptions import OLMoConfigurationError
from ..io import is_url
from ..utils import get_default_device
from .callbacks import Callback
from .checkpoint import Checkpointer
from .trainer import LoadStrategy, Trainer
from .utils import Duration, DurationUnit


@dataclass
class TrainerConfig(Config):
    """
    A configuration class for easily building :class:`~olmo_core.train.trainer.Trainer` instances.
    """

    save_folder: str
    global_batch_size: int
    microbatch_size: int

    work_dir: Optional[str] = None
    load_path: Optional[str] = None
    load_strategy: LoadStrategy = LoadStrategy.if_available

    device: Optional[str] = None
    save_overwrite: bool = False
    max_duration: Duration = field(
        default_factory=lambda: Duration(value=1, unit=DurationUnit.epochs)
    )
    cancel_check_interval: int = 25
    hard_stop: Optional[Duration] = None
    metrics_collect_interval: int = 5
    callbacks: Dict[str, Callback] = field(default_factory=dict)
    fused_loss: bool = False
    z_loss_multiplier: Optional[float] = None
    autocast_precision: Optional[DType] = None
    data_seed: int = 0
    data_loader_workers: int = 0
    data_loader_prefetch_factor: Optional[int] = None

    def with_callback(self, name: str, callback: Callback) -> "TrainerConfig":
        """
        Add another callback.

        :param name: A name to assign the callback. Must be unique.
        :param callback: The callback to add.
        """
        if name in self.callbacks:
            raise OLMoConfigurationError(f"A callback with name '{name}' already exists")
        self.callbacks[name] = callback
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
            save_overwrite=kwargs["save_overwrite"],
            process_group=checkpointer_pg,
        )
        device = kwargs.pop("device", None)
        autocast_precision: Optional[DType] = kwargs.pop("autocast_precision", None)
        work_dir = kwargs.pop("work_dir", None)
        if work_dir is None:
            if not is_url(self.save_folder):
                work_dir = self.save_folder
            else:
                work_dir = os.path.join(tempfile.gettempdir(), os.path.basename(self.save_folder))

        collator = DataCollator(pad_token_id=dataset.pad_token_id)

        return Trainer(
            model=model,
            optim=optim,
            dataset=dataset,
            collator=collator,
            checkpointer=checkpointer,
            train_sequence_length=dataset.sequence_length,
            autocast_precision=None if autocast_precision is None else autocast_precision.as_pt(),
            work_dir=Path(work_dir),
            device=torch.device(device) if device is not None else get_default_device(),
            dp_process_group=dp_process_group,
            **kwargs,
        )
