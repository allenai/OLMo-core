import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.optim import Optimizer

from ..config import Config, DType
from ..data import DataLoaderBase
from ..distributed.parallel import get_dp_process_group
from ..exceptions import OLMoConfigurationError
from ..io import is_url
from ..utils import get_default_device
from .callbacks import Callback, CallbackConfig
from .checkpoint import CheckpointerConfig
from .common import Duration, LoadStrategy
from .trainer import Trainer


@dataclass
class TrainerConfig(Config):
    """
    A configuration class for easily building :class:`Trainer` instances.

    .. seealso::
        See the :class:`Trainer` documentation for a description of the fields.
    """

    save_folder: str
    rank_microbatch_size: int

    work_dir: Optional[str] = None
    load_path: Optional[str] = None
    load_strategy: LoadStrategy = LoadStrategy.if_available
    load_key_mapping: Optional[Dict[str, str]] = None
    checkpointer: CheckpointerConfig = field(default_factory=CheckpointerConfig)

    device: Optional[str] = None
    save_overwrite: bool = False
    max_duration: Duration = field(default_factory=lambda: Duration.epochs(2))
    cancel_check_interval: int = 25
    hard_stop: Optional[Duration] = None
    metrics_collect_interval: int = 5
    callbacks: Dict[str, Callback] = field(default_factory=dict)
    fused_loss: bool = False
    compile_loss: bool = False
    z_loss_multiplier: Optional[float] = None
    autocast_precision: Optional[DType] = None
    async_bookkeeping: Optional[bool] = None

    def add_callback(self, name: str, callback: Callback):
        """
        Add another callback.
        """
        if name in self.callbacks:
            raise OLMoConfigurationError(f"A callback with name '{name}' already exists")
        self.callbacks[name] = callback

    def with_callback(self, name: str, callback: Callback) -> "TrainerConfig":
        """
        Add another callback, returning the trainer config.

        :param name: A name to assign the callback. Must be unique.
        :param callback: The callback to add.
        """
        self.add_callback(name, callback)
        return self

    def build(
        self,
        model: nn.Module,
        optim: Optimizer,
        data_loader: DataLoaderBase,
        *,
        mesh: Optional[DeviceMesh] = None,
        dp_process_group: Optional[dist.ProcessGroup] = None,
        checkpointer_pg: Optional[dist.ProcessGroup] = None,
    ) -> Trainer:
        """
        Build the corresponding trainer.

        :param model: The model to train.
        :param optim: The optimizer to use.
        :param data_loader: The data loader to train on.
        :param mesh: An optional ``DeviceMesh`` that defines the data parallel dimensions. Ideally
            you should create this mesh using :func:`~olmo_core.distributed.parallel.build_device_mesh()`
            or equivalently :meth:`olmo_core.nn.transformer.TransformerConfig.build_mesh()`.
            Alternatively you can pass the ``dp_process_group`` instead.
        :param dp_process_group: The data parallel process group.
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)

        if dp_process_group is None and mesh is not None:
            dp_process_group = get_dp_process_group(mesh)

        device = kwargs.pop("device", None)

        autocast_precision: Optional[DType] = kwargs.pop("autocast_precision", None)

        work_dir = kwargs.pop("work_dir", None)
        if work_dir is None:
            if not is_url(self.save_folder):
                work_dir = self.save_folder
            else:
                work_dir = os.path.join(tempfile.gettempdir(), os.path.basename(self.save_folder))

        checkpointer_kwargs = {}
        if self.checkpointer.save_overwrite is None:
            checkpointer_kwargs["save_overwrite"] = self.save_overwrite
        if self.checkpointer.work_dir is None:
            checkpointer_kwargs["work_dir"] = work_dir
        checkpointer = kwargs.pop("checkpointer").build(
            process_group=checkpointer_pg, **checkpointer_kwargs
        )

        all_callbacks = kwargs.pop("callbacks")
        callbacks = {k: cb for k, cb in all_callbacks.items() if not isinstance(cb, CallbackConfig)}
        callback_configs = {
            k: cb for k, cb in all_callbacks.items() if isinstance(cb, CallbackConfig)
        }

        trainer = Trainer(
            model=model,
            optim=optim,
            data_loader=data_loader,
            checkpointer=checkpointer,
            autocast_precision=None if autocast_precision is None else autocast_precision.as_pt(),
            work_dir=Path(work_dir),
            device=torch.device(device) if device is not None else get_default_device(),
            dp_process_group=dp_process_group,
            callbacks=callbacks,
            **kwargs,
        )

        for cb_name, cb_config in callback_configs.items():
            cb = cb_config.build(trainer)
            if cb is not None:
                trainer.add_callback(cb_name, cb)

        return trainer
