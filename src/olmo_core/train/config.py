import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist

from olmo_core.eval.task_groups import TASK_GROUPS

from ..config import Config
from ..data import DataLoaderBase, TokenizerConfig
from ..exceptions import OLMoConfigurationError
from ..io import is_url
from ..utils import get_default_device
from .callbacks import Callback, CallbackConfig
from .checkpoint import CheckpointerConfig
from .common import Duration, LoadStrategy
from .train_module import TrainModule
from .trainer import Trainer


@dataclass
class TrainerConfig(Config):
    """
    A configuration class for easily building :class:`Trainer` instances.

    .. seealso::
        See the :class:`Trainer` documentation for a description of the fields.
    """

    save_folder: str

    work_dir: Optional[str] = None
    load_path: Optional[str] = None
    load_strategy: LoadStrategy = LoadStrategy.if_available
    load_optim_state: Optional[bool] = None
    load_trainer_state: Optional[bool] = None
    checkpointer: CheckpointerConfig = field(default_factory=CheckpointerConfig)

    device: Optional[str] = None
    save_overwrite: bool = False
    max_duration: Duration = field(default_factory=lambda: Duration.epochs(1))
    cancel_check_interval: int = 25
    hard_stop: Optional[Duration] = None
    metrics_collect_interval: int = 5
    callbacks: Dict[str, Callback] = field(default_factory=dict)
    async_bookkeeping: Optional[bool] = None
    bookkeeping_soft_timeout: int = 30
    no_checkpoints: bool = False
    no_evals: bool = False

    def add_callback(self, name: str, callback: Callback):
        """
        Add another callback.
        """
        if name in self.callbacks:
            raise OLMoConfigurationError(f"A callback with name '{name}' already exists")
        self.callbacks[name] = callback

    def add_callbacks(self, callbacks: Dict[str, Callback]):
        """
        Add a set of callbacks.
        """
        for name, callback in callbacks.items():
            self.add_callback(name, callback)

    def with_callback(self, name: str, callback: Callback) -> "TrainerConfig":
        """
        Return a new trainer config with an additional callback.

        :param name: A name to assign the callback. Must be unique.
        :param callback: The callback to add.
        """
        out = replace(self, callbacks=deepcopy(self.callbacks))
        out.add_callback(name, callback)
        return out

    def with_recommended_evals(
        self,
        tokenizer: TokenizerConfig,
        sequence_length: int,
        cluster: str,
        task_set: str = "full",
        eval_interval: int = 10_000,
    ) -> "TrainerConfig":
        """
        Return a new trainer config with added callbacks for downstream evaluation and validation sets.
        """
        from olmo_core.data import DataMix, NumpyPaddedFSLDatasetConfig
        from olmo_core.internal.common import get_root_dir, get_work_dir
        from olmo_core.train.callbacks import (
            DownstreamEvaluatorCallbackConfig,
            LMEvaluatorCallbackConfig,
        )

        try:
            tasks = TASK_GROUPS[task_set]
        except KeyError as e:
            raise ValueError(f"Task set not recognized: {task_set}") from e

        tasks.sort()

        return self.with_callback(
            "downstream_evaluator",
            DownstreamEvaluatorCallbackConfig(
                tasks=tasks, tokenizer=tokenizer, eval_interval=eval_interval
            ),
        ).with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=get_root_dir(cluster),
                    sequence_length=sequence_length,
                    tokenizer=tokenizer,
                    work_dir=get_work_dir(get_root_dir(cluster)),
                ),
                eval_interval=eval_interval,
            ),
        )

    def build(
        self,
        train_module: TrainModule,
        data_loader: DataLoaderBase,
        *,
        dp_process_group: Optional[dist.ProcessGroup] = None,
        checkpointer_pg: Optional[dist.ProcessGroup] = None,
    ) -> Trainer:
        """
        Build the corresponding trainer.

        :param train_module: The train module to fit.
        :param data_loader: The data loader to train on.
        :param dp_process_group: The data parallel process group. Defaults to
            :data:`olmo_core.train.train_module.TrainModule.dp_process_group`.
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)

        if dp_process_group is None:
            dp_process_group = train_module.dp_process_group

        device = kwargs.pop("device", None)

        work_dir = kwargs.pop("work_dir", None)
        if work_dir is None:
            if not is_url(self.save_folder):
                work_dir = self.save_folder
            else:
                work_dir = os.path.join(tempfile.gettempdir(), os.path.basename(self.save_folder))
        elif is_url(work_dir):
            raise OLMoConfigurationError(
                f"Trainer 'work_dir' must be a local path, not a URL ('{work_dir}')"
            )

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
            train_module=train_module,
            data_loader=data_loader,
            checkpointer=checkpointer,
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
