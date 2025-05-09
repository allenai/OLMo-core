import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.optim import Scheduler, SkipStepAdamW

from ..common import Duration
from ..train_module import TransformerPipelineTrainModule, TransformerTrainModule
from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class BatchSizeSchedulerCallback(Callback):
    """
    A callback for setting a batch size scheduler over the course of a training run.
    Also adjusts the base learning rate with Adam optimizers for transformer train modules by a factor of
    ``sqrt(new_batch_size / current_batch_size)``.
    """

    schedule: List[Tuple[int, Duration]] = dataclasses.field(default_factory=list)
    """
    Defines the schedule. Each tuple in the schedule represents a target batch size after a point
    in training.
    """

    def __post_init__(self):
        if not self.schedule:
            return

        if len(set([duration.unit for _, duration in self.schedule])) > 1:
            raise OLMoConfigurationError(
                "batch size scheduler must use consistent units for all points in the schedule"
            )

        batch_size, event_start = self.schedule[0]
        if event_start.value != 0:
            raise OLMoConfigurationError("batch size schedule must start at 0")

        for next_batch_size, next_event_start in self.schedule[1:]:
            if next_event_start.value <= event_start.value:
                raise OLMoConfigurationError(
                    "subsequent events in the batch size schedule must be configured to occur after previous events"
                )

            if next_batch_size > batch_size:
                if next_batch_size % batch_size != 0:
                    raise OLMoConfigurationError(
                        "invalid batch size schedule, batch size can only be increased by an integer factor"
                    )
            elif next_batch_size < batch_size:
                if batch_size % next_batch_size != 0:
                    raise OLMoConfigurationError(
                        "invalid batch size schedule, batch size can only be decreased by an integer factor"
                    )
            else:
                raise OLMoConfigurationError(
                    "invalid batch size schedule, duplicate batch size in schedule"
                )

            batch_size = next_batch_size
            event_start = next_event_start

    @property
    def current_batch_size(self) -> int:
        return self.trainer.data_loader.global_batch_size

    def post_attach(self):
        self._maybe_update_batch_size_and_lr()

    def post_checkpoint_loaded(self):
        self._maybe_update_batch_size_and_lr()

    def pre_load_batch(self):
        self._maybe_update_batch_size_and_lr()

    def _maybe_update_batch_size_and_lr(self):
        # Find latest event in the schedule to apply.
        for target_batch_size, event_start in reversed(self.schedule):
            if event_start.due(
                step=self.trainer.global_step,
                tokens=self.trainer.global_train_tokens_seen,
                epoch=self.trainer.epoch,
            ):
                if target_batch_size > self.current_batch_size:
                    assert target_batch_size % self.current_batch_size == 0
                    # When increasing the batch size, need to wait until 'batches_processed' is divisible
                    # by the factor that batch size is being increased by.
                    ratio = target_batch_size // self.current_batch_size
                    if self.trainer.data_loader.batches_processed % ratio != 0:
                        self._update_batch_size_and_lr(target_batch_size)
                elif target_batch_size < self.current_batch_size:
                    assert self.current_batch_size % target_batch_size == 0
                    self._update_batch_size_and_lr(target_batch_size)

                break

    def _update_batch_size_and_lr(self, batch_size: int):
        log.info(f"Changing global batch size to {batch_size:,d}...")
        ratio = batch_size / self.current_batch_size
        lr_adjustment_factor = math.sqrt(ratio)
        self.trainer.data_loader.global_batch_size = batch_size

        optimizers: Optional[List[torch.optim.Optimizer]] = None
        scheduler: Optional[Scheduler] = None
        if isinstance(self.trainer.train_module, TransformerTrainModule):
            optimizers = [self.trainer.train_module.optim]
            scheduler = self.trainer.train_module.scheduler
        elif isinstance(self.trainer.train_module, TransformerPipelineTrainModule):
            optimizers = self.trainer.train_module.optimizers
            scheduler = self.trainer.train_module.scheduler

        if not optimizers:
            log.warning(
                f"Unable to adjust learning rate for {self.trainer.train_module.__class__.__name__} train module class"
            )
            return

        for optim in optimizers:
            if not isinstance(optim, (torch.optim.Adam, torch.optim.AdamW, SkipStepAdamW)):
                log.warning(
                    f"Unable to adjust learning rate for {optim.__class__.__name__} optimizer"
                )
                return

        log.info(
            f"Adjusting base learning rate by a factor of {lr_adjustment_factor:.4f} = sqrt({ratio})"
        )
        for optim_idx, optim in enumerate(optimizers):
            for group_idx, group in enumerate(optim.param_groups):
                new_lr: Union[float, torch.Tensor]
                if scheduler is not None:
                    if group.get(scheduler.initial_lr_field) is None:
                        group[scheduler.initial_lr_field] = group[scheduler.lr_field]
                    group[scheduler.initial_lr_field] *= lr_adjustment_factor
                    new_lr = group[scheduler.initial_lr_field]
                else:
                    group["lr"] *= lr_adjustment_factor
                    new_lr = group["lr"]
                log.info(
                    f"Set base LR for optimizer {optim_idx+1}, group {group_idx+1} to {float(new_lr):.8f}"
                )
