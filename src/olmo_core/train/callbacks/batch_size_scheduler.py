import dataclasses
import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.optim import INITIAL_LR_FIELD, LR_FIELD, SkipStepAdamW
from olmo_core.optim.scheduler import WSD, ConstantScheduler, Scheduler

from ..common import Duration
from ..train_module import TransformerPipelineTrainModule, TransformerTrainModule
from .callback import Callback
from .speed_monitor import SpeedMonitorCallback

log = logging.getLogger(__name__)


@dataclass
class BatchSizeSchedulerCallback(Callback):
    """
    A callback for setting a batch size scheduler over the course of a training run.
    Also adjusts the base learning rate with Adam optimizers for transformer train modules by a factor of
    ``sqrt(new_batch_size / current_batch_size)``.
    """

    batch_sizes: List[int] = dataclasses.field(default_factory=list)
    """
    Defines the batch sizes to apply, in order.
    """

    schedule: List[Duration] = dataclasses.field(default_factory=list)
    """
    Defines the schedule at which to apply each batch size.
    """

    def __post_init__(self):
        if len(self.batch_sizes) != len(self.schedule):
            raise OLMoConfigurationError(
                "batch_sizes and schedules should have the same number of items"
            )

        if not self.schedule:
            return

        if len({duration.unit for duration in self.schedule}) > 1:
            raise OLMoConfigurationError(
                "batch size scheduler must use consistent units for all points in the schedule"
            )

        batch_size, event_start = self.batch_sizes[0], self.schedule[0]
        if event_start.value != 0:
            raise OLMoConfigurationError("batch size schedule must start at 0")

        for next_batch_size, next_event_start in zip(self.batch_sizes[1:], self.schedule[1:]):
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
        if not self.schedule:
            return

        scheduler: Optional[Scheduler] = None
        if isinstance(self.trainer.train_module, TransformerTrainModule):
            scheduler = self.trainer.train_module.scheduler
        elif isinstance(self.trainer.train_module, TransformerPipelineTrainModule):
            scheduler = self.trainer.train_module.scheduler

        # If we have an LR scheduler, we need to make sure that the value it uses for `t_max`
        # (the end point of the schedule) won't be changed by this callback, since that will
        # lead to unexpected behavior in the schedule.
        # For example, if the scheduler doesn't have its own `self.t_max` set and its unit are in
        # steps, then it takes `t_max` to be `trainer.max_steps`, which will change when this callback
        # updates the batch size (unless the trainer's max duration is set in steps).
        if (
            scheduler is not None
            and not isinstance(scheduler, (ConstantScheduler, WSD))
            and getattr(scheduler, "t_max", None) is None
            and self.trainer.max_duration.unit != "steps"
            and scheduler.units != "tokens"
        ):
            raise OLMoConfigurationError(
                f"Batch size scheduler requires the {scheduler.__class__.__name__} LR scheduler's "
                "units to be set to 'tokens' unless 't_max' is set."
            )

        self._maybe_update_batch_size_and_lr()

    def post_checkpoint_loaded(self, *args):
        del args
        # NOTE: we set the "initial_lr_field" correctly after loading a checkpoint because
        # the "initial_lr_field" in the optimizer state always gets reset to the value from
        # the config, not the checkpoint.
        self._maybe_update_batch_size_and_lr()

    def pre_load_batch(self):
        self._maybe_update_batch_size_and_lr()

    def post_step(self):
        self.trainer.record_metric("train/global batch size", self.current_batch_size)

    def _maybe_update_batch_size_and_lr(self):
        # Find latest event in the schedule to apply.
        for target_batch_size, event_start in reversed(list(zip(self.batch_sizes, self.schedule))):
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
                    if self.trainer.data_loader.batches_processed % ratio == 0:
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
            raise NotImplementedError(
                f"Unable to adjust learning rate for {self.trainer.train_module.__class__.__name__} train module class"
            )

        for optim in optimizers:
            if not isinstance(optim, (torch.optim.Adam, torch.optim.AdamW, SkipStepAdamW)):
                raise NotImplementedError(
                    f"Unable to adjust learning rate for {optim.__class__.__name__} optimizer"
                )

        log.info(
            f"Adjusting base learning rate by a factor of {lr_adjustment_factor:.4f} = sqrt({ratio})"
        )
        for optim_idx, optim in enumerate(optimizers):
            for group_idx, group in enumerate(optim.param_groups):
                lr_field = LR_FIELD if scheduler is None else scheduler.lr_field
                initial_lr_field = (
                    INITIAL_LR_FIELD if scheduler is None else scheduler.initial_lr_field
                )

                if group.get(initial_lr_field) is None:
                    group[initial_lr_field] = group[lr_field]

                group[initial_lr_field] *= lr_adjustment_factor

                # Only set the actual LR if there's no scheduler. Schedulers update the LR based
                # on the initial LR.
                if scheduler is None:
                    group[lr_field] = group[initial_lr_field]

                log.info(
                    f"Set base LR for optimizer {optim_idx+1}, group {group_idx+1} to {float(group[initial_lr_field]):.8f}"
                )

        for callback in self.trainer.callbacks.values():
            if isinstance(callback, SpeedMonitorCallback):
                log.info("Resetting speed monitor...")
                callback.reset()
