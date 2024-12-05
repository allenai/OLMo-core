import logging
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from math import cos, pi, sqrt
from typing import List, Optional, Union

import torch

from ..exceptions import OLMoConfigurationError

log = logging.getLogger(__name__)


@dataclass
class Scheduler(metaclass=ABCMeta):
    """
    Learning rate scheduler base class.
    """

    lr_field: str = "lr"
    initial_lr_field: str = "initial_lr"

    @abstractmethod
    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        """
        Get the learning rate for a step given the initial/max learning rate and the maximum
        number of steps.
        """
        raise NotImplementedError


@dataclass
class ConstantScheduler(Scheduler):
    """
    Constant learning rate schedule, basically a no-op.
    """

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        del step, max_steps
        return initial_lr


@dataclass
class LinearWarmupDecoratorScheduler(Scheduler):
    """
    A scheduler that can wrap other schedulers to add a linear warmup phase.
    """

    inner: Scheduler = field(default_factory=ConstantScheduler)
    warmup_steps: int = 2000
    warmup_min_lr: float = 0.0

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        if step <= self.warmup_steps:
            lr_at_intercept = self.inner.get_lr(initial_lr, 1, max_steps - self.warmup_steps)
            return self.warmup_min_lr + (lr_at_intercept - self.warmup_min_lr) * (step / self.warmup_steps)
        else:
            return self.inner.get_lr(initial_lr, step - self.warmup_steps, max_steps - self.warmup_steps)


@dataclass
class LinearScheduler(Scheduler):
    """
    Linear learning rate schedule.
    """

    alpha_f: float = 0.1
    t_max: Optional[int] = None

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f

        if step >= max_steps:
            return eta_min
        else:
            return initial_lr - (initial_lr - eta_min) * (step / max_steps)


@dataclass
class InvSqrtScheduler(Scheduler):
    """
    Inverse square root learning rate (LR) schedule.

    To enable having a customizable LR warmup (or no warmup), this schedule removes
    the LR warmup that is part of traditional formulations of the inverse square
    root scheduler. Instead, it supports skipping the beginning of the inverse square
    root using `step_offset`.
    """

    alpha_f: float = 0.1
    step_offset: int = 1
    """
    Step provided is increased by the given offset for inverse square root calculations.
    Largers offsets give less steep LR decay, since `get_lr` is normalized to
    give the initial learning rate at step 0.
    """

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        del max_steps

        eta_min = initial_lr * self.alpha_f
        return eta_min + (initial_lr - eta_min) * sqrt(self.step_offset / (step + self.step_offset))


@dataclass
class SequentialScheduler(Scheduler):
    """
    A scheduler that calls a sequence of schedulers sequentially during the optimization process.
    """

    schedulers: List[Scheduler] = field(default_factory=lambda: [ConstantScheduler()])
    schedulers_max_steps: List[int] = field(default_factory=list)
    """
    A list of the steps for which each scheduler runs. The last scheduler is assumed to run until the
    end of training, so any value provided for it is ignored.
    """

    def __post_init__(self):
        if len(self.schedulers_max_steps) == len(self.schedulers):
            log.info(
                "Max steps are set for the last scheduler in sequential scheduling. "
                "The last scheduler is assumed to run until the end of training, so this value is ignored."
            )
            self.schedulers_max_steps.pop()

        if len(self.schedulers_max_steps) + 1 != len(self.schedulers):
            raise OLMoConfigurationError(
                f"Max steps must be set for all schedulers except the last when using {SequentialScheduler.__name__}"
            )

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        assert 0 <= step <= max_steps

        # Call schedulers sequentially until the step is within the max steps
        # of the scheduler or the last scheduler is reached
        for scheduler, scheduler_max_steps in zip(self.schedulers[:-1], self.schedulers_max_steps, strict=True):
            if step <= scheduler_max_steps:
                return scheduler.get_lr(initial_lr, step, min(max_steps, scheduler_max_steps))

            # The next scheduler's initial LR should be the final LR of the current schedule
            initial_lr = scheduler.get_lr(initial_lr, scheduler_max_steps, scheduler_max_steps)

            step -= scheduler_max_steps
            max_steps -= scheduler_max_steps

        assert max_steps > 0
        return self.schedulers[-1].get_lr(initial_lr, step, max_steps)


@dataclass
class CosWithWarmup(Scheduler):
    """
    Cosine learning rate schedule with a warmup.
    """

    warmup_steps: int = 2000
    alpha_f: float = 0.1
    t_max: Optional[int] = None
    warmup_min_lr: float = 0.0

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f
        if step < self.warmup_steps:
            return _linear_warmup(initial_lr, step, self.warmup_steps, self.warmup_min_lr)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - self.warmup_steps
            max_steps = max_steps - self.warmup_steps
            return eta_min + (initial_lr - eta_min) * (1 + cos(pi * step / max_steps)) / 2


def _linear_warmup(
    initial_lr: Union[float, torch.Tensor], step: int, warmup_steps: int, warmup_min_lr: float = 0.0
) -> Union[float, torch.Tensor]:
    if isinstance(initial_lr, float):  # not worth the potential host-device sync if it's a tensor
        assert 0 <= warmup_min_lr < initial_lr
    return warmup_min_lr + (initial_lr - warmup_min_lr) * min(step, warmup_steps) / warmup_steps
