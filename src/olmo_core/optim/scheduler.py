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
class ConstantWithWarmup(Scheduler):
    """
    Constant learning rate schedule with a warmup.
    """

    warmup_steps: Optional[int] = 2000
    warmup_fraction: Optional[float] = None
    warmup_min_lr: float = 0.0

    def __post_init__(self):
        if (self.warmup_fraction is None) == (self.warmup_steps is None):
            raise OLMoConfigurationError(
                "Either warmup_fraction or warmup_steps must be specified."
            )
        if self.warmup_fraction is not None and (
            self.warmup_fraction < 0 or self.warmup_fraction > 1
        ):
            raise OLMoConfigurationError("warmup_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        if self.warmup_steps is None:
            assert self.warmup_fraction is not None
            warmup_steps = round(max_steps * self.warmup_fraction)
        else:
            warmup_steps = self.warmup_steps

        if step <= warmup_steps:
            return _linear_warmup(initial_lr, step, warmup_steps, self.warmup_min_lr)

        del step, max_steps
        return initial_lr


@dataclass
class WSD(Scheduler):
    """
    Warmup-stable-decay scheduler
    """

    warmup_steps: Optional[int] = 2000
    warmup_fraction: Optional[float] = None
    decay_steps: Optional[int] = None
    decay_fraction: Optional[float] = 0.1
    warmup_min_lr: float = 0.0
    decay_min_lr: float = 0.0

    def __post_init__(self):
        if (self.warmup_fraction is None) == (self.warmup_steps is None):
            raise OLMoConfigurationError(
                "Either warmup_fraction or warmup_steps must be specified."
            )
        if self.warmup_fraction is not None and (
            self.warmup_fraction < 0 or self.warmup_fraction > 1
        ):
            raise OLMoConfigurationError("warmup_fraction must be between 0 and 1.")

        if (self.decay_fraction is None) == (self.decay_steps is None):
            raise OLMoConfigurationError("Either decay_fraction or decay_steps must be specified.")
        if self.decay_fraction is not None and (self.decay_fraction < 0 or self.decay_fraction > 1):
            raise OLMoConfigurationError("decay_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        if self.warmup_steps is None:
            assert self.warmup_fraction is not None
            warmup_steps = round(max_steps * self.warmup_fraction)
        else:
            warmup_steps = self.warmup_steps

        if step <= warmup_steps:
            return _linear_warmup(initial_lr, step, warmup_steps, self.warmup_min_lr)

        if self.decay_steps is None:
            assert self.decay_fraction is not None
            decay_steps = round(max_steps * self.decay_fraction)
        else:
            decay_steps = self.decay_steps

        if step >= max_steps - decay_steps:
            return _linear_decay(initial_lr, max_steps - step, decay_steps, self.decay_min_lr)

        del step, max_steps
        return initial_lr


@dataclass
class LinearWithWarmup(Scheduler):
    """
    Linear learning rate schedule with a warmup.
    """

    alpha_f: float = 0.1
    t_max: Optional[int] = None
    warmup_steps: Optional[int] = 2000
    warmup_fraction: Optional[float] = None
    warmup_min_lr: float = 0.0

    def __post_init__(self):
        if (self.warmup_fraction is None) == (self.warmup_steps is None):
            raise OLMoConfigurationError(
                "Either warmup_fraction or warmup_steps must be specified."
            )
        if self.warmup_fraction is not None and (
            self.warmup_fraction < 0 or self.warmup_fraction > 1
        ):
            raise OLMoConfigurationError("warmup_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f

        if self.warmup_steps is None:
            assert self.warmup_fraction is not None
            warmup_steps = round(max_steps * self.warmup_fraction)
        else:
            warmup_steps = self.warmup_steps

        if step < warmup_steps:
            return _linear_warmup(initial_lr, step, warmup_steps, self.warmup_min_lr)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - warmup_steps
            max_steps = max_steps - warmup_steps
            return initial_lr - (initial_lr - eta_min) * (step / max_steps)


@dataclass
class InvSqrtWithWarmup(Scheduler):
    """
    Inverse square root learning rate (LR) schedule with a warmup.
    """

    alpha_f: float = 0.1
    warmup_steps: Optional[int] = 2000
    warmup_fraction: Optional[float] = None
    warmup_min_lr: float = 0.0

    def __post_init__(self):
        if (self.warmup_fraction is None) == (self.warmup_steps is None):
            raise OLMoConfigurationError(
                "Either warmup_fraction or warmup_steps must be specified."
            )
        if self.warmup_fraction is not None and (
            self.warmup_fraction < 0 or self.warmup_fraction > 1
        ):
            raise OLMoConfigurationError("warmup_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        if self.warmup_steps is None:
            assert self.warmup_fraction is not None
            warmup_steps = round(max_steps * self.warmup_fraction)
        else:
            warmup_steps = self.warmup_steps

        if step < warmup_steps:
            return _linear_warmup(initial_lr, step, warmup_steps, self.warmup_min_lr)
        del max_steps

        eta_min = initial_lr * self.alpha_f
        return eta_min + (initial_lr - eta_min) * sqrt(warmup_steps / step)


@dataclass
class CosWithWarmup(Scheduler):
    """
    Cosine learning rate schedule with a warmup.
    """

    warmup_steps: Optional[int] = 2000
    warmup_fraction: Optional[float] = None
    alpha_f: float = 0.1
    t_max: Optional[int] = None
    warmup_min_lr: float = 0.0

    def __post_init__(self):
        if (self.warmup_fraction is None) == (self.warmup_steps is None):
            raise OLMoConfigurationError(
                "Either warmup_fraction or warmup_steps must be specified."
            )
        if self.warmup_fraction is not None and (
            self.warmup_fraction < 0 or self.warmup_fraction > 1
        ):
            raise OLMoConfigurationError("warmup_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f

        if self.warmup_steps is None:
            assert self.warmup_fraction is not None
            warmup_steps = round(max_steps * self.warmup_fraction)
        else:
            warmup_steps = self.warmup_steps

        if step < warmup_steps:
            return _linear_warmup(initial_lr, step, warmup_steps, self.warmup_min_lr)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - warmup_steps
            max_steps = max_steps - warmup_steps
            return eta_min + (initial_lr - eta_min) * (1 + cos(pi * step / max_steps)) / 2


@dataclass
class CosWithWarmupAndLinearDecay(CosWithWarmup):
    """
    Cosine learning rate schedule with a warmup, cut short at the end and followed by a linear decay.
    """

    decay_steps: Optional[int] = None
    decay_fraction: Optional[float] = 0.1
    decay_min_lr: float = 0.0

    def __post_init__(self):
        if (self.decay_fraction is None) == (self.decay_steps is None):
            raise OLMoConfigurationError("Either decay_fraction or decay_steps must be specified.")
        if self.decay_fraction is not None and (self.decay_fraction < 0 or self.decay_fraction > 1):
            raise OLMoConfigurationError("decay_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        if self.decay_steps is None:
            assert self.decay_fraction is not None
            decay_steps = round(max_steps * self.decay_fraction)
        else:
            decay_steps = self.decay_steps

        if step >= max_steps - decay_steps:
            final_cosine_lr = super().get_lr(initial_lr, max_steps - decay_steps, max_steps)
            return _linear_decay(final_cosine_lr, max_steps - step, decay_steps, self.decay_min_lr)

        return super().get_lr(initial_lr, step, max_steps)


def _linear_warmup(
    initial_lr: Union[float, torch.Tensor], step: int, warmup_steps: int, warmup_min_lr: float = 0.0
) -> Union[float, torch.Tensor]:
    if isinstance(initial_lr, float):  # not worth the potential host-device sync if it's a tensor
        assert 0 <= warmup_min_lr < initial_lr
    return warmup_min_lr + (initial_lr - warmup_min_lr) * min(step, warmup_steps) / warmup_steps


def _linear_decay(
    initial_lr: Union[float, torch.Tensor],
    step_from_end: int,
    decay_steps: int,
    decay_min_lr: float = 0.0,
) -> Union[float, torch.Tensor]:
    if isinstance(initial_lr, float):  # not worth the potential host-device sync if it's a tensor
        assert 0 <= decay_min_lr < initial_lr

    return (
        decay_min_lr + (initial_lr - decay_min_lr) * min(step_from_end, decay_steps) / decay_steps
    )


@dataclass
class SequentialScheduler(Scheduler):
    """
    A scheduler that calls a sequence of schedulers sequentially during the optimization
    process. The initial LR of a scheduler in the sequence is set to the final LR of the
    previous scheduler.
    """

    schedulers: List[Scheduler] = field(default_factory=lambda: [ConstantWithWarmup()])
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
        for scheduler, scheduler_max_steps in zip(
            self.schedulers[:-1], self.schedulers_max_steps, strict=True
        ):
            if step <= scheduler_max_steps:
                return scheduler.get_lr(initial_lr, step, min(max_steps, scheduler_max_steps))

            # The next scheduler's initial LR should be the final LR of the current schedule
            initial_lr = scheduler.get_lr(initial_lr, scheduler_max_steps, scheduler_max_steps)

            step -= scheduler_max_steps
            max_steps -= scheduler_max_steps

        assert max_steps > 0
        return self.schedulers[-1].get_lr(initial_lr, step, max_steps)
