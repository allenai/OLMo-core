import logging
import warnings
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from math import cos, pi, sqrt
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

from ..config import StrEnum
from ..exceptions import OLMoConfigurationError

if TYPE_CHECKING:
    from olmo_core.train import Trainer

log = logging.getLogger(__name__)


class SchedulerUnits(StrEnum):
    steps = "steps"
    tokens = "tokens"


@dataclass
class Scheduler(metaclass=ABCMeta):
    """
    Learning rate scheduler base class.
    """

    lr_field: str = "lr"
    initial_lr_field: str = "initial_lr"
    units: SchedulerUnits = SchedulerUnits.steps

    @abstractmethod
    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        """
        Get the learning rate for a step given the initial/max learning rate and the maximum
        number of steps.
        """
        raise NotImplementedError

    def set_lr(self, group: Dict[str, Any], trainer: "Trainer") -> Union[float, torch.Tensor]:
        """
        Set the learning rate on an optimizer param group given a trainer's state.
        """
        if (lr_field := self.lr_field) not in group and (
            initial_lr_field := self.initial_lr_field
        ) not in group:
            group_fields_list = "\n - ".join(
                [f"{k}: {v}" for k, v in group.items() if k != "params"]
            )
            raise RuntimeError(
                f"learning rate field '{lr_field}' and initial learning rate field "
                f"'{initial_lr_field}' not found in optimizer param group "
                f"with {len(group['params'])} parameter(s):\n"
                f" - {group_fields_list}"
            )

        # Ensure 'initial_lr' is set.
        if group.get(self.initial_lr_field) is None:
            group[self.initial_lr_field] = group["lr"]

        # Set new LR.
        if self.units == SchedulerUnits.steps:
            new_lr = self.get_lr(
                group[self.initial_lr_field],
                trainer.global_step,
                trainer.max_steps,
            )
        elif self.units == SchedulerUnits.tokens:
            new_lr = self.get_lr(
                group[self.initial_lr_field],
                trainer.global_train_tokens_seen,
                trainer.max_tokens,
            )
        else:
            raise NotImplementedError(self.units)

        if isinstance(current_lr := group.get(self.lr_field), torch.Tensor):
            current_lr.fill_(new_lr)
        else:
            group[self.lr_field] = new_lr

        return new_lr


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

    warmup: Optional[int] = None
    warmup_steps: Optional[int] = None  # deprecated, use 'warmup' instead.
    warmup_fraction: Optional[float] = None
    warmup_min_lr: float = 0.0

    def __post_init__(self):
        if self.warmup is None and self.warmup_steps is not None:
            self.warmup = self.warmup_steps
            self.warmup_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.warmup_steps' is deprecated, please use '.warmup' instead.",
                DeprecationWarning,
            )

        if (self.warmup_fraction is None) == (self.warmup is None):
            raise OLMoConfigurationError("Either 'warmup_fraction' or 'warmup' must be specified.")

        if self.warmup_fraction is not None and (
            self.warmup_fraction < 0 or self.warmup_fraction > 1
        ):
            raise OLMoConfigurationError("'warmup_fraction' must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(max_steps * self.warmup_fraction)
        else:
            warmup = self.warmup

        if step <= warmup:
            return _linear_warmup(initial_lr, step, warmup, self.warmup_min_lr)

        del step, max_steps
        return initial_lr


@dataclass
class WSD(Scheduler):
    """
    Warmup-stable-decay scheduler
    """

    warmup: Optional[int] = None
    warmup_steps: Optional[int] = None  # deprecated, use 'warmup' instead.
    warmup_fraction: Optional[float] = None
    decay: Optional[int] = None
    decay_steps: Optional[int] = None  # deprecated, use 'decay' instead.
    decay_fraction: Optional[float] = 0.1
    warmup_min_lr: float = 0.0
    decay_min_lr: float = 0.0

    def __post_init__(self):
        if self.warmup is None and self.warmup_steps is not None:
            self.warmup = self.warmup_steps
            self.warmup_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.warmup_steps' is deprecated, please use '.warmup' instead.",
                DeprecationWarning,
            )

        if (self.warmup_fraction is None) == (self.warmup is None):
            raise OLMoConfigurationError("Either 'warmup_fraction' or 'warmup' must be specified.")

        if self.warmup_fraction is not None and (
            self.warmup_fraction < 0 or self.warmup_fraction > 1
        ):
            raise OLMoConfigurationError("warmup_fraction must be between 0 and 1.")

        if self.decay is None and self.decay_steps is not None:
            self.decay = self.decay_steps
            self.decay_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.decay_steps' is deprecated, please use '.decay' instead.",
                DeprecationWarning,
            )

        if (self.decay_fraction is None) == (self.decay is None):
            raise OLMoConfigurationError("Either 'decay_fraction' or 'decay' must be specified.")

        if self.decay_fraction is not None and (self.decay_fraction < 0 or self.decay_fraction > 1):
            raise OLMoConfigurationError("decay_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(max_steps * self.warmup_fraction)
        else:
            warmup = self.warmup

        if step <= warmup:
            return _linear_warmup(initial_lr, step, warmup, self.warmup_min_lr)

        if self.decay is None:
            assert self.decay_fraction is not None
            decay = round(max_steps * self.decay_fraction)
        else:
            decay = self.decay

        if step >= max_steps - decay:
            return _linear_decay(initial_lr, max_steps - step, decay, self.decay_min_lr)

        del step, max_steps
        return initial_lr


@dataclass
class LinearWithWarmup(Scheduler):
    """
    Linear learning rate schedule with a warmup.
    """

    alpha_f: float = 0.1
    t_max: Optional[int] = None
    warmup: Optional[int] = None
    warmup_steps: Optional[int] = None  # deprecated, use 'warmup' instead.
    warmup_fraction: Optional[float] = None
    warmup_min_lr: float = 0.0

    def __post_init__(self):
        if self.warmup is None and self.warmup_steps is not None:
            self.warmup = self.warmup_steps
            self.warmup_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.warmup_steps' is deprecated, please use '.warmup' instead.",
                DeprecationWarning,
            )

        if (self.warmup_fraction is None) == (self.warmup is None):
            raise OLMoConfigurationError("Either 'warmup_fraction' or 'warmup' must be specified.")

        if self.warmup_fraction is not None and (
            self.warmup_fraction < 0 or self.warmup_fraction > 1
        ):
            raise OLMoConfigurationError("warmup_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f

        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(max_steps * self.warmup_fraction)
        else:
            warmup = self.warmup

        if step < warmup:
            return _linear_warmup(initial_lr, step, warmup, self.warmup_min_lr)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - warmup
            max_steps = max_steps - warmup
            return initial_lr - (initial_lr - eta_min) * (step / max_steps)


@dataclass
class InvSqrtWithWarmup(Scheduler):
    """
    Inverse square root learning rate (LR) schedule with a warmup.
    """

    alpha_f: float = 0.1
    warmup: Optional[int] = None
    warmup_steps: Optional[int] = None  # deprecated, use 'warmup' instead.
    warmup_fraction: Optional[float] = None
    warmup_min_lr: float = 0.0

    def __post_init__(self):
        if self.warmup is None and self.warmup_steps is not None:
            self.warmup = self.warmup_steps
            self.warmup_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.warmup_steps' is deprecated, please use '.warmup' instead.",
                DeprecationWarning,
            )

        if (self.warmup_fraction is None) == (self.warmup is None):
            raise OLMoConfigurationError("Either 'warmup_fraction' or 'warmup' must be specified.")

        if self.warmup_fraction is not None and (
            self.warmup_fraction < 0 or self.warmup_fraction > 1
        ):
            raise OLMoConfigurationError("warmup_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(max_steps * self.warmup_fraction)
        else:
            warmup = self.warmup

        if step < warmup:
            return _linear_warmup(initial_lr, step, warmup, self.warmup_min_lr)
        del max_steps

        eta_min = initial_lr * self.alpha_f
        return eta_min + (initial_lr - eta_min) * sqrt(warmup / step)


@dataclass
class CosWithWarmup(Scheduler):
    """
    Cosine learning rate schedule with a warmup.
    """

    warmup: Optional[int] = None
    warmup_steps: Optional[int] = None  # deprecated, use 'warmup' instead.
    warmup_fraction: Optional[float] = None
    alpha_f: float = 0.1
    t_max: Optional[int] = None
    warmup_min_lr: float = 0.0

    def __post_init__(self):
        if self.warmup is None and self.warmup_steps is not None:
            self.warmup = self.warmup_steps
            self.warmup_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.warmup_steps' is deprecated, please use '.warmup' instead.",
                DeprecationWarning,
            )

        if (self.warmup_fraction is None) == (self.warmup is None):
            raise OLMoConfigurationError("Either 'warmup_fraction' or 'warmup' must be specified.")

        if self.warmup_fraction is not None and (
            self.warmup_fraction < 0 or self.warmup_fraction > 1
        ):
            raise OLMoConfigurationError("warmup_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        max_steps = max_steps if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f

        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(max_steps * self.warmup_fraction)
        else:
            warmup = self.warmup

        if step < warmup:
            return _linear_warmup(initial_lr, step, warmup, self.warmup_min_lr)
        elif step >= max_steps:
            return eta_min
        else:
            step = step - warmup
            max_steps = max_steps - warmup
            return eta_min + (initial_lr - eta_min) * (1 + cos(pi * step / max_steps)) / 2


@dataclass
class CosWithWarmupAndLinearDecay(CosWithWarmup):
    """
    Cosine learning rate schedule with a warmup, cut short at the end and followed by a linear decay.
    """

    decay: Optional[int] = None
    decay_steps: Optional[int] = None  # deprecated, use 'decay' instead.
    decay_fraction: Optional[float] = 0.1
    decay_min_lr: float = 0.0

    def __post_init__(self):
        if self.decay is None and self.decay_steps is not None:
            self.decay = self.decay_steps
            self.decay_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.decay_steps' is deprecated, please use '.decay' instead.",
                DeprecationWarning,
            )

        if (self.decay_fraction is None) == (self.decay is None):
            raise OLMoConfigurationError("Either 'decay_fraction' or 'decay' must be specified.")

        if self.decay_fraction is not None and (self.decay_fraction < 0 or self.decay_fraction > 1):
            raise OLMoConfigurationError("'decay_fraction' must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        if self.decay is None:
            assert self.decay_fraction is not None
            decay = round(max_steps * self.decay_fraction)
        else:
            decay = self.decay

        if step >= max_steps - decay:
            final_cosine_lr = super().get_lr(initial_lr, max_steps - decay, max_steps)
            return _linear_decay(final_cosine_lr, max_steps - step, decay, self.decay_min_lr)

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
    schedulers_max: Optional[List[int]] = None
    """
    A list of the steps or token counts for which each scheduler runs.
    The last scheduler is assumed to run until the end of training, so any value provided for it is ignored.
    """
    schedulers_max_steps: Optional[List[int]] = None  # deprecated, use 'schedulers_max' instead.

    def __post_init__(self):
        if self.schedulers_max is None and self.schedulers_max_steps is not None:
            self.schedulers_max = self.schedulers_max_steps
            self.schedulers_max_steps = None
            warnings.warn(
                f"'{self.__class__.__name__}.schedulers_max_steps' is deprecated, please use '.schedulers_max' instead.",
                DeprecationWarning,
            )

        if self.schedulers_max is None:
            raise OLMoConfigurationError("'schedulers_max' must be specified")

        if len(self.schedulers_max) == len(self.schedulers):
            log.info(
                "Max steps are set for the last scheduler in sequential scheduling. "
                "The last scheduler is assumed to run until the end of training, so this value is ignored."
            )
            self.schedulers_max.pop()

        if len(self.schedulers_max) + 1 != len(self.schedulers):
            raise OLMoConfigurationError(
                f"Max steps must be set for all schedulers except the last when using '{self.__class__.__name__}'"
            )

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], step: int, max_steps: int
    ) -> Union[float, torch.Tensor]:
        assert 0 <= step <= max_steps
        assert self.schedulers_max is not None

        # Call schedulers sequentially until the step is within the max steps
        # of the scheduler or the last scheduler is reached
        for scheduler, scheduler_max in zip(self.schedulers[:-1], self.schedulers_max, strict=True):
            if step <= scheduler_max:
                return scheduler.get_lr(initial_lr, step, min(max_steps, scheduler_max))

            # The next scheduler's initial LR should be the final LR of the current schedule
            initial_lr = scheduler.get_lr(initial_lr, scheduler_max, scheduler_max)

            step -= scheduler_max
            max_steps -= scheduler_max

        assert max_steps > 0
        return self.schedulers[-1].get_lr(initial_lr, step, max_steps)
