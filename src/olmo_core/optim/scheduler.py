from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from math import cos, pi
from typing import Optional


@dataclass
class Scheduler(metaclass=ABCMeta):
    """
    Learning rate scheduler base class.
    """

    lr_field: str = "lr"
    initial_lr_field: str = "initial_lr"

    @abstractmethod
    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
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

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
        del step, max_steps
        return initial_lr


@dataclass
class CosWithWarmup(Scheduler):
    """
    Cosine learning rate schedule with a warmup.
    """

    warmup_steps: int = 2000
    alpha_f: float = 0.1
    t_max: Optional[int] = None
    warmup_min_lr: float = 0.0

    def get_lr(self, initial_lr: float, step: int, max_steps: int) -> float:
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
    initial_lr: float, step: int, warmup_steps: int, warmup_min_lr: float = 0.0
) -> float:
    assert 0 <= warmup_min_lr < initial_lr
    return warmup_min_lr + (initial_lr - warmup_min_lr) * min(step, warmup_steps) / warmup_steps
