import logging
import warnings
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from math import cos, pi, sqrt
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

from ..config import Config, StrEnum
from ..exceptions import OLMoConfigurationError
from .config import INITIAL_LR_FIELD, LR_FIELD

if TYPE_CHECKING:
    from olmo_core.train import Trainer

log = logging.getLogger(__name__)


class SchedulerUnits(StrEnum):
    steps = "steps"
    tokens = "tokens"


@dataclass
class Scheduler(Config, metaclass=ABCMeta):
    """
    Learning rate scheduler base class.
    """

    lr_field: str = LR_FIELD
    initial_lr_field: str = INITIAL_LR_FIELD
    units: SchedulerUnits = SchedulerUnits.steps

    @abstractmethod
    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        """
        Get the learning rate given the initial/max learning rate, current step/token count, and the maximum
        number of steps/tokens.
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
            group[self.initial_lr_field] = group[self.lr_field]

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
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        del current, t_max
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
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(t_max * self.warmup_fraction)
        else:
            warmup = self.warmup

        if current <= warmup:
            return _linear_warmup(initial_lr, current, warmup, self.warmup_min_lr)

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
            raise OLMoConfigurationError(
                "Either 'decay_fraction' or 'decay' must be specified. Never both."
            )

        if self.decay_fraction is not None and (self.decay_fraction < 0 or self.decay_fraction > 1):
            raise OLMoConfigurationError("decay_fraction must be between 0 and 1.")

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(t_max * self.warmup_fraction)
        else:
            warmup = self.warmup

        if current <= warmup:
            return _linear_warmup(initial_lr, current, warmup, self.warmup_min_lr)

        if self.decay is None:
            assert self.decay_fraction is not None
            decay = round(t_max * self.decay_fraction)
        else:
            decay = self.decay

        if current >= t_max - decay:
            return _linear_decay(initial_lr, t_max - current, decay, self.decay_min_lr)

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
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        t_max = t_max if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f

        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(t_max * self.warmup_fraction)
        else:
            warmup = self.warmup

        if current < warmup:
            return _linear_warmup(initial_lr, current, warmup, self.warmup_min_lr)
        elif current >= t_max:
            return eta_min
        else:
            current = current - warmup
            t_max = t_max - warmup
            return initial_lr - (initial_lr - eta_min) * (current / t_max)


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
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(t_max * self.warmup_fraction)
        else:
            warmup = self.warmup

        if current < warmup:
            return _linear_warmup(initial_lr, current, warmup, self.warmup_min_lr)

        eta_min = initial_lr * self.alpha_f
        return eta_min + (initial_lr - eta_min) * sqrt(warmup / current)


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
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        t_max = t_max if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f

        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(t_max * self.warmup_fraction)
        else:
            warmup = self.warmup

        if current < warmup:
            return _linear_warmup(initial_lr, current, warmup, self.warmup_min_lr)
        elif current >= t_max:
            return eta_min
        else:
            current = current - warmup
            t_max = t_max - warmup
            return eta_min + (initial_lr - eta_min) * (1 + cos(pi * current / t_max)) / 2


@dataclass
class HalfCosWithWarmup(Scheduler):
    """
    Second half of a cosine learning rate schedule, with a warmup before that.
    Note: This assumes that the peak LR set is for the full cosine schedule.
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
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        t_max = t_max if self.t_max is None else self.t_max
        eta_min = initial_lr * self.alpha_f

        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(t_max * self.warmup_fraction)
        else:
            warmup = self.warmup

        if current < warmup:
            max_lr = eta_min + (initial_lr - eta_min) / 2
            return _linear_warmup(max_lr, current, warmup, self.warmup_min_lr)
        elif current >= t_max:
            return eta_min
        else:
            current = current - warmup
            t_max = t_max - warmup
            current += t_max
            t_max *= 2
            return eta_min + (initial_lr - eta_min) * (1 + cos(pi * current / t_max)) / 2


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
        super().__post_init__()

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
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        if self.decay is None:
            assert self.decay_fraction is not None
            decay = round(t_max * self.decay_fraction)
        else:
            decay = self.decay

        if current >= t_max - decay:
            final_cosine_lr = super().get_lr(initial_lr, t_max - decay, t_max)
            return _linear_decay(final_cosine_lr, t_max - current, decay, self.decay_min_lr)

        return super().get_lr(initial_lr, current, t_max)


def _linear_warmup(
    initial_lr: Union[float, torch.Tensor], current: int, warmup: int, warmup_min_lr: float = 0.0
) -> Union[float, torch.Tensor]:
    if isinstance(initial_lr, float):  # not worth the potential host-device sync if it's a tensor
        assert 0 <= warmup_min_lr < initial_lr
    return warmup_min_lr + (initial_lr - warmup_min_lr) * min(current, warmup) / warmup


def _linear_decay(
    initial_lr: Union[float, torch.Tensor],
    step_from_end: int,
    decay: int,
    decay_min_lr: float = 0.0,
) -> Union[float, torch.Tensor]:
    if isinstance(initial_lr, float):  # not worth the potential host-device sync if it's a tensor
        assert 0 <= decay_min_lr < initial_lr

    return decay_min_lr + (initial_lr - decay_min_lr) * min(step_from_end, decay) / decay


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
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        assert 0 <= current <= t_max
        assert self.schedulers_max is not None

        # Call schedulers sequentially until the current step/token count is within the max steps/token count
        # of the scheduler or the last scheduler is reached.
        for scheduler, scheduler_max in zip(self.schedulers[:-1], self.schedulers_max, strict=True):
            if current <= scheduler_max:
                return scheduler.get_lr(initial_lr, current, min(t_max, scheduler_max))

            # The next scheduler's initial LR should be the final LR of the current schedule
            initial_lr = scheduler.get_lr(initial_lr, scheduler_max, scheduler_max)

            current -= scheduler_max
            t_max -= scheduler_max

        assert t_max > 0
        return self.schedulers[-1].get_lr(initial_lr, current, t_max)


def _invprop_decay(
    eta_max: Union[float, torch.Tensor],
    eta_min: float,
    t: int,
    T: int,
) -> Union[float, torch.Tensor]:
    """
    Inverse-proportional decay where 1/lr interpolates linearly.
    
    Formula: 1/lr(t) = (t/T) * (1/eta_min) + (1 - t/T) * (1/eta_max)
    """
    if T == 0:
        return eta_min if eta_min > 0 else 0.0
    
    if t >= T:
        return eta_min if eta_min > 0 else 0.0
    
    if t == 0:
        return eta_max
    
    # For eta_min <= 0, use a very small positive value to approximate zero
    # This prevents the formula from collapsing to zero immediately
    effective_eta_min = max(eta_min, eta_max * 0.1)
    
    progress = t / T
    inv_lr = progress / effective_eta_min + (1 - progress) / eta_max
    
    return 1 / inv_lr


@dataclass
class WSDS(Scheduler):
    """
    Warmup–Stable–Decay—Simplified (WSD‑S) scheduler for continual pretraining.

    Design:
      • A single warmup at the very beginning (period 0).
      • Multiple periods, each with a long STABLE phase followed by a short DECAY phase.
      • Period 0 = warmup + stable + decay. Periods 1..N = stable + decay.
      • At the boundary between periods, LR is *reset* to eta_max (the 'initial_lr' for the group).
      • End of each period: LR is set to EXACTLY 0.0 (for eta_min <= 0), which is ideal for checkpointing.
      • 'period_lengths' defines each period's total length in chosen 'units' (steps or tokens).

    Phases per period i (length L_i):
      - if i == 0:
          warmup length W = self.warmup or round(self.warmup_fraction * L_0)
          decay  length D = self.decay  or round(self.decay_fraction  * L_i)
          stable length   = L_0 - W - D
        else:
          warmup length W = 0
          decay  length D = self.decay or round(self.decay_fraction * L_i)
          stable length   = L_i - D

    Notes:
      - If warmup + decay > L_0, they are proportionally shrunk to fit.
      - If decay > L_i for any period, it's clamped to L_i.
      - For eta_min <= 0, the inverse-proportional decay returns 0 at t = T and
        stays finite (using epsilon) for t < T.

    Attributes
    ----------
    period_lengths : List[int]
        Length of EACH period in chosen 'units' (steps or tokens). Must be >0.
    warmup : Optional[int]
        Warmup length for period 0 in absolute 'units'. Exactly one of (warmup, warmup_fraction) must be set.
    warmup_fraction : Optional[float]
        Warmup fraction of period 0 (0..1). Mutually exclusive with 'warmup'.
    decay : Optional[int]
        Decay length per period (absolute). Exactly one of (decay, decay_fraction) must be set.
    decay_fraction : Optional[float]
        Decay fraction per period (0..1). Mutually exclusive with 'decay'.
    warmup_min_lr : float
        LR at start of warmup.
    decay_min_lr : float
        LR at end of decay. Can be 0.0 (recommended for "checkpoint when LR=0").
    """

    # --- configuration ---
    period_lengths: List[int] = field(default_factory=list)

    warmup: Optional[int] = None
    warmup_fraction: Optional[float] = None

    decay: Optional[int] = None
    decay_fraction: Optional[float] = None

    warmup_min_lr: float = 0.0
    decay_min_lr: float = 0.0

    # --- cached derived state ---
    _cum_period_end: List[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        # Units validated by base class.
        if not self.period_lengths:
            raise OLMoConfigurationError("'period_lengths' must be provided and non-empty.")
        if any(p <= 0 for p in self.period_lengths):
            raise OLMoConfigurationError("All entries in 'period_lengths' must be > 0.")

        # warmup validation (applies ONLY to the first period)
        if (self.warmup is None) == (self.warmup_fraction is None):
            raise OLMoConfigurationError("Exactly one of 'warmup' or 'warmup_fraction' must be specified.")
        if self.warmup_fraction is not None and not (0.0 <= self.warmup_fraction <= 1.0):
            raise OLMoConfigurationError("'warmup_fraction' must be in [0, 1].")

        # decay validation (applies to ALL periods)
        if (self.decay is None) == (self.decay_fraction is None):
            raise OLMoConfigurationError("Exactly one of 'decay' or 'decay_fraction' must be specified.")
        if self.decay_fraction is not None and not (0.0 <= self.decay_fraction <= 1.0):
            raise OLMoConfigurationError("'decay_fraction' must be in [0, 1].")

        # Precompute cumulative ends so we can detect period boundaries quickly.
        self._cum_period_end = []
        s = 0
        for L in self.period_lengths:
            s += int(L)
            self._cum_period_end.append(s)

        # Validate warmup+decay for the first period.
        L0 = self.period_lengths[0]
        W = self._resolve_warmup(L0)
        D0 = self._resolve_decay(L0)
        if W + D0 > L0:
            # Proportionally shrink (keep ratio W:D0) to fit in L0.
            total = W + D0
            if total == 0:
                # degenerate; make everything stable
                pass
            else:
                W = round(W * L0 / total)
                D0 = L0 - W
            # Store the adjusted warmup/decay values into fractions so that we recompute consistently later
            if self.warmup is not None:
                self.warmup = W
            else:
                self.warmup_fraction = W / L0
            if self.decay is not None:
                self.decay = D0
            else:
                self.decay_fraction = D0 / L0

    def _resolve_warmup(self, L0: int) -> int:
        if self.warmup is not None:
            return min(max(int(self.warmup), 0), int(L0))
        else:
            return min(max(int(round(self.warmup_fraction * L0)), 0), int(L0))

    def _resolve_decay(self, Li: int) -> int:
        if self.decay is not None:
            return min(max(int(self.decay), 0), int(Li))
        else:
            return min(max(int(round(self.decay_fraction * Li)), 0), int(Li))

    def _find_period(self, x: int) -> int:
        """
        Return the 0-based period index such that:
            cumulative_end[period-1] < x <= cumulative_end[period]
        If x <= period_lengths[0], this returns 0.
        If x > sum(period_lengths), returns len(period_lengths)-1, and we will clamp positions.
        """
        for idx, end in enumerate(self._cum_period_end):
            if x <= end:
                return idx
        return len(self._cum_period_end) - 1

    def period_boundaries(self) -> List[int]:
        """
        Inclusive boundaries in the same units as 'period_lengths', i.e., the set of x
        where LR is exactly 0.0 (end of period).
            boundary_k = sum_{i=0..k} period_lengths[i]
        """
        return list(self._cum_period_end)

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        """
        Return the LR at 'current' (in 'units').
        Contract:
          • At *end of each period* (current == boundary), LR == 0 if decay_min_lr <= 0 (exact).
          • Right after a boundary (start of next period), LR resets to initial_lr (stable phase).
        """
        # If we've exceeded all periods, stay at 0 (if eta_min <= 0) or at eta_min otherwise.
        total = self._cum_period_end[-1]
        if current >= total:
            return 0.0 if self.decay_min_lr <= 0.0 else self.decay_min_lr

        # Determine period and position; clamp 'pos_in_period' into [0, L_i] (inclusive)
        pidx = self._find_period(current)
        start = 0 if pidx == 0 else self._cum_period_end[pidx - 1]
        Li = self.period_lengths[pidx]
        # inclusive position; allow pos = Li to evaluate exact endpoint/lr=0
        pos = min(max(current - start, 0), Li)

        # Determine W (warmup only for p==0) and D (decay for all)
        if pidx == 0:
            W = self._resolve_warmup(Li)
            D = self._resolve_decay(Li)
            S = max(Li - W - D, 0)
            # Phases: [0, W) warmup, [W, W+S) stable, [W+S, W+S+D] decay (inclusive end)
            if pos < W:
                # warmup from warmup_min_lr -> initial_lr
                return _linear_warmup(initial_lr, pos, W, self.warmup_min_lr)
            elif pos < W + S:
                return initial_lr
            else:
                t = pos - (W + S)
                T = max(D, 0)
                return _invprop_decay(initial_lr, self.decay_min_lr, t, T)
        else:
            W = 0
            D = self._resolve_decay(Li)
            S = max(Li - D, 0)
            # Phases: [0, S) stable, [S, S+D] decay (inclusive end)
            if pos < S:
                # reset to peak LR at start of each period
                return initial_lr
            else:
                t = pos - S
                T = max(D, 0)
                return _invprop_decay(initial_lr, self.decay_min_lr, t, T)

