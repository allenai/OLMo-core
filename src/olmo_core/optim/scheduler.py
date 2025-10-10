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


@dataclass
class RepeatedWSD(Scheduler):
    """
    Repeated Warmup-Stable-Decay (WSD) scheduler for scaling law experiments.
    
    This scheduler repeats a WSD pattern every `cycle_length` steps/tokens, ensuring that
    each cycle ends with the learning rate decayed to its minimum value. This is designed
    for ladder/scaling law runs where checkpoints are saved at the end of each cycle
    when LR is at its minimum.
    
    Each cycle consists of:
    1. Warmup phase: LR increases from warmup_min_lr to initial_lr
    2. Stable phase: LR remains at initial_lr
    3. Decay phase: LR decreases from initial_lr to decay_min_lr
    
    The cycle ends exactly when LR reaches decay_min_lr, making it ideal for checkpointing.
    """
    
    cycle_length: Optional[int] = None
    """Number of steps/tokens for each WSD cycle. Checkpoint is saved at the end of each cycle."""
    
    warmup: Optional[int] = None
    warmup_steps: Optional[int] = None  # deprecated, use 'warmup' instead.
    warmup_fraction: Optional[float] = None
    """Fraction of each cycle used for warmup, or absolute number of steps/tokens."""
    
    decay: Optional[int] = None
    decay_steps: Optional[int] = None  # deprecated, use 'decay' instead.
    decay_fraction: Optional[float] = 0.1
    """Fraction of each cycle used for decay, or absolute number of steps/tokens."""
    
    warmup_min_lr: float = 0.0
    """Minimum LR at the start of each warmup phase."""
    
    decay_min_lr: float = 0.0
    """Minimum LR at the end of each decay phase (checkpoint point)."""
    
    restart_warmup_from_min: bool = True
    """If True, each cycle starts warmup from warmup_min_lr. If False, starts from decay_min_lr of previous cycle."""

    def __post_init__(self):
        # Check that cycle_length was provided
        if self.cycle_length is None:
            raise OLMoConfigurationError("'cycle_length' must be specified.")

        if self.cycle_length <= 0:
            raise OLMoConfigurationError("'cycle_length' must be positive.")
            
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
            raise OLMoConfigurationError("'decay_fraction' must be between 0 and 1.")
            
        # Validate that warmup + decay don't exceed cycle length
        if self.warmup is not None and self.decay is not None:
            if self.warmup + self.decay > self.cycle_length:
                raise OLMoConfigurationError(
                    f"warmup ({self.warmup}) + decay ({self.decay}) exceeds cycle_length ({self.cycle_length})"
                )
        elif self.warmup_fraction is not None and self.decay_fraction is not None:
            if self.warmup_fraction + self.decay_fraction > 1.0:
                raise OLMoConfigurationError(
                    f"warmup_fraction ({self.warmup_fraction}) + decay_fraction ({self.decay_fraction}) exceeds 1.0"
                )

    def get_lr(
        self, initial_lr: Union[float, torch.Tensor], current: int, t_max: int
    ) -> Union[float, torch.Tensor]:
        """
        Get learning rate for the current step within repeated WSD cycles.
        
        Each cycle of length `cycle_length` follows a WSD pattern, with the cycle
        ending exactly when LR reaches decay_min_lr (ideal for checkpointing).
        """
        # Determine which cycle we're in and position within that cycle
        cycle_idx = current // self.cycle_length
        position_in_cycle = current % self.cycle_length
        
        # Calculate warmup and decay lengths for this cycle
        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(self.cycle_length * self.warmup_fraction)
        else:
            warmup = min(self.warmup, self.cycle_length)
            
        if self.decay is None:
            assert self.decay_fraction is not None
            decay = round(self.cycle_length * self.decay_fraction)
        else:
            decay = min(self.decay, self.cycle_length)
            
        # Ensure warmup + decay don't exceed cycle length
        if warmup + decay > self.cycle_length:
            # Proportionally reduce both to fit
            total = warmup + decay
            warmup = round(warmup * self.cycle_length / total)
            decay = self.cycle_length - warmup
            
        # Calculate stable phase length
        stable = self.cycle_length - warmup - decay
        
        # Determine which phase we're in within the current cycle
        if position_in_cycle < warmup:
            # Warmup phase
            if self.restart_warmup_from_min or cycle_idx == 0:
                start_lr = self.warmup_min_lr
            else:
                start_lr = self.decay_min_lr
            return _linear_warmup(initial_lr, position_in_cycle, warmup, start_lr)
            
        elif position_in_cycle < warmup + stable:
            # Stable phase
            return initial_lr
            
        else:
            decay_position = position_in_cycle - warmup - stable
            lr_fraction = 1.0 - (decay_position / decay)
            return self.decay_min_lr + (initial_lr - self.decay_min_lr) * lr_fraction
            
    def get_cycle_info(self, current: int) -> Dict[str, Any]:
        """
        Get information about the current position within the cycle.
        Useful for logging and debugging.
        
        Returns:
            Dictionary containing:
            - cycle_idx: Current cycle number (0-indexed)
            - position_in_cycle: Position within current cycle
            - phase: Current phase ('warmup', 'stable', or 'decay')
            - is_checkpoint_step: Whether this is a checkpoint step (end of cycle)
        """
        cycle_idx = current // self.cycle_length
        position_in_cycle = current % self.cycle_length
        
        # Calculate phase boundaries
        if self.warmup is None:
            assert self.warmup_fraction is not None
            warmup = round(self.cycle_length * self.warmup_fraction)
        else:
            warmup = min(self.warmup, self.cycle_length)
            
        if self.decay is None:
            assert self.decay_fraction is not None
            decay = round(self.cycle_length * self.decay_fraction)
        else:
            decay = min(self.decay, self.cycle_length)
            
        if warmup + decay > self.cycle_length:
            total = warmup + decay
            warmup = round(warmup * self.cycle_length / total)
            decay = self.cycle_length - warmup
            
        stable = self.cycle_length - warmup - decay
        
        # Determine phase
        if position_in_cycle < warmup:
            phase = "warmup"
        elif position_in_cycle < warmup + stable:
            phase = "stable"
        else:
            phase = "decay"
            
        # Check if this is a checkpoint step (last step of cycle)
        is_checkpoint_step = (position_in_cycle == self.cycle_length - 1)
        
        return {
            "cycle_idx": cycle_idx,
            "position_in_cycle": position_in_cycle,
            "phase": phase,
            "is_checkpoint_step": is_checkpoint_step,
        }