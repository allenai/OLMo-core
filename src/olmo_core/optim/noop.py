from dataclasses import dataclass
from typing import Optional, Type

import torch

from .config import OptimConfig
from .skip_step_optimizer import SkipStepOptimizer


class NoOpOptimizer(SkipStepOptimizer):
    """
    A no-op optimizer that performs no parameter updates but maintains all step skipping logic.

    This optimizer is useful for gathering statistics from training without actually modifying
    the model parameters. It tracks losses and gradient norms, computes step factors based on
    rolling statistics, but does not apply any updates to the model.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        rolling_interval_length: int = 128,
        sigma_factor: int = 6,
    ) -> None:
        defaults = dict(lr=lr)
        super().__init__(
            params,
            defaults,
            rolling_interval_length=rolling_interval_length,
            sigma_factor=sigma_factor,
        )
        self._step_skipped: Optional[torch.Tensor] = None

    @property
    def step_skipped(self) -> torch.Tensor:
        if self._step_skipped is not None:
            return self._step_skipped
        else:
            return torch.tensor(0.0)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        # Compute step factor to maintain step skipping logic
        step_factor = self.get_step_factor()
        self._step_skipped = 1 - step_factor

        # Iterate through parameters to maintain optimizer structure
        # but perform no updates
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Initialize state if needed (for consistency)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)

                # Increment step counter
                state["step"] += step_factor


@dataclass
class NoOpConfig(OptimConfig):
    """
    Configuration class for building a :class:`NoOpOptimizer`.

    This optimizer performs no parameter updates but maintains step skipping logic
    for gathering statistics during training.
    """

    lr: float = 1e-3
    """Learning rate (not used for updates, but maintained for compatibility)."""

    rolling_interval_length: int = 128
    """
    The length of the rolling interval to use for computing the mean and standard deviation
    of the loss and gradient norm.
    """

    sigma_factor: int = 6
    """
    The number of standard deviations above the mean loss/grad norm to skip a step.
    """

    @classmethod
    def optimizer(cls) -> Type[NoOpOptimizer]:
        return NoOpOptimizer
