from dataclasses import dataclass
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from .config import OptimConfig
from .skip_step_optimizer import SkipStepOptimizer


def lion_step(
    p: nn.Parameter,
    *,
    lr: float,
    weight_decay: float,
    exp_avg: torch.Tensor,
    betas: Tuple[float, float],
    step_factor: torch.Tensor,
):
    if p.grad is None:
        return

    beta1, beta2 = betas

    # Perform step weight decay
    p.data.mul_(1 - step_factor * (lr * weight_decay))

    # Weight update
    update = exp_avg * beta1 + p.grad * (1 - beta1)
    update.mul_(step_factor)
    signed_update = torch.sign(update)
    p.add_(signed_update, alpha=-lr)

    # Decay the momentum running average coefficient
    exp_avg.mul_(1 - step_factor * (1 - beta2)).add_(step_factor * p.grad, alpha=1 - beta2)


class Lion(Optimizer):
    """
    An implementation of the Lion optimizer.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["step_factor"] = torch.tensor(1.0, device=p.device)

                lion_step(
                    p,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    betas=group["betas"],
                    exp_avg=state["exp_avg"],
                    step_factor=state["step_factor"],
                )


class SkipStepLion(SkipStepOptimizer):
    """
    A "skip step" version of :class:`Lion`.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        rolling_interval_length: int = 128,
        sigma_factor: int = 6,
    ) -> None:
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
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

        step_factor = self.get_step_factor()
        self._step_skipped = 1 - step_factor
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                lion_step(
                    p,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    betas=group["betas"],
                    exp_avg=state["exp_avg"],
                    step_factor=step_factor,
                )


@dataclass
class LionConfig(OptimConfig):
    """
    Configuration class for building a :class:`Lion` optimizer.
    """

    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0

    @classmethod
    def optimizer(cls) -> Type[Lion]:
        return Lion


@dataclass
class SkipStepLionConfig(OptimConfig):
    """
    Configuration class for building a :class:`SkipStepLion` optimizer.
    """

    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0
    rolling_interval_length: int = 128
    sigma_factor: int = 6

    @classmethod
    def optimizer(cls) -> Type[SkipStepLion]:
        return SkipStepLion
