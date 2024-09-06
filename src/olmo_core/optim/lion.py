from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from .config import OptimConfig
from .skip_step_optimizer import SkipStepOptimizer


def lion_step(
    p: torch.nn.Parameter,
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


@torch.compile(fullgraph=False)
def compiled_lion_step(
    p: torch.nn.Parameter,
    *,
    lr: float,
    weight_decay: float,
    exp_avg: torch.Tensor,
    betas: Tuple[float, float],
    step_factor: torch.Tensor,
):
    lion_step(
        p, lr=lr, weight_decay=weight_decay, exp_avg=exp_avg, betas=betas, step_factor=step_factor
    )


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
        compile: bool = False,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.compile = compile

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        step_func = lion_step if not self.compile else compiled_lion_step
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["step_factor"] = torch.tensor(1.0, device=p.device)

                step_func(
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
        compile: bool = False,
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
        self.compile = compile

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        step_factor = self.get_step_factor()
        step_func = lion_step if not self.compile else compiled_lion_step
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                step_func(
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
    compile: bool = False

    def build(self, model: nn.Module) -> Lion:
        kwargs = self.as_dict()
        kwargs.pop("group_overrides")
        optim = Lion(self.build_groups(model), **kwargs)
        for group in optim.param_groups:
            group.setdefault("initial_lr", self.lr)
        return optim


@dataclass
class SkipStepLionConfig(LionConfig):
    """
    Configuration class for building a :class:`SkipStepLion` optimizer.
    """

    rolling_interval_length: int = 128
    sigma_factor: int = 6