import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from ..distributed.utils import get_local_tensor
from .config import OptimConfig


def adamw_step(
    p: torch.Tensor,
    grad: torch.Tensor,
    *,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    step_factor: torch.Tensor,
):
    beta1, beta2 = betas

    # Perform step weight decay.
    p.mul_(1 - step_factor * (lr * weight_decay))

    # Decay the first and second moment running average coefficient.
    exp_avg.lerp_(grad, step_factor * (1 - beta1))
    exp_avg_sq.mul_(1 - step_factor * (1 - beta2))
    exp_avg_sq.add_(step_factor * grad * grad, alpha=1 - beta2)

    bias_correction1 = 1 - beta1**step
    bias_correction2 = 1 - beta2**step

    step_size = lr / bias_correction1

    bias_correction2_sqrt = math.sqrt(bias_correction2)
    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

    update = -step_size * torch.div(exp_avg, denom)
    update.mul_(step_factor)
    p.add_(update)


@torch.compile(fullgraph=False)
def compiled_adamw_step(
    p: torch.Tensor,
    grad: torch.Tensor,
    *,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int,
    step_factor: torch.Tensor,
):
    adamw_step(
        p,
        grad,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq,
        step=step,
        step_factor=step_factor,
    )


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        compile: bool = False,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.compile = compile

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        step_func = adamw_step if not self.compile else compiled_adamw_step
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] = torch.tensor(0.0)
                    state["step_factor"] = torch.tensor(1.0, device=p.device)

                # Ensure 'step' tensor always on CPU to avoid host-device sync.
                if state["step"].device.type != "cpu":
                    state["step"] = state["step"].cpu()

                # Update step.
                state["step"] += 1

                # NOTE: No host-device sync since 'step' tensor will always be on CPU.
                step = state["step"].item()

                step_func(
                    get_local_tensor(p),
                    get_local_tensor(p.grad),
                    lr=group["lr"],
                    betas=group["betas"],
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                    exp_avg=get_local_tensor(state["exp_avg"]),
                    exp_avg_sq=get_local_tensor(state["exp_avg_sq"]),
                    step=step,
                    step_factor=state["step_factor"],
                )


@dataclass
class AdamWConfig(OptimConfig):  # NOTE: omagaconf doesn't like "OptimConfig[torch.optim.AdamW]"
    """
    Configuration class for building an :class:`AdamW` optimizer.
    """

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    compile: bool = False

    def build(self, model: nn.Module) -> AdamW:
        kwargs = self.as_dict()
        kwargs.pop("group_overrides")
        optim = AdamW(self.build_groups(model), **kwargs)
        for group in optim.param_groups:
            group.setdefault("initial_lr", self.lr)
        return optim
