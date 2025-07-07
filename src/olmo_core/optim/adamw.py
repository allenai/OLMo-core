from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.optim.optimizer

from ..config import DType
from .config import OptimConfig
from .skip_step_optimizer import SkipStepOptimizer
import logging
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.distributed.algorithms.ddp_comm_hooks.ddp_zero_hook import (
    hook_with_zero_step,
    hook_with_zero_step_interleaved,
)

log = logging.getLogger(__name__)

def adamw_step(
    p: nn.Parameter,
    *,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: torch.Tensor,
    step_factor: torch.Tensor,
):
    if p.grad is None:
        return

    beta1, beta2 = betas

    # Perform step weight decay.
    p.mul_(1 - step_factor * (lr * weight_decay))

    # Decay the first and second moment running average coefficient.
    exp_avg.lerp_(p.grad.type_as(exp_avg), (step_factor * (1 - beta1)).type_as(exp_avg))
    exp_avg_sq.mul_(1 - step_factor * (1 - beta2))
    exp_avg_sq.add_(step_factor * p.grad * p.grad, alpha=1 - beta2)

    bias_correction1 = 1 - beta1 ** (step + 1)
    bias_correction2 = 1 - beta2 ** (step + 1)

    step_size = lr / bias_correction1

    denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)

    update = -step_size * torch.div(exp_avg, denom)
    update.mul_(step_factor)
    p.add_(update)
    step.add_(step_factor)


def foreach_adamw_step(
    params: list[torch.Tensor],
    grads: list[torch.Tensor],
    exp_avgs: list[torch.Tensor],
    exp_avg_sqs: list[torch.Tensor],
    steps: list[torch.Tensor],
    *,
    lr: float,
    betas: Tuple[float, float],
    eps: float,
    weight_decay: float,
    step_factor: torch.Tensor,
):
    """Perform a single AdamW update with multi-tensor (*foreach*) kernels."""
    if not params:
        return  # nothing to do

    beta1, beta2 = betas

    # Perform step weight decay.
    torch._foreach_mul_(params, 1 - step_factor * (lr * weight_decay))

    grads = [g.type_as(ea) for g, ea in zip(grads, exp_avgs)]

    # Decay the first and second moment running average coefficient.
    # foreach_lerp_ has issues when DTensor is enabled (see https://github.com/pytorch/pytorch/issues/132017).
    # Implement the lerp(a, b, w) = a + w * (b - a) with basic _foreach_mul_/add_ ops instead:
    w1 = step_factor * (1 - beta1)
    torch._foreach_mul_(exp_avgs, 1.0 - w1)
    torch._foreach_add_(exp_avgs, torch._foreach_mul(grads, w1))

    grad_squares = torch._foreach_mul(grads, grads)

    w2 = step_factor * (1 - beta2)
    torch._foreach_mul_(exp_avg_sqs, 1.0 - w2)
    torch._foreach_add_(exp_avg_sqs, torch._foreach_mul(grad_squares, w2))

    steps_t = torch.stack(steps)
    bias_corrections1 = 1 - torch.pow(beta1, steps_t + 1)
    bias_corrections2 = 1 - torch.pow(beta2, steps_t + 1)

    step_sizes = lr / bias_corrections1

    denoms = torch._foreach_sqrt(exp_avg_sqs)
    torch._foreach_div_(denoms, bias_corrections2.sqrt().unbind())
    torch._foreach_add_(denoms, eps)

    updates = torch._foreach_div(exp_avgs, denoms)
    torch._foreach_mul_(updates, (-step_factor * step_sizes).unbind())
    torch._foreach_add_(params, updates)
    torch._foreach_add_(steps, [step_factor] * len(steps))


class SkipStepAdamW(SkipStepOptimizer):
    """
    A "skip step" version of :class:`AdamW`.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        rolling_interval_length: int = 128,
        sigma_factor: int = 6,
        dtype: Optional[Union[torch.dtype, DType]] = None,
        foreach: bool = False,
    ) -> None:
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(
            params,
            defaults,
            rolling_interval_length=rolling_interval_length,
            sigma_factor=sigma_factor,
        )
        if isinstance(dtype, DType):
            dtype = dtype.as_pt()
        self.dtype = dtype
        self.foreach = foreach
        self._step_skipped: Optional[torch.Tensor] = None

    @property
    def step_skipped(self) -> torch.Tensor:
        if self._step_skipped is not None:
            return self._step_skipped
        else:
            return torch.tensor(0.0)

    @torch.no_grad()
    def step(self, closure=None) -> None:
        if self.foreach:
            self._step_foreach(closure)
        else:
            self._step(closure)

    def _step(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        step_factor = self.get_step_factor()
        self._step_skipped = 1 - step_factor
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    raise RuntimeError(
                        f"DEBUG: Parameter {p} has no gradient, skipping optimization step for this parameter."
                    )
                    # continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p, dtype=self.dtype)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype)

                adamw_step(
                    p,
                    lr=group["lr"],
                    betas=group["betas"],
                    eps=group["eps"],
                    weight_decay=group["weight_decay"],
                    exp_avg=state["exp_avg"],
                    exp_avg_sq=state["exp_avg_sq"],
                    step=state["step"],
                    step_factor=step_factor,
                )

    def _step_foreach(self, closure=None) -> None:
        if closure is not None:
            with torch.enable_grad():
                closure()

        step_factor = self.get_step_factor()
        self._step_skipped = 1 - step_factor
        for group in self.param_groups:
            params_with_grad: list[torch.Tensor] = []
            grads: list[torch.Tensor] = []
            exp_avgs: list[torch.Tensor] = []
            exp_avg_sqs: list[torch.Tensor] = []
            steps_list = []  # create list outside loops

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p, dtype=self.dtype)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.dtype)

                params_with_grad.append(p)
                grads.append(p.grad)
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                steps_list.append(state["step"])

            if not params_with_grad:
                continue  # nothing to update in this group

            foreach_adamw_step(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                steps_list,
                lr=group["lr"],
                betas=group["betas"],
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                step_factor=step_factor,
            )
            # grads_size = sum([g.numel() * g.element_size() for g in grads])/1024**3
            # exp_avgs_size = sum([ea.numel() * ea.element_size() for ea in exp_avgs])/1024**3
            # exp_avg_sqs_size = sum([eas.numel() * eas.element_size() for eas in exp_avg_sqs])/1024**3

@dataclass
class AdamWConfig(OptimConfig):  # NOTE: omagaconf doesn't like "OptimConfig[torch.optim.AdamW]"
    """
    Configuration class for building an :class:`torch.optim.AdamW` optimizer.
    """

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    foreach: Optional[bool] = None
    fused: Optional[bool] = None

    @classmethod
    def optimizer(cls) -> Type[torch.optim.AdamW]:
        return torch.optim.AdamW


@dataclass
class SkipStepAdamWConfig(OptimConfig):
    """
    Configuration class for building a :class:`SkipStepAdamW` optimizer.
    """

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    dtype: Optional[DType] = None

    foreach: bool = True
    """
    Whether to use multi-tensor (*foreach*) kernels for the AdamW update.
    Faster than the non-foreach version.
    """

    rolling_interval_length: int = 128
    """
    The length of the rolling interval to use for computing the mean and standard deviation of the loss.
    """

    sigma_factor: int = 6
    """
    The number of standard deviations above the mean loss to skip a step.
    """

    @classmethod
    def optimizer(cls) -> Type[SkipStepAdamW]:
        return SkipStepAdamW



# --------------------------------------------------------------------------- #
# ZeRO-1 sharded Skip-Step AdamW
# --------------------------------------------------------------------------- #
from torch.distributed.optim import ZeroRedundancyOptimizer
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
from .config import LR_FIELD, INITIAL_LR_FIELD
from ..utils import get_default_device, move_to_device
Opt = TypeVar("Opt", bound=torch.optim.Optimizer)

from ..train.train_module import TrainModule

@dataclass
class ZeroAdamWConfig(OptimConfig):
    """
    Config for the ZeRO-1 sharded Skip-Step AdamW optimizer.
    """

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    foreach: Optional[bool] = None
    fused: Optional[bool] = None
    # optimizer_class: torch.optim.Optimizer = torch.optim.AdamW
    
    @classmethod
    def optimizer(cls):
        return ZeroRedundancyOptimizer
    

    def build(self, model: nn.Module, train_module: TrainModule, strict: bool = True) -> Opt:
        """
        Build the optimizer.

        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        kwargs = self.as_dict()
        kwargs.pop("group_overrides")
        kwargs.pop("compile")
        kwargs.pop("fixed_fields")

        optim: torch.optim.Optimizer = self.optimizer()(
            self.build_groups(model, strict=strict), 
            optimizer_class=torch.optim.AdamW,
            process_group=train_module.dp_process_group,
            **kwargs
        )

        # Set 'lr' and 'initial_lr' in each group if needed.
        fixed_fields_per_group: List[Dict[str, Any]] = [{} for _ in optim.param_groups]
        for fixed_fields, group in zip(fixed_fields_per_group, optim.param_groups):
            lr: Optional[float] = None
            if LR_FIELD in group:
                lr = group[LR_FIELD]
            elif hasattr(self, LR_FIELD):
                lr = getattr(self, LR_FIELD)

            if lr is not None:
                if self.compile:
                    # 'lr' should be a tensor.
                    group[LR_FIELD] = move_to_device(torch.tensor(lr), self.device)
                else:
                    group[LR_FIELD] = lr
                group.setdefault(INITIAL_LR_FIELD, lr)

            for k in self.fixed_fields:
                if k in group:
                    fixed_fields[k] = group[k]

        log.info(
            f"Building {self.optimizer().__name__} optimizer with {len(optim.param_groups)} param group(s)..."
        )
        for g_idx, group in enumerate(optim.param_groups):
            group_fields_list = "\n - ".join(
                [f"{k}: {v}" for k, v in optim.param_groups[g_idx].items() if k != "params"]
            )
            if group_fields_list:
                log.info(
                    f"Group {g_idx}, {len(group['params'])} parameter(s):\n - {group_fields_list}"
                )
            else:
                log.info(f"Group {g_idx}, {len(group['params'])} parameter(s)")

        if self.compile:
            log.info("Compiling optimizer step...")
            optim.step = torch.compile(optim.step)

        # Register hook to reset fixed fields after loading a checkpoint.
        def reset_fixed_fields(opt: torch.optim.Optimizer):
            for fixed_fields, group in zip(fixed_fields_per_group, opt.param_groups):
                group.update(fixed_fields)

        optim.register_load_state_dict_post_hook(reset_fixed_fields)

        return cast(Opt, optim)
    