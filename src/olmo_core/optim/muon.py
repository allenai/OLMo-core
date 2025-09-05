import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel

from olmo_core.distributed.utils import is_distributed
from olmo_core.utils import get_default_device

from .config import OptimConfig


class MuonAdamW(torch.optim.Optimizer):
    """
    Muon optimizer with AdamW fallback for non-matrix parameters.
    Muon (Momentum Orthogonalized by Newton-schulz) is an optimizer that uses
    orthogonalization of updates for matrix parameters to achieve better
    conditioning. For non-matrix parameters and certain excluded layers
    (embeddings, output heads), it falls back to AdamW.
    Reference:
        "Muon: Momentum Orthogonalized by Newton-schulz"
        https://github.com/KellerJordan/Muon
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        lr: Learning rate (default: 0.01)
        betas: Coefficients for computing running averages (default: (0.95, 0.95))
        weight_decay: Weight decay coefficient (default: 0.0)
        ns_steps: Number of Newton-Schulz iterations (default: 5)
        nesterov: Whether to use Nesterov momentum (default: True)
        eps: Term added to denominator for AdamW (default: 1e-8)
        record_update_metrics: Whether to record update metrics (default: False)
        selective_updates: Whether to use selective weight updates (default: False)
    Note:
        - Matrix parameters (2D+) use Muon unless they contain 'embed' or 'head' in name
        - Non-matrix parameters always use AdamW
        - Weight decay is applied AdamW-style (decoupled)
    """

    def __init__(
        self,
        params,
        lr=0.01,
        betas=(0.95, 0.95),
        weight_decay=0.0,
        ns_steps=5,
        nesterov=True,
        eps=1e-8,
        record_update_metrics=False,
        selective_updates=False,
        device=None,
    ):
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param_group in params:
                if "use_muon" not in param_group:
                    param_group["use_muon"] = True
        else:
            params = [{"params": params, "use_muon": True}]
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            nesterov=nesterov,
            eps=eps,
            use_muon=True,
        )
        super().__init__(params, defaults)
        self._record_update_metrics = record_update_metrics
        self._selective_updates = selective_updates
        self._collecting_metrics = True
        self._device = device
        self._update_norms = None
        self._update_maxs = None
        self._update_param_names = None

    def zeropower_via_newtonschulz5(self, G, steps: int):
        """
        Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
        """
        assert G.ndim >= 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        transposed = False
        if G.size(-2) > G.size(-1):
            X = X.mT
            transposed = True
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X
        if transposed:
            X = X.mT
        return X

    def get_state_for_param(self, param: nn.Parameter) -> Dict[str, Optional[torch.Tensor]]:
        """Return optimizer state for a parameter."""
        state = self.state[param]
        if not state:
            return {}
        result = {}
        if "momentum_buffer" in state:
            result["momentum_buffer"] = state["momentum_buffer"]
        if "exp_avg" in state:
            result["exp_avg"] = state["exp_avg"]
        if "exp_avg_sq" in state:
            result["exp_avg_sq"] = state["exp_avg_sq"]
        return result

    def _clean_param_name(self, name: str) -> str:
        """Stub for cleaning parameter names; implement as needed for olmo-core."""
        return name.replace(".", "_")

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        if closure is not None:
            with torch.enable_grad():
                closure()
        device = get_default_device() if self._device is None else self._device
        update_norms = []
        update_maxs = []
        update_param_names = []
        collecting_metrics = self._collecting_metrics and self._record_update_metrics
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            eps = group["eps"]
            use_muon = group["use_muon"]
            for i, p in enumerate(group["params"]):
                if "param_names" in group and i < len(group["param_names"]):
                    name = self._clean_param_name(group["param_names"][i])
                else:
                    name = f"param_{i}"
                if p.grad is None:
                    if collecting_metrics:
                        update_param_names.append(name)
                        update_norms.append(torch.tensor([0.0], device=device))
                        update_maxs.append(torch.tensor([0.0], device=device))
                    continue
                mask = (
                    (p.grad != 0)
                    if self._selective_updates
                    else torch.ones_like(p, dtype=torch.bool)
                )
                p.mul_(1 - mask * (lr * weight_decay))
                grad = p.grad
                state = self.state[p]
                # Determine whether to use Muon or AdamW for this parameter
                # We use Muon for matrix parameters unless explicitly disabled
                should_use_muon = (
                    use_muon
                    and p.ndim >= 2
                    and not ("embed" in name.lower() or "head" in name.lower())
                )
                if should_use_muon:
                    # Initialize momentum buffer if needed
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    momentum_buffer = state["momentum_buffer"]
                    momentum_buffer.lerp_(grad, mask * (1 - beta1))
                    if nesterov:
                        update = momentum_buffer * beta1 + grad * (1 - beta1)
                    else:
                        update = momentum_buffer.clone()
                    if isinstance(mask, torch.Tensor):
                        update.mul_(mask)
                    orig_shape = update.shape
                    if update.ndim == 4:
                        update = update.view(update.shape[0], -1)
                    update = self.zeropower_via_newtonschulz5(update, steps=ns_steps)
                    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
                    if len(orig_shape) == 4:
                        update = update.view(orig_shape)
                else:
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(grad)
                        state["exp_avg_sq"] = torch.zeros_like(grad)
                        state["step"] = 0
                    state["step"] += 1
                    step = state["step"]
                    if self._selective_updates:
                        state["exp_avg"].lerp_(grad, mask * (1 - beta1))
                        state["exp_avg_sq"].mul_(1 - mask * (1 - beta2)).addcmul_(
                            grad, grad, value=mask * (1 - beta2)
                        )
                    else:
                        state["exp_avg"].lerp_(grad, 1 - beta1)
                        state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    # Compute AdamW update
                    denom = (state["exp_avg_sq"].sqrt() / math.sqrt(bias_correction2)).add_(eps)
                    update = state["exp_avg"] / bias_correction1 / denom
                    if isinstance(mask, torch.Tensor):
                        update.mul_(mask)
                p.add_(update, alpha=-lr)
                if collecting_metrics:
                    update_param_names.append(name)
                    update_norms.append(
                        torch.linalg.vector_norm(update, 2.0, dtype=torch.float32).unsqueeze(0)
                    )
                    update_maxs.append(update.abs().max().unsqueeze(0))
        if collecting_metrics:
            self._update_norms = update_norms
            self._update_maxs = update_maxs
            self._update_param_names = update_param_names

    def get_post_step_metrics(
        self, module: nn.Module, process_group: Optional[dist.ProcessGroup] = None
    ) -> Dict[str, torch.Tensor]:
        """Get metrics about the optimization step."""
        if not (self._record_update_metrics and self._collecting_metrics):
            return {}
        device = get_default_device() if self._device is None else self._device
        dst_rank = 0
        if process_group is not None:
            dst_rank = dist.get_global_rank(process_group, 0)
        param_names = self._update_param_names
        update_norms = self._update_norms
        update_maxs = self._update_maxs
        if param_names is None or update_norms is None or update_maxs is None:
            return {}
        if is_distributed() and isinstance(module, FullyShardedDataParallel):
            all_norms = torch.cat(update_norms).to(device) ** 2.0
            dist.reduce(all_norms, dst_rank, op=dist.ReduceOp.SUM, group=process_group)
            update_norms = (all_norms ** (0.5)).squeeze(0).split(1)
            all_maxs = torch.cat(update_maxs).to(device)
            dist.reduce(all_maxs, dst_rank, op=dist.ReduceOp.MAX, group=process_group)
            update_maxs = all_maxs.split(1)
        metrics = {}
        for param_name, update_norm, update_max in zip(param_names, update_norms, update_maxs):
            metrics[f"update/{param_name}.norm"] = update_norm.squeeze(0)
            metrics[f"update/{param_name}.max"] = update_max.squeeze(0)
        self._update_norms = None
        self._update_maxs = None
        self._update_param_names = None
        return metrics


@dataclass
class MuonAdamWConfig(OptimConfig):
    """
    Configuration class for building a :class:`MuonAdamW` optimizer.
    """

    lr: float = 0.01
    betas: Tuple[float, float] = (0.95, 0.95)
    weight_decay: float = 0.0
    ns_steps: int = 5
    nesterov: bool = True
    eps: float = 1e-8
    record_update_metrics: bool = False
    selective_updates: bool = False

    @classmethod
    def optimizer(cls) -> Type[MuonAdamW]:
        return MuonAdamW
