# Adapted from https://github.com/KellerJordan/Muon/blob/master/muon.py
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple, Type, Union

import torch

from olmo_core.distributed.utils import all_gather, get_rank, get_world_size
from olmo_core.optim.config import OptimConfig, OptimGroupOverride
from olmo_core.optim.skip_step_optimizer import SkipStepOptimizer

try:
    from quack.gemm_interface import gemm_symmetric  # type: ignore[reportMissingImports]
except ModuleNotFoundError:

    def gemm_symmetric(X, Y, out=None):
        if out is None:
            out = torch.empty_like(X)
        torch.matmul(X, Y, out=out)
        return out


log = logging.getLogger(__name__)


@torch.compile(dynamic=False, fullgraph=True)
def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Adapted from https://github.com/KellerJordan/Muon/blob/master/muon.py
    """
    assert gemm_symmetric is not None
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    buf1 = torch.empty(X.size(0), X.size(0), dtype=X.dtype, device=X.device)
    buf2 = torch.empty(X.size(0), X.size(0), dtype=X.dtype, device=X.device)

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    # Perform the NS iterations
    for _ in range(steps):
        # symmul usage adapted from
        # - https://www.lakernewhouse.com/assets/writing/faster-symmul-with-thunderkittens.pdf
        # - https://github.com/nil0x9/flash-muon/
        gemm_symmetric(X, X.mT, out=buf1)
        gemm_symmetric(buf1, buf1.mT, out=buf2)
        B = b * buf1 + c * buf2
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(
    grad,
    momentum,
    *,
    beta=0.95,
    nesterov=True,
    step_factor: torch.Tensor = torch.tensor(1.0),
):
    a, b = grad.size(-2), grad.size(-1)
    momentum.lerp_(grad, step_factor * (1 - beta))
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=5)
    # update *= max(1, a / b) ** 0.5  # original scaling
    update *= 0.2 * max(a, b) ** 0.5  # moonlight scaling
    return update


def adam_update(grad, buf1, buf2, step, betas, eps, step_factor: torch.Tensor):
    buf1.lerp_(grad, step_factor * (1 - betas[0]))
    buf2.lerp_(grad.square(), step_factor * (1 - betas[1]))
    buf1c = buf1 / (1 - betas[0] ** (step + 1))
    buf2c = buf2 / (1 - betas[1] ** (step + 1))
    update = buf1c / (buf2c.sqrt() + eps)
    update.mul_(step_factor)
    step.add_(step_factor)
    return update


class SkipStepMuon(SkipStepOptimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    Example usage:
    ```python
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """

    def __init__(
        self,
        param_groups,
        *,
        lr,
        momentum,
        weight_decay,
        rolling_interval_length: int = 128,
        sigma_factor: int = 6,
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, use_muon=True)
        for group in param_groups:
            if group.get("use_muon", False):
                # group params by size, largest to smallest
                group["params"] = sorted(group["params"], key=lambda x: x.numel(), reverse=True)
        super().__init__(param_groups, defaults)
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
            if group.get("use_muon", False):
                params = group["params"]
                world_size = get_world_size()
                params_pad = params + [torch.empty_like(params[-1])] * (
                    world_size - len(params) % world_size
                )
                for base_i in range(len(params))[::world_size]:
                    if base_i + get_rank() < len(params):
                        p = params[base_i + get_rank()]
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(
                            p.grad,
                            state["momentum_buffer"],
                            beta=group["momentum"],
                            step_factor=step_factor,
                        )
                        p.mul_(1 - step_factor * (group["lr"] * group["weight_decay"]))
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    params_pad[base_i : base_i + world_size] = all_gather(
                        params_pad[base_i + get_rank()]
                    )

            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = torch.zeros((), dtype=torch.float32, device=p.device)
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                        step_factor=step_factor,
                    )
                    p.mul_(1 - step_factor * (group["lr"] * group["weight_decay"]))
                    p.add_(update, alpha=-group["lr"])
        return


@dataclass
class SkipStepMuonConfig(OptimConfig):
    """
    Configuration class for building a :class:`SkipStepMuon` optimizer.
    """

    rolling_interval_length: int = 128
    """
    The length of the rolling interval to use for computing the mean and standard deviation of the loss.
    """

    sigma_factor: int = 6
    """
    The number of standard deviations above the mean loss to skip a step.
    """

    # Default values for Muon groups
    lr: float = 0.02
    momentum: float = 0.95
    weight_decay: float = 0.0

    # Default values for Adam groups, private so they are not included in the config dict
    _adam_lr: float = 1e-3
    _adam_betas: Tuple[float, float] = (0.9, 0.999)
    _adam_eps: float = 1e-8
    _adam_weight_decay: float = 1e-2
    _adam_embed_weight_decay: float = 0.0

    @classmethod
    def optimizer(cls) -> Type[SkipStepMuon]:
        return SkipStepMuon

    def default_group_overrides(self, model: torch.nn.Module) -> list[OptimGroupOverride]:
        """
        Split the model parameters into Adam and Muon groups.
        Only >=2d, internal parameters are meant to be optimized with Muon.
        """
        embed_param_names = [n for n, p in model.named_parameters() if "embed" in n]
        scalar_param_names = [n for n, p in model.named_parameters() if p.ndim < 2]
        head_param_names = [
            n for n, p in model.named_parameters() if "lm_head" in n and p.ndim >= 2
        ]
        adam_param_names = set(embed_param_names + scalar_param_names + head_param_names)
        hidden_matrix_param_names = [
            n for n, _ in model.named_parameters() if n not in adam_param_names
        ]
        assert all(
            p.ndim >= 2 for n, p in model.named_parameters() if n in hidden_matrix_param_names
        )

        adam_override = OptimGroupOverride(
            params=head_param_names + scalar_param_names,
            opts=dict(
                lr=self._adam_lr,
                betas=self._adam_betas,
                eps=self._adam_eps,
                weight_decay=self._adam_weight_decay,
                use_muon=False,
            ),
        )
        embed_override = OptimGroupOverride(
            params=embed_param_names,
            opts=dict(
                lr=self._adam_lr,
                betas=self._adam_betas,
                eps=self._adam_eps,
                weight_decay=self._adam_embed_weight_decay,
                use_muon=False,
            ),
        )
        return [adam_override, embed_override]

    def build_groups(
        self, model: torch.nn.Module, strict: bool = True
    ) -> Union[Iterable[torch.Tensor], list[dict[str, Any]]]:
        """
        Build parameters groups.

        :param model: The model to optimize.
        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """
        all_params: dict[str, torch.Tensor] = OrderedDict()
        frozen_params: set = set()
        for n, p in model.named_parameters():
            if p.requires_grad:
                all_params[n] = p
            else:
                frozen_params.add(n)

        if self.group_overrides is None:
            self.group_overrides = self.default_group_overrides(model)

        group_overrides = [
            self._expand_param_globs(go, all_params, frozen_params, g_idx, strict=strict)
            for g_idx, go in enumerate(self.group_overrides or [])
        ]

        # Treat no overrides as its own override group
        overridden_param_names = {name for go in group_overrides for name in go.params}
        default_override = OptimGroupOverride(
            [name for name in all_params.keys() if name not in overridden_param_names], {}
        )
        group_overrides.append(default_override)

        return [
            {"params": [all_params[param_name] for param_name in go.params], **go.opts}
            for go in group_overrides
            if len(go.params) > 0
        ]

    def build(self, model: torch.nn.Module, strict: bool = True) -> SkipStepMuon:
        """
        Build the optimizer.

        :param model: The model to optimize.
        :param strict: If ``True`` an error is raised if a pattern in ``group_overrides`` doesn't
            match any parameter.
        """

        return super().build(model, strict=strict)
