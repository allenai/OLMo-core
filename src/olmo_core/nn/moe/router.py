from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module

import olmo_core.ops.moe as ops
from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.utils import (
    distribute_like,
    get_local_tensor,
    is_distributed,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.ops import AutoAuxiliaryLoss

from ..buffer_cache import BufferCache
from .loss import MoELoadBalancingLossGranularity, load_balancing_loss, router_z_loss

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

__all__ = [
    "MoERouter",
    "MoELinearRouter",
    "MoERouterConfig",
    "MoERouterType",
    "MoERouterGatingFunction",
]


# NOTE: To enable end-to-end benchmarking without convergence we
# support a flag to force the router to assign items/tokens uniformly
# across the experts. We do this with a custom autograd operation
# so that PyTorch still executes the full set of router operation.
class _UniformExpertAssignment(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, num_experts: int):
        del ctx
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)


_uniform_expert_assignment: Callable[
    [torch.Tensor, int], torch.Tensor
] = _UniformExpertAssignment.apply  # type: ignore


class MoERouterType(StrEnum):
    """
    An enumeration of the different MoE router implementations.
    """

    default = "default"
    """
    ➡️ :class:`MoELinearRouter`
    """


class MoERouterGatingFunction(StrEnum):
    softmax = "softmax"
    sigmoid = "sigmoid"


@dataclass
class MoERouterConfig(Config):
    """
    A configuration class for easily building any of the different MoE router modules.
    """

    name: MoERouterType = MoERouterType.default
    """
    The name of the implementation.
    """
    top_k: int = 1
    jitter_eps: Optional[float] = None
    normalize_expert_weights: Optional[float] = None
    uniform_expert_assignment: bool = False
    bias_gamma: Optional[float] = None
    gating_function: MoERouterGatingFunction = MoERouterGatingFunction.softmax
    dtype: Optional[DType] = None

    def num_params(self, d_model: int, num_experts: int) -> int:
        """
        The number of params that the module will have once built.

        :param d_model: The model dimensionality.
        """
        num_params = 0
        if self.name == MoERouterType.default:
            num_params += d_model * num_experts
        else:
            raise NotImplementedError

        return num_params

    def build(
        self,
        d_model: int,
        num_experts,
        *,
        lb_loss_weight: Optional[float] = None,
        lb_loss_granularity: MoELoadBalancingLossGranularity = MoELoadBalancingLossGranularity.local_batch,
        z_loss_weight: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        init_device: str = "cpu",
    ) -> "MoERouter":
        """
        Build the corresponding MoE router module.

        :param d_model: The model dimensionality.
        :param num_experts: The number of experts.
        :param init_device: The device initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.update(
            d_model=d_model,
            num_experts=num_experts,
            init_device=init_device,
            lb_loss_weight=lb_loss_weight,
            lb_loss_granularity=lb_loss_granularity,
            z_loss_weight=z_loss_weight,
        )
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype.as_pt()
        elif dtype is not None:
            kwargs["dtype"] = dtype

        try:
            if self.name == MoERouterType.default:
                return MoELinearRouter(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


class MoERouter(nn.Module):
    """
    A base class for MoE router modules.

    :param d_model: The model dimensionality (hidden size).
    :param num_experts: The total number of experts.
    :param top_k: The number of experts to assign to each item/token.
    :param jitter_eps: Controls the amount of noise added to the input during training.
    :param normalize_expert_weights: The type of norm (e.g. ``2.0`` for L2 norm) to use to normalize
        the expert weights.
    :param uniform_expert_assignment: Force uniform assignment. Useful for benchmarking.
    :param bias_gamma: If set to a positive float, experts scores for top-k routing will be adjusted
        by a bias following the "auxiliary-loss-free load balancing" strategy from DeepSeek-v3.
        A reasonable value is on the order of 0.0001.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_experts: int,
        top_k: int = 1,
        jitter_eps: Optional[float] = None,
        normalize_expert_weights: Optional[float] = None,
        uniform_expert_assignment: bool = False,
        bias_gamma: Optional[float] = None,
        gating_function: MoERouterGatingFunction = MoERouterGatingFunction.softmax,
        lb_loss_weight: Optional[float] = None,
        lb_loss_granularity: MoELoadBalancingLossGranularity = MoELoadBalancingLossGranularity.local_batch,
        z_loss_weight: Optional[float] = None,
        init_device: str = "cpu",
    ):
        super().__init__()
        # We don't take the 'BufferCache' as an argument to ensure it's not a shared cache.
        self._cache = BufferCache()

        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_eps = jitter_eps
        self.normalize_expert_weights = normalize_expert_weights
        self.uniform_expert_assignment = uniform_expert_assignment
        self.bias_gamma = bias_gamma
        self.gating_function = gating_function
        self.lb_loss_weight = lb_loss_weight
        self.lb_loss_granularity = lb_loss_granularity
        self.z_loss_weight = z_loss_weight
        self.group: Optional[dist.ProcessGroup] = None
        self.cp_mesh: Optional[dist.DeviceMesh] = None

        if self.bias_gamma is not None:
            assert self.bias_gamma > 0
            self.register_buffer("score_bias", torch.zeros(self.num_experts, device=init_device))
            # NOTE: use buffer cache to accumulate 'batch_size_per_expert' and "hide" it from FSDP
            # so that FSDP doesn't try to distribute it.
            self._cache["score_bias_batch_size_per_expert"] = torch.zeros(self.num_experts)
        else:
            self.register_buffer("score_bias", None)

    def reset_parameters(self):
        if self.bias_gamma is not None:
            assert self.score_bias is not None
            assert self._cache is not None
            score_bias = cast(torch.Tensor, self.score_bias)
            score_bias.zero_()
            self._cache["batch_size_per_expert"] = torch.zeros(
                self.num_experts, device=score_bias.device
            )

    def _maybe_accumulate_batch_size_per_expert(self, batch_size_per_expert: torch.Tensor):
        if self.bias_gamma is None or not self.training or not torch.is_grad_enabled():
            return
        self._accumulate_metric("score_bias_batch_size_per_expert", batch_size_per_expert)

    @torch.no_grad()
    def _accumulate_metric(self, name: str, value: torch.Tensor):
        value = value.detach()
        if name not in self._cache:
            self._cache[name] = torch.zeros_like(value)
        elif self._cache[name].device != value.device:
            self._cache[name] = self._cache[name].to(value.device)
        self._cache[name] += value

    @torch.no_grad()
    def post_batch(self, dry_run: bool = False):
        if self.bias_gamma is None or not self.training:
            return

        assert self.score_bias is not None
        assert self._cache is not None
        score_bias = cast(torch.Tensor, self.score_bias)
        batch_size_per_expert = self._cache["score_bias_batch_size_per_expert"]

        # Maybe reduce across the process group.
        if is_distributed():
            dist.all_reduce(batch_size_per_expert, group=self.group)

        ideal_batch_size_per_expert = batch_size_per_expert.mean(
            dim=0, keepdim=True, dtype=torch.float32
        )
        bias_delta = self.bias_gamma * (ideal_batch_size_per_expert - batch_size_per_expert).sign()
        # NOTE: have to be careful here to manage the case where `score_bias` is a DTensor.
        bias_delta = distribute_like(score_bias, bias_delta)

        if not dry_run:
            get_local_tensor(score_bias).add_(get_local_tensor(bias_delta))

        # Reset the accumulator.
        batch_size_per_expert.zero_()

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        if self.jitter_eps is None or not self.training:
            return x
        else:
            low = 1.0 - self.jitter_eps
            high = 1.0 + self.jitter_eps
            noise = torch.rand_like(x)
            return x * (low + noise * (high - low))

    def get_top_k(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        expert_weights: torch.Tensor
        expert_indices: torch.Tensor
        if self.bias_gamma is None:
            if self.top_k == 1:
                expert_weights, expert_indices = scores.max(dim=-1, keepdim=True)
            else:
                expert_weights, expert_indices = torch.topk(scores, self.top_k, dim=-1)
        else:
            assert self.score_bias is not None
            with torch.no_grad():
                _, expert_indices = torch.topk(
                    scores + self.score_bias.unsqueeze(0), self.top_k, dim=-1  # type: ignore
                )
            expert_weights = scores.gather(-1, expert_indices)

        if self.uniform_expert_assignment:
            expert_indices = _uniform_expert_assignment(expert_indices, self.num_experts)
            expert_weights = scores.gather(-1, expert_indices)

        return expert_weights, expert_indices

    @abstractmethod
    def get_expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given the input ``x`` of shape ``(*, d_model)``, compute the un-normalized expert scores.

        :returns: The expert logits, shape ``(*, num_experts)``.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        from olmo_core.train.common import ReduceType

        out: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]] = {}

        # Load imbalance.
        batch_size_per_expert = self._cache["batch_size_per_expert"]
        out["load imbalance"] = (
            batch_size_per_expert.max() / batch_size_per_expert.mean(dtype=torch.float),
            ReduceType.max,
        )

        # Load balancing loss.
        if self.lb_loss_weight is not None:
            out["load balancing loss"] = (self._cache["load balancing loss"], ReduceType.mean)
            out["load balancing loss (unscaled)"] = (
                self._cache["load balancing loss (unscaled)"],
                ReduceType.mean,
            )

        # Router Z loss.
        if self.z_loss_weight is not None:
            out["router Z loss"] = (self._cache["router Z loss"], ReduceType.mean)
            out["router Z loss (unscaled)"] = (
                self._cache["router Z loss (unscaled)"],
                ReduceType.mean,
            )

        if reset:
            self.reset_metrics()

        return out

    def reset_metrics(self):
        for key in list(self._cache.keys()):
            if (
                key != "score_bias_batch_size_per_expert"
            ):  # this one gets reset from '.post_batch()'
                self._cache[key] = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given the input ``x`` of shape ``(B, S, d_model)``, compute the experts assignment.

        :returns: The input, potentially with losses attached for autograd,
            the expert weights of shape ``(B, S, top_k)``,
            the expert indices of shape ``(B, S, top_k)``,
            the total number of items routed to each expert, with shape ``(num_experts,)``.
        """
        B, S, _ = x.shape

        # shape: (batch_size, seq_len, d_model)
        x = self.jitter(x)

        # shape: (batch_size, seq_len, num_experts)
        logits = self.get_expert_logits(x)

        # shape: (batch_size, seq_len, num_experts)
        if self.gating_function == MoERouterGatingFunction.softmax:
            scores = logits.softmax(dim=-1)
        elif self.gating_function == MoERouterGatingFunction.sigmoid:
            scores = F.sigmoid(logits)  # NOTE: we normalize these *after* top-k, see below
        else:
            raise NotImplementedError(self.gating_function)

        # shape: (batch_size, seq_len, top_k)
        expert_weights, expert_indices = self.get_top_k(scores)

        if self.normalize_expert_weights is not None:
            expert_weights = expert_weights.div(
                torch.norm(
                    expert_weights,
                    p=self.normalize_expert_weights,
                    dim=-1,
                    keepdim=True,
                )
            )

        # Make sure scores are normalized, otherwise load balancing loss doesn't work.
        if self.gating_function == MoERouterGatingFunction.sigmoid:
            scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)

        with torch.no_grad():
            # Histogram the expert ids to identify the number of items/tokens routed to each expert.
            # shape: (batch_size, seq_len, num_experts)
            batched_batch_size_per_expert = ops.batched_histc(expert_indices, self.num_experts)
            # shape: (batch_size, num_experts)
            batched_batch_size_per_expert = batched_batch_size_per_expert.sum(dim=1)
            # shape: (num_experts,)
            batch_size_per_expert = batched_batch_size_per_expert.sum(dim=0)

        # Maybe apply auxiliary losses and accumulate metrics.
        if self.training and torch.is_grad_enabled():
            if self.lb_loss_weight is not None:
                lb_loss = load_balancing_loss(
                    num_experts=self.num_experts,
                    top_k=self.top_k,
                    expert_scores=scores,
                    batch_size_per_expert=batch_size_per_expert,
                    batched_batch_size_per_expert=batched_batch_size_per_expert,
                    granularity=self.lb_loss_granularity,
                    sp_mesh=self.cp_mesh,
                )
                if loss_div_factor is not None:
                    lb_loss = lb_loss / loss_div_factor
                else:
                    lb_loss = lb_loss / (B * S)
                scaled_lb_loss = self.lb_loss_weight * lb_loss
                AutoAuxiliaryLoss.apply(x, scaled_lb_loss)
                self._accumulate_metric("load balancing loss", scaled_lb_loss)
                self._accumulate_metric("load balancing loss (unscaled)", lb_loss)

            if self.z_loss_weight is not None:
                z_loss = router_z_loss(expert_logits=logits)
                if loss_div_factor is not None:
                    z_loss = z_loss / loss_div_factor
                else:
                    z_loss = z_loss / (B * S)
                scaled_z_loss = self.z_loss_weight * z_loss
                AutoAuxiliaryLoss.apply(x, scaled_z_loss)
                self._accumulate_metric("router Z loss", scaled_z_loss)
                self._accumulate_metric("router Z loss (unscaled)", z_loss)

            self._accumulate_metric("batch_size_per_expert", batch_size_per_expert)
            if self.bias_gamma is not None:
                self._accumulate_metric("score_bias_batch_size_per_expert", batch_size_per_expert)

        return x, expert_weights, expert_indices, batch_size_per_expert

    @abstractmethod
    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        raise NotImplementedError

    def apply_cp(self, cp_mesh: DeviceMesh):
        self.cp_mesh = cp_mesh


class MoELinearRouter(MoERouter):
    """
    A simple, learned, linear router.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        **kwargs,
    ):
        super().__init__(init_device=init_device, **kwargs)
        # NOTE: this parameter needs to have a large enough first dimension (which would be num experts)
        # in order to be sharded over big world sizes with FSDP. So we flatten it to a single dimension tensor.
        # And for that reason we don't support a 'bias' option.
        self.weight = nn.Parameter(
            torch.empty(self.num_experts * self.d_model, device=init_device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        nn.init.trunc_normal_(self.weight, std=0.02, a=-3 * 0.02, b=3 * 0.02)

    def extra_repr(self):
        return f"in_features={self.d_model}, num_experts={self.num_experts}"

    def get_expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, get_local_tensor(self.weight).view(self.num_experts, self.d_model))

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        del float8_enabled
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Shard(1),),
                use_local_output=True,
            ),
        )

        self.register_parameter(
            "weight", nn.Parameter(distribute_tensor(self.weight, tp_mesh, [Replicate()]))
        )
