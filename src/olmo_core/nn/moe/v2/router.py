  
import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module

import olmo_core.ops.moe as ops
from olmo_core.config import Config, DType, StrEnum
from olmo_core.distributed.utils import (
    _HiddenTensor,
    distribute_like,
    get_local_tensor,
    hide_from_torch,
    is_distributed,
    unhide_from_torch,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import get_default_device

from ..loss import MoELoadBalancingLossGranularity, load_balancing_loss, router_z_loss
from ..router import MoERouterGatingFunction, _uniform_expert_assignment

import nvtx
if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType
    
    

@dataclass
class MoERouterConfigV2(Config):
    """
    A configuration class for easily building any of the different MoE router modules.
    """

    d_model: int
    
    num_experts: int

    top_k: int
    jitter_eps: Optional[float] = None
    normalize_expert_weights: Optional[float] = None
    uniform_expert_assignment: bool = False
    bias_gamma: Optional[float] = None
    gating_function: MoERouterGatingFunction = MoERouterGatingFunction.softmax
    dtype: Optional[DType] = None
    record_routing_batch_size: bool = False
    lb_loss_weight: Optional[float] = None
    lb_loss_granularity: MoELoadBalancingLossGranularity = MoELoadBalancingLossGranularity.local_batch
    z_loss_weight: Optional[float] = None
    orth_loss_weight: Optional[float] = None
    record_routing_batch_size: bool = False
    restore_weight_scale: bool = False # if True, multiply the router weights by topK so that the scores have similar scale as dense models.
    
    def num_params(self) -> int:
        """
        The number of params that the module will have once built.

        :param d_model: The model dimensionality.
        """
        num_params = 0

        num_params += self.d_model * self.num_experts

        return num_params

    def build(
        self,
        init_device: str = "cpu",
    ) -> "MoERouterV2":
        """
        Build the corresponding MoE router module.

        :param d_model: The model dimensionality.
        :param num_experts: The number of experts.
        :param init_device: The device initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)

        if self.dtype is not None:
            kwargs["dtype"] = self.dtype.as_pt()

        return MoERouterV2(**kwargs, init_device=init_device)
        
class MoERouterV2(nn.Module):
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
        top_k: int,
        jitter_eps: Optional[float] = None,
        normalize_expert_weights: Optional[float] = None,
        uniform_expert_assignment: bool = False,
        bias_gamma: Optional[float] = None,
        gating_function: MoERouterGatingFunction = MoERouterGatingFunction.softmax,
        lb_loss_weight: Optional[float] = None,
        lb_loss_granularity: MoELoadBalancingLossGranularity = MoELoadBalancingLossGranularity.local_batch,
        z_loss_weight: Optional[float] = None,
        orth_loss_weight: Optional[float] = None,
        init_device: str = "cpu",
        record_routing_batch_size: bool = False,
        restore_weight_scale: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
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
        self.orth_loss_weight = orth_loss_weight
        self.group: Optional[dist.ProcessGroup] = None
        self.cp_mesh: Optional[DeviceMesh] = None
        self.tp_mesh: Optional[DeviceMesh] = None
        self.record_routing_batch_size = record_routing_batch_size
        self.restore_weight_scale = restore_weight_scale


        if self.bias_gamma is not None:
            assert self.bias_gamma > 0
            self.register_buffer("score_bias", torch.zeros(self.num_experts, device=init_device))
        else:
            self.register_buffer("score_bias", None)

        # NOTE: we don't use buffers for t hese because we don't want FSDP to manage them, and we
        # don't use a BufferCache because `torch.compile()` doesn't handle that well when we're modifying
        # values in the cache.
        self._batch_size_per_expert = hide_from_torch(
            torch.zeros(self.num_experts, device=init_device)
        )
        self._score_bias_batch_size_per_expert: Optional[_HiddenTensor] = None
        self._load_balancing_loss: Optional[_HiddenTensor] = None
        self._z_loss: Optional[_HiddenTensor] = None
        self._orth_loss: Optional[_HiddenTensor] = None
        
        self.weight = nn.Parameter(
            torch.empty(self.num_experts * self.d_model, device=init_device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        self._batch_size_per_expert = hide_from_torch(
            torch.zeros(self.num_experts, device=self.device)
        )

        if self.bias_gamma is not None:
            assert self.score_bias is not None
            score_bias = cast(torch.Tensor, self.score_bias)
            score_bias.zero_()
            self._score_bias_batch_size_per_expert = hide_from_torch(
                torch.zeros(self.num_experts, device=self.device)
            )

        if self.lb_loss_weight is not None:
            self._load_balancing_loss = hide_from_torch(torch.zeros([], device=self.device))

        if self.z_loss_weight is not None:
            self._z_loss = hide_from_torch(torch.zeros([], device=self.device))
            
        if self.orth_loss_weight is not None:
            self._orth_loss = hide_from_torch(torch.zeros([], device=self.device))
            
        nn.init.trunc_normal_(self.weight, std=0.02, a=-3 * 0.02, b=3 * 0.02)

    @property
    def device(self) -> torch.device:
        return self.weight.device if self.weight.device.type != "meta" else torch.device("cpu")

    def extra_repr(self):
        return f"in_features={self.d_model}, num_experts={self.num_experts}"

    @property
    def score_bias_batch_size_per_expert(self) -> Optional[torch.Tensor]:
        if self.bias_gamma is not None:
            if self._score_bias_batch_size_per_expert is None:
                self._score_bias_batch_size_per_expert = hide_from_torch(
                    torch.zeros(self.num_experts, device=self.device)
                )
            elif self._score_bias_batch_size_per_expert.device != self.device:
                self._score_bias_batch_size_per_expert = self._score_bias_batch_size_per_expert.to(
                    self.device
                )
        return (
            None
            if self._score_bias_batch_size_per_expert is None
            else unhide_from_torch(self._score_bias_batch_size_per_expert)
        )

    @score_bias_batch_size_per_expert.setter
    def score_bias_batch_size_per_expert(self, value: torch.Tensor):
        self._score_bias_batch_size_per_expert = hide_from_torch(value)

    @property
    def batch_size_per_expert(self) -> torch.Tensor:
        if self._batch_size_per_expert.device != self.device:
            self._batch_size_per_expert = self._batch_size_per_expert.to(self.device)
        return unhide_from_torch(self._batch_size_per_expert)

    @batch_size_per_expert.setter
    def batch_size_per_expert(self, value: torch.Tensor):
        self._batch_size_per_expert = hide_from_torch(value)

    @property
    def load_balancing_loss(self) -> Optional[torch.Tensor]:
        if self.lb_loss_weight is not None:
            if self._load_balancing_loss is None:
                self._load_balancing_loss = hide_from_torch(torch.zeros([], device=self.device))
            elif self._load_balancing_loss.device != self.device:
                self._load_balancing_loss = self._load_balancing_loss.to(self.device)
        return (
            None
            if self._load_balancing_loss is None
            else unhide_from_torch(self._load_balancing_loss)
        )

    @load_balancing_loss.setter
    def load_balancing_loss(self, value: torch.Tensor):
        self._load_balancing_loss = hide_from_torch(value)

    @property
    def z_loss(self) -> Optional[torch.Tensor]:
        if self.z_loss_weight is not None:
            if self._z_loss is None:
                self._z_loss = hide_from_torch(torch.zeros([], device=self.device))
            elif self._z_loss.device != self.device:
                self._z_loss = self._z_loss.to(self.device)
        return None if self._z_loss is None else unhide_from_torch(self._z_loss)

    @z_loss.setter
    def z_loss(self, value: torch.Tensor):
        self._z_loss = hide_from_torch(value)

    @property
    def orth_loss(self) -> Optional[torch.Tensor]:
        if self.orth_loss_weight is not None:
            if self._orth_loss is None:
                self._orth_loss = hide_from_torch(torch.zeros([], device=self.device))
            elif self._orth_loss.device != self.device:
                self._orth_loss = self._orth_loss.to(self.device)
        return None if self._orth_loss is None else unhide_from_torch(self._orth_loss)
    
    @orth_loss.setter
    def orth_loss(self, value: torch.Tensor):
        self._orth_loss = hide_from_torch(value)
        
    @torch.no_grad()
    def post_batch(self, dry_run: bool = False):
        if self.bias_gamma is None or not self.training:
            return

        assert self.score_bias is not None
        assert self.score_bias_batch_size_per_expert is not None
        score_bias = cast(torch.Tensor, self.score_bias)
        batch_size_per_expert = self.score_bias_batch_size_per_expert

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

    @nvtx.annotate("MoERouter.get_top_k", color='blue')
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

    def get_expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x.float(), get_local_tensor(self.weight).view(self.num_experts, self.d_model).float()
        )

    @torch.no_grad()
    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        from olmo_core.train.common import ReduceType

        out: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]] = {}

        # Load imbalance.
        batch_size_per_expert = self.batch_size_per_expert
        out["load imbalance"] = (
            batch_size_per_expert.max() / batch_size_per_expert.mean(dtype=torch.float),
            ReduceType.max,
        )
        
        # record the number of tokens routed to each routed expert.
        if self.record_routing_batch_size:
            for i in range(self.num_experts):
                out[f"expert {i} assigned tokens"] = (
                    self.batch_size_per_expert[i],
                    ReduceType.mean,
                )

        # Load balancing loss.
        if self.lb_loss_weight is not None:
            assert self.load_balancing_loss is not None
            out["load balancing loss"] = (
                self.lb_loss_weight * self.load_balancing_loss,
                ReduceType.mean,
            )
            out["load balancing loss unscaled"] = (
                self.load_balancing_loss.clone(),
                ReduceType.mean,
            )

        # Router Z loss.
        if self.z_loss_weight is not None:
            assert self.z_loss is not None
            out["router Z loss"] = (self.z_loss_weight * self.z_loss, ReduceType.mean)
            out["router Z loss unscaled"] = (self.z_loss.clone(), ReduceType.mean)

        if self.orth_loss_weight is not None:
            assert self.orth_loss is not None
            out["router orthogonal loss"] = (self.orth_loss_weight * self.orth_loss, ReduceType.mean)
            out["router orthogonal loss unscaled"] = (self.orth_loss.clone(), ReduceType.mean)

        if reset:
            self.reset_metrics()

        return out

    def reset_metrics(self):
        if (bz_per_expert := self.batch_size_per_expert) is not None:
            bz_per_expert.zero_()
        if (lb_loss := self.load_balancing_loss) is not None:
            lb_loss.zero_()
        if (z_loss := self.z_loss) is not None:
            z_loss.zero_()
        if (orth_loss := self.orth_loss) is not None:
            orth_loss.zero_()

    @nvtx.annotate("MoERouter.forward", color='blue')
    def forward(
        self,
        x: torch.Tensor,
        scores_only: bool,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    ) -> Tuple[torch.Tensor,  Optional[torch.Tensor],  Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Given the input ``x`` of shape ``(B, S, d_model)``, compute the experts assignment.

        :returns: The expert weights of shape ``(B, S, top_k)``,
            the expert indices of shape ``(B, S, top_k)``,
            the total number of items routed to each expert, with shape ``(num_experts,)``,
            and optionally the auxiliary losses.
        """
        # shape: (batch_size, seq_len, d_model)
        x = self.jitter(x)

        # shape: (batch_size, seq_len, num_experts)
        logits = self.get_expert_logits(x).float()

        # shape: (batch_size, seq_len, num_experts)
        if self.gating_function == MoERouterGatingFunction.softmax:
            scores = logits.softmax(dim=-1)
        elif self.gating_function == MoERouterGatingFunction.sigmoid:
            scores = F.sigmoid(logits)
            # to avoid NaNs in the load balancing loss
            # if all logits of a token are very negative for all experts, sigmoid gives 0 for all experts, causing NaNs when we div by the sum.
            scores = scores + 1e-7  
        else:
            raise NotImplementedError(self.gating_function)

        if scores_only:
            if self.normalize_expert_weights is not None:
                scores = scores.div(
                    torch.norm(
                        scores,
                        p=self.normalize_expert_weights,
                        dim=-1,
                        keepdim=True,
                    )
                )
            # If we only need the scores, return them directly.
            return scores, None, None, None

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
        
        # if used together normalize_expert_weights=True, 
        # then the weights for each token sum to TOP_K instead of 1.0
        if self.restore_weight_scale:
            expert_weights = expert_weights * self.top_k

        with torch.no_grad():
            # Histogram the expert ids to identify the number of items/tokens routed to each expert.
            # shape: (batch_size, seq_len, num_experts)
            batched_batch_size_per_expert = ops.batched_histc(expert_indices, self.num_experts)
            # shape: (batch_size, num_experts)
            batched_batch_size_per_expert = batched_batch_size_per_expert.sum(dim=1)
            # shape: (num_experts,)
            batch_size_per_expert = batched_batch_size_per_expert.sum(dim=0)

        # Maybe compute auxiliary losses and accumulate metrics.
        aux_loss: Optional[torch.Tensor] = None
        if self.training and torch.is_grad_enabled():
            with torch.autocast(enabled=False, device_type=x.device.type):
                if self.lb_loss_weight is not None:
                    assert self.load_balancing_loss is not None

                    # Make sure scores are normalized, otherwise load balancing loss doesn't work well.
                    if self.gating_function == MoERouterGatingFunction.sigmoid:
                        scores = scores / scores.sum(dim=-1, keepdim=True)

                    lb_loss = load_balancing_loss(
                        num_experts=self.num_experts,
                        top_k=self.top_k,
                        expert_scores=scores,
                        batch_size_per_expert=batch_size_per_expert,
                        batched_batch_size_per_expert=batched_batch_size_per_expert,
                        granularity=self.lb_loss_granularity,
                        loss_div_factor=loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )
                    self.load_balancing_loss += lb_loss.detach()

                    scaled_lb_loss = self.lb_loss_weight * lb_loss
                    aux_loss = scaled_lb_loss

                if self.z_loss_weight is not None:
                    assert self.z_loss is not None

                    z_loss = router_z_loss(
                        expert_logits=logits,
                        loss_div_factor=loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )
                    self.z_loss += z_loss.detach()

                    scaled_z_loss = self.z_loss_weight * z_loss
                    aux_loss = scaled_z_loss if aux_loss is None else aux_loss + scaled_z_loss

                if self.orth_loss_weight is not None:
                    # TODO: should only compute orthogonal loss on the last micro batch because the loss is only computed on the router weights.
                    assert self.orth_loss is not None
                    
                    # NOTE: loss_div_factor is the total number of tokens in the global batch.
                    # orth_loss_div_factor  is approximately the number of micro batches in the global batch.
                    # since the orthogonal loss is computed `num_micro_batches` times (should have the same loss each time since it does not chagne wrt data), we need to scale it down.
                    if loss_div_factor is None: 
                        raise NotImplementedError(
                            "Orthogonal loss requires a loss_div_factor to be set."
                        )
                    # orth_loss_factor = (logits.size(0) * logits.size(1)) / loss_div_factor  # --> divide by num_micro_batches
                    # or 
                    orth_loss_factor = 1 / loss_div_factor  # --> divide by num_tokens in micro batch
                    
                    orth_loss = self.compute_orthogonal_loss() * orth_loss_factor
                    
                    self.orth_loss += orth_loss.detach()

                    scaled_orth_loss = self.orth_loss_weight * orth_loss
                    aux_loss = (
                        scaled_orth_loss if aux_loss is None else aux_loss + scaled_orth_loss
                    )
            self.batch_size_per_expert += batch_size_per_expert
            if self.bias_gamma is not None:
                assert self.score_bias_batch_size_per_expert is not None
                self.score_bias_batch_size_per_expert += batch_size_per_expert

        return expert_weights, expert_indices, batch_size_per_expert, aux_loss

    def compute_orthogonal_loss(self) -> torch.Tensor:
        """
        Computes the orthogonal loss for the router.
        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError("Orthogonal loss computation is not implemented.")

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
        self.tp_mesh = tp_mesh
        self.register_parameter(
            "weight", nn.Parameter(distribute_tensor(self.weight, tp_mesh, [Replicate()]))
        )

    def apply_cp(self, cp_mesh: DeviceMesh):
        self.cp_mesh = cp_mesh

