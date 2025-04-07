from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import get_local_tensor

__all__ = ["MoELoss", "MoELoadBalancingLoss", "MoERouterZLoss", "MoELoadBalancingLossGranularity"]


class MoELoss(metaclass=ABCMeta):
    def __init__(self):
        self.group: Optional[dist.ProcessGroup] = None  # usually the data parallel group
        self.sp_mesh: Optional[dist.DeviceMesh] = None  # the sequence parallel mesh

    @abstractmethod
    def update(
        self,
        *,
        expert_logits: torch.Tensor,
        expert_scores: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
        batched_batch_size_per_expert: torch.Tensor,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def compute(
        self, total_bz: Union[int, float, torch.Tensor], reset: bool = True, **kwargs
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class MoELoadBalancingLossGranularity(StrEnum):
    local_batch = "local_batch"
    instance = "instance"


def load_balancing_loss(
    *,
    num_experts: int,
    top_k: int,
    expert_scores: torch.Tensor,
    batch_size_per_expert: torch.Tensor,
    batched_batch_size_per_expert: torch.Tensor,
    granularity: MoELoadBalancingLossGranularity,
    sp_mesh: Optional[dist.DeviceMesh] = None,
) -> torch.Tensor:
    loss: torch.Tensor
    if granularity == MoELoadBalancingLossGranularity.instance:
        B = batched_batch_size_per_expert.shape[0]
        # shape: (B, num_experts)
        batched_batch_size_per_expert = batched_batch_size_per_expert.type_as(expert_scores)
        if sp_mesh is not None:
            dist.all_reduce(batched_batch_size_per_expert, group=sp_mesh.get_group())
            batched_batch_size_per_expert = DTensor.from_local(
                batched_batch_size_per_expert, sp_mesh, (Replicate(),)
            )
            # NOTE: assumes equal sequence splits across group
            # shape: (B * S, num_experts) -> (B, S, num_experts,) -> (B, 1, num_experts)
            expert_scores = expert_scores.view(B, -1, num_experts).sum(dim=1, keepdim=True)
            # shape: (B, 1, num_experts) -> (B, num_experts)
            expert_scores = DTensor.from_local(expert_scores, sp_mesh, (Shard(1),)).sum(dim=1)
        else:
            # shape: (B * S, num_experts) -> (B, S, num_experts,) -> (B, num_experts)
            expert_scores = expert_scores.view(B, -1, num_experts).sum(dim=1)
        loss = (expert_scores * batched_batch_size_per_expert).sum()
    elif granularity == MoELoadBalancingLossGranularity.local_batch:
        # shape: (num_experts,)
        batch_size_per_expert = batch_size_per_expert.type_as(expert_scores)
        # shape: (B, S, num_experts) -> (B * S, num_experts)
        expert_scores = expert_scores.view(-1, num_experts)
        if sp_mesh is not None:
            # NOTE: torch.dot doesn't work on DTensor
            dist.all_reduce(batch_size_per_expert, group=sp_mesh.get_group())
            # NOTE: assumes equal sequence splits across group
            # shape: (B * S, num_experts) -> (1, num_experts)
            expert_scores = expert_scores.sum(dim=0, keepdim=True)
            # shape: (1, num_experts) -> (num_experts,)
            expert_scores = get_local_tensor(
                DTensor.from_local(expert_scores, sp_mesh, (Shard(0),)).sum(dim=0)
            )
            loss = torch.dot(batch_size_per_expert, expert_scores)
            # Wrap back in DTensor.
            loss = DTensor.from_local(loss, sp_mesh, (Replicate(),))
        else:
            # shape: (B * S, num_experts) -> (num_experts,)
            expert_scores = expert_scores.sum(dim=0)
            loss = torch.dot(batch_size_per_expert, expert_scores)
    else:
        raise NotImplementedError(granularity)

    scale = num_experts / top_k

    return scale * loss


class MoELoadBalancingLoss(MoELoss):
    """
    Implements the load balancing loss from Switch Transformers.
    """

    def __init__(
        self,
        *,
        loss_weight: float,
        num_experts: int,
        top_k: int,
        granularity: MoELoadBalancingLossGranularity = MoELoadBalancingLossGranularity.local_batch,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.num_experts = num_experts
        self.top_k = top_k
        self.granularity = granularity
        self.loss: Optional[torch.Tensor] = None

    def update(
        self,
        *,
        expert_scores: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
        batched_batch_size_per_expert: torch.Tensor,
        **kwargs,
    ):
        del kwargs

        loss = load_balancing_loss(
            num_experts=self.num_experts,
            top_k=self.top_k,
            expert_scores=expert_scores,
            batch_size_per_expert=batch_size_per_expert,
            batched_batch_size_per_expert=batched_batch_size_per_expert,
            granularity=self.granularity,
            sp_mesh=self.sp_mesh,
        )

        if self.loss is None:
            self.loss = loss
        else:
            self.loss += loss

    def compute(
        self, total_bz: Union[int, float, torch.Tensor], reset: bool = True, **kwargs
    ) -> Dict[str, torch.Tensor]:
        del kwargs

        if self.loss is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}.update()' needs to be called before '.compute()'"
            )

        scale = self.loss_weight / total_bz
        lb_loss = scale * self.loss

        if reset:
            self.reset()

        return {
            "load balancing loss": lb_loss,
            "load balancing loss (unscaled)": lb_loss.detach().float() / self.loss_weight,
        }

    def reset(self):
        self.loss = None


def router_z_loss(*, expert_logits: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(expert_logits, dim=-1).square().sum()


class MoERouterZLoss(MoELoss):
    def __init__(self, *, loss_weight: float, num_experts: int):
        super().__init__()
        self.loss_weight = loss_weight
        self.num_experts = num_experts
        self.loss: Optional[torch.Tensor] = None

    def update(self, *, expert_logits: torch.Tensor, **kwargs):
        del kwargs
        loss = router_z_loss(expert_logits=expert_logits)
        if self.loss is None:
            self.loss = loss
        else:
            self.loss += loss

    def compute(
        self, total_bz: Union[int, float, torch.Tensor], reset: bool = True, **kwargs
    ) -> Dict[str, torch.Tensor]:
        del kwargs
        if self.loss is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}.update()' needs to be called before '.compute()'"
            )

        scale = self.loss_weight / total_bz
        lb_z_loss = scale * self.loss

        if reset:
            self.reset()

        return {
            "router Z loss": lb_z_loss,
            "router Z loss (unscaled)": lb_z_loss.detach().float() / self.loss_weight,
        }

    def reset(self):
        self.loss = None
