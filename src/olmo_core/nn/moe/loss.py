from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import get_local_tensor

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

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

    def compute_metrics(
        self, total_bz: Union[int, float, torch.Tensor], reset: bool = True, **kwargs
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        del total_bz, reset, kwargs
        return {}

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class MoELoadBalancingLossGranularity(StrEnum):
    local_batch = "local_batch"
    instance = "instance"


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
        from olmo_core.train.common import ReduceType

        self.loss_weight = loss_weight
        self.num_experts = num_experts
        self.top_k = top_k
        self.granularity = granularity
        self.loss: Optional[torch.Tensor] = None
        self.detached_loss: Optional[torch.Tensor] = None
        self.reduction = ReduceType.mean

    def update(
        self,
        *,
        expert_scores: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
        batched_batch_size_per_expert: torch.Tensor,
        **kwargs,
    ):
        del kwargs

        loss: torch.Tensor
        if self.granularity == MoELoadBalancingLossGranularity.instance:
            B = batched_batch_size_per_expert.shape[0]
            # shape: (B, num_experts)
            batched_batch_size_per_expert = batched_batch_size_per_expert.type_as(expert_scores)
            if self.sp_mesh is not None:
                dist.all_reduce(batched_batch_size_per_expert, group=self.sp_mesh.get_group())
                batched_batch_size_per_expert = DTensor.from_local(
                    batched_batch_size_per_expert, self.sp_mesh, (Replicate(),)
                )
                # NOTE: assumes equal sequence splits across group
                # shape: (B * S, num_experts) -> (B, S, num_experts,) -> (B, 1, num_experts)
                expert_scores = expert_scores.view(B, -1, self.num_experts).mean(
                    dim=1, keepdim=True
                )
                # shape: (B, 1, num_experts) -> (B, num_experts)
                expert_scores = DTensor.from_local(expert_scores, self.sp_mesh, (Shard(1),)).mean(
                    dim=1
                )
            else:
                # shape: (B * S, num_experts) -> (B, S, num_experts,) -> (B, num_experts)
                expert_scores = expert_scores.view(B, -1, self.num_experts).mean(dim=1)
            loss = (expert_scores * batched_batch_size_per_expert).sum()
        elif self.granularity == MoELoadBalancingLossGranularity.local_batch:
            # shape: (num_experts,)
            batch_size_per_expert = batch_size_per_expert.type_as(expert_scores)
            # shape: (B, S, num_experts) -> (B * S, num_experts)
            expert_scores = expert_scores.view(-1, self.num_experts)
            if self.sp_mesh is not None:
                # NOTE: torch.dot doesn't work on DTensor
                dist.all_reduce(batch_size_per_expert, group=self.sp_mesh.get_group())
                # NOTE: assumes equal sequence splits across group
                # shape: (B * S, num_experts) -> (1, num_experts)
                expert_scores = expert_scores.mean(dim=0, keepdim=True)
                # shape: (1, num_experts) -> (num_experts,)
                expert_scores = get_local_tensor(
                    DTensor.from_local(expert_scores, self.sp_mesh, (Shard(0),)).mean(dim=0)
                )
                loss = torch.dot(batch_size_per_expert, expert_scores)
                # Wrap back in DTensor.
                loss = DTensor.from_local(loss, self.sp_mesh, (Replicate(),))
            else:
                # shape: (B * S, num_experts) -> (num_experts,)
                expert_scores = expert_scores.mean(dim=0)
                loss = torch.dot(batch_size_per_expert, expert_scores)
        else:
            raise NotImplementedError(self.granularity)

        if self.loss is None:
            self.loss = loss
        else:
            self.loss += loss
        self.detached_loss = self.loss.detach()

    def compute(
        self, total_bz: Union[int, float, torch.Tensor], reset: bool = True, **kwargs
    ) -> Dict[str, torch.Tensor]:
        del kwargs

        if self.loss is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}.update()' needs to be called before '.compute()'"
            )

        scale = (self.num_experts * self.loss_weight) / (total_bz * self.top_k)
        lb_loss = scale * self.loss

        if reset:
            self.loss = None

        return {"load balancing loss": lb_loss}

    def compute_metrics(
        self, total_bz: Union[int, float, torch.Tensor], reset: bool = True, **kwargs
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        del kwargs

        if self.detached_loss is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}.update()' needs to be called before '.compute_metrics()'"
            )

        scale = self.num_experts / (total_bz * self.top_k)
        lb_loss = scale * self.detached_loss

        if reset:
            self.detached_loss = None

        return {"load balancing loss (unscaled)": (lb_loss, self.reduction)}

    def reset(self):
        self.loss = None
        self.detached_loss = None


class MoERouterZLoss(MoELoss):
    def __init__(self, *, loss_weight: float, num_experts: int):
        from olmo_core.train.common import ReduceType

        super().__init__()
        self.loss_weight = loss_weight
        self.num_experts = num_experts
        self.loss: Optional[torch.Tensor] = None
        self.detached_loss: Optional[torch.Tensor] = None
        self.reduction = ReduceType.mean

    def update(self, *, expert_logits: torch.Tensor, **kwargs):
        del kwargs
        loss = torch.logsumexp(expert_logits, dim=-1).square().sum()
        if self.loss is None:
            self.loss = loss
        else:
            self.loss += loss
        self.detached_loss = self.loss.detach()

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
            self.loss = None

        return {"router Z loss": lb_z_loss}

    def compute_metrics(
        self, total_bz: Union[int, float, torch.Tensor], reset: bool = True, **kwargs
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        del kwargs

        if self.detached_loss is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}.update()' needs to be called before '.compute_metrics()'"
            )

        lb_z_loss = self.detached_loss / total_bz

        if reset:
            self.detached_loss = None

        return {"router Z loss (unscaled)": (lb_z_loss, self.reduction)}

    def reset(self):
        self.loss = None
        self.detached_loss = None
