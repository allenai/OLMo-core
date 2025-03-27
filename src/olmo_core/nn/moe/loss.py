from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import get_local_tensor

__all__ = ["MoELoss", "MoELoadBalancingLoss", "MoERouterZLoss", "MoELoadBalancingLossGranularity"]


class MoELoss(metaclass=ABCMeta):
    def __init__(self):
        self.group: Optional[dist.ProcessGroup] = None
        self.sp_mesh: Optional[dist.DeviceMesh] = None

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

    #  def post_batch(self, dry_run: bool = False, training: bool = True):
    #      del dry_run, training


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

        loss: torch.Tensor
        if self.granularity == MoELoadBalancingLossGranularity.instance:
            B = batched_batch_size_per_expert.shape[0]
            # shape: (B, num_experts)
            batched_batch_size_per_expert = batched_batch_size_per_expert.type_as(expert_scores)
            if self.sp_mesh is not None:
                dist.all_reduce(batched_batch_size_per_expert, group=self.sp_mesh.get_group())
                # NOTE: assumes equal sequence splits across group
                # shape: (B * S, num_experts) -> (B, S, num_experts,) -> (B, 1, num_experts)
                expert_scores = expert_scores.view(B, -1, self.num_experts).mean(
                    dim=1, keepdim=True
                )
                # shape: (B, 1, num_experts) -> (B, num_experts)
                expert_scores = get_local_tensor(
                    DTensor.from_local(expert_scores, self.sp_mesh, (Shard(1),)).mean(dim=1)
                )
            else:
                # shape: (B * S, num_experts) -> (B, S, num_experts,) -> (B, num_experts)
                expert_scores = expert_scores.view(B, -1, self.num_experts).mean(dim=1)
            loss = (expert_scores * batched_batch_size_per_expert).sum()
        elif self.granularity == MoELoadBalancingLossGranularity.local_batch:
            # shape: (num_experts,)
            batch_size_per_expert = batch_size_per_expert.type_as(expert_scores)
            # shape: (B * S, num_experts) -> (num_experts,)
            expert_scores = expert_scores.mean(dim=0)
            loss = torch.dot(batch_size_per_expert, expert_scores)
        else:
            raise NotImplementedError(self.granularity)

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

        scale = (self.num_experts * self.loss_weight) / (total_bz * self.top_k)
        lb_loss = scale * self.loss

        if reset:
            self.reset()

        return {"load balancing loss": lb_loss}

    def reset(self):
        self.loss = None

    #  def post_batch(self, dry_run: bool = False, training: bool = True):
    #      if not training:
    #          return

    #      if self.target_load_imbalance is not None:
    #          # TODO: 'self.loss_weight' needs to be stored in checkpoint
    #          assert self.batch_size_per_expert is not None
    #          assert self.min_loss_weight is not None
    #          assert self.max_loss_weight is not None
    #          assert self.loss_weight_delta is not None
    #          if not isinstance(self.loss_weight, torch.Tensor):
    #              self.loss_weight = move_to_device(
    #                  torch.tensor(self.loss_weight), self.batch_size_per_expert.device
    #              )

    #          if is_distributed():
    #              dist.all_reduce(self.batch_size_per_expert, group=self.group)

    #          ideal_batch_size_per_expert = self.batch_size_per_expert.mean(dtype=torch.float32)
    #          actual_load_imbalance = self.batch_size_per_expert.max() / ideal_batch_size_per_expert

    #          if not dry_run:
    #              self.loss_weight.add_(
    #                  self.loss_weight_delta
    #                  * (actual_load_imbalance - self.target_load_imbalance).sign()
    #              )
    #              torch.clamp_(self.loss_weight, min=self.min_loss_weight, max=self.max_loss_weight)

    #          self.batch_size_per_expert.zero_()


class MoERouterZLoss(MoELoss):
    def __init__(self, *, loss_weight: float, num_experts: int):
        super().__init__()
        self.loss_weight = loss_weight
        self.num_experts = num_experts
        self.loss: Optional[torch.Tensor] = None

    def update(self, *, expert_logits: torch.Tensor, **kwargs):
        del kwargs
        loss = torch.logsumexp(expert_logits, dim=-1).square().sum()
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
        return {"router Z loss": lb_loss}

    def reset(self):
        self.loss = None
