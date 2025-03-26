from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Union

import torch
import torch.distributed as dist

from olmo_core.distributed.utils import is_distributed
from olmo_core.utils import move_to_device

__all__ = ["MoELoss", "MoELoadBalancingLoss", "MoERouterZLoss"]


class MoELoss(metaclass=ABCMeta):
    def __init__(self):
        self.pp_group: Optional[dist.ProcessGroup] = None

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

    def post_batch(self, dry_run: bool = False, training: bool = True):
        del dry_run, training


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
        min_loss_weight: Optional[float] = None,
        max_loss_weight: Optional[float] = None,
        target_load_imbalance: Optional[float] = None,
        loss_weight_delta: Optional[float] = None,
        instance_level: bool = False,
    ):
        super().__init__()

        if target_load_imbalance is not None:
            if min_loss_weight is None:
                min_loss_weight = loss_weight
            assert max_loss_weight is not None
            assert min_loss_weight <= loss_weight <= max_loss_weight
            if loss_weight_delta is None:
                loss_weight_delta = (max_loss_weight - min_loss_weight) / 50

        self.loss_weight = loss_weight
        self.min_loss_weight = min_loss_weight
        self.max_loss_weight = max_loss_weight
        self.target_load_imbalance = target_load_imbalance
        self.loss_weight_delta = loss_weight_delta
        self.num_experts = num_experts
        self.top_k = top_k
        self.instance_level = instance_level
        self.loss: Optional[torch.Tensor] = None
        self.batch_size_per_expert: Optional[torch.Tensor] = None

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
        if self.instance_level:
            B = batched_batch_size_per_expert.shape[0]
            # shape: (B * S, num_experts) -> (B, S, num_experts,) -> (B, num_experts)
            expert_scores = expert_scores.view(B, -1, self.num_experts).mean(dim=1)
            # shape: (B, num_experts)
            batched_batch_size_per_expert = batched_batch_size_per_expert.type_as(expert_scores)
            loss = (expert_scores * batched_batch_size_per_expert).sum()
        else:
            # shape: (B * S, num_experts) -> (num_experts,)
            expert_scores = expert_scores.mean(dim=0)
            batch_size_per_expert = batch_size_per_expert.type_as(expert_scores)
            loss = torch.dot(batch_size_per_expert, expert_scores)

        if self.loss is None:
            self.loss = loss
        else:
            self.loss += loss

        if self.target_load_imbalance is not None:
            if self.batch_size_per_expert is None:
                self.batch_size_per_expert = torch.zeros_like(batch_size_per_expert)
            self.batch_size_per_expert += batch_size_per_expert.detach()

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

    def post_batch(self, dry_run: bool = False, training: bool = True):
        if not training:
            return

        if self.target_load_imbalance is not None:
            # TODO: 'self.loss_weight' needs to be stored in checkpoint
            assert self.batch_size_per_expert is not None
            assert self.min_loss_weight is not None
            assert self.max_loss_weight is not None
            assert self.loss_weight_delta is not None
            if not isinstance(self.loss_weight, torch.Tensor):
                self.loss_weight = move_to_device(
                    torch.tensor(self.loss_weight), self.batch_size_per_expert.device
                )

            if is_distributed():
                dist.all_reduce(self.batch_size_per_expert, group=self.pp_group)

            ideal_batch_size_per_expert = self.batch_size_per_expert.mean(dtype=torch.float32)
            actual_load_imbalance = self.batch_size_per_expert.max() / ideal_batch_size_per_expert

            if not dry_run:
                self.loss_weight.add_(
                    self.loss_weight_delta
                    * (actual_load_imbalance - self.target_load_imbalance).sign()
                )
                torch.clamp_(self.loss_weight, min=self.min_loss_weight, max=self.max_loss_weight)

            self.batch_size_per_expert.zero_()


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
