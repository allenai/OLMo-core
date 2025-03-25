from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Union

import torch

__all__ = ["MoELoss", "MoELoadBalancingLoss", "MoERouterZLoss"]


class MoELoss(metaclass=ABCMeta):
    @abstractmethod
    def update(
        self,
        *,
        expert_logits: torch.Tensor,
        expert_scores: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
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


class MoELoadBalancingLoss(MoELoss):
    """
    Implements the load balancing loss from Switch Transformers.
    """

    def __init__(self, *, loss_weight: float, num_experts: int, top_k: int):
        self.loss_weight = loss_weight
        self.num_experts = num_experts
        self.top_k = top_k
        self.loss: Optional[torch.Tensor] = None

    def update(
        self,
        *,
        expert_scores: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
        **kwargs,
    ):
        del kwargs
        # shape: (batch_size, num_experts) -> (num_experts,)
        expert_scores = expert_scores.mean(dim=0)
        loss = torch.dot(batch_size_per_expert.type_as(expert_scores), expert_scores)
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


class MoERouterZLoss(MoELoss):
    def __init__(self, *, loss_weight: float, num_experts: int):
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
