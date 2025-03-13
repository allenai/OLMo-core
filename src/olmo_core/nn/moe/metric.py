from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import torch

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

__all__ = ["MoEMetric", "MoELoadImbalanceMetric"]


class MoEMetric(metaclass=ABCMeta):
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
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class MoELoadImbalanceMetric(MoEMetric):
    def __init__(self, *, num_experts: int, top_k: int):
        from olmo_core.train.common import ReduceType

        self.num_experts = num_experts
        self.top_k = top_k
        self.batch_size_per_expert: Optional[torch.Tensor] = None
        self.reduction = ReduceType.max

    @torch.no_grad()
    def update(
        self,
        *,
        batch_size_per_expert: torch.Tensor,
        **kwargs,
    ):
        del kwargs
        if self.batch_size_per_expert is None:
            self.batch_size_per_expert = torch.zeros_like(batch_size_per_expert)
        self.batch_size_per_expert += batch_size_per_expert.detach()

    @torch.no_grad()
    def compute(
        self, total_bz: Union[int, float, torch.Tensor], reset: bool = True, **kwargs
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        del kwargs
        if self.batch_size_per_expert is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}.update()' needs to be called before '.compute()'"
            )

        ideal_bz_per_expert = total_bz * (self.top_k / self.num_experts)
        load_imbalance = self.batch_size_per_expert.max() / ideal_bz_per_expert

        if reset:
            self.reset()

        return {"load imbalance": (load_imbalance, self.reduction)}

    def reset(self):
        self.batch_size_per_expert = None
