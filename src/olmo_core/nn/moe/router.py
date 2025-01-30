from abc import abstractmethod
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn


# NOTE: To enable end-to-end benchmarking without convergence we
# support a flag to force the router to assign tokens uniformly
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


class MoERouter(nn.Module):
    """
    A base class for MoE router modules.

    :param d_model: The model dimensionality (hidden size).
    :param num_experts: The total number of experts.
    :param top_k: The number of experts to assign to each token.
    :param jitter_eps: Controls the amount of noise added to the input during training.
    :param normalize_expert_weights: The type of norm (e.g. ``2.0`` for L2 norm) to use to normalize
        the expert weights.
    :param uniform_expert_assignment: Force uniform assignment. Useful for benchmarking.
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
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_eps = jitter_eps
        self.normalize_expert_weights = normalize_expert_weights
        self.uniform_expert_assignment = uniform_expert_assignment

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        if self.jitter_eps is None or not self.training:
            return x
        else:
            low = 1.0 - self.jitter_eps
            high = 1.0 + self.jitter_eps
            noise = torch.rand_like(x)
            return x * (low + noise * (high - low))

    def get_top_k(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.top_k == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, self.top_k, dim=-1)

    @abstractmethod
    def get_expert_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given the input ``x`` of shape ``(*, d_model)``, compute the expert scores.

        :returns: The expert scores, shape ``(*, num_experts)``.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given the input ``x`` of shape ``(batch_size, seq_len, d_model)``, compute the
        experts assignment.

        :returns: The scores of shape ``(batch_size, seq_len, num_experts)``, the expert weights
            of shape ``(batch_size, seq_len, top_k)``, and the expert indices of shape
            ``(batch_size, seq_len, top_k)``.
        """
        # shape: (batch_size, seq_len, d_model)
        x = self.jitter(x)

        # shape: (batch_size * seq_len, num_experts)
        scores = self.get_expert_scores(x.view(-1, self.d_model))

        # shape: (batch_size * seq_len, top_k)
        expert_weights, expert_indices = self.get_top_k(scores)

        if self.normalize_expert_weights is not None:
            expert_weights.div_(
                torch.norm(
                    expert_weights,
                    p=self.normalize_expert_weights,
                    dim=-1,
                    keepdim=True,
                )
            )

        if self.uniform_expert_assignment:
            expert_indices = _uniform_expert_assignment(expert_indices, self.num_experts)

        return scores, expert_weights, expert_indices


class MoELinearRouter(MoERouter):
    """
    A simple, learned, linear router.
    """

    def __init__(
        self,
        *,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.w_score = nn.Linear(
            self.d_model, self.num_experts, bias=bias, dtype=dtype, device=init_device
        )

    def get_expert_scores(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.w_score(x.view(-1, self.d_model))
        # TODO: save router logits for Z-loss
        return logits.softmax(dim=-1)
