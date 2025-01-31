from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn

from ...config import Config, DType, StrEnum
from ...exceptions import OLMoConfigurationError

__all__ = ["MoERouter", "MoELinearRouter", "MoERouterConfig", "MoERouterType"]


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


@dataclass
class MoERouterConfig(Config):
    """
    A configuration class for easily building any of the different MoE router modules.
    """

    name: MoERouterType = MoERouterType.default
    """
    The name of the implementation.
    """
    num_experts: int = 1
    top_k: int = 1
    jitter_eps: Optional[float] = None
    normalize_expert_weights: Optional[float] = None
    uniform_expert_assignment: bool = False
    bias: bool = True
    dtype: DType = DType.float32

    def num_params(self, d_model: int) -> int:
        """
        The number of params that the module will have once built.

        :param d_model: The model dimensionality.
        """
        num_params = 0
        if self.name == MoERouterType.default:
            num_params += d_model * self.num_experts
            if self.bias:
                num_params += self.num_experts
        else:
            raise NotImplementedError

        return num_params

    def build(self, d_model: int, *, init_device: str = "cpu") -> "MoERouter":
        """
        Build the corresponding MoE router module.

        :param d_model: The model dimensionality.
        :param init_device: The device initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.update(
            dtype=kwargs.pop("dtype").as_pt(),
            d_model=d_model,
            init_device=init_device,
        )

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
    def get_expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given the input ``x`` of shape ``(*, d_model)``, compute the un-normalized expert scores.

        :returns: The expert logits, shape ``(*, num_experts)``.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given the input ``x`` of shape ``(*, d_model)``, compute the
        experts assignment.

        :returns: The logits of shape ``(N, num_experts)``, the expert weights
            of shape ``(N, top_k)``, and the expert indices of shape ``(N, top_k)``.
        """
        # shape: (batch_size, seq_len, d_model)
        x = self.jitter(x)

        # shape: (batch_size * seq_len, num_experts)
        logits = self.get_expert_logits(x.view(-1, self.d_model))

        # shape: (batch_size * seq_len, top_k)
        expert_weights, expert_indices = self.get_top_k(logits)

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

        return logits, expert_weights, expert_indices


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

    def get_expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_score(x.view(-1, self.d_model))
