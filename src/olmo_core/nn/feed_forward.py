import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config, DType, StrEnum
from ..exceptions import OLMoConfigurationError
from .functional import l2_normalize

__all__ = ["FeedForwardConfig", "FeedForwardType", "FeedForward", "NormalizedFeedForward"]


class FeedForwardType(StrEnum):
    """
    An enumeration of the different feed-forward / MLP implementations.
    """

    default = "default"
    """
    :class:`FeedForward`.
    """

    normalized = "normalized"
    """
    :class:`NormalizedFeedForward`.
    """


@dataclass
class FeedForwardConfig(Config):
    """
    A config for building :class:`FeedForward` modules.

    See :class:`FeedForward` for parameter descriptions.
    """

    hidden_size: int
    name: FeedForwardType = FeedForwardType.default
    bias: bool = True
    dtype: DType = DType.float32

    def build(self, d_model: int, init_device: str = "cpu") -> "FeedForward":
        if self.name == FeedForwardType.default:
            return FeedForward(
                d_model=d_model,
                hidden_size=self.hidden_size,
                bias=self.bias,
                dtype=self.dtype.as_pt(),
                init_device=init_device,
            )
        else:
            if self.bias:
                raise OLMoConfigurationError(f"'bias' is invalid for '{self.name}' feed-forward")
            return NormalizedFeedForward(
                d_model=d_model,
                hidden_size=self.hidden_size,
                dtype=self.dtype.as_pt(),
                init_device=init_device,
            )


class FeedForward(nn.Module):
    """
    Basic feed-forward module with SwiGLU activation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(d_model, hidden_size, bias=bias, dtype=dtype, device=init_device)
        self.w2 = nn.Linear(hidden_size, d_model, bias=bias, dtype=dtype, device=init_device)
        self.w3 = nn.Linear(d_model, hidden_size, bias=bias, dtype=dtype, device=init_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the feed-forward on the input ``x``.

        :param x: The input of shape ``(*, d_model)``.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class NormalizedFeedForward(FeedForward):
    """
    An nGPT feed-forward implementation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__(
            d_model=d_model,
            hidden_size=hidden_size,
            dtype=dtype,
            init_device=init_device,
            bias=False,
        )
        self.sw_init_value = 1.0
        self.sw_init_scaling = 1.0
        self.sw1 = torch.nn.Parameter(
            self.sw_init_scaling * torch.ones(hidden_size, dtype=dtype, device=init_device)
        )
        self.sw3 = torch.nn.Parameter(
            self.sw_init_scaling * torch.ones(hidden_size, dtype=dtype, device=init_device)
        )
        self.sqrt_d_model = math.sqrt(d_model)

    def reset_parameters(self):
        nn.init.ones_(self.sw1)
        self.sw1.mul_(self.sw_init_scaling)
        nn.init.ones_(self.sw3)
        self.sw3.mul_(self.sw_init_scaling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sw1 = self.sw1 * ((self.sw_init_value / self.sw_init_scaling) * self.sqrt_d_model)
        sw3 = self.sw3 * (self.sw_init_value / self.sw_init_scaling)
        return self.w2(F.silu(sw1 * self.w1(x)) * (sw3 * self.w3(x)))

    @torch.no_grad()
    def normalize_matrices(self):
        """
        Normalize the weights in all matrices. This should be called after each optimizer step, which
        the :class:`~olmo_core.train.callbacks.MatrixNormalizerCallback` will handle for you.
        """
        self._normalize_matrix(self.w1.weight)
        self._normalize_matrix(self.w2.weight, dim=0)
        self._normalize_matrix(self.w3.weight)

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(l2_normalize(w, dim=dim))
