from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config, DType

__all__ = ["FeedForwardConfig", "FeedForward"]


@dataclass
class FeedForwardConfig(Config):
    """
    A config for building :class:`FeedForward` modules.

    See :class:`FeedForward` for parameter descriptions.
    """

    hidden_size: int
    bias: bool = True
    dtype: DType = DType.float32

    def build(self, d_model: int, init_device: str = "cpu") -> "FeedForward":
        return FeedForward(
            d_model=d_model,
            hidden_size=self.hidden_size,
            bias=self.bias,
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
        self.w1 = nn.Linear(d_model, hidden_size, bias=bias, dtype=dtype, device=init_device)
        self.w2 = nn.Linear(hidden_size, d_model, bias=bias, dtype=dtype, device=init_device)
        self.w3 = nn.Linear(d_model, hidden_size, bias=bias, dtype=dtype, device=init_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the feed-forward on the input ``x``.

        :param x: The input of shape ``(*, d_model)``.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
