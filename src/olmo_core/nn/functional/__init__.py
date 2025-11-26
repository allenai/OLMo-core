"""
Common ``nn`` function implementations.
"""

import torch

from .cross_entropy_loss import *
from .cross_entropy_loss import cce_loss

__all__ = [
    "cross_entropy_loss",
    "fused_linear_cross_entropy_loss",
    "l2_normalize",
    "cce_loss",
]


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # NOTE: could also use F.normalize(), but that doesn't work with DTensor at the moment.
    return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32).type_as(x)
