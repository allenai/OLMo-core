"""
Common ``nn`` function implementations.
"""

import torch

from .cross_entropy_loss import *
from .flash_attn_api import (
    dispatch_flash_attn,
    dispatch_flash_attn_qkvpacked,
    dispatch_ring_flash_attn,
    dispatch_ring_flash_attn_qkvpacked,
)

__all__ = [
    "cross_entropy_loss",
    "fused_cross_entropy_loss",
    "l2_normalize",
    "dispatch_flash_attn",
    "dispatch_flash_attn_qkvpacked",
    "dispatch_ring_flash_attn",
    "dispatch_ring_flash_attn_qkvpacked",
]


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # NOTE: could also use F.normalize(), but that doesn't work with DTensor at the moment.
    return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32).type_as(x)
