"""
Common ``nn`` function implementations.
"""

import torch

from .cross_entropy_loss import *
from .flash_attention import (
    flash_attn,
    flash_attn_qkvpacked,
    flash_attn_varlen,
    flash_attn_varlen_qkvpacked,
    zigzag_ring_flash_attn,
    zigzag_ring_flash_attn_qkvpacked,
    zigzag_ring_flash_attn_varlen,
    zigzag_ring_flash_attn_varlen_qkvpacked,
)

__all__ = [
    "cross_entropy_loss",
    "fused_cross_entropy_loss",
    "l2_normalize",
    "flash_attn",
    "flash_attn_qkvpacked",
    "flash_attn_varlen",
    "flash_attn_varlen_qkvpacked",
    "zigzag_ring_flash_attn",
    "zigzag_ring_flash_attn_qkvpacked",
    "zigzag_ring_flash_attn_varlen",
    "zigzag_ring_flash_attn_varlen_qkvpacked",
]


def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # NOTE: could also use F.normalize(), but that doesn't work with DTensor at the moment.
    return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32).type_as(x)
