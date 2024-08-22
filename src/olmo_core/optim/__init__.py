from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..config import Config
from .scheduler import ConstantScheduler, CosWithWarmup, Scheduler

__all__ = ["AdamWConfig", "Scheduler", "CosWithWarmup", "ConstantScheduler"]


@dataclass
class AdamWConfig(Config):
    """
    Configuration class for building an :class:`~torch.optim.AdamW` optimizer.
    """

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-2
    amsgrad: bool = False
    maximize: bool = False
    foreach: Optional[bool] = None
    capturable: bool = False
    differentiable: bool = False
    fused: Optional[bool] = None

    def build(self, model: nn.Module) -> torch.optim.AdamW:
        return torch.optim.AdamW(model.parameters(), **self.as_dict())
