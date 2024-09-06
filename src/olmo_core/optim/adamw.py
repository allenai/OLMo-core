from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .config import OptimConfig


@dataclass
class AdamWConfig(OptimConfig):  # NOTE: omagaconf doesn't like "OptimConfig[torch.optim.AdamW]"
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
        kwargs = self.as_dict()
        kwargs.pop("group_overrides")
        optim = torch.optim.AdamW(self.build_groups(model), **kwargs)
        for group in optim.param_groups:
            group.setdefault("initial_lr", self.lr)
        return optim
