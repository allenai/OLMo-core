from dataclasses import dataclass
from typing import Optional, Tuple, Type

import torch

from ..nn.mup import MuPOptimizerType
from .config import OptimConfig


@dataclass
class AdamConfig(OptimConfig):  # NOTE: omagaconf doesn't like "OptimConfig[torch.optim.AdamW]"
    """
    Configuration class for building an :class:`torch.optim.Adam` optimizer.
    """

    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    foreach: Optional[bool] = None
    fused: Optional[bool] = None

    @classmethod
    def mup_optimizer_type(cls) -> Optional[MuPOptimizerType]:
        return MuPOptimizerType.adam

    @classmethod
    def optimizer(cls) -> Type[torch.optim.Adam]:
        return torch.optim.Adam
