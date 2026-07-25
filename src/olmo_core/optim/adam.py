from dataclasses import dataclass

import torch

from .config import OptimConfig


@OptimConfig.register("adam")
@dataclass
class AdamConfig(OptimConfig[torch.optim.Adam]):
    """
    Configuration class for building an :class:`torch.optim.Adam` optimizer.
    """

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    foreach: bool | None = None
    fused: bool | None = None

    @classmethod
    def optimizer(cls) -> type[torch.optim.Adam]:
        return torch.optim.Adam
