from dataclasses import dataclass
from typing import Type

import torch
from muon import MuonWithAuxAdam

from .config import OptimConfig


@dataclass
class MuonWithAuxAdamConfig(OptimConfig):  # NOTE: omagaconf doesn't like "OptimConfig[torch.optim.AdamW]"
    """
    Configuration class for building an :class:`MuonWithAuxAdam` optimizer.
    See https://github.com/KellerJordan/Muon for more details.
    """

    lr: float = 1e-3
    momentum: float = 0.95
    weight_decay: float = 1e-2
    use_muon: bool = True

    @classmethod
    def optimizer(cls) -> Type[torch.optim.AdamW]:
        return MuonWithAuxAdam
