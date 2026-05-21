"""
Compatibility module for old ``olmo_core.nn.fla.model`` config paths.
"""

from dataclasses import dataclass, field
from typing import Any, Dict

import torch

from olmo_core.config import Config


class FLAModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        return self.model(x)


@dataclass
class FLAModelConfig(Config):
    fla_model_name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def build(self) -> FLAModel:
        raise NotImplementedError("Legacy FLAModelConfig is only kept for config decoding")


__all__ = ["FLAModel", "FLAModelConfig"]
