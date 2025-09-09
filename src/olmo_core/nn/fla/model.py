from dataclasses import dataclass
from typing import Any, Dict
from olmo_core.config import Config

import torch
import fla.models
from transformers import AutoModelForCausalLM


class FLAModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        # FIXME: adapt inputs/outputs as needed
        return self.model(x)


@dataclass
class FLAModelConfig(Config):
    fla_model_name: str
    kwargs: Dict[str, Any]

    def build(self) -> FLAModel:
        config_cls = getattr(fla.models, self.fla_model_name + "Config", None)
        assert config_cls is not None, f"Unknown FLA model name: {self.fla_model_name}"
        config = config_cls(**self.kwargs)
        model = AutoModelForCausalLM.from_config(config)
        return FLAModel(model)
