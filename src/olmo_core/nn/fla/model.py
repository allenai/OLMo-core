from dataclasses import dataclass
from typing import Any, Dict

import fla.models
import torch
from transformers import AutoModelForCausalLM

from olmo_core.config import Config


class FLAModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        # TODO: Run the model, ignoring the lm_head
        # The transformer part will do the lm_head
        return NotImplementedError()


@dataclass
class FLAModelConfig(Config):
    fla_model_name: str
    kwargs: Dict[str, Any]

    def build(self) -> FLAModel:
        config_cls = getattr(fla.models, self.fla_model_name + "Config", None)
        assert config_cls is not None, f"Unknown FLA model name: {self.fla_model_name}"
        config = config_cls(**self.kwargs)
        causal_lm = AutoModelForCausalLM.from_config(config)
        return FLAModel(causal_lm.model)
