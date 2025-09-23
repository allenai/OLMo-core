from dataclasses import dataclass
import logging
import torch
from torch import nn
from dataclasses import field
from olmo_core.config import Config, DType

log = logging.getLogger(__name__)


class FLA(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

        self.kv_cache_manager = None

    def init_kv_cache_manager(self, batch_size: int):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor):
        if self.kv_cache_manager is not None and self.kv_cache_manager.current_position() == 0:
            raise NotImplementedError() # prefill
        elif self.kv_cache_manager is not None:
            raise NotImplementedError() # generate step
        else:
            return self.inner(x)[0] # returns out, ?, cache


@dataclass
class FLAConfig(Config):
    name: str
    fla_layer_kwargs: dict = field(default_factory=dict)
    dtype: DType = DType.float32

    def build(self, d_model: int, init_device) -> FLA:
        import fla.layers
        layer = getattr(fla.layers, self.name)(
            hidden_size=d_model,
            **self.fla_layer_kwargs,
        ).to(device=init_device, dtype=self.dtype.as_pt())

        return FLA(layer)

    def num_params(self):
        raise NotImplementedError()