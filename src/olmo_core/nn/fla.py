import logging
from dataclasses import dataclass, field
from typing import Optional

import fla.layers
import torch
from torch import nn

from olmo_core.config import Config, DType

log = logging.getLogger(__name__)


class FLA(nn.Module):
    def __init__(self, inner: fla.layers.ABCAttention):
        super().__init__()
        self.inner = inner

        self.kv_cache_manager = None

    def init_kv_cache_manager(self, batch_size: int):
        self.kv_cache_manager = FLACacheManager()

    def forward(self, x: torch.Tensor, **_kwargs) -> torch.Tensor:
        # FIXME: Right now we just ignore the kwargs.

        if self.kv_cache_manager is not None:
            # Use the cache manager with past_key_values API
            cache = self.kv_cache_manager.cache

            # Call the inner FLA layer with cache
            out, _, new_cache = self.inner(x, past_key_values=cache, use_cache=True)

            # Update the cache manager
            self.kv_cache_manager.cache = new_cache

            return out
        else:
            return self.inner(x)[0]  # returns out, ?, cache


class FLACacheManager(nn.Module):
    def __init__(self):
        super().__init__()

        self.zero_cache()

    def zero_cache(self):
        from fla.models.utils import Cache
        # FLA layers manage their own cache through the Cache object
        # We just store a reference to it
        self.cache = Cache()

    def reallocate(self, batch_size: int):
        from fla.models.utils import Cache

        self.cache = Cache()

    def is_reusable(self, batch_size: int) -> bool:
        # FLA library doesn't provide a simple way to check cache compatibility
        # So we just recreate it to be safe
        return False

    def reset(self, batch_size: int):
        if self.is_reusable(batch_size):
            self.zero_cache()
        else:
            log.debug("Unreusable FLA cache, reallocating")
            self.reallocate(batch_size)


@dataclass
class FLAConfig(Config):
    name: str
    fla_layer_kwargs: dict = field(default_factory=dict)
    dtype: DType = DType.float32

    def build(self, d_model: int, n_heads: int, init_device) -> FLA:
        layer = getattr(fla.layers, self.name)(
            hidden_size=d_model,
            num_heads=n_heads,
            layer_idx=0, # for cache
            **self.fla_layer_kwargs,
        ).to(device=init_device, dtype=self.dtype.as_pt())

        return FLA(layer)

    def num_params(self):
        raise NotImplementedError()