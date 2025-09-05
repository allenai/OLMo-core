from dataclasses import dataclass
import logging
from types import SimpleNamespace
from mamba_ssm.modules.mamba2 import Mamba2 as _Mamba2
import torch
from torch import nn

from olmo_core.config import Config, DType

log = logging.getLogger(__name__)


class Mamba(_Mamba2):
    def __init__(self, *args, **kwargs):
        # layer_idx not None required to enable caching
        super().__init__(*args, **kwargs, layer_idx=0)  # type: ignore
        # not cast to dtype in _Mamba2.__init__
        self.D = nn.Parameter(self.D.to(self.conv1d.weight.dtype))

        self.mamba_cache_manager = None

    def init_mamba_cache_manager(self, batch_size: int):
        self.mamba_cache_manager = MambaCacheManager(
            self,
            batch_size,
            dtype=self.conv1d.weight.dtype,
        )

    def forward(self, x: torch.Tensor):  # type: ignore
        if self.mamba_cache_manager is not None and self.mamba_cache_manager.current_position() == 0:
            # writes to cache inplace
            # Mamba2 expects it in this format
            inference_params = SimpleNamespace(
                key_value_memory_dict={0: (self.mamba_cache_manager.conv_state, self.mamba_cache_manager.ssm_state)},
                seqlen_offset=self.mamba_cache_manager.current_position(),
            )
            out = super().forward(x, inference_params=inference_params)
            self.mamba_cache_manager.update_seqlen(x.shape[1])
            return out
        elif self.mamba_cache_manager is not None:
            return self.step(x, (self.mamba_cache_manager.conv_state, self.mamba_cache_manager.ssm_state))
        else:
            return super().forward(x)

    def step(self, hidden_states, inference_params):  # type: ignore
        assert self.mamba_cache_manager is not None

        conv_state, ssm_state = inference_params

        # Mamba2 'Only support[s] decoding with 1 token at a time for now' so we have to loop
        results = []
        for i in range(hidden_states.shape[1]):
            result, conv_state, ssm_state = super().step(
                hidden_states[:, [i]], conv_state, ssm_state
            )
            results.append(result)

        result = torch.cat(results, dim=1)

        # Update the state cache in-place
        inference_params[0].copy_(conv_state)
        inference_params[1].copy_(ssm_state)
        self.mamba_cache_manager.update_seqlen(hidden_states.shape[1])
        return result

class MambaCacheManager(nn.Module):
    def __init__(
        self,
        layer: Mamba,
        batch_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        self.layer = layer

        self.conv_state, self.ssm_state = layer.allocate_inference_cache(
            batch_size, max_seqlen=None, dtype=dtype # max seqlen not used
        )
        self.cache_seqlens = torch.zeros((), dtype=torch.int32, device=self.conv_state.device)

    def update_seqlen(self, seqlen: int):
        self.cache_seqlens.add_(seqlen)

    def current_position(self) -> torch.Tensor:
        return self.cache_seqlens

    def zero_cache(self):
        self.conv_state.zero_()
        self.ssm_state.zero_()

    def reallocate(self, batch_size: int):
        # TODO(benjaminm): support / properly propagate dtype
        self.conv_state, self.ssm_state = self.layer.allocate_inference_cache(
            batch_size, max_seqlen=None, dtype=self.conv_state.dtype
        )

    def is_reusable(self, batch_size: int) -> bool:
        return self.conv_state.shape[0] == batch_size and self.ssm_state.shape[0] >= batch_size

    def reset(self, batch_size: int):
        if self.is_reusable(batch_size):
            self.zero_cache()
        else:
            log.debug("Unreusable Mamba cache, reallocating")
            self.reallocate(batch_size)


@dataclass
class MambaConfig(Config):
    chunk_size: int
    d_conv: int
    d_state: int
    expand: int
    dtype: DType = DType.float32

    def build(self, d_model: int, init_device) -> Mamba:
        return Mamba(
            d_model,
            chunk_size=self.chunk_size,
            d_conv=self.d_conv,
            d_state=self.d_state,
            expand=self.expand,
            device=init_device,
            dtype=self.dtype.as_pt(),
        )