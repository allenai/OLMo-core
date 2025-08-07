from typing import Optional, Tuple
from types import SimpleNamespace
from dataclasses import dataclass
import gc
import logging

from mamba_ssm.modules.mamba2 import Mamba2
import torch

from olmo_core.config import Config

log = logging.getLogger(__name__)

class MambaWrapper(Mamba2):
    def allocate_kv_cache(
        self, batch_size: int, max_seq_len: int, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO(benjaminm): support / properly propagate dtype
        return self.allocate_inference_cache(batch_size, max_seq_len)

    def free_kv_cache(self):
        if hasattr(self, "conv_state") and self.conv_state is not None:
            del self.conv_state
        if hasattr(self, "ssm_state") and self.ssm_state is not None:
            del self.ssm_state
        if hasattr(self, "seqlen_offset") and self.seqlen_offset is not None:
            del self.seqlen_offset
        gc.collect()
        torch.cuda.empty_cache()

    def reset_kv_cache(
        self,
        use_cache: bool,
        batch_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        dtype: Optional[torch.dtype] = torch.float32,
    ) -> None:
        # Fast path when KV caching is disabled.
        if not use_cache:
            self.conv_state, self.ssm_state = None, None
            return
        
        if batch_size is None:
            raise ValueError("batch_size must be provided if use_cache is True")
        if max_seq_len is None:
            raise ValueError("max_seq_len must be provided if use_cache is True")
        if dtype is None:
            raise ValueError("dtype must be provided if use_cache is True")

        # Attempt to reuse an existing cache that satisfies the requested size and dtype requirements.
        if (
            hasattr(self, "conv_state")
            and hasattr(self, "ssm_state")
            and self.conv_state is not None
            and self.ssm_state is not None
            and self.conv_state.shape[0] == batch_size
            and self.conv_state.dtype == dtype
            and self.ssm_state.shape[0] == batch_size
            and self.ssm_state.dtype == dtype
        ):
            # The kv cache is reusable, so we just reset it.
            self.conv_state.zero_()
            self.ssm_state.zero_()
            return

        if (
            hasattr(self, "conv_state")
            and self.conv_state is not None
            and hasattr(self, "ssm_state")
            and self.ssm_state is not None
        ):
            # cache exists, why did reuse fail?
            reasons = []
            if self.conv_state.shape[0] != batch_size:
                reasons.append(f"batch_size mismatch: {self.conv_state.shape[0]} != {batch_size}")
            if self.ssm_state.dtype != dtype:
                reasons.append(f"dtype mismatch: {self.ssm_state.dtype} != {dtype}")
            if reasons:
                log.info(f"KV cache reuse failed: {', '.join(reasons)}")

        self.free_kv_cache()  # free the old cache to avoid OOMs
        self.conv_state, self.ssm_state = self.allocate_kv_cache(
            batch_size, max_seq_len, dtype
        )
        self.seqlen_offset = 0

    # from HNet repo
    def _step(self, hidden_states, inference_params):  # type: ignore
        # Don't use _get_states_from_cache because we want to assert that they exist
        conv_state, ssm_state = inference_params # init class of Mamba2 accepts layer_idx
        result, conv_state, ssm_state = super().step(
            hidden_states, conv_state, ssm_state
        )

        # Update the state cache in-place
        inference_params[0].copy_(conv_state)
        inference_params[1].copy_(ssm_state)
        self.seqlen_offset += 1
        return result

    def forward(self, x: torch.Tensor, prefill_kv_cache: bool = False):  # type: ignore
        has_cache = hasattr(self, "conv_state") and self.conv_state is not None and hasattr(self, "ssm_state") and self.ssm_state is not None

        if has_cache and prefill_kv_cache:
            # writes to cache inplace
            # Mamba2 excepts it in this format
            inference_params = SimpleNamespace(
                key_value_memory_dict={0: (self.conv_state, self.ssm_state)},
                seqlen_offset=self.seqlen_offset,
            )

            return super().forward(x, inference_params=inference_params)
        elif has_cache:
            return self._step(x, (self.conv_state, self.ssm_state))
        else:
            return super().forward(x)

@dataclass
class MambaConfig(Config):
    chunk_size: int
    d_conv: int
    d_state: int
    expand: int

    def build(self, d_model: int, init_device) -> MambaWrapper:
        return MambaWrapper(
            d_model,
            chunk_size=self.chunk_size,
            d_conv=self.d_conv,
            d_state=self.d_state,
            expand=self.expand,
            device=init_device,
            # we store cache separately per layer so no need to set this correctly
            # (otherwise used to distinguish between layer caches)
            layer_idx=0,
        )