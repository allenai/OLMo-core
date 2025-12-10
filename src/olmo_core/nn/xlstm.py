from dataclasses import dataclass
import logging
from typing import Optional, cast
import torch
from torch import nn
from xlstm.xlstm_large.model import mLSTMLayer, mLSTMLayerConfig, mLSTMLayerStateType, soft_cap, mLSTMBackendConfig

from olmo_core.config import Config, DType
from olmo_core.nn.bolmo.utils import MaskState

log = logging.getLogger(__name__)


class XLSTM(mLSTMLayer):
    def __init__(self, config):
        super().__init__(config)  # type: ignore

        self.xlstm_cache_manager = None

    def init_xlstm_cache_manager(self, batch_size: int):
        self.xlstm_cache_manager = XLSTMCacheManager()

    # original forward adapted to support sequence_start_indices
    # i.e. set the forget gate to zero at the start of sequence
    def _original_forward(
        self, x: torch.Tensor,
        state: mLSTMLayerStateType | None = None,
        sequence_start_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, mLSTMLayerStateType | None]:
        assert x.ndim == 3, f"Input must have shape [B, S, D], got {x.shape}"
        B, S, _ = x.shape
        if self.config.weight_mode == "single":
            q = self.q(x)
            k = self.k(x)
            v = self.v(x)
            o_preact = self.ogate_preact(x)
            i_preact = soft_cap(
                self.igate_preact(x), cap_value=self.config.gate_soft_cap
            )
            f_preact = soft_cap(
                self.fgate_preact(x), cap_value=self.config.gate_soft_cap
            )
        elif self.config.weight_mode == "fused":
            qkv_opreact = self.qkv_opreact(x)
            q, k, v, o_preact = torch.tensor_split(
                qkv_opreact,
                (
                    self.qk_dim,
                    2 * self.qk_dim,
                    2 * self.qk_dim + self.v_dim,
                ),
                dim=-1,
            )

            if_preact = soft_cap(
                self.ifgate_preact(x), cap_value=self.config.gate_soft_cap
            )
            i_preact, f_preact = torch.tensor_split(
                if_preact, (self.config.num_heads,), dim=-1
            )
        else:
            raise ValueError(f"Unknown weight_mode: {self.config.weight_mode}")

        q = q.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        k = k.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)
        v = v.reshape(B, S, self.config.num_heads, -1).transpose(1, 2)

        if sequence_start_indices is not None:
            f_preact[torch.arange(B, device=f_preact.device), sequence_start_indices] = -100_000

        i_preact = i_preact.transpose(1, 2)
        f_preact = f_preact.transpose(1, 2)
        if state is None:
            c_initial, n_initial, m_initial = None, None, None
        else:
            c_initial, n_initial, m_initial = state

        h, state = self.mlstm_backend(
            q=q,
            k=k,
            v=v,
            i=i_preact,
            f=f_preact,
            c_initial=c_initial,
            n_initial=n_initial,
            m_initial=m_initial,
        )
        expected_h_shape = (
            B,
            self.config.num_heads,
            S,
            self.v_dim // self.config.num_heads,
        )
        assert (
            h.shape == expected_h_shape
        ), f"Got {h.shape}, expected {expected_h_shape}"

        h = h.transpose(1, 2)
        h_norm = self.multihead_norm(h)
        h_norm = h_norm.reshape(B, S, -1)

        h_out = self.ogate_act_fn(o_preact) * h_norm

        y = self.out_proj(h_out)
        return y, state

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        sequence_start_indices: Optional[torch.Tensor] = None,
        cache_mask: Optional[MaskState] = None
    ):
        if self.training:
            self.mlstm_backend.config.mode = "train"
        else:
            self.mlstm_backend.config.mode = "inference"

        if self.xlstm_cache_manager is not None:
            prev_mode = self.mlstm_backend.config.mode
            state = self.xlstm_cache_manager.state

            if cache_mask is not None:
                state_for_model = cast(mLSTMLayerStateType, tuple(cache_mask.selective_get(x, inv=True) for x in state) if state is not None else None)
            else:
                state_for_model = state

            h, new_state = self._original_forward(
                x,
                state=state_for_model,
                sequence_start_indices=sequence_start_indices
            )
            assert new_state is not None

            if state is None or cache_mask is None:
                state = new_state
            else:
                if cache_mask is not None:
                    for i in range(len(state)):
                        cache_mask.selective_put(new_state[i], state[i], inv=True)

            self.xlstm_cache_manager.state = state  # type: ignore
            self.mlstm_backend.config.mode = prev_mode

            return h
        else:
            h, _ = super().forward(x)
            return h


@dataclass
class XLSTMConfig(Config):
    num_heads: int
    dtype: DType = DType.float32

    def build(self, d_model: int, init_device) -> XLSTM:
        return XLSTM(mLSTMLayerConfig(
            embedding_dim=d_model,
            num_heads=self.num_heads,
            mlstm_backend=mLSTMBackendConfig(
                chunkwise_kernel="chunkwise--triton_limit_chunk",
                sequence_kernel="native_sequence__triton",
                step_kernel="triton",
                mode="train",
                return_last_states=True,
                autocast_kernel_dtype="float32",
                chunk_size=128,
            )
        )).to(device=init_device, dtype=self.dtype.as_pt())

    def num_params(self):
        raise NotImplementedError()


class XLSTMCacheManager(nn.Module):
    def __init__(self):
        super().__init__()

        # not designed to be managed externally - cant easily allocate beforehand
        # so we just init to none and let the prefill allocate the state
        self.state: Optional[mLSTMLayerStateType] = None

    def zero_cache(self):
        raise NotImplementedError()

    def reallocate(self, batch_size: int):
        self.state = None

    def is_reusable(self, batch_size: int) -> bool:
        # not implemented
        return False

    def reset(self, batch_size: int):
        if self.is_reusable(batch_size):
            self.zero_cache()
        else:
            log.debug("Unreusable XLSTM cache, reallocating")
            self.reallocate(batch_size)