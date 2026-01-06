# Modified from https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import logging
import math
import warnings
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange, repeat
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm  # , ShortConvolution
from fla.modules.convolution import causal_conv1d, causal_conv1d_update, causal_conv1d_update_cuda

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import functional as F

from olmo_core.distributed.parallel import RingContextParallelStyle
from olmo_core.distributed.parallel.context_parallel import (
    UlyssesContextParallelStyle,
    all_to_all_cp2hp,
    all_to_all_hp2cp,
)

if TYPE_CHECKING:
    from fla.models.utils import Cache
    from transformers.processing_utils import Unpack


@torch.compile
def elu_p1(x):
    return (F.elu(x, 1.0, False) + 1.0).to(x)


@torch.compile
def sum_norm(x):
    return (x / x.sum(-1, keepdim=True)).to(x)


def _to_channel_parallel(x: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
    """
    Transform from sequence-parallel to channel-parallel for conv in CP mode.
    [B, T/CP, C] -> [B, T, C/CP]
    """
    world_size = dist.get_world_size(cp_group)
    B, t_local, C = x.shape
    c_local = C // world_size
    # Reshape to [B, T/CP, C, 1] to match [B, T/CP, H, D] expected by cp2hp
    # Here H=C (channels treated as heads) and D=1
    x_4d = x.view(B, t_local, C, 1)
    # cp2hp: [B, T/CP, H, D] -> [B, T, H/CP, D] = [B, T, C/CP, 1]
    out_4d = all_to_all_cp2hp(x_4d, cp_group)
    # Flatten back to 3D: [B, T, C/CP]
    return out_4d.reshape(B, t_local * world_size, c_local)


def _to_seq_parallel(x: torch.Tensor, orig_C: int, cp_group: dist.ProcessGroup) -> torch.Tensor:
    """
    Transform from channel-parallel to sequence-parallel after conv in CP mode.
    [B, T, C/CP] -> [B, T/CP, C]
    """
    world_size = dist.get_world_size(cp_group)
    B, t_full, c_local = x.shape
    t_local = t_full // world_size
    # Reshape to [B, T, C/CP, 1] to match [B, T, H/CP, D] expected by hp2cp
    # Here H/CP = C/CP (channel-partitioned) and D=1
    x_4d = x.view(B, t_full, c_local, 1)
    # hp2cp: [B, T, H/CP, D] -> [B, T/CP, H, D] = [B, T/CP, C, 1]
    out_4d = all_to_all_hp2cp(x_4d, cp_group)
    # Flatten back to 3D: [B, T/CP, C]
    return out_4d.reshape(B, t_local, orig_C)


class GatedDeltaNet(nn.Module):
    """
    The layer implementaion for [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464).  # noqa

    Similar to Mamba2, each layer contains around 6*hidden_size*hidden_size parameters.

    Parameter alloation when use_gate=True:
        - 0.75 * hidden_size * hidden_size for the q_proj and k_proj each
        - 1.5 * hidden_size * hidden_size for the v_proj, g_proj and o_proj each
        - Others are ignorably small.
        - In total = 0.75 * 2 + 1.5 * 3 = 6 * hidden_size * hidden_size
    NOTE: num_heads * head_dim = 0.75 * hidden_size, please make sure to set the correct num_heads and head_dim.

    Parameter allocation when use_gate=False:
        - 1 * hidden_size * hidden_size for the q_proj and k_proj each
        - 2 * hidden_size * hidden_size for the v_proj and o_proj each
        - Others are ignorably small.
        - In total = 1 * 2 + 2 * 2 = 6 * hidden_size * hidden_size

    Args:
        hidden_size (int, Optional):
            The hidden size of the input. Default: 2048.
        expand_v (float, Optional):
            The expansion ratio for the value dim. Default: 2.0.
        head_dim (int, Optional):
            The dimension of each head. Default: 256.
        num_heads (int, Optional):
            The number of heads. Default: 4.
        num_v_heads (int, Optional):
            The number of heads for the value projection, equal to `num_heads` if `None`.
            GVA is applied if `num_v_heads` > `num_heads`. Default: `None`.
        mode (str, Optional):
            Which Gated DeltaNet kernel to use.
            Currently available: `chunk` and `fused_recurrent`.
            Default: `chunk`.
        use_beta (bool, Optional):
            Whether to use beta. Default: `True`.
        use_gate (bool, Optional):
            Whether to use output gate. Default: `True`.
        use_short_conv (bool, Optional):
            Whether to use short convolutions. Default: `True`.
        allow_neg_eigval (bool, Optional):
            Allow negative eigenvalues. Default: `False`. If set to `True`, the beta will be multiplied by 2.
            See reference: [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
        conv_size (int, Optional):
            The kernel size of the short convolution, only used when `use_short_conv` is `True`. Default: 4.
        conv_bias (bool, Optional):
            Whether to use bias in the short convolution, only used when `use_short_conv` is `True`. Default: `False`.
        layer_idx (int, Optional):
            The index of the layer. Default: None.
        norm_eps (float, Optional):
            The epsilon value for the normalization layer. Default: 1e-5.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        expand_v: float = 2,
        head_dim: int = 256,
        num_heads: int = 6,
        num_v_heads: int = None,
        mode: str = "chunk",
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: int = None,
        norm_eps: float = 1e-5,
        **kwargs,
    ) -> GatedDeltaNet:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size
        self.expand_v = expand_v

        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads

        self.head_k_dim = head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.num_heads * self.head_k_dim)
        self.value_dim = int(self.num_v_heads * self.head_v_dim)
        self.layer_idx = layer_idx

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(
            self.num_v_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5
        ):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.num_v_heads * self.head_dim * expand_v}, which is invalid for nn.Linear.",
            )
        if self.num_v_heads > self.num_heads and self.num_v_heads % self.num_heads != 0:
            raise ValueError(
                f"num_v_heads={self.num_v_heads} must be divisible by num_heads={self.num_heads}.",
            )

        if not math.isclose(head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={head_dim}. "
                f"Resulting head_v_dim would be {head_dim * expand_v}, which is invalid for FusedRMSNormGated.",
            )
        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.a_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.num_v_heads, bias=False)

        A = torch.empty(self.num_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_v_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation="silu",
            )
        else:
            warnings.warn(
                "ShortConvolution is crucial to the performance. "
                "Do not turn it off, i.e., setting `use_short_conv=False` unless you know what you are doing.",
            )
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps, dtype=torch.float32)
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self._cp_mesh: DeviceMesh | None = None
        self._cp_group: dist.ProcessGroup | None = None
        self.cp_enabled = False
        self.uly: Optional[UlyssesContextParallelStyle] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        # change to inference mode.
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode
        if self.cp_enabled:
            assert mode == "chunk", "Only chunk mode is supported in context parallelism."
        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens")
        if self.cp_enabled:
            if batch_size != 1:
                raise AssertionError("Context parallelism only supports batch_size == 1")
            # For Ulysses CP, the kernel runs on the full gathered sequence after all-to-all.
            # cu_seqlens must reflect the full sequence, not the sharded local sequence.
            total_len = q_len * dist.get_world_size(self._cp_group)
            if cu_seqlens is not None:
                # cu_seqlens was provided (this is the full unsharded cu_doc_lens from the model)
                # Validate that it covers the full sequence length
                if int(cu_seqlens[-1]) != total_len:
                    raise RuntimeError(
                        f"cu_seqlens[-1]={int(cu_seqlens[-1])} does not match expected full sequence "
                        f"length {total_len} for Ulysses CP. Make sure cu_doc_lens_full is passed."
                    )
            else:
                # No cu_seqlens provided, create synthetic one for the full sequence
                cu_seqlens = torch.tensor(
                    [0, total_len],
                    dtype=torch.int32,
                    device=hidden_states.device,
                )
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

            # Get projected values
            q_proj = self.q_proj(hidden_states)  # [B, T/CP, key_dim]
            k_proj = self.k_proj(hidden_states)  # [B, T/CP, key_dim]
            v_proj = self.v_proj(hidden_states)  # [B, T/CP, value_dim]

            if self.cp_enabled and self.uly is not None:
                assert self._cp_group is not None
                # For conv, we need full sequence. Swap from seq-parallel to channel-parallel.
                # [B, T/CP, C] -> [B, T, C/CP]
                q_proj = _to_channel_parallel(q_proj, self._cp_group)
                k_proj = _to_channel_parallel(k_proj, self._cp_group)
                v_proj = _to_channel_parallel(v_proj, self._cp_group)

            q, conv_state_q = self.q_conv1d(
                x=q_proj,
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=k_proj,
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=v_proj,
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )

            if self.cp_enabled and self.uly is not None:
                assert self._cp_group is not None
                # Swap back to seq-parallel (partitioned sequence, full channels)
                q = _to_seq_parallel(q, self.key_dim, self._cp_group)
                k = _to_seq_parallel(k, self.key_dim, self._cp_group)
                v = _to_seq_parallel(v, self.value_dim, self._cp_group)
        else:
            q = F.silu(self.q_proj(hidden_states))
            k = F.silu(self.k_proj(hidden_states))
            v = F.silu(self.v_proj(hidden_states))

        q, k = map(lambda x: rearrange(x, "... (h d) -> ... h d", d=self.head_k_dim), (q, k))
        v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)

        if self.num_v_heads > self.num_heads:
            q, k = map(
                lambda x: repeat(x, "... h d -> ... (h g) d", g=self.num_v_heads // self.num_heads),
                (q, k),
            )

        beta = self.b_proj(hidden_states).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        g = -self.A_log.float().exp() * F.softplus(
            self.a_proj(hidden_states).float() + self.dt_bias
        )

        recurrent_state = last_state["recurrent_state"] if last_state is not None else None
        if not self.cp_enabled:
            logging.warning(
                "[GatedDeltaNet] Running without CP (cp_enabled=%s, uly=%s, mode=%s, q_len=%s)",
                self.cp_enabled,
                self.uly is not None,
                mode,
                q_len,
            )
        if self.cp_enabled and self.uly is not None:
            assert self._cp_group is not None

            # Transform from context-parallel to head-parallel partitioning
            # [B, T/CP, H, D] -> [B, T, H/CP, D]
            q = all_to_all_cp2hp(q, self._cp_group)
            k = all_to_all_cp2hp(k, self._cp_group)
            v = all_to_all_cp2hp(v, self._cp_group)
            g = all_to_all_cp2hp(g.unsqueeze(-1), self._cp_group).squeeze(-1)
            beta = all_to_all_cp2hp(beta.unsqueeze(-1), self._cp_group).squeeze(-1)

            # Debugging: ensure lengths match expected padded full sequence (log every time)
            world_size = dist.get_world_size(self._cp_group)
            expected_full = q_len * world_size
            cu_last = int(cu_seqlens[-1]) if cu_seqlens is not None else None
            seq_len_after_a2a = q.shape[1]
            if cu_last is None:
                logging.warning(
                    "[GatedDeltaNet CP] cu_seqlens is None; expected_full=%s, seq_len_after_a2a=%s",
                    expected_full,
                    seq_len_after_a2a,
                )
            elif cu_last != expected_full or cu_last != seq_len_after_a2a:
                logging.warning(
                    "[GatedDeltaNet CP] Length mismatch before kernel: cu_seqlens[-1]=%s, "
                    "expected_full=%s, seq_len_after_a2a=%s",
                    cu_last,
                    expected_full,
                    seq_len_after_a2a,
                )
            logging.warning(
                "[GatedDeltaNet CP] Shapes before kernel (rank=%s): "
                "q=%s k=%s v=%s g=%s beta=%s cu_last=%s expected_full=%s seq_after_a2a=%s",
                dist.get_rank(self._cp_group) if dist.is_initialized() else "n/a",
                tuple(q.shape),
                tuple(k.shape),
                tuple(v.shape),
                tuple(g.shape),
                tuple(beta.shape),
                cu_last,
                expected_full,
                seq_len_after_a2a,
            )

            o, recurrent_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )

            # Transform back from head-parallel to context-parallel partitioning
            # [B, T, H/CP, D] -> [B, T/CP, H, D]
            o = all_to_all_hp2cp(o, self._cp_group)
        elif mode == "chunk":
            o, recurrent_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v)
                if self.use_short_conv
                else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        if self.use_gate:
            g = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
            o = self.o_norm(o, g)
        else:
            o = self.o_norm(o)
        o = rearrange(o, "b t h d -> b t (h d)")
        o = self.o_proj(o)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o, None, past_key_values

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        if ring is not None:
            raise NotImplementedError("Ring context parallelism is not supported for GatedDeltaNet")
        if uly is None:
            raise ValueError("Ulysses context parallelism is required for GatedDeltaNet CP")

        # Ulysses CP requires divisibility by CP world size for:
        # 1. num_v_heads - for head partitioning in the recurrent kernel
        # 2. key_dim and value_dim - for channel partitioning in the conv layers
        cp_world_size = cp_mesh.size()
        if self.num_v_heads % cp_world_size != 0:
            raise ValueError(
                f"Ulysses context parallelism requires num_v_heads ({self.num_v_heads}) "
                f"to be divisible by CP world size ({cp_world_size}). "
                f"Consider adjusting num_v_heads or CP degree."
            )
        if self.use_short_conv:
            if self.key_dim % cp_world_size != 0:
                raise ValueError(
                    f"Ulysses context parallelism requires key_dim ({self.key_dim}) "
                    f"to be divisible by CP world size ({cp_world_size}). "
                    f"key_dim = num_heads * head_dim = {self.num_heads} * {self.head_dim}."
                )
            if self.value_dim % cp_world_size != 0:
                raise ValueError(
                    f"Ulysses context parallelism requires value_dim ({self.value_dim}) "
                    f"to be divisible by CP world size ({cp_world_size}). "
                    f"value_dim = num_v_heads * head_v_dim = {self.num_v_heads} * {self.head_v_dim}."
                )

        self.uly = uly

        group = cp_mesh.get_group()
        if cp_mesh.size() == 1:
            group = None

        self._cp_mesh = cp_mesh
        self._cp_group = group
        self.cp_enabled = True

        # Shard conv weights across CP ranks for channel-parallel execution.
        # After _to_channel_parallel, each rank processes C/CP channels with full sequence.
        # The conv weights need to be sharded on dim 0 (the channel dimension).
        if self.use_short_conv and group is not None:
            self.q_conv1d.apply_cp(cp_mesh)
            self.k_conv1d.apply_cp(cp_mesh)
            self.v_conv1d.apply_cp(cp_mesh)


class ShortConvolution(nn.Conv1d):
    """Short convolution layer for efficient causal convolution operations.

    This class implements a depthwise separable 1D convolution with causal padding,
    designed for efficient sequence processing. It supports multiple backends (Triton/CUDA)
    and optional activation functions.

    Args:
        hidden_size (int): Number of input/output channels (must be equal for depthwise conv)
        kernel_size (int): Size of the convolution kernel
        bias (bool, optional): Whether to include learnable bias. Defaults to False.
        activation (Optional[str], optional): Activation function ('silu' or 'swish'). Defaults to 'silu'.
        backend (Optional[str], optional): Backend implementation ('triton' or 'cuda'). Defaults to 'triton'.
        device (Optional[torch.device], optional): Device to place the layer on. Defaults to None.
        dtype (Optional[torch.dtype], optional): Data type for layer parameters. Defaults to None.
        **kwargs: Additional keyword arguments (deprecated 'use_fast_conv1d' supported for compatibility)

    Attributes:
        hidden_size (int): Number of channels
        activation (Optional[str]): Selected activation function
        backend (str): Actual backend being used (may differ from input due to availability)

    Note:
        - Uses depthwise convolution (groups=hidden_size) for efficiency
        - Applies causal padding (kernel_size-1) to ensure no future information leakage
        - Falls back to Triton backend if CUDA backend is unavailable
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        activation: str | None = "silu",
        backend: str | None = "triton",
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs,
    ):
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            device=device,
            dtype=dtype,
        )

        self.hidden_size = hidden_size
        self.activation = None
        self.cp_enabled = False

        if activation is not None:
            assert activation in ["silu", "swish"], f"Activation `{activation}` not supported yet."
            self.activation = activation

        if "use_fast_conv1d" in kwargs:
            warnings.warn(
                "The `use_fast_conv1d` parameter is deprecated and will be ignored. "
                "Please use the `backend` parameter instead.",
            )
        import os

        self.backend = os.environ.get("FLA_CONV_BACKEND", backend)
        if backend not in ["cuda", "triton"]:
            raise ValueError(f"Invalid backend: {backend}, must be one of ['cuda', 'triton']")
        if backend == "cuda":
            if causal_conv1d_fn is None:
                warnings.warn(
                    "The `backend` parameter is set to `cuda`, but `causal_conv1d_fn` is not available. "
                    "Switching to the Triton implementation instead. "
                    "Consider installing `causal_conv1d` to enable the CUDA backend.",
                )
                self.backend = "triton"

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}"
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        if self.activation is not None:
            s += ", activation={activation}"
        s += f", backend={self.backend}"
        return s.format(**self.__dict__)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        cache: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        chunk_indices: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (`torch.Tensor`):
                Tensor of shape `[B, T, D]`. `B` must be 1 if `cu_seqlens` is provided.
            residual (`Optional[torch.Tensor]`):
                Residual tensor of shape `[B, T, D]`. Default: `None`.
            mask (`Optional[torch.Tensor]`):
                Attention mask dealing with padded positions.
            cache (`Optional[torch.Tensor]`):
                Previous cache tensor of shape `[N, D, W]`, where `W` is the kernel size.
                If provided, the cache is updated **inplace**.
            output_final_state (Optional[bool]):
                Whether to output the final state of shape `[N, D, W]`. Default: `False`.
            cu_seqlens (Optional[torch.LongTensor]):
                Cumulative sequence lengths for each batch. Used for varlen. Default: `None`.
                Shape: [B+1]
            chunk_indices (Optional[torch.LongTensor]):
                Chunk indices for variable-length sequences. Default: `None`.

        Returns:
            Tensor of shape `[B, T, D]`.
        """

        B, T, *_ = x.shape
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        if mask is not None:
            if cu_seqlens is not None:
                raise ValueError("`mask` and `cu_seqlens` cannot be provided at the same time")
            x = x.mul_(mask.unsqueeze(-1))

        # in decoding phase, the cache (if provided) is updated inplace
        if B * T == N:
            y, cache = self.step(
                x=x,
                residual=residual,
                cache=cache,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
            )
            return y, cache

        # cuda backend do not support:
        # 1. both `cu_seqlens` and `cache` being provided
        # 2. both `cu_seqlens` and `output_final_state` being provided
        if self.backend == "cuda" and (
            (cu_seqlens is not None and cache is not None)
            or (cu_seqlens is not None and output_final_state)
        ):
            warnings.warn(
                "The CUDA backend does not support both `cu_seqlens` and `cache` being provided, "
                "or both `cu_seqlens` and `output_final_state` being provided. "
                "Switching to the Triton backend instead. ",
                stacklevel=2,
            )
            self.backend = "triton"

        # Get weight and bias - slice if CP is enabled
        weight = self.weight
        bias = self.bias
        if self.cp_enabled:
            # Slice to local C/CP channels
            weight = weight[self._cp_channel_start : self._cp_channel_end]
            if bias is not None:
                bias = bias[self._cp_channel_start : self._cp_channel_end]

        weight = rearrange(weight, "d 1 w -> d w")

        # Debug: verify shapes match
        expected_channels = weight.shape[0]
        actual_channels = x.shape[-1]
        assert expected_channels == actual_channels, (
            f"Channel mismatch in ShortConvolution: weight has {expected_channels} channels, "
            f"input has {actual_channels} channels. "
            f"hidden_size={self.hidden_size}, cp_enabled={self.cp_enabled}, "
            f"cp_range=[{getattr(self, '_cp_channel_start', None)}:{getattr(self, '_cp_channel_end', None)}]"
        )

        return _causal_conv1d(
            x=x,
            weight=weight,
            bias=bias,
            residual=residual,
            initial_state=cache,
            output_final_state=output_final_state,
            activation=self.activation,
            backend=self.backend,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            **kwargs,
        )

    def step(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        cache: torch.Tensor,
        output_final_state: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
    ):
        B, _, D, W = *x.shape, self.kernel_size[0]
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        if output_final_state and cache is None:
            cache = x.new_zeros(N, D, W)

        # Get weight and bias - slice if CP is enabled
        weight = self.weight
        bias = self.bias
        if self.cp_enabled:
            weight = weight[self._cp_channel_start : self._cp_channel_end]
            if bias is not None:
                bias = bias[self._cp_channel_start : self._cp_channel_end]

        # Rearrange weight and ensure contiguous for Triton/CUDA kernels
        weight = rearrange(weight, "d 1 w -> d w").contiguous()
        bias = bias.contiguous() if bias is not None else None

        # NOTE: we follow the fast mode that updates the cache in-place
        if self.backend == "triton":
            return causal_conv1d_update(
                x=x.contiguous(),
                cache=cache,
                residual=residual,
                weight=weight,
                bias=bias,
                activation=self.activation,
            )

        shape = x.shape
        x = x.squeeze(0) if cu_seqlens is not None else x.squeeze(1)
        # equivalent to:
        # cache.copy_(cache.roll(shifts=-1, dims=-1))
        # cache[:, :, -1] = x
        # y = torch.sum(cache * rearrange(self.weight, "d 1 w -> d w"), dim=-1)
        y = causal_conv1d_update_cuda(
            x=x.contiguous(),
            conv_state=cache,
            weight=weight,
            bias=bias,
            activation=self.activation,
        )
        y = y.view(shape)
        if residual is not None:
            y.add_(residual)
        return y, cache

    @property
    def state_size(self) -> int:
        return self.hidden_size * self.kernel_size

    def apply_cp(self, cp_mesh: DeviceMesh):
        """
        Configure conv for Ulysses-style context parallelism.

        Instead of sharding parameters (which conflicts with FSDP), we keep the full
        parameters and slice to the local C/CP channels during forward based on CP rank.

        This way:
        - FSDP handles the full parameters normally (no DTensor conflicts)
        - Checkpoints save/load the full parameters
        - Forward pass uses only the local slice for the C/CP channels this rank processes
        """

        if cp_mesh.size() == 1:
            return

        # Store CP info for slicing in forward
        self._cp_mesh = cp_mesh
        self._cp_world_size = cp_mesh.size()
        # Use get_local_rank() to get rank within the CP group (0 to CP-1),
        # not the global rank across all processes.
        self._cp_rank = cp_mesh.get_local_rank()
        self.cp_enabled = True

        # Compute the local channel range
        local_channels = self.hidden_size // self._cp_world_size
        self._cp_channel_start = self._cp_rank * local_channels
        self._cp_channel_end = self._cp_channel_start + local_channels


@torch._dynamo.disable()
def _causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    activation: str | None,
    backend: str | None,
    cu_seqlens: torch.LongTensor | None,
    chunk_indices: torch.LongTensor | None,
    **kwargs: Unpack[dict],
) -> tuple[torch.Tensor, torch.Tensor]:
    return causal_conv1d(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        initial_state=initial_state,
        output_final_state=output_final_state,
        activation=activation,
        backend=backend,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        **kwargs,
    )
