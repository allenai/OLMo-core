"""Native OLMo-core integration for FLA's Gated DeltaNet 2 kernel."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement
from torch.nn import functional as F

from olmo_core.config import DType
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.flash_linear_attn_api import dispatch_chunk_gdn2, has_fla
from olmo_core.nn.attention.ring import RingContextParallelStyle, UlyssesContextParallelStyle
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.convolution import CausalConv1d
from olmo_core.nn.feed_forward import ActivationFunction

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod


class GatedDeltaNet2(SequenceMixer):
    """Gated DeltaNet 2 sequence mixer backed by FLA's chunk training kernel.

    GDN2 replaces GDN1's scalar erase/write gate and scalar decay with separate
    channel-wise erase and write gates and a channel-wise decay. This module
    follows FLA's reference layer while retaining OLMo-core initialization,
    configuration, packed-document convolution, and sequence-mixer interfaces.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_v_heads: int | None = None,
        head_dim: int | None = None,
        expand_v: float = 1.0,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        disable_recompute: bool = False,
        norm_eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ) -> None:
        super().__init__()
        if not has_fla():
            raise RuntimeError("GatedDeltaNet2 requires flash-linear-attention with fla.ops.gdn2")
        from fla.modules import FusedRMSNormGated

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_v_heads = n_v_heads if n_v_heads is not None else n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.expand_v = expand_v
        self.allow_neg_eigval = allow_neg_eigval
        self.conv_size = conv_size
        self.disable_recompute = disable_recompute

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * expand_v)
        self.key_dim = self.n_heads * self.head_k_dim
        self.value_dim = self.n_v_heads * self.head_v_dim

        if not math.isclose(self.head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError("expand_v must produce an integer value-head dimension")
        if self.n_v_heads < self.n_heads or self.n_v_heads % self.n_heads:
            raise ValueError("n_v_heads must be a multiple of n_heads and at least as large")
        if self.head_k_dim > 256:
            raise ValueError("FLA's GDN2 chunk kernel requires head_dim <= 256")

        factory = {"dtype": dtype, "device": init_device}
        self.w_q = nn.Linear(d_model, self.key_dim, bias=False, **factory)
        self.w_k = nn.Linear(d_model, self.key_dim, bias=False, **factory)
        self.w_v = nn.Linear(d_model, self.value_dim, bias=False, **factory)

        # FLA's canonical low-rank channel-wise decay projection.
        self.f_proj_1 = nn.Linear(d_model, self.head_v_dim, bias=False, **factory)
        self.f_proj_2 = nn.Linear(self.head_v_dim, self.key_dim, bias=False, **factory)
        self.w_b = nn.Linear(d_model, self.key_dim, bias=False, **factory)
        self.w_w = nn.Linear(d_model, self.value_dim, bias=False, **factory)

        # FLA keeps recurrent timescales in fp32 even under autocast.
        self.A_log = nn.Parameter(
            torch.empty(self.n_heads, dtype=torch.float32, device=init_device)
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.key_dim, dtype=torch.float32, device=init_device)
        )

        self.q_conv1d = CausalConv1d(
            hidden_size=self.key_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu.value,
            dtype=dtype,
            init_device=init_device,
        )
        self.k_conv1d = CausalConv1d(
            hidden_size=self.key_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu.value,
            dtype=dtype,
            init_device=init_device,
        )
        self.v_conv1d = CausalConv1d(
            hidden_size=self.value_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu.value,
            dtype=dtype,
            init_device=init_device,
        )

        self.g_proj_1 = nn.Linear(d_model, self.head_v_dim, bias=False, **factory)
        self.g_proj_2 = nn.Linear(self.head_v_dim, self.value_dim, bias=True, **factory)
        self.o_norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=norm_eps,
            activation="sigmoid",
            device=torch.device(init_device),
            dtype=dtype,
        )
        self.w_out = nn.Linear(self.value_dim, d_model, bias=False, **factory)

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        batch_size, seq_len, _ = x.shape

        q = self.q_conv1d(x=self.w_q(x), cu_seqlens=cu_doc_lens)
        k = self.k_conv1d(x=self.w_k(x), cu_seqlens=cu_doc_lens)
        v = self.v_conv1d(x=self.w_v(x), cu_seqlens=cu_doc_lens)

        g = F.softplus(self.f_proj_2(self.f_proj_1(x)).float() + self.dt_bias)
        b = self.w_b(x).sigmoid()
        w = self.w_w(x).sigmoid()

        q = q.view(batch_size, seq_len, self.n_heads, self.head_k_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_k_dim)
        g = g.view(batch_size, seq_len, self.n_heads, self.head_k_dim)
        b = b.view(batch_size, seq_len, self.n_heads, self.head_k_dim)
        v = v.view(batch_size, seq_len, self.n_v_heads, self.head_v_dim)
        w = w.view(batch_size, seq_len, self.n_v_heads, self.head_v_dim)
        g = -self.A_log.float().exp().view(1, 1, self.n_heads, 1) * g

        if self.n_v_heads > self.n_heads:
            repeat_factor = self.n_v_heads // self.n_heads
            q = q.repeat_interleave(repeat_factor, dim=-2)
            k = k.repeat_interleave(repeat_factor, dim=-2)
            g = g.repeat_interleave(repeat_factor, dim=-2)
            b = b.repeat_interleave(repeat_factor, dim=-2)
        if self.allow_neg_eigval:
            b = b * 2.0

        o, _ = dispatch_chunk_gdn2(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            use_qk_l2norm_in_kernel=True,
            disable_recompute=self.disable_recompute,
            cu_seqlens=cu_doc_lens,
        )
        output_gate = self.g_proj_2(self.g_proj_1(x)).view(
            batch_size, seq_len, self.n_v_heads, self.head_v_dim
        )
        return self.w_out(self.o_norm(o, output_gate).view(batch_size, seq_len, -1))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ) -> None:
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not yet implemented for GatedDeltaNet2")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ) -> None:
        del cp_mesh, ring, uly
        raise NotImplementedError("Context parallelism is not yet implemented for GatedDeltaNet2")

    @torch.no_grad()
    def init_weights(
        self,
        *,
        init_method: "InitMethod",
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        from olmo_core.nn.transformer.init import InitMethod, init_linear

        if init_method == InitMethod.fan_in:
            raise NotImplementedError(
                f"init method '{init_method}' is not supported for GatedDeltaNet2"
            )
        if init_method == InitMethod.normalized:
            std = d_model**-0.5

        for linear in (
            self.w_q,
            self.w_k,
            self.w_v,
            self.f_proj_1,
            self.f_proj_2,
            self.w_b,
            self.w_w,
            self.g_proj_1,
            self.g_proj_2,
        ):
            init_linear(linear, std=std, generator=generator)
        for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
            init_linear(conv, std=std, generator=generator)

        self.A_log.copy_(nn.init.uniform_(self.A_log, a=1, b=16, generator=generator).log())
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            nn.init.uniform_(self.dt_bias, generator=generator)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.o_norm.reset_parameters()

        if init_method == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5
        init_linear(self.w_out, std=std, generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        del seq_len
        training_factor = 3
        linear_flops = (
            2
            * training_factor
            * sum(
                module.weight.numel()
                for module in (
                    self.w_q,
                    self.w_k,
                    self.w_v,
                    self.f_proj_1,
                    self.f_proj_2,
                    self.w_b,
                    self.w_w,
                    self.g_proj_1,
                    self.g_proj_2,
                    self.w_out,
                )
            )
        )
        conv_flops = (
            2 * training_factor * self.conv_size * (self.key_dim + self.key_dim + self.value_dim)
        )
        # Decay, erase, write, and read all operate on the recurrent K x V state.
        state_size = self.n_v_heads * self.head_k_dim * self.head_v_dim
        recurrent_flops = 2 * training_factor * 5 * state_size
        return int(linear_flops + conv_flops + recurrent_flops)


@SequenceMixerConfig.register("gated_delta_net_2")
@dataclass
class GatedDeltaNet2Config(SequenceMixerConfig[GatedDeltaNet2]):
    n_heads: int = 16
    n_v_heads: Optional[int] = None
    head_dim: Optional[int] = None
    expand_v: float = 1.0
    allow_neg_eigval: bool = False
    conv_size: int = 4
    conv_bias: bool = False
    disable_recompute: bool = False
    norm_eps: float = 1e-5
    dtype: DType = DType.float32

    def num_params(self, d_model: int) -> int:
        n_v_heads = self.n_v_heads or self.n_heads
        head_dim = self.head_dim or d_model // self.n_heads
        head_v_dim = int(head_dim * self.expand_v)
        key_dim = self.n_heads * head_dim
        value_dim = n_v_heads * head_v_dim

        params = 0
        params += d_model * key_dim * 2  # q, k
        params += d_model * value_dim  # v
        params += d_model * head_v_dim + head_v_dim * key_dim  # decay bottleneck
        params += d_model * key_dim  # channel-wise erase
        params += d_model * value_dim  # channel-wise write
        params += self.n_heads + key_dim  # A_log, dt_bias
        params += self.conv_size * (key_dim * 2 + value_dim)
        if self.conv_bias:
            params += key_dim * 2 + value_dim
        params += d_model * head_v_dim + head_v_dim * value_dim + value_dim  # output gate
        params += head_v_dim  # RMSNorm weight
        params += value_dim * d_model  # output projection
        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> GatedDeltaNet2:
        del layer_idx, n_layers, cache
        return GatedDeltaNet2(
            d_model=d_model,
            n_heads=self.n_heads,
            n_v_heads=self.n_v_heads,
            head_dim=self.head_dim,
            expand_v=self.expand_v,
            allow_neg_eigval=self.allow_neg_eigval,
            conv_size=self.conv_size,
            conv_bias=self.conv_bias,
            disable_recompute=self.disable_recompute,
            norm_eps=self.norm_eps,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )
