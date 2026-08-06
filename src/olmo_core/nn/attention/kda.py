import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement

from olmo_core.config import DType
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.flash_linear_attn_api import dispatch_chunk_kda, has_fla
from olmo_core.nn.attention.ring import RingContextParallelStyle, UlyssesContextParallelStyle
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.convolution import CausalConv1d

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod


class KimiDeltaAttention(SequenceMixer):
    """Kimi Delta Attention recurrent sequence mixer."""

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_v_heads: int | None = None,
        head_dim: int | None = None,
        expand_v: float = 1.0,
        allow_neg_eigval: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        assert has_fla()
        from fla.modules import FusedRMSNormGated

        if safe_gate and lower_bound is None:
            raise ValueError("safe_gate requires lower_bound")
        if not safe_gate and lower_bound is not None:
            raise ValueError("lower_bound requires safe_gate")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_v_heads = n_v_heads if n_v_heads is not None else n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.expand_v = expand_v
        self.allow_neg_eigval = allow_neg_eigval
        self.safe_gate = safe_gate
        self.lower_bound = lower_bound
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = self.n_heads * self.head_k_dim
        self.value_dim = self.n_v_heads * self.head_v_dim
        self.gate_dim = self.n_v_heads * self.head_k_dim

        if not math.isclose(self.head_dim * self.expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(f"expand_v={expand_v} does not produce an integer value head dimension")
        if self.n_v_heads < self.n_heads or self.n_v_heads % self.n_heads != 0:
            raise ValueError("n_v_heads must be a multiple of n_heads")

        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False, dtype=dtype, device=init_device)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False, dtype=dtype, device=init_device)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False, dtype=dtype, device=init_device)

        self.q_conv1d = CausalConv1d(
            hidden_size=self.key_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            dtype=dtype,
            init_device=init_device,
        )
        self.k_conv1d = CausalConv1d(
            hidden_size=self.key_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            dtype=dtype,
            init_device=init_device,
        )
        self.v_conv1d = CausalConv1d(
            hidden_size=self.value_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            dtype=dtype,
            init_device=init_device,
        )

        self.f_proj = nn.Sequential(
            nn.Linear(d_model, self.head_v_dim, bias=False, dtype=dtype, device=init_device),
            nn.Linear(
                self.head_v_dim,
                self.gate_dim,
                bias=False,
                dtype=dtype,
                device=init_device,
            ),
        )
        self.b_proj = nn.Linear(
            d_model, self.n_v_heads, bias=False, dtype=dtype, device=init_device
        )

        self.A_log = nn.Parameter(
            torch.empty(self.n_v_heads, dtype=torch.float32, device=init_device)
        )
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.dt_bias = nn.Parameter(
            torch.empty(self.gate_dim, dtype=torch.float32, device=init_device)
        )
        self.dt_bias._no_weight_decay = True  # type: ignore[attr-defined]

        self.g_proj = nn.Sequential(
            nn.Linear(d_model, self.head_v_dim, bias=False, dtype=dtype, device=init_device),
            nn.Linear(
                self.head_v_dim,
                self.value_dim,
                bias=True,
                dtype=dtype,
                device=init_device,
            ),
        )
        self.o_norm = FusedRMSNormGated(
            self.head_v_dim,
            activation="sigmoid",
            eps=norm_eps,
            device=torch.device(init_device),
            dtype=dtype,
        )
        self.o_proj = nn.Linear(
            self.value_dim, d_model, bias=False, dtype=dtype, device=init_device
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        batch_size, seq_len, _ = x.shape

        q = self.q_conv1d(self.q_proj(x), cu_seqlens=cu_doc_lens)
        k = self.k_conv1d(self.k_proj(x), cu_seqlens=cu_doc_lens)
        v = self.v_conv1d(self.v_proj(x), cu_seqlens=cu_doc_lens)
        g = self.f_proj(x)
        beta = self.b_proj(x)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_k_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_k_dim)
        v = v.view(batch_size, seq_len, self.n_v_heads, self.head_v_dim)
        g = g.view(batch_size, seq_len, self.n_v_heads, self.head_k_dim)

        if self.n_v_heads > self.n_heads:
            repeat_factor = self.n_v_heads // self.n_heads
            q = q.repeat_interleave(repeat_factor, dim=-2)
            k = k.repeat_interleave(repeat_factor, dim=-2)

        output, _ = dispatch_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            allow_neg_eigval=self.allow_neg_eigval,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
            cu_seqlens=cu_doc_lens,
        )
        output_gate = self.g_proj(x).view(
            batch_size, seq_len, self.n_v_heads, self.head_v_dim
        )
        return self.o_proj(self.o_norm(output, output_gate).reshape(batch_size, seq_len, -1))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not yet implemented for KimiDeltaAttention")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        del cp_mesh, ring, uly
        raise NotImplementedError("Context parallelism is not yet implemented for KimiDeltaAttention")

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
                f"init method '{init_method}' is not supported for KimiDeltaAttention"
            )
        if init_method == InitMethod.normalized:
            std = d_model**-0.5

        linears = (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.f_proj[0],
            self.f_proj[1],
            self.b_proj,
            self.g_proj[0],
            self.g_proj[1],
        )
        for linear in linears:
            init_linear(linear, std=std, generator=generator)
        for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
            init_linear(conv, std=std, generator=generator)

        if self.safe_gate:
            self.A_log.zero_()
        else:
            self.A_log.copy_(
                nn.init.uniform_(self.A_log, a=1, b=16, generator=generator).log()
            )
        dt = torch.exp(
            nn.init.uniform_(self.dt_bias, generator=generator)
            * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        output_std = std
        if init_method == InitMethod.llama:
            output_std /= (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            output_std /= (2 * (block_idx + 1)) ** 0.5
        elif init_method == InitMethod.normalized:
            output_std /= (2 * num_blocks) ** 0.5
        init_linear(self.o_proj, std=output_std, generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        del seq_len
        projection_flops = 2 * sum(
            module.weight.numel()
            for module in (
                self.q_proj,
                self.k_proj,
                self.v_proj,
                self.f_proj[0],
                self.f_proj[1],
                self.b_proj,
                self.g_proj[0],
                self.g_proj[1],
                self.o_proj,
            )
        )
        conv_flops = 2 * self.conv_size * (2 * self.key_dim + self.value_dim)
        state_size = self.n_v_heads * self.head_k_dim * self.head_v_dim
        recurrent_flops = 2 * 4 * state_size
        return int(projection_flops + conv_flops + recurrent_flops)


@SequenceMixerConfig.register("kimi_delta_attention")
@dataclass
class KimiDeltaAttentionConfig(SequenceMixerConfig[KimiDeltaAttention]):
    """Configuration for :class:`KimiDeltaAttention`."""

    n_heads: int = 16
    n_v_heads: Optional[int] = None
    head_dim: Optional[int] = None
    expand_v: float = 1.0
    allow_neg_eigval: bool = False
    safe_gate: bool = False
    lower_bound: Optional[float] = None
    conv_size: int = 4
    conv_bias: bool = False
    norm_eps: float = 1e-5
    dtype: DType = DType.float32

    def num_params(self, d_model: int) -> int:
        n_v_heads = self.n_v_heads or self.n_heads
        head_dim = self.head_dim or d_model // self.n_heads
        head_v_dim = int(head_dim * self.expand_v)
        key_dim = self.n_heads * head_dim
        value_dim = n_v_heads * head_v_dim
        gate_dim = n_v_heads * head_dim

        params = 2 * d_model * key_dim
        params += d_model * value_dim
        params += self.conv_size * (2 * key_dim + value_dim)
        if self.conv_bias:
            params += 2 * key_dim + value_dim
        params += d_model * head_v_dim + head_v_dim * gate_dim
        params += d_model * n_v_heads
        params += n_v_heads + gate_dim
        params += d_model * head_v_dim + head_v_dim * value_dim + value_dim
        params += head_v_dim
        params += value_dim * d_model
        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> KimiDeltaAttention:
        del layer_idx, n_layers, cache
        return KimiDeltaAttention(
            d_model=d_model,
            n_heads=self.n_heads,
            n_v_heads=self.n_v_heads,
            head_dim=self.head_dim,
            expand_v=self.expand_v,
            allow_neg_eigval=self.allow_neg_eigval,
            safe_gate=self.safe_gate,
            lower_bound=self.lower_bound,
            conv_size=self.conv_size,
            conv_bias=self.conv_bias,
            norm_eps=self.norm_eps,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )