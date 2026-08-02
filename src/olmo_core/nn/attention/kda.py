"""Native OLMo-core integration for Kimi Delta Attention (KDA)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
from olmo_core.nn.feed_forward import ActivationFunction

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod


class KimiDeltaAttention(SequenceMixer):
    """Kimi Delta Attention using Moonshot's released Triton training kernel.

    This follows the parameterization of the released Kimi-Linear checkpoint:
    KDA uses a vector-valued decay for every key channel and a scalar delta
    gate for every value head. The surrounding OLMo-core adapter retains packed
    document convolutions, initialization, and sequence-mixer interfaces.
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
        norm_eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ) -> None:
        super().__init__()
        if not has_fla():
            raise RuntimeError(
                "KimiDeltaAttention requires flash-linear-attention with fla.ops.kda"
            )
        from fla.modules import FusedRMSNormGated

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_v_heads = n_v_heads if n_v_heads is not None else n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.expand_v = expand_v
        self.allow_neg_eigval = allow_neg_eigval
        self.conv_size = conv_size

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * expand_v)
        self.key_dim = self.n_heads * self.head_k_dim
        self.value_dim = self.n_v_heads * self.head_v_dim
        self.gate_dim = self.n_heads * self.head_k_dim

        if not math.isclose(self.head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError("expand_v must produce an integer value-head dimension")
        if self.n_v_heads != self.n_heads:
            raise ValueError("the pinned Kimi reference layer requires n_v_heads == n_heads")
        if self.head_k_dim > 256:
            raise ValueError("FLA's KDA chunk kernel requires head_dim <= 256")

        factory = {"dtype": dtype, "device": init_device}
        self.w_q = nn.Linear(d_model, self.key_dim, bias=False, **factory)
        self.w_k = nn.Linear(d_model, self.key_dim, bias=False, **factory)
        self.w_v = nn.Linear(d_model, self.value_dim, bias=False, **factory)

        # Kimi's low-rank projection produces one decay logit per key channel.
        self.f_proj_1 = nn.Linear(d_model, self.head_v_dim, bias=False, **factory)
        self.f_proj_2 = nn.Linear(self.head_v_dim, self.gate_dim, bias=False, **factory)
        self.w_b = nn.Linear(d_model, self.n_heads, bias=False, **factory)
        self.A_log = nn.Parameter(
            torch.empty(self.n_heads, dtype=torch.float32, device=init_device)
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.gate_dim, dtype=torch.float32, device=init_device)
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
        cu_doc_lens: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        batch_size, seq_len, _ = x.shape
        output_shape = (batch_size, seq_len, self.d_model)
        if cu_doc_lens is not None and batch_size > 1:
            # FLA's variable-length KDA and causal-convolution kernels represent packed
            # sequences as a single flattened batch, with document boundaries supplied by
            # ``cu_seqlens``. OLMo-core's mask builder likewise flattens boundaries across the
            # entire batch, so flatten the payload here and restore its shape after KDA.
            x = x.reshape(1, batch_size * seq_len, self.d_model)
            batch_size, seq_len = 1, batch_size * seq_len

        q = self.q_conv1d(x=self.w_q(x), cu_seqlens=cu_doc_lens)
        k = self.k_conv1d(x=self.w_k(x), cu_seqlens=cu_doc_lens)
        v = self.v_conv1d(x=self.w_v(x), cu_seqlens=cu_doc_lens)
        raw_decay = self.f_proj_2(self.f_proj_1(x))
        beta = self.w_b(x).float().sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        q = q.view(batch_size, seq_len, self.n_heads, self.head_k_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_k_dim)
        v = v.view(batch_size, seq_len, self.n_v_heads, self.head_v_dim)
        raw_decay = raw_decay.view(batch_size, seq_len, self.n_v_heads, self.head_k_dim)

        o, _ = dispatch_chunk_kda(
            q=q,
            k=k,
            v=v,
            g=raw_decay,
            beta=beta,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            cu_seqlens=cu_doc_lens,
        )
        output_gate = self.g_proj_2(self.g_proj_1(x)).view(
            batch_size, seq_len, self.n_v_heads, self.head_v_dim
        )
        return self.w_out(self.o_norm(o, output_gate).view(batch_size, seq_len, -1)).view(
            output_shape
        )

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Placement | None = None,
        output_layout: Placement | None = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ) -> None:
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not yet implemented for KDA")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: RingContextParallelStyle | None = None,
        uly: UlyssesContextParallelStyle | None = None,
    ) -> None:
        del cp_mesh, ring, uly
        raise NotImplementedError("Context parallelism is not yet implemented for KDA")

    @torch.no_grad()
    def init_weights(
        self,
        *,
        init_method: InitMethod,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: torch.Generator | None = None,
    ) -> None:
        from olmo_core.nn.transformer.init import InitMethod, init_linear

        if init_method == InitMethod.fan_in:
            raise NotImplementedError(f"init method '{init_method}' is not supported for KDA")
        if init_method == InitMethod.normalized:
            std = d_model**-0.5

        for linear in (
            self.w_q,
            self.w_k,
            self.w_v,
            self.f_proj_1,
            self.f_proj_2,
            self.w_b,
            self.g_proj_1,
            self.g_proj_2,
        ):
            init_linear(linear, std=std, generator=generator)
        for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
            init_linear(conv, std=std, generator=generator)

        self.A_log.copy_(nn.init.uniform_(self.A_log, a=1, b=16, generator=generator).log())
        # Match FLA 0.4.1's released KDA layer initialization.
        self.dt_bias.zero_()
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
                    self.g_proj_1,
                    self.g_proj_2,
                    self.w_out,
                )
            )
        )
        conv_flops = 2 * training_factor * self.conv_size * (self.key_dim * 2 + self.value_dim)
        state_size = self.n_v_heads * self.head_k_dim * self.head_v_dim
        # Decay, delta read/erase/write, and recurrent read all touch the KxV state.
        recurrent_flops = 2 * training_factor * 5 * state_size
        return int(linear_flops + conv_flops + recurrent_flops)


@SequenceMixerConfig.register("kimi_delta_attention")
@dataclass
class KimiDeltaAttentionConfig(SequenceMixerConfig[KimiDeltaAttention]):
    """Configuration for :class:`KimiDeltaAttention`.

    :param n_heads: The number of key/query heads.
    :param n_v_heads: The number of value heads. The pinned KDA kernel currently requires this
        to equal ``n_heads``. Defaults to ``n_heads`` when unset.
    :param head_dim: The key/query head dimension. Defaults to ``d_model // n_heads``.
    :param expand_v: Multiplier applied to ``head_dim`` to determine the value head dimension.
    :param allow_neg_eigval: Whether to scale the delta gate to allow negative eigenvalues.
    :param conv_size: The kernel size of the causal convolutions applied to Q, K, and V.
    :param conv_bias: Whether the causal convolutions include bias parameters.
    :param norm_eps: Epsilon used by the gated RMS normalization on the output.
    :param dtype: The parameter dtype.
    """

    n_heads: int = 16
    n_v_heads: int | None = None
    head_dim: int | None = None
    expand_v: float = 1.0
    allow_neg_eigval: bool = False
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
        gate_dim = self.n_heads * head_dim

        params = 0
        params += d_model * key_dim * 2  # q, k
        params += d_model * value_dim  # v
        params += d_model * head_v_dim + head_v_dim * gate_dim  # decay bottleneck
        params += d_model * self.n_heads  # scalar delta gate
        params += self.n_heads + gate_dim  # A_log, dt_bias
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
        cache: BufferCache | None = None,
    ) -> KimiDeltaAttention:
        del layer_idx, n_layers, cache
        return KimiDeltaAttention(
            d_model=d_model,
            n_heads=self.n_heads,
            n_v_heads=self.n_v_heads,
            head_dim=self.head_dim,
            expand_v=self.expand_v,
            allow_neg_eigval=self.allow_neg_eigval,
            conv_size=self.conv_size,
            conv_bias=self.conv_bias,
            norm_eps=self.norm_eps,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )
