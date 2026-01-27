import math
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement
from torch.nn import functional as F

from olmo_core.config import DType
from olmo_core.distributed.parallel.context_parallel import (
    all_to_all_cp2hp,
    all_to_all_hp2cp,
)
from olmo_core.nn.attention import AttentionBase
from olmo_core.nn.attention.flash_linear_attn_api import dispatch_chunk_gated_delta_rule, has_fla
from olmo_core.nn.attention.ring import RingContextParallelStyle, UlyssesContextParallelStyle
from olmo_core.nn.buffer_cache import BufferCache
from olmo_core.nn.config import ModuleConfig
from olmo_core.nn.convolution import CausalConv1d
from olmo_core.nn.feed_forward import ActivationFunction


@dataclass
class RecurrentConfig(ModuleConfig):
    """
    Base configuration class for recurrent sequence mixing modules.

    This is an abstract base class - use concrete implementations like
    :class:`GatedDeltaNetConfig` to build recurrent modules.
    """

    @abstractmethod
    def num_params(self, d_model: int) -> int:
        """
        The number of params that the recurrent implementation will have once built.

        :param d_model: The model dimensionality.
        """
        raise NotImplementedError

    @abstractmethod
    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> AttentionBase:
        """
        Build the corresponding recurrent module.

        :param d_model: The model dimensionality.
        :param layer_idx: The layer index.
        :param n_layers: The total number of layers.
        :param init_device: The device to initialize the parameters on, e.g. "cpu", "meta".
        :param cache: Optional buffer cache.
        """
        raise NotImplementedError


@dataclass
class GatedDeltaNetConfig(RecurrentConfig):
    """
    Configuration for :class:`GatedDeltaNet`.

    See :class:`GatedDeltaNet` for a description of the configuration options.
    """

    n_heads: int = 16
    """
    The number of attention heads.
    """
    n_kv_heads: Optional[int] = None
    """
    The number of key/value heads. If ``None``, defaults to ``n_heads``.
    """
    head_dim: Optional[int] = None
    """
    The dimension of each head. If ``None``, defaults to ``d_model // n_heads``.
    """
    expand_v: float = 2.0
    """
    The expansion ratio for the value dimension.
    """
    allow_neg_eigval: bool = True
    """
    Allow negative eigenvalues in the recurrent dynamics.
    """
    conv_size: int = 4
    """
    The kernel size of the short convolution.
    """
    conv_bias: bool = False
    """
    Whether to use bias in the short convolution.
    """
    norm_eps: float = 1e-5
    """
    The epsilon value for the normalization layer.
    """
    dtype: DType = DType.float32
    """
    The default data type to use for parameters.
    """

    def num_params(self, d_model: int) -> int:
        """
        The number of params that the GatedDeltaNet will have once built.

        :param d_model: The model dimensionality.
        """
        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads or n_heads
        head_dim = self.head_dim or d_model // n_heads
        head_v_dim = int(head_dim * self.expand_v)
        key_dim = n_heads * head_dim
        value_dim = n_kv_heads * head_v_dim

        params = 0

        # Linear projections: w_q, w_k, w_v, w_a, w_b, w_g, w_out
        params += d_model * key_dim  # w_q
        params += d_model * key_dim  # w_k
        params += d_model * value_dim  # w_v
        params += d_model * n_kv_heads  # w_a
        params += d_model * n_kv_heads  # w_b
        params += d_model * value_dim  # w_g
        params += value_dim * d_model  # w_out

        # A_log and dt_bias parameters
        params += n_kv_heads  # A_log
        params += n_kv_heads  # dt_bias

        # Short convolutions (kernel_size * hidden_size for each)
        params += self.conv_size * key_dim  # q_conv1d
        params += self.conv_size * key_dim  # k_conv1d
        params += self.conv_size * value_dim  # v_conv1d
        if self.conv_bias:
            params += key_dim  # q_conv1d bias
            params += key_dim  # k_conv1d bias
            params += value_dim  # v_conv1d bias

        # FusedRMSNormGated (weight only, no bias)
        params += head_v_dim  # o_norm

        return params

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> "GatedDeltaNet":
        """
        Build the GatedDeltaNet module.

        :param d_model: The model dimensionality.
        :param layer_idx: The layer index (unused).
        :param n_layers: The total number of layers (unused).
        :param init_device: The device to initialize the parameters on, e.g. "cpu", "meta".
        :param cache: Optional buffer cache (unused).
        """
        del layer_idx, n_layers, cache  # Unused

        return GatedDeltaNet(
            d_model=d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            expand_v=self.expand_v,
            allow_neg_eigval=self.allow_neg_eigval,
            conv_size=self.conv_size,
            conv_bias=self.conv_bias,
            norm_eps=self.norm_eps,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )


class GatedDeltaNet(AttentionBase):
    """
    The layer implementation for `Gated Delta Networks <https://arxiv.org/abs/2412.06464>`_.

    This is a linear attention variant that uses a gated delta rule for recurrent
    state updates, providing efficient O(n) sequence modeling.

    :param d_model: The model hidden size.
    :param n_heads: The number of attention heads.
    :param n_kv_heads: The number of key/value heads. If ``None``, defaults to ``n_heads``.
        GQA is applied if ``n_kv_heads`` < ``n_heads``.
    :param head_dim: The dimension of each head. If ``None``, defaults to ``d_model // n_heads``.
    :param expand_v: The expansion ratio for the value dim. Default: 2.0.
    :param allow_neg_eigval: Allow negative eigenvalues. Default: ``True``. If set to ``True``, the beta
        will be multiplied by 2. See reference: `Unlocking State-Tracking in Linear RNNs Through Negative
        Eigenvalues <https://arxiv.org/abs/2411.12537>`_.
    :param conv_size: The kernel size of the short convolution. Default: 4.
    :param conv_bias: Whether to use bias in the short convolution. Default: ``False``.
    :param norm_eps: The epsilon value for the normalization layer. Default: 1e-5.
    :param dtype: The default data type to use for parameters.
    :param init_device: The device to initialize weights on.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        expand_v: float = 2.0,
        allow_neg_eigval: bool = True,
        conv_size: int = 4,
        conv_bias: bool = False,
        norm_eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        assert has_fla()
        from fla.modules import FusedRMSNormGated

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads
        self.expand_v = expand_v
        self.allow_neg_eigval = allow_neg_eigval
        self.conv_size = conv_size

        self.head_k_dim = self.head_dim
        self.head_v_dim = int(self.head_dim * self.expand_v)
        self.key_dim = int(self.n_heads * self.head_k_dim)
        self.value_dim = int(self.n_kv_heads * self.head_v_dim)

        # Consistency check: Ensure expand_v produces integer values
        if not math.isclose(
            self.n_kv_heads * self.head_dim * expand_v, self.value_dim, rel_tol=1e-5
        ):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by key_dim={self.key_dim}. "
                f"Resulting value_dim would be {self.n_kv_heads * self.head_dim * expand_v}, which is invalid for nn.Linear.",
            )
        if self.n_kv_heads > self.n_heads and self.n_kv_heads % self.n_heads != 0:
            raise ValueError(
                f"n_kv_heads={self.n_kv_heads} must be divisible by n_heads={self.n_heads}.",
            )

        if not math.isclose(self.head_dim * expand_v, self.head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"expand_v={expand_v} does not produce an integer value when multiplied by head_dim={self.head_dim}. "
                f"Resulting head_v_dim would be {self.head_dim * expand_v}, which is invalid for FusedRMSNormGated.",
            )

        self.w_q = nn.Linear(d_model, self.key_dim, bias=False, dtype=dtype, device=init_device)
        self.w_k = nn.Linear(d_model, self.key_dim, bias=False, dtype=dtype, device=init_device)
        self.w_v = nn.Linear(d_model, self.value_dim, bias=False, dtype=dtype, device=init_device)
        self.w_a = nn.Linear(d_model, self.n_kv_heads, bias=False, dtype=dtype, device=init_device)
        self.w_b = nn.Linear(d_model, self.n_kv_heads, bias=False, dtype=dtype, device=init_device)

        A = torch.empty(self.n_kv_heads, dtype=torch.float32, device=init_device).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        # hard coded for now
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.n_kv_heads, device=init_device) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min),
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True  # type: ignore[attr-defined]

        self.q_conv1d = CausalConv1d(
            hidden_size=self.key_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu,
            init_device=init_device,
            # TODO: why no explicit dtype?
        )
        self.k_conv1d = CausalConv1d(
            hidden_size=self.key_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu,
            init_device=init_device,
        )
        self.v_conv1d = CausalConv1d(
            hidden_size=self.value_dim,
            kernel_size=conv_size,
            bias=conv_bias,
            activation=ActivationFunction.silu,
            init_device=init_device,
        )
        self.w_g = nn.Linear(d_model, self.value_dim, bias=False, dtype=dtype, device=init_device)
        self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps, device=init_device)  # type: ignore
        self.w_out = nn.Linear(self.value_dim, d_model, bias=False, dtype=dtype, device=init_device)

        self.cp_enabled = False

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply gated delta network sequence mixing to the input.

        :param x: The input of shape ``(batch_size, seq_len, d_model)``.
        :param cu_doc_lens: Cumulative document lengths in the input ``x``, a 1D
            :class:`torch.int32` tensor that should always have one more element than there
            are documents (the first element in the tensor should always be ``0``).

        :returns: The output with shape ``(batch_size, seq_len, d_model)``.
        """
        del kwargs  # Ignore any extra kwargs passed from attention interface
        B, T, _ = x.shape

        # shape: (batch_size, seq_len, n_heads * head_k_dim),
        #        (batch_size, seq_len, n_kv_heads * head_k_dim),
        #        (batch_size, seq_len, n_kv_heads * head_v_dim)
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)

        if self.cp_enabled and self.uly is not None:
            assert self._cp_group is not None
            # For conv, we need full sequence. Swap from seq-parallel to channel-parallel.
            # [B, T/CP, C] -> [B, T, C/CP]
            q = _to_channel_parallel(q, self._cp_group)
            k = _to_channel_parallel(k, self._cp_group)
            v = _to_channel_parallel(v, self._cp_group)

        q = self.q_conv1d(x=q, cu_seqlens=cu_doc_lens)
        k = self.k_conv1d(x=k, cu_seqlens=cu_doc_lens)
        v = self.v_conv1d(x=v, cu_seqlens=cu_doc_lens)

        if self.cp_enabled and self.uly is not None:
            assert self._cp_group is not None
            # Swap back to seq-parallel (partitioned sequence, full channels)
            q = _to_seq_parallel(q, self.key_dim, self._cp_group)
            k = _to_seq_parallel(k, self.key_dim, self._cp_group)
            v = _to_seq_parallel(v, self.value_dim, self._cp_group)

        q = q.view(B, T, -1, self.head_k_dim)
        k = k.view(B, T, -1, self.head_k_dim)
        v = v.view(B, T, -1, self.head_v_dim)

        if self.n_kv_heads > self.n_heads:
            repeat_factor = self.n_kv_heads // self.n_heads
            q = q.repeat_interleave(repeat_factor, dim=-2)
            k = k.repeat_interleave(repeat_factor, dim=-2)

        beta = self.w_b(x).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # shape: (batch_size, seq_len, n_kv_heads)
        g = -self.A_log.float().exp() * F.softplus(self.w_a(x).float() + self.dt_bias)

        if self.cp_enabled and self.uly is not None:
            assert self._cp_group is not None

            # Transform from context-parallel to head-parallel partitioning
            # [B, T/CP, H, D] -> [B, T, H/CP, D]
            q = all_to_all_cp2hp(q, self._cp_group)
            k = all_to_all_cp2hp(k, self._cp_group)
            v = all_to_all_cp2hp(v, self._cp_group)
            g = all_to_all_cp2hp(g.unsqueeze(-1), self._cp_group).squeeze(-1)
            beta = all_to_all_cp2hp(beta.unsqueeze(-1), self._cp_group).squeeze(-1)

        o, _ = dispatch_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,  # pyright: ignore[reportCallIssue]
            beta=beta,  # pyright: ignore[reportCallIssue]
            cu_seqlens=cu_doc_lens,  # pyright: ignore[reportCallIssue]
            use_qk_l2norm_in_kernel=True,  # pyright: ignore[reportCallIssue]
        )

        if self.cp_enabled and self.uly is not None:
            assert self._cp_group is not None
            # Transform back from head-parallel to context-parallel partitioning
            # [B, T, H/CP, D] -> [B, T/CP, H, D]
            o = all_to_all_hp2cp(o, self._cp_group)

        g = self.w_g(x).view(B, T, -1, self.head_v_dim)

        # shape: (batch_size, seq_len, d_model)
        return self.w_out(self.o_norm(o, g).view(B, T, -1))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not yet implemented for GatedDeltaNet")

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
        # 1. n_kv_heads - for head partitioning in the recurrent kernel
        # 2. key_dim and value_dim - for channel partitioning in the conv layers
        cp_world_size = cp_mesh.size()
        if cp_world_size == 1:
            return

        if self.n_kv_heads % cp_world_size != 0:
            raise ValueError(
                f"Ulysses context parallelism requires n_kv_heads ({self.n_kv_heads}) "
                f"to be divisible by CP world size ({cp_world_size}). "
                f"Consider adjusting n_kv_heads or CP degree."
            )
        if self.q_conv1d is not None:
            if self.key_dim % cp_world_size != 0:
                raise ValueError(
                    f"Ulysses context parallelism requires key_dim ({self.key_dim}) "
                    f"to be divisible by CP world size ({cp_world_size}). "
                    f"key_dim = n_heads * head_dim = {self.n_heads} * {self.head_dim}."
                )
            if self.value_dim % cp_world_size != 0:
                raise ValueError(
                    f"Ulysses context parallelism requires value_dim ({self.value_dim}) "
                    f"to be divisible by CP world size ({cp_world_size}). "
                    f"value_dim = n_kv_heads * head_v_dim = {self.n_kv_heads} * {self.head_v_dim}."
                )

        self.uly = uly
        self._cp_mesh = cp_mesh
        self._cp_group = cp_mesh.get_group()
        self.cp_enabled = True

        if self.q_conv1d is not None:
            self.q_conv1d.apply_cp(cp_mesh)
        if self.k_conv1d is not None:
            self.k_conv1d.apply_cp(cp_mesh)
        if self.v_conv1d is not None:
            self.v_conv1d.apply_cp(cp_mesh)

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Compute FLOPs per token for Gated Delta Net.

        This accounts for:
        - Linear projections (w_q, w_k, w_v, w_a, w_b, w_g, w_out)
        - Short convolutions (q, k, v)
        - Gated delta rule recurrent computation
        - Gated RMS normalization
        """
        del seq_len
        # Linear projection FLOPs (2 ops per multiply-add)
        linear_flops = 2 * sum(
            m.weight.numel()
            for m in (self.w_q, self.w_k, self.w_v, self.w_a, self.w_b, self.w_g, self.w_out)
        )

        # Short convolution FLOPs (2 ops per multiply-add, kernel_size taps per output)
        conv_flops = (
            2
            * self.conv_size
            * (
                self.key_dim  # q_conv1d
                + self.key_dim  # k_conv1d
                + self.value_dim  # v_conv1d
            )
        )

        # Gated delta rule recurrent computation per token:
        # - Outer product k ⊗ v: n_kv_heads * head_k_dim * head_v_dim
        # - State decay: n_kv_heads * head_k_dim * head_v_dim
        # - Beta scaling: n_kv_heads * head_k_dim * head_v_dim
        # - Query-state matmul: n_kv_heads * head_k_dim * head_v_dim
        # Each is 2 FLOPs per element (multiply-add or similar)
        state_size = self.n_kv_heads * self.head_k_dim * self.head_v_dim
        recurrent_flops = 2 * 4 * state_size

        return int(linear_flops + conv_flops + recurrent_flops)


def _to_channel_parallel(x: torch.Tensor, cp_group: dist.ProcessGroup) -> torch.Tensor:
    """
    Transform from sequence-parallel to channel-parallel for conv in CP mode.
    [B, T/CP, C] -> [B, T, C/CP]
    """
    world_size = dist.get_world_size(cp_group)
    B, t_local, C = x.shape
    c_local = C // world_size
    # Reshape to [B, T/CP, C, 1] to match [B, T/CP, H, D] expected by cp2hp
    x_4d = x.view(B, t_local, C, 1)
    # cp2hp: [B, T/CP, H, D] -> [B, T, H/CP, D]
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
    x_4d = x.view(B, t_full, c_local, 1)
    # hp2cp: [B, T, H/CP, D] -> [B, T/CP, H, D]
    out_4d = all_to_all_hp2cp(x_4d, cp_group)
    # Flatten back to 3D: [B, T/CP, C]
    return out_4d.reshape(B, t_local, orig_C)
