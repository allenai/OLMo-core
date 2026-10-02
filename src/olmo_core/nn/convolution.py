from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.nn.attention.flash_linear_attn_api import dispatch_causal_conv1d

__all__ = ["CausalConv1d"]


class CausalConv1d(nn.Conv1d):
    """
    CausalConv1d (aka short convolution) layer for efficient causal convolution operations.
    This implements a depthwise separable 1D convolution with causal padding.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        kernel_size: int,
        bias: bool = False,
        backend: Literal["triton", "cuda"] = "triton",
        dtype: torch.dtype | None = None,
        init_device: str = "cpu",
        activation: Literal["silu", "swish"] | None = "silu",
    ):
        """
        :param hidden_size: Number of input/output channels (must be equal for depthwise conv).
        :param kernel_size: Size of the convolution kernel.
        :param bias: Whether to include learnable bias.
        :param backend: Backend implementation ('triton' or 'cuda').
        :param dtype: The data type of the convolution weights and bias.
        :param init_device: The device to initialize the parameters on, e.g. "cpu", "meta".
        :param activation: Activation function ('silu' or 'swish').
        """
        super().__init__(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
            device=init_device,
            dtype=dtype,
        )
        self.hidden_size = hidden_size
        self.backend = backend
        self.activation = activation
        self.cp_enabled = False

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            ``batch_size`` must be 1 if ``cu_seqlens`` is provided.
            When CP is enabled, input should be channel-parallel: ``(batch_size, seq_len, hidden_size/CP)``.
        :param cu_seqlens: Cumulative sequence lengths for variable-length sequences.
            Shape: ``(num_seqs + 1,)``.
        :returns: Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
            When CP is enabled, output is channel-parallel: ``(batch_size, seq_len, hidden_size/CP)``.
        """
        weight = self.weight
        bias = self.bias

        if self.cp_enabled:
            weight = weight[self._cp_channel_slice]
            if bias is not None:
                bias = bias[self._cp_channel_slice]

        output = dispatch_causal_conv1d(
            x=x,
            weight=weight.squeeze(1),
            bias=bias,
            activation=self.activation,
            backend=self.backend,
            cu_seqlens=cu_seqlens,
        )
        return output[0]

    def forward_soft_keep(self, x: torch.Tensor, keep: torch.Tensor, chunk: int = 128) -> torch.Tensor:
        """
        The causal conv under DIFFERENTIABLE token removal (``GatedDeltaNet(soft_keep=...)``).

        Removing token ``s`` from the row changes every later token's window: it holds the previous
        ``kernel_size - 1`` *kept* tokens. That window is written with registers
        ``R_r(t) = p_t * R_{r-1}(t-1) + (1 - p_t) * R_r(t-1)`` (``R_0(t) = x_t``), i.e. ``R_r(t)`` is the
        r-th most recent kept input at or before ``t``, and the output is
        ``act(w[K-1] x_t + sum_r w[K-1-r] R_r(t-1) + bias)``. For ``p in {0, 1}`` this is EXACTLY the
        conv of the compacted row at the kept positions; in between it interpolates smoothly and is
        differentiable in ``p``. Each register is a first-order scalar-gated scan, computed in
        ``chunk``-sized blocks with non-positive exponents only (stable); activation-checkpointed so
        the (T, C) float32 intermediates are not kept for backward. No CP / ``cu_seqlens`` support.

        :param x: ``(B, T, C)`` conv input.
        :param keep: ``(B, T)`` keep probability in ``[0, 1]``.
        :returns: ``(B, T, C)`` in ``x.dtype``.
        """
        from torch.utils.checkpoint import checkpoint

        weight, bias = self._local_weight_bias()
        return checkpoint(_soft_keep_conv, x, keep, weight.squeeze(1), bias, self.activation, chunk, use_reentrant=False)

    @property
    def state_width(self) -> int:
        """The width of the cached conv state needed for single-step decoding: ``kernel_size - 1``."""
        return self.kernel_size[0] - 1

    def _local_weight_bias(self):
        weight = self.weight
        bias = self.bias
        if self.cp_enabled:
            weight = weight[self._cp_channel_slice]
            if bias is not None:
                bias = bias[self._cp_channel_slice]
        return weight, bias

    def prefill_state(self, x: torch.Tensor) -> torch.Tensor:
        """
        Capture the conv state for cached decoding from a prefill input.

        :param x: The prefill conv input of shape ``(batch_size, seq_len, hidden_size)`` (the same
            tensor passed to :meth:`forward`).
        :returns: The last ``kernel_size - 1`` inputs, channel-first and left-padded with zeros if
            ``seq_len < kernel_size - 1``, of shape ``(batch_size, hidden_size, kernel_size - 1)``.
        """
        w = self.state_width
        xt = x.transpose(1, 2)  # (B, hidden, seq_len)
        if xt.shape[-1] < w:
            xt = F.pad(xt, (w - xt.shape[-1], 0))
        return xt[:, :, -w:].contiguous()

    def step(self, x_t: torch.Tensor, conv_state: torch.Tensor) -> torch.Tensor:
        """
        Single-step causal convolution for cached decoding, updating ``conv_state`` in place.

        Mirrors the reference ``causal_conv1d_update``: the new input is concatenated onto the
        cached window, convolved, and the trailing ``kernel_size - 1`` inputs are written back.

        :param x_t: The new conv input of shape ``(batch_size, 1, hidden_size)``.
        :param conv_state: The cached window of shape ``(batch_size, hidden_size, kernel_size - 1)``,
            updated in place.
        :returns: The conv output of shape ``(batch_size, 1, hidden_size)``.
        """
        weight, bias = self._local_weight_bias()
        hidden_size = weight.shape[0]
        xt = x_t.transpose(1, 2)  # (B, hidden, 1)
        window = torch.cat([conv_state, xt], dim=-1)  # (B, hidden, kernel_size)
        conv_state.copy_(window[:, :, -self.state_width :])
        out = F.conv1d(
            window.to(weight.dtype), weight, bias, padding=0, groups=hidden_size
        )  # (B, hidden, 1)
        if self.activation in ("silu", "swish"):
            out = F.silu(out)
        return out.transpose(1, 2).to(x_t.dtype)  # (B, 1, hidden)

    def apply_cp(self, cp_mesh: DeviceMesh):
        """
        Configure convolution for Ulysses-style (channel-parallel) context parallelism.

        Instead of sharding parameters (which conflicts with ``FSDP``), we keep the full
        parameters and slice to the local ``C/CP`` channels during forward based on CP rank.
        Since convolutions tend to have a small number of parameters, the extra memory overhead
        of keeping the full parameters on each rank is minimal.

        :param cp_mesh: The context parallel device mesh.
        """
        if cp_mesh.size() == 1:
            return

        local_channels = self.hidden_size // cp_mesh.size()
        start = cp_mesh.get_local_rank() * local_channels
        self._cp_channel_slice = slice(start, start + local_channels)
        self.cp_enabled = True


def _gated_scan(a_log: torch.Tensor, b: torch.Tensor, x: torch.Tensor, chunk: int) -> torch.Tensor:
    """``h_t = exp(a_log_t) * h_{t-1} + b_t * x_t`` (``h_{-1} = 0``) over ``(B, T, C)`` float32, blockwise:
    intra-block decay matrices ``exp(cum_t - cum_s)`` (s <= t) plus a block-level carry matmul."""
    B, T, C = x.shape
    pad = (-T) % chunk
    if pad:
        a_log = F.pad(a_log, (0, pad))
        b = F.pad(b, (0, pad))
        x = F.pad(x, (0, 0, 0, pad))
    n = (T + pad) // chunk
    cum = a_log.view(B, n, chunk).cumsum(-1)  # (B, n, L), non-increasing
    tri = torch.ones(chunk, chunk, dtype=torch.bool, device=x.device).tril()
    D = torch.exp((cum[..., :, None] - cum[..., None, :]).masked_fill(~tri, float("-inf")))  # (B, n, L, L)
    intra = D @ (b[..., None] * x).view(B, n, chunk, C)  # (B, n, L, C)
    g_end = cum[..., -1].cumsum(-1)  # (B, n) total log-decay through the end of each block
    trib = torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
    E = torch.exp((g_end[..., :, None] - g_end[..., None, :]).masked_fill(~trib, float("-inf")))  # (B, n, n)
    carry = E @ intra[:, :, -1]  # (B, n, C) state at the end of each block
    h_prev = F.pad(carry, (0, 0, 1, 0))[:, :-1]  # state entering each block
    out = intra + torch.exp(cum)[..., None] * h_prev[:, :, None, :]
    return out.reshape(B, n * chunk, C)[:, :T]


def _soft_keep_conv(x, keep, w, bias, activation, chunk):
    K = w.shape[-1]
    dt = torch.float64 if x.dtype == torch.float64 else torch.float32
    p = keep.to(dt)
    a_log = torch.log(torch.clamp(1.0 - p, min=1e-6))  # kept (p=1) -> ~no carry-over; dropped (p=0) -> 0
    src = x.to(dt)
    wf = w.to(dt)
    y = src * wf[:, K - 1]
    for r in range(1, K):
        reg = _gated_scan(a_log, p, src, chunk)  # r-th most recent kept input at or before t
        reg_prev = F.pad(reg, (0, 0, 1, 0))[:, :-1]  # ... before t
        y = y + reg_prev * wf[:, K - 1 - r]
        src = reg_prev
    if bias is not None:
        y = y + bias.to(dt)
    if activation in ("silu", "swish"):
        y = F.silu(y)
    elif activation is not None:
        raise ValueError(activation)
    return y.to(x.dtype)

