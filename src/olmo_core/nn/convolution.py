from typing import Literal, Optional

import torch
import torch.nn as nn
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
