from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

__all__ = ["CausalConv1d"]


class CausalConv1d(nn.Conv1d):
    """
    CausalConv1d (aka short convolution) layer for efficient causal convolution operations.
    This implements a depthwise separable 1D convolution with causal padding.

    Modified from: https://github.com/fla-org/flash-linear-attention/blob/3cf180339b8a1cbad823f553541cd531d18670ea/fla/modules/conv/short_conv.py#L19
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
        activation: Optional[str] = "silu",
    ):
        """
        :param hidden_size: Number of input/output channels (must be equal for depthwise conv).
        :param kernel_size: Size of the convolution kernel.
        :param bias: Whether to include learnable bias.
        :param backend: Backend implementation ('triton' or 'cuda').
        :param dtype: The data type of the input and output.
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
        self.activation: Optional[str] = None
        self.backend = backend

        if activation is not None:
            if activation not in ("silu", "swish"):
                raise ValueError(f"Activation `{activation}` not supported, use 'silu' or 'swish'")
            self.activation = activation

        self.cp_enabled = False

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the causal convolution on the input.

        :param x: Input tensor of shape ``(batch_size, seq_len, hidden_size)``.
            ``batch_size`` must be 1 if ``cu_seqlens`` is provided.
        :param cu_seqlens: Cumulative sequence lengths for variable-length sequences.
            Shape: ``(num_seqs + 1,)``.

        :returns: Output tensor of shape ``(batch_size, seq_len, hidden_size)``.
        """
        from fla.modules.convolution import causal_conv1d

        weight = self.weight
        bias = self.bias

        if self.cp_enabled:
            # Slice to local C/CP channels
            weight = weight[self._cp_channel_start : self._cp_channel_end]
            if bias is not None:
                bias = bias[self._cp_channel_start : self._cp_channel_end]

        return causal_conv1d(
            x=x,
            weight=weight.squeeze(1),
            bias=bias,
            activation=self.activation,
            backend=self.backend,
            cu_seqlens=cu_seqlens,
        )

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

        # Store CP info for slicing
        self._cp_mesh = cp_mesh
        self._cp_world_size = cp_mesh.size()
        self._cp_rank = cp_mesh.get_local_rank()
        local_channels = self.hidden_size // self._cp_world_size
        self._cp_channel_start = self._cp_rank * local_channels
        self._cp_channel_end = self._cp_channel_start + local_channels
        self.cp_enabled = True
