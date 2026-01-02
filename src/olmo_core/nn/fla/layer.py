import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import (
    DTensor,
    Replicate,
    Shard,
    distribute_module,
    distribute_tensor,
)
from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module
from torch.distributed.tensor.placement_types import Placement

from olmo_core.config import Config, DType

log = logging.getLogger(__name__)


class GatedDeltaNetParallel(ParallelStyle):
    """
    A ParallelStyle that applies tensor parallelism to a GatedDeltaNet module.

    This handles:
    - q_proj, k_proj, v_proj, a_proj, b_proj, g_proj: columnwise parallel (Shard(0) on weight)
    - o_proj: rowwise parallel (Shard(1) on weight)
    - A_log, dt_bias: Shard(0) on the num_heads dimension
    - q_conv1d, k_conv1d, v_conv1d: replicated (Triton kernels can't handle DTensor)
    - o_norm: replicated (operates on head_v_dim, not sharded dimension)

    Keyword Args:
        input_layouts: The DTensor layout of input tensor (how it arrives), default: Replicate().
        output_layouts: The DTensor layout of output tensor (how it should leave), default: Replicate().
        use_local_output: Whether to convert output back to local tensor, default: True.
    """

    def __init__(
        self,
        *,
        input_layouts: Optional[Placement] = None,
        output_layouts: Optional[Placement] = None,
        use_local_output: bool = True,
    ):
        super().__init__()
        # input_layouts: how the input arrives (for annotation)
        self.input_layouts = (input_layouts or Replicate(),)
        # desired_input_layouts: what the module needs (Replicate for colwise parallel)
        self.desired_input_layouts = (Replicate(),)
        # output_layouts: desired output placement
        self.output_layouts = (output_layouts or Replicate(),)
        self.use_local_output = use_local_output

    def _partition_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh):
        """Partition the GatedDeltaNet module's parameters."""
        # Columnwise parallel projections: shard weight on dim 0, bias on dim 0
        colwise_projs = ["q_proj", "k_proj", "v_proj", "a_proj", "b_proj", "g_proj"]
        for proj_name in colwise_projs:
            if hasattr(module, proj_name):
                proj = getattr(module, proj_name)
                if proj is not None and isinstance(proj, nn.Linear):
                    proj.weight = nn.Parameter(
                        distribute_tensor(proj.weight, device_mesh, [Shard(0)])
                    )
                    if proj.bias is not None:
                        proj.bias = nn.Parameter(
                            distribute_tensor(proj.bias, device_mesh, [Shard(0)])
                        )

        # Rowwise parallel output projection: shard weight on dim 1, replicate bias
        if hasattr(module, "o_proj") and module.o_proj is not None:
            o_proj = module.o_proj
            if isinstance(o_proj, nn.Linear):
                o_proj.weight = nn.Parameter(
                    distribute_tensor(o_proj.weight, device_mesh, [Shard(1)])
                )
                if o_proj.bias is not None:
                    o_proj.bias = nn.Parameter(
                        distribute_tensor(o_proj.bias, device_mesh, [Replicate()])
                    )

        # A_log and dt_bias: shard on dim 0 to match colwise-sharded a_proj output
        for param_name in ["A_log", "dt_bias"]:
            if hasattr(module, param_name):
                param = getattr(module, param_name)
                if param is not None and isinstance(param, nn.Parameter):
                    new_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
                    setattr(module, param_name, new_param)

        # Convolutions (q_conv1d, k_conv1d, v_conv1d): keep replicated
        # FLA's ShortConvolution uses Triton kernels that access .data directly
        for conv_name in ["q_conv1d", "k_conv1d", "v_conv1d"]:
            if hasattr(module, conv_name):
                conv = getattr(module, conv_name)
                if conv is not None:
                    for p_name, param in conv.named_parameters():
                        if not isinstance(param, DTensor):
                            new_param = nn.Parameter(
                                distribute_tensor(param, device_mesh, [Replicate()])
                            )
                            conv.register_parameter(p_name, new_param)

        # o_norm: replicate (operates on head_v_dim, not the sharded dimension)
        if hasattr(module, "o_norm") and module.o_norm is not None:
            for p_name, param in module.o_norm.named_parameters():
                if not isinstance(param, DTensor):
                    new_param = nn.Parameter(distribute_tensor(param, device_mesh, [Replicate()]))
                    module.o_norm.register_parameter(p_name, new_param)

    @staticmethod
    def _prepare_input_fn(input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        # Annotate input with input_layouts, then redistribute to desired_input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts, run_check=False
            )

        # Redistribute to desired layout if needed (colwise parallel needs Replicate input)
        if input_tensor.placements != desired_input_layouts:
            input_tensor = input_tensor.redistribute(
                placements=desired_input_layouts, async_op=True
            )
        return input_tensor

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # After rowwise parallel o_proj, output has Partial placement, needs reduction
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=partial(
                self._prepare_input_fn, self.input_layouts, self.desired_input_layouts
            ),
            output_fn=partial(self._prepare_output_fn, self.output_layouts, self.use_local_output),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_layouts={self.input_layouts}, "
            f"desired_input_layouts={self.desired_input_layouts}, "
            f"output_layouts={self.output_layouts}, "
            f"use_local_output={self.use_local_output})"
        )


class FLA(nn.Module):
    def __init__(self, inner: "fla.layers.ABCAttention"):
        super().__init__()
        self.inner = inner

        self.kv_cache_manager = None

    def init_kv_cache_manager(self, batch_size: int):
        raise NotImplementedError()

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        **_kwargs,
    ) -> torch.Tensor:
        # FIXME: Right now we just ignore the kwargs.

        if self.kv_cache_manager is not None and self.kv_cache_manager.current_position() == 0:
            raise NotImplementedError()  # prefill
        elif self.kv_cache_manager is not None:
            raise NotImplementedError()  # generate step
        else:
            return self.inner(x, cu_seqlens=cu_doc_lens)[0]  # returns out, ?, cache

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        """
        Apply tensor parallelism to the FLA layer.

        Currently only GatedDeltaNet is supported. GatedDeltaNet has:
        - q_proj, k_proj, v_proj: input projections (columnwise parallel)
        - a_proj, b_proj: gating projections (columnwise parallel)
        - g_proj: optional gate projection (columnwise parallel, only when use_gate=True)
        - A_log, dt_bias: per-head parameters (sharded on head dimension)
        - q_conv1d, k_conv1d, v_conv1d: short convolutions (replicated, Triton kernels)
        - o_norm: output normalization (replicated, operates on head_v_dim)
        - o_proj: output projection (rowwise parallel)
        """
        if float8_enabled:
            raise NotImplementedError("float8 is not yet supported for FLA layers")

        inner = self.inner
        inner_type = type(inner).__name__
        if inner_type != "GatedDeltaNet":
            raise NotImplementedError(
                f"Tensor parallelism is only supported for GatedDeltaNet, but got {inner_type}. "
                "Please file an issue if you need TP support for other FLA layer types."
            )

        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan={
                "inner": GatedDeltaNetParallel(
                    input_layouts=input_layout,
                    output_layouts=output_layout,
                    use_local_output=use_local_output,
                )
            },
        )

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Calculate FLOPs per token for FLA (Flash Linear Attention) layer.

        This accounts for:
        - Linear projections (Q, K, V, A, B, optional G, and output)
        - Linear attention computation (O(n) complexity instead of O(n^2))
        - Optional short convolutions
        """
        # TODO(tylerr): this is a quick and dirty estimate, not thoroughly checked.

        # Get inner layer attributes
        inner = self.inner
        num_heads = getattr(inner, "num_heads", None)
        head_dim = getattr(inner, "head_dim", None)
        if num_heads is None or head_dim is None:
            raise ValueError("num_heads and head_dim must be set")

        # Count convolution parameters separately to avoid double-counting
        conv_params = 0
        conv_flops = 0
        for conv_name in ["q_conv1d", "k_conv1d", "v_conv1d"]:
            if hasattr(inner, conv_name) and getattr(inner, conv_name) is not None:
                conv = getattr(inner, conv_name)
                if hasattr(conv, "weight"):
                    # Count conv parameters
                    conv_params += conv.weight.numel()
                    if conv.bias is not None:
                        conv_params += conv.bias.numel()

                    # Conv1d computation FLOPs account for sliding over sequence
                    # Weight shape: (out_channels, in_channels/groups, kernel_size)
                    # For depthwise conv: groups == in_channels == out_channels
                    kernel_size = conv.weight.shape[-1] if len(conv.weight.shape) > 0 else 1
                    out_channels = conv.weight.shape[0] if len(conv.weight.shape) > 0 else 1

                    # Computation: kernel_size * (in_channels/groups) * out_channels * seq_len
                    # For depthwise: in_channels/groups = 1, so = kernel_size * out_channels * seq_len
                    # 6x multiplier for forward+backward
                    in_channels_per_group = (
                        conv.weight.shape[1] if len(conv.weight.shape) > 1 else 1
                    )
                    conv_flops += 6 * kernel_size * in_channels_per_group * out_channels * seq_len

        # 6 FLOPs per parameter (2 ops * 3 for forward+backward)
        # Exclude convolution parameters since we count their computation separately
        all_params = sum(p.numel() for p in self.parameters())
        param_flops = 6 * (all_params - conv_params)

        # Linear attention computation (O(1) per token, O(n) total)
        # For linear attention mechanisms like GatedDeltaNet:
        # - Uses recurrence/state-space models: each token updates a fixed-size state
        # - Per-token computation is constant (doesn't scale with seq_len)
        # - Core operations per token:
        #   * Q, K, V processing: ~3 * num_heads * head_dim
        #   * State update (recurrence): ~num_heads * head_dim
        #   * Output computation: ~num_heads * head_dim
        # - Total: ~5 * num_heads * head_dim per token
        # - 12x multiplier: accounts for forward+backward pass (2x) and various ops (6x)
        # Note: This is O(1) per token vs O(seq_len) per token for quadratic attention
        attn_flops = 12 * num_heads * head_dim

        return param_flops + attn_flops + conv_flops


@dataclass
class FLAConfig(Config):
    name: str
    fla_layer_kwargs: dict = field(default_factory=dict)
    dtype: DType = DType.float32

    def build(self, d_model: int, n_heads: int, init_device) -> FLA:
        import fla.layers

        layer = getattr(fla.layers, self.name)(
            hidden_size=d_model,
            num_heads=n_heads,
            **self.fla_layer_kwargs,
        ).to(device=init_device, dtype=self.dtype.as_pt())

        return FLA(layer)

    def num_params(self):
        raise NotImplementedError()
