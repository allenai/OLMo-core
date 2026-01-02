import logging
from dataclasses import dataclass, field
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
from olmo_core.nn.utils import get_tp_wrappers

log = logging.getLogger(__name__)


class ShardParameter(ParallelStyle):
    """
    A ParallelStyle that shards a specific parameter (not a submodule) on a given dimension.

    This is used for parameters like A_log and dt_bias in GatedDeltaNet that are
    nn.Parameter attributes directly on the module, not inside submodules like nn.Linear.
    """

    def __init__(self, param_name: str, shard_dim: int = 0):
        super().__init__()
        self.param_name = param_name
        self.shard_dim = shard_dim

    def _partition_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh):
        if hasattr(module, self.param_name):
            param = getattr(module, self.param_name)
            if param is not None and not isinstance(param, DTensor):
                new_param = nn.Parameter(
                    distribute_tensor(param, device_mesh, [Shard(self.shard_dim)])
                )
                setattr(module, self.param_name, new_param)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(module, device_mesh, partition_fn=self._partition_fn)


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

        IMPORTANT: All parallel styles use use_local_output=True because FLA's internal
        Triton kernels (convolutions, etc.) cannot work with DTensors - they access .data directly.
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

        rowwise_parallel, colwise_parallel, prepare_module_input = get_tp_wrappers(
            float8_enabled=float8_enabled
        )

        # Build parallelization plan for GatedDeltaNet
        # All styles use use_local_output=True because FLA's Triton kernels need local tensors
        plan: dict[str, ParallelStyle] = {}

        # Handle input: redistribute to Replicate if needed, output local tensor
        plan["inner"] = prepare_module_input(
            input_layouts=None if input_layout is None else (input_layout,),
            desired_input_layouts=(Replicate(),),
            use_local_output=True,  # FLA needs local tensors
        )

        # Input projections (columnwise parallel - shard output dimension)
        # All use use_local_output=True so Triton kernels get local tensors
        for proj_name in ["q_proj", "k_proj", "v_proj", "a_proj", "b_proj"]:
            if hasattr(inner, proj_name) and getattr(inner, proj_name) is not None:
                plan[f"inner.{proj_name}"] = colwise_parallel(use_local_output=True)

        # g_proj: optional gate projection (only when use_gate=True)
        if hasattr(inner, "g_proj") and inner.g_proj is not None:
            plan["inner.g_proj"] = colwise_parallel(use_local_output=True)

        # Output projection (rowwise parallel - shard input dimension)
        # This one handles the final output layout
        if hasattr(inner, "o_proj") and inner.o_proj is not None:
            plan["inner.o_proj"] = rowwise_parallel(
                output_layouts=output_layout,
                use_local_output=use_local_output,
            )

        parallelize_module(module=self, device_mesh=tp_mesh, parallelize_plan=plan)

        # A_log and dt_bias: shard on dim 0 to match colwise-sharded a_proj output
        # These are nn.Parameter attributes, not submodules, so we handle them separately
        for param_name in ["A_log", "dt_bias"]:
            if hasattr(inner, param_name):
                param = getattr(inner, param_name)
                if param is not None and not isinstance(param, DTensor):
                    new_param = nn.Parameter(distribute_tensor(param, tp_mesh, [Shard(0)]))
                    setattr(inner, param_name, new_param)

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
