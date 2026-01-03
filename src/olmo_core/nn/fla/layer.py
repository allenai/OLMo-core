import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import (
    DTensor,
    Shard,
    distribute_module,
    distribute_tensor,
)
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement

from olmo_core.config import Config, DType

log = logging.getLogger(__name__)


class ShardParameters(ParallelStyle):
    """
    A ParallelStyle that shards multiple parameters on the module as DTensors.

    This is used for parameters like A_log and dt_bias in GatedDeltaNet that are
    nn.Parameter attributes directly on the module, not inside submodules like nn.Linear.

    Use with the "" (empty string) key in a parallelize_plan to apply to the current module.

    Args:
        param_specs: List of (param_name, shard_dim) tuples specifying which parameters
            to shard and on which dimension.

    Example:
        plan = {
            "": ShardParameters([("A_log", 0), ("dt_bias", 0)]),
            "A_log_id": PrepareModuleOutput(..., use_local_output=True),
        }
    """

    def __init__(self, param_specs: list[tuple[str, int]]):
        super().__init__()
        self.param_specs = param_specs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        for param_name, shard_dim in self.param_specs:
            if hasattr(module, param_name):
                param = getattr(module, param_name)
                if param is not None and not isinstance(param, DTensor):
                    new_param = nn.Parameter(
                        distribute_tensor(param, device_mesh, [Shard(shard_dim)])
                    )
                    setattr(module, param_name, new_param)
        return module


class ShardModule(ParallelStyle):
    """
    A ParallelStyle that shards all parameters of a module as DTensors.

    This uses distribute_module to shard parameters. The module will have DTensor
    parameters for checkpoint compatibility, and PyTorch's DTensor dispatch will
    handle the computation.

    Args:
        shard_dim: Dimension to shard all parameters on. Default: 0.
    """

    def __init__(self, shard_dim: int = 0):
        super().__init__()
        self.shard_dim = shard_dim

    def _partition_fn(self, name: str, module: nn.Module, device_mesh: DeviceMesh):
        # Shard all parameters on the specified dimension
        for param_name, param in module.named_parameters():
            if param is not None and not isinstance(param, DTensor):
                new_param = nn.Parameter(
                    distribute_tensor(param, device_mesh, [Shard(self.shard_dim)])
                )
                module.register_parameter(param_name, new_param)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
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
        self.inner.apply_tp(
            tp_mesh,
            input_layout,
            output_layout,
            use_local_output,
            float8_enabled,
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
        if self.name == "GatedDeltaNet":
            from olmo_core.nn.fla.gated_deltanet import GatedDeltaNet

            layer = GatedDeltaNet(
                hidden_size=d_model,
                num_heads=n_heads,
                **self.fla_layer_kwargs,
            ).to(device=init_device, dtype=self.dtype.as_pt())
        else:
            raise NotImplementedError(f"Layer {self.name} not implemented")
            import fla.layers

            layer = getattr(fla.layers, self.name)(
                hidden_size=d_model,
                num_heads=n_heads,
                **self.fla_layer_kwargs,
            ).to(device=init_device, dtype=self.dtype.as_pt())

        return FLA(layer)

    def num_params(self):
        raise NotImplementedError()
