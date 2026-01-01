import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard

from olmo_core.config import Config, DType
from olmo_core.nn.utils import get_tp_wrappers

log = logging.getLogger(__name__)


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
        - q_conv1d, k_conv1d, v_conv1d: short convolutions (when use_short_conv=True)
        - o_norm: output normalization (operates on head_v_dim, no sharding needed)
        - o_proj: output projection (rowwise parallel)
        """
        inner = self.inner
        inner_type = type(inner).__name__

        # Only support GatedDeltaNet for now
        # TODO: implementation specific factorization of FLA variants instead of just one wrapper class
        if inner_type != "GatedDeltaNet":
            raise NotImplementedError(
                f"Tensor parallelism is only supported for GatedDeltaNet, "
                f"but got {inner_type}. Please file an issue if you need TP support "
                f"for other FLA layer types."
            )

        rowwise_parallel, colwise_parallel, prepare_module_input = get_tp_wrappers(
            float8_enabled=float8_enabled
        )

        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan=prepare_module_input(
                input_layouts=None if input_layout is None else (input_layout,),
                desired_input_layouts=(Replicate(),),
            ),
        )

        # Build parallelization plan for GatedDeltaNet
        plan = {}

        # Input projections (columnwise parallel - shard output dimension)
        # q_proj, k_proj, v_proj: standard attention-like projections
        # a_proj, b_proj: GatedDeltaNet-specific gating projections (output num_heads)
        for proj_name in ["q_proj", "k_proj", "v_proj", "a_proj", "b_proj"]:
            assert hasattr(inner, proj_name) and getattr(inner, proj_name) is not None
            plan[f"inner.{proj_name}"] = colwise_parallel()

        # g_proj: optional gate projection (only when use_gate=True)
        if hasattr(inner, "g_proj") and inner.g_proj is not None:
            plan["inner.g_proj"] = colwise_parallel()

        # Output projection (rowwise parallel - shard input dimension)
        assert hasattr(inner, "o_proj") and getattr(inner, "o_proj") is not None
        plan["inner.o_proj"] = rowwise_parallel(
            output_layouts=output_layout, use_local_output=use_local_output
        )

        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan=plan,
        )

        # Shard per-head parameters to match sharded a_proj/b_proj outputs.
        # A_log and dt_bias are [num_heads] tensors used with a_proj output in:
        #   g = -A_log.exp() * softplus(a_proj(x) + dt_bias)
        # They must be sharded on dim 0 to match the colwise-sharded a_proj output.
        inner.A_log = nn.Parameter(
            distribute_tensor(cast(torch.Tensor, inner.A_log.data), tp_mesh, [Shard(0)])
        )
        inner.dt_bias = nn.Parameter(
            distribute_tensor(cast(torch.Tensor, inner.dt_bias.data), tp_mesh, [Shard(0)])
        )

        # Shard ShortConvolution layers (when use_short_conv=True).
        #
        # FLA's ShortConvolution uses custom Triton kernels that access weight.data directly
        # and cannot handle DTensor. We keep conv weights as local (sliced) tensors for runtime,
        # but register state dict hooks to convert to/from DTensor for checkpoint compatibility.
        tp_rank = dist.get_rank(tp_mesh.get_group())
        tp_world_size = dist.get_world_size(tp_mesh.get_group())

        # Track which conv params need DTensor conversion for checkpointing
        conv_tp_params: Dict[str, Tuple[str, torch.Size]] = {}  # param_name -> (conv_name, global_shape)

        for conv_name in ["q_conv1d", "k_conv1d", "v_conv1d"]:
            if not (hasattr(inner, conv_name) and getattr(inner, conv_name) is not None):
                continue

            conv = getattr(inner, conv_name)
            if not (hasattr(conv, "weight") and hasattr(conv, "groups")):
                raise NotImplementedError(
                    f"Don't know how to shard {conv_name} of type {type(conv).__name__}"
                )

            # Expect depthwise Conv1d: groups == in_channels == out_channels
            in_channels = int(getattr(conv, "in_channels", conv.weight.shape[0]))
            out_channels = int(getattr(conv, "out_channels", conv.weight.shape[0]))
            groups = int(conv.groups)
            if not (groups == in_channels == out_channels):
                raise NotImplementedError(
                    f"TP sharding only implemented for depthwise Conv1d in {conv_name}, "
                    f"but got in={in_channels}, out={out_channels}, groups={groups}"
                )

            if out_channels % tp_world_size != 0:
                raise ValueError(
                    f"{conv_name} out_channels={out_channels} must be divisible by tp_world_size={tp_world_size}"
                )

            local_channels = out_channels // tp_world_size
            start = tp_rank * local_channels
            end = start + local_channels

            # Record global shape before slicing for checkpoint hooks
            conv_tp_params[f"{conv_name}.weight"] = (conv_name, conv.weight.shape)
            if conv.bias is not None:
                conv_tp_params[f"{conv_name}.bias"] = (conv_name, conv.bias.shape)

            # Conv1d weight shape is (out_channels, in_channels/groups, kernel_size).
            # For depthwise conv, in_channels/groups == 1, so slicing dim 0 is correct.
            w_local = conv.weight.detach()[start:end].contiguous()
            conv.weight = nn.Parameter(w_local)
            if conv.bias is not None:
                b_local = conv.bias.detach()[start:end].contiguous()
                conv.bias = nn.Parameter(b_local)

            conv.in_channels = local_channels
            conv.out_channels = local_channels
            conv.groups = local_channels

            # Some implementations keep a separate attribute used by kernels.
            if hasattr(conv, "hidden_size"):
                conv.hidden_size = local_channels

        # Register state dict hooks to handle conv params as DTensors for checkpoint compatibility.
        # This allows loading checkpoints saved with different TP configurations.
        if conv_tp_params:
            self._conv_tp_params = conv_tp_params
            self._tp_mesh = tp_mesh

            def state_dict_hook(
                module: "FLA",
                state_dict: Dict[str, Any],
                prefix: str,
                _local_metadata: Dict[str, Any],
            ) -> None:
                """Convert local conv tensors to DTensors when saving state dict."""
                for param_name in module._conv_tp_params:
                    key = f"{prefix}inner.{param_name}"
                    if key in state_dict:
                        local_tensor = state_dict[key]
                        state_dict[key] = distribute_tensor(
                            local_tensor, module._tp_mesh, [Shard(0)]
                        )

            def load_state_dict_hook(
                module: "FLA",
                state_dict: Dict[str, Any],
                prefix: str,
                _local_metadata: Dict[str, Any],
                _strict: bool,
                _missing_keys: List[str],
                _unexpected_keys: List[str],
                _error_msgs: List[str],
            ) -> None:
                """Convert DTensors to local tensors when loading state dict."""
                for param_name, (_, global_shape) in module._conv_tp_params.items():
                    key = f"{prefix}inner.{param_name}"
                    if key in state_dict:
                        tensor = state_dict[key]
                        if isinstance(tensor, DTensor):
                            # Already a DTensor from checkpoint - extract local shard
                            state_dict[key] = tensor.to_local()
                        elif tensor.shape != global_shape:
                            # Already a local tensor with correct shape - nothing to do
                            pass
                        else:
                            # Full tensor from non-TP checkpoint - slice it
                            tp_rank = dist.get_rank(module._tp_mesh.get_group())
                            tp_world_size = dist.get_world_size(module._tp_mesh.get_group())
                            local_size = tensor.shape[0] // tp_world_size
                            start = tp_rank * local_size
                            end = start + local_size
                            state_dict[key] = tensor[start:end].contiguous()

            self._register_state_dict_hook(state_dict_hook)
            self._register_load_state_dict_pre_hook(load_state_dict_hook, with_module=True)

        # o_norm: normalizes over head_v_dim (last dimension), not the head dimension.
        # Since heads are sharded but each shard has complete head_v_dim vectors,
        # o_norm can operate locally without sharding its weights.

        for name, param in inner.named_parameters():
            if isinstance(param, DTensor):
                log.info(f"{name}: is_dtensor=True, placements={param.placements}")
            else:
                log.info(f"{name}: is_dtensor=False")

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
