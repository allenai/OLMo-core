import logging
from dataclasses import dataclass, field
from typing import Optional, cast

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

    def forward(self, x: torch.Tensor, **_kwargs) -> torch.Tensor:
        # FIXME: Right now we just ignore the kwargs.

        if self.kv_cache_manager is not None and self.kv_cache_manager.current_position() == 0:
            raise NotImplementedError()  # prefill
        elif self.kv_cache_manager is not None:
            raise NotImplementedError()  # generate step
        else:
            return self.inner(x)[0]  # returns out, ?, cache

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
        # IMPORTANT: FLA's ShortConvolution runs custom Triton kernels, which do not work with DTensor
        # parameters. Instead we *materialize per-rank local Conv1d parameters* by slicing along the
        # channel dimension and updating the Conv1d metadata (in/out/groups). This matches the
        # colwise-sharded outputs of q_proj/k_proj/v_proj.
        tp_rank = dist.get_rank(tp_mesh.get_group())
        tp_world_size = dist.get_world_size(tp_mesh.get_group())

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

        # o_norm: normalizes over head_v_dim (last dimension), not the head dimension.
        # Since heads are sharded but each shard has complete head_v_dim vectors,
        # o_norm can operate locally without sharding its weights.

        for name, param in inner.named_parameters():
            if isinstance(param, DTensor):
                log.info(f"{name}: is_dtensor=True, placements={param.placements}")
            else:
                log.info(f"{name}: is_dtensor=False")


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
