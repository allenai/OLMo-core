import logging
from dataclasses import dataclass, field
from typing import Optional

import fla.layers
import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.placement_types import Placement, Replicate, Shard

from olmo_core.config import Config, DType
from olmo_core.nn.utils import get_tp_wrappers

log = logging.getLogger(__name__)


class FLA(nn.Module):
    def __init__(self, inner: fla.layers.ABCAttention):
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
        - g_proj: optional gate projection (columnwise parallel)
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
        # g_proj: optional gate projection (only when use_gate=True)
        for proj_name in ["q_proj", "k_proj", "v_proj", "a_proj", "b_proj", "g_proj"]:
            assert hasattr(inner, proj_name) and getattr(inner, proj_name) is not None
            plan[f"inner.{proj_name}"] = colwise_parallel()

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
        inner.A_log = nn.Parameter(distribute_tensor(inner.A_log.data, tp_mesh, [Shard(0)]))
        inner.dt_bias = nn.Parameter(distribute_tensor(inner.dt_bias.data, tp_mesh, [Shard(0)]))


@dataclass
class FLAConfig(Config):
    name: str
    fla_layer_kwargs: dict = field(default_factory=dict)
    dtype: DType = DType.float32

    def build(self, d_model: int, n_heads: int, init_device) -> FLA:
        layer = getattr(fla.layers, self.name)(
            hidden_size=d_model,
            num_heads=n_heads,
            **self.fla_layer_kwargs,
        ).to(device=init_device, dtype=self.dtype.as_pt())

        return FLA(layer)

    def num_params(self):
        raise NotImplementedError()
