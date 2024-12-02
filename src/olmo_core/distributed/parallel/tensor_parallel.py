from functools import partial
from typing import Optional

import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Shard, distribute_module
from torch.distributed.tensor.parallel import SequenceParallel as _SequenceParallel
from torch.distributed.tensor.placement_types import Placement


class SequenceParallel(_SequenceParallel):
    def __init__(
        self,
        *,
        sequence_dim: int = 1,
        use_local_output: bool = False,
        output_layouts: Optional[Placement] = None,
    ):
        super().__init__(sequence_dim=sequence_dim, use_local_output=use_local_output)
        self.output_layouts = (output_layouts or Shard(1),)

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        del mod, device_mesh
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._replicate_module_fn,
            partial(self._prepare_input_fn, self.sequence_sharding),  # type: ignore
            partial(self._prepare_output_fn, self.output_layouts, self.use_local_output),  # type: ignore
        )
