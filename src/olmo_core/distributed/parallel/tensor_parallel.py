import logging
from dataclasses import dataclass
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Shard, distribute_module
from torch.distributed.tensor.parallel import SequenceParallel as _SequenceParallel

from olmo_core.config import Config

log = logging.getLogger(__name__)


@dataclass
class TensorParallelConfig(Config):
    """
    Configuration class for tensor parallelism (TP).
    """

    degree: int
    """
    The TP degree.
    """

    enable_async: bool = False
    """
    Enable experimental async tensor parallelism.
    """

    def maybe_enable_async_tp(self, tp_mesh: DeviceMesh):
        if self.enable_async:
            # https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/
            log.info("Enabling async tensor parallel")

            from torch.distributed._symmetric_memory import enable_symm_mem_for_group

            torch._inductor.config._micro_pipeline_tp = True  # type: ignore
            enable_symm_mem_for_group(tp_mesh.get_group().group_name)


class SequenceParallel(_SequenceParallel):
    def __init__(
        self,
        *,
        sequence_dim: int = 1,
        use_local_output: bool = False,
        output_layouts: Optional[Placement] = None,
    ):
        super().__init__(sequence_dim=sequence_dim, use_local_output=use_local_output)
        self.output_layouts = (output_layouts or Shard(sequence_dim),)

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
