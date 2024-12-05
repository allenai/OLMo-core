import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._tensor import DTensor
from torch.distributed.pipelining.schedules import _PipelineSchedule

from olmo_core.config import Config


@dataclass
class PipelineParallelConfig(Config):
    """
    Configuration class for pipeline parallelism (PP).
    """

    degree: int
    """
    The PP degree.
    """


class PipelineSchedule:
    def __init__(
        self,
        *,
        model_parts: List[nn.Module],
        pp_mesh: DeviceMesh,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ):
        self.model_parts = model_parts
        self.pp_mesh = pp_mesh
        self._loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
        self._base_schedule: Optional[_PipelineSchedule] = None

        if loss_fn is not None:
            self.loss_fn = loss_fn

    def _lazy_init(self):
        if self._base_schedule is None:
            self._base_schedule = self.build_base_schedule()

    @property
    def loss_fn(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self._loss_fn is None:
            raise RuntimeError("pipeline schedule's loss function has not been set yet!")
        return self._loss_fn

    @loss_fn.setter
    def loss_fn(self, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self._loss_fn = loss_fn
        self._base_schedule = None
        self._lazy_init()

    @property
    def base_schedule(self) -> _PipelineSchedule:
        if self._base_schedule is None:
            raise RuntimeError("pipeline base schedule has not been built yet!")
        return self._base_schedule

    def build_base_schedule(self) -> _PipelineSchedule:
        raise NotImplementedError

    def step(
        self,
        *args,
        target: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        is_last_stage = self.pp_mesh.get_local_rank() == self.pp_mesh.size() - 1
        if self.pp_mesh.get_local_rank() == 0:
            self.base_schedule.step(*args, **kwargs)
            return None, None
        elif is_last_stage:
            losses: List[torch.Tensor] = []
            output = self.base_schedule.step(target=target, losses=losses)
            return output, torch.stack(losses)
        else:
            self.base_schedule.step()
            return None, None

    def clip_grad_norm_(
        self, max_norm: float, norm_type: float = 2.0, foreach: Optional[bool] = None
    ) -> torch.Tensor:
        parameters = [p for m in self.model_parts for p in m.parameters()]
        grads = [p.grad for p in parameters if p.grad is not None]

        total_norm = nn.utils.get_total_norm(grads, norm_type, False, True)
        if isinstance(total_norm, DTensor):
            # Will reach here if PP + other parallelism is used. If only using PP, total_norm will be a local tensor.
            # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
            # We can simply reduce the DTensor to get the total norm in this tensor's process group
            # and then convert it to a local tensor
            total_norm = total_norm.full_tensor()

        # TODO: cleanup maybe using DTensor
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=self.pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=self.pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

        torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach=foreach)
        return total_norm
