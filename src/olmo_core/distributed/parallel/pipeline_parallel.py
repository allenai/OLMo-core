import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._tensor import DTensor
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    _PipelineSchedule,
    get_schedule_class,
)

from olmo_core.config import Config, StrEnum
from olmo_core.exceptions import OLMoConfigurationError


class PipelineScheduleType(StrEnum):
    """
    An enumeration of the different pipeline schedules available.
    """

    # See torch.distributed.pipelining.schedules.get_schedule_class for a list of available.

    single_1F1B = "1F1B"
    interleaved_1F1B = "Interleaved1F1B"
    gpipe = "GPipe"
    looped_bfs = "LoopedBFS"
    interleaved_zero_bubble = "InterleavedZeroBubble"

    @property
    def is_single_stage(self) -> bool:
        try:
            return issubclass(get_schedule_class(self), PipelineScheduleSingle)
        except ValueError as e:
            raise OLMoConfigurationError(f"Invalid pipeline schedule '{self}'") from e

    @property
    def is_multi_stage(self) -> bool:
        return not self.is_single_stage


@dataclass
class PipelineParallelConfig(Config):
    """
    Configuration class for pipeline parallelism (PP).
    """

    degree: int
    """
    The PP degree.
    """

    schedule: PipelineScheduleType
    """
    The name of the schedule.
    """

    def stage_ids_this_rank(
        self, pp_rank: int, num_stages: int, style: str = "loop"
    ) -> Tuple[int, ...]:
        """
        Compute the stage ids for the stages that will run on this pp rank for either a looped or
        V style schedule.
        """
        if num_stages % self.degree != 0:
            raise OLMoConfigurationError(
                f"num_stages {num_stages} must be evenly divisible by pipeline size {self.degree}"
            )

        stages_per_rank = num_stages // self.degree
        if style == "loop":
            return tuple(pp_rank + s * self.degree for s in range(stages_per_rank))
        elif style == "v":
            assert (
                stages_per_rank == 2
            ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
            stage_v_pairs = list(
                zip(range(self.degree), range(num_stages - 1, self.degree - 1, -1))
            )
            return stage_v_pairs[pp_rank]
        else:
            raise NotImplementedError(style)


class PipelineSchedule:
    """
    A thin wrapper around PyTorch pipeline schedule classes.

    :param n_microbatches: How many microbatches to split the global training batch into.
        If global training batch size must be evenly divisible by this.
        If not specified, the default will be the number of pipeline stages.
    """

    def __init__(
        self,
        *,
        model_parts: List[nn.Module],
        stages: List[PipelineStage],
        pp_mesh: DeviceMesh,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        schedule_name: PipelineScheduleType,
        n_microbatches: Optional[int] = None,
    ):
        self.model_parts = model_parts
        self.stages = stages
        self.pp_mesh = pp_mesh
        self.loss_fn = loss_fn

        try:
            schedule_class = get_schedule_class(schedule_name)
        except ValueError as e:
            raise OLMoConfigurationError(f"Invalid pipeline schedule name '{schedule_name}'") from e

        if n_microbatches is None:
            n_microbatches = pp_mesh.size()

        schedule: _PipelineSchedule
        if issubclass(schedule_class, PipelineScheduleSingle):
            if len(model_parts) > 1:
                raise OLMoConfigurationError(
                    f"Expected a single stage for '{schedule_name}' pipeline schedule"
                )
            schedule = schedule_class(
                stages[0], n_microbatches=n_microbatches, loss_fn=self.loss_fn
            )
        elif issubclass(schedule_class, PipelineScheduleMulti):
            schedule = schedule_class(
                stages,  # type: ignore[arg-type]
                n_microbatches=n_microbatches,
                loss_fn=self.loss_fn,
            )
        else:
            raise NotImplementedError(schedule_class)

        self.base_schedule = schedule

    @property
    def is_first_stage(self) -> bool:
        return self.pp_mesh.get_local_rank() == 0

    @property
    def is_last_stage(self) -> bool:
        return self.pp_mesh.get_local_rank() == self.pp_mesh.size() - 1

    def step(
        self,
        *args,
        target: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        if self.is_first_stage:
            self.base_schedule.step(*args, **kwargs)
            return None, None
        elif self.is_last_stage:
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
