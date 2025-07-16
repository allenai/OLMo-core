from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    _PipelineSchedule,
    get_schedule_class,
)

from olmo_core.config import Config, StrEnum
from olmo_core.exceptions import OLMoConfigurationError


class PipelineSplitStyle(StrEnum):
    loop = "loop"
    v = "v"


class PipelineScheduleType(StrEnum):
    """
    An enumeration of the different pipeline schedules available.

    .. warning::
        The zero-bubble variants have several issues at the moment including not being compatible
        with ``torch.compile``.

    """

    # See torch.distributed.pipelining.schedules.get_schedule_class for a list of available.

    single_1F1B = "1F1B"
    interleaved_1F1B = "Interleaved1F1B"
    gpipe = "GPipe"
    looped_bfs = "LoopedBFS"
    interleaved_zero_bubble = "InterleavedZeroBubble"
    zbv_zero_bubble = "ZBVZeroBubble"
    custom_1F1B = "Custom1F1B"
    custom_interleaved_1F1B = "CustomInterleaved1F1B"

    @property
    def is_single_stage(self) -> bool:
        try:
            if self.startswith("Custom"):
                if self == self.custom_1F1B:
                    return True
                elif self == self.custom_interleaved_1F1B:
                    return False
            return issubclass(get_schedule_class(self), PipelineScheduleSingle)
        except ValueError as e:
            raise OLMoConfigurationError(f"Invalid pipeline schedule '{self}'") from e

    @property
    def is_multi_stage(self) -> bool:
        return not self.is_single_stage

    @property
    def default_style(self) -> PipelineSplitStyle:
        if self == self.zbv_zero_bubble:
            return PipelineSplitStyle.v
        else:
            return PipelineSplitStyle.loop


@dataclass
class PipelineParallelConfig(Config):
    """
    Configuration class for pipeline parallelism (PP).
    """

    degree: int
    """
    The PP degree.
    """

    schedule: PipelineScheduleType = PipelineScheduleType.interleaved_1F1B
    """
    The name of the schedule.
    """

    style: Optional[PipelineSplitStyle] = None
    """
    The split style.
    """

    def infer_style(self) -> PipelineSplitStyle:
        if self.style is not None:
            return self.style
        else:
            return self.schedule.default_style

    def final_stage_rank(self) -> int:
        style = self.infer_style()
        if style == PipelineSplitStyle.loop:
            return self.degree - 1
        elif style == PipelineSplitStyle.v:
            return 0
        else:
            raise NotImplementedError(style)

    def rank_completion_order(self) -> Iterable[int]:
        """
        The order that ranks within the PP group will complete a batch.
        """
        style = self.infer_style()
        if style == PipelineSplitStyle.loop:
            return range(self.degree - 1, -1, -1)
        elif style == PipelineSplitStyle.v:
            return range(self.degree)
        else:
            raise NotImplementedError(style)

    def stage_ids_this_rank(self, pp_rank: int, num_stages: int) -> Tuple[int, ...]:
        """
        Compute the stage ids for the stages that will run on this pp rank for either a looped or
        V style schedule.
        """
        style = self.infer_style()
        if num_stages % self.degree != 0:
            raise OLMoConfigurationError(
                f"num_stages {num_stages} must be evenly divisible by pipeline size {self.degree}"
            )

        stages_per_rank = num_stages // self.degree
        if style == PipelineSplitStyle.loop:
            return tuple(pp_rank + s * self.degree for s in range(stages_per_rank))
        elif style == PipelineSplitStyle.v:
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
        schedule_name: PipelineScheduleType,
        loss_fn: Optional[Callable[[Any, torch.Tensor], torch.Tensor]] = None,
        num_microbatches: Optional[int] = None,
    ):
        self.model_parts = model_parts
        self.stages = stages
        self.pp_mesh = pp_mesh
        self.loss_fn = loss_fn

        try:
            if schedule_name.startswith("Custom"):
                if schedule_name == PipelineScheduleType.custom_1F1B:
                    # Custom 1F1B schedule
                    from olmo_core.train.train_module.transformer.pipeline_schedule import (
                        CustomSchedule1F1B,
                    )
                    schedule_class = CustomSchedule1F1B
                elif schedule_name == PipelineScheduleType.custom_interleaved_1F1B:
                    # Custom interleaved 1F1B schedule
                    from olmo_core.train.train_module.transformer.pipeline_schedule import (
                        CustomScheduleInterleaved1F1B,
                    )
                    schedule_class = CustomScheduleInterleaved1F1B
            else:
                # pytorch native PP schedule
                schedule_class = get_schedule_class(schedule_name)
        except ValueError as e:
            raise OLMoConfigurationError(f"Invalid pipeline schedule name '{schedule_name}'") from e

        if num_microbatches is None:
            num_microbatches = pp_mesh.size()

        schedule: _PipelineSchedule
        if issubclass(schedule_class, PipelineScheduleSingle):
            if len(model_parts) > 1:
                raise OLMoConfigurationError(
                    f"Expected a single stage for '{schedule_name}' pipeline schedule"
                )
            schedule = schedule_class(
                stages[0], n_microbatches=num_microbatches, loss_fn=self.loss_fn
            )
        elif issubclass(schedule_class, PipelineScheduleMulti):
            schedule = schedule_class(
                stages,  # type: ignore[arg-type]
                n_microbatches=num_microbatches,
                loss_fn=self.loss_fn,
            )
        else:
            raise NotImplementedError(schedule_class)

        self.base_schedule = schedule
        self.num_microbatches = num_microbatches

    @cached_property
    def has_first_stage(self) -> bool:
        for stage in self.stages:
            if stage.is_first:
                return True
        return False

    @cached_property
    def has_last_stage(self) -> bool:
        for stage in self.stages:
            if stage.is_last:
                return True
        return False

    def step(
        self,
        *args,
        target: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        """
        :param args: Only passed to first stage.
        :param kwargs: Passed to all stages.
        """
        losses: Optional[List[torch.Tensor]] = None
        if self.has_last_stage and self.loss_fn is not None:
            losses = []
        else:
            target = None

        if not self.has_first_stage:
            args = () # If there is no first stage, we need to provide an empty tuple for single_1F1B

        output = self.base_schedule.step(*args, target=target, losses=losses, **kwargs)
        return output, None if losses is None else torch.stack(losses)
