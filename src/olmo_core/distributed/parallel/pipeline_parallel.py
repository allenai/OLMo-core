from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.pipelining import PipelineStage


from olmo_core.config import Config, StrEnum
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.lm_head import LMOutputWithLoss

import logging
logger = logging.getLogger(__name__)

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
            if self == self.custom_1F1B:
                return True
            elif self == self.custom_interleaved_1F1B:
                return False
            else:
                raise OLMoConfigurationError(f"Invalid pipeline schedule '{self}'")

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

import os
import re
from typing import Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
def _parse_cell(op: Any):
    """Return (kind, mb) where kind in {'F','B','idle'} and mb is Optional[int]."""
    if op is None:
        return "idle", None
    if isinstance(op, tuple):
        # allow e.g. ('F', 7) or ('B', 3)
        stage_id, k, mb = op
        return (str(k).upper() if k in ('F','B') else 'idle', int(mb))
    s = str(op).strip()
    if s.lower() == "idle":
        return "idle", None
    # accept 'F7', 'B12', '1F7', 'stage3B4', etc.
    m = re.search(r'([FBfb])\s*([0-9]+)', s)
    if m:
        return m.group(1).upper(), int(m.group(2))
    return "idle", None


def draw_pipeline_timeline(pp_order: Dict[int, List[Any]],
                           title: str = "Pipeline schedule",
                           figsize_per_cell: float = 0.35,
                           outpath: Optional[str] = None):

    ranks = sorted(pp_order.keys())
    assert ranks == list(range(len(ranks))), "pp_order keys should be 0..N-1"
    num_ranks = len(ranks)
    num_steps = len(pp_order[0])

    # pad rows to same length
    for r in ranks:
        if len(pp_order[r]) < num_steps:
            pp_order[r] = pp_order[r] + [None] * (num_steps - len(pp_order[r]))
        elif len(pp_order[r]) > num_steps:
            raise ValueError(f"pp_order[{r}] has length {len(pp_order[r])} > {num_steps}")

    # ---- per-column width: 2 if any rank does backward at that step, else 1
    def is_bwd(cell):
        if cell is None:
            return False
        _, kind, _, need_offload, need_reload = cell
        return getattr(kind, "name", "") in {"FULL_BACKWARD", "BACKWARD", "BWD"}

    # col_w = [2 if any(is_bwd(pp_order[r][t]) for r in ranks) else 1 for t in range(num_steps)]
    # x_edges = [0]
    # for w in col_w:
    #     x_edges.append(x_edges[-1] + w)
    # total_w = x_edges[-1]
    total_w = []
    for r in ranks:
        w = 0
        for t in range(num_steps):
            if is_bwd(pp_order[r][t]):
                w += 2
            else:
                w += 1
        total_w.append(w)
    
    # assert all rows have the same total width
    if not all(w == total_w[0] for w in total_w):
        assert False, f"Not all rows have the same total width: {total_w}"
    total_w = total_w[0]
        # ---- figure sizing
    width = max(8, total_w * figsize_per_cell)
    height = max(2.5, num_ranks * 0.55 * 2 + 1.4)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    ax.set_title(title, fontsize=14, pad=10)

    # colors
    C_FWD   = "#3B64C3"
    C_FWD_2 = "#A5B2E7"
    C_BWD_2 = "#B9D992"
    C_BWD   = "#355113"
    C_IDLE  = "#9E9E9E"

    # ---- draw
    x_offset = 0
    for iy, r in enumerate(ranks):
        y = num_ranks - 1 - iy
        y = y * 2 
        x_offset = 0
        for t, op in enumerate(pp_order[r]):
            # x0 = x_edges[t]
            # w_col = col_w[t]

            if op is None: # idle
                stage_id, kind, mb, need_offload, need_reload = None, None, None, False, False
            else:
                stage_id, kind, mb, need_offload, need_reload = op

            text_color = 'white'
            if kind is None:
                color, width_rect, txt = C_IDLE, 1, ""
            elif kind.name == "FULL_BACKWARD_CONT":
                continue # skip drawing the continuation cell
            elif kind.name == "FORWARD":
                color, width_rect, txt = C_FWD, 1, (str(op) if mb is not None else "")
                # text_color = 'white'
                assert stage_id is not None
                if stage_id >= num_ranks:
                    color = C_FWD_2
                    text_color = 'black'
            elif kind.name == "FULL_BACKWARD":
                color, width_rect, txt = C_BWD, 2, (str(op) if mb is not None else "")
                # text_color = 'white'
                assert stage_id is not None
                if stage_id >= num_ranks:
                    color = C_BWD_2
                    text_color = 'black'
            else:
                raise AssertionError(f"Unexpected kind: {kind}")

            # forward/idle occupy the left-half when the column was doubled
            rect = Rectangle((x_offset, y), width_rect, 1, facecolor=color, edgecolor="black", linewidth=0.3)
            ax.add_patch(rect)

            if txt:

                ax.text(x_offset + width_rect/2, y + 0.5, txt,
                        va="center", ha="center",
                        fontsize=8,
                        color=text_color)
                
            x_offset += width_rect
    total_w = x_offset
    # ---- axes / grid
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, num_ranks * 2)
    ax.set_yticks([num_ranks * 2 - 1 - 0.5 - i * 2 for i in range(num_ranks)])
    ax.set_yticklabels([f"Rank {i}" for i in ranks])
    ax.set_xlabel("Time")
    ax.set_ylabel("Pipeline stage")

    # ticks: every ~20 labels max on the stretched x-scale
    ax.set_xticks(range(0, total_w, max(1, total_w // 20)))
    # minor vertical grid at every unit (this also shows the midline when a column is doubled)
    ax.set_xticks(range(total_w + 1), minor=True)
    ax.set_yticks(range(num_ranks * 2), minor=True)
    ax.grid(which='minor', linestyle=':', linewidth=0.3, color="#666666", alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=9)

    # ---- legend OUTSIDE the axes (no overlap)
    fwd_patch  = Rectangle((0,0),1,1, facecolor=C_FWD,  edgecolor='black', linewidth=0.3)
    bwd_patch  = Rectangle((0,0),1,1, facecolor=C_BWD,  edgecolor='black', linewidth=0.3)
    idle_patch = Rectangle((0,0),1,1, facecolor=C_IDLE, edgecolor='black', linewidth=0.3)
    ax.legend([fwd_patch, bwd_patch, idle_patch],
              ["Forward", "Backward", "Idle"],
              loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0., frameon=False)

    ax.set_aspect('equal')
    for s in ax.spines.values():
        s.set_visible(False)

    if outpath:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        fig.savefig(outpath, dpi=200)  # you can add bbox_inches="tight" if you place legend below
    return fig, ax


def debug_save_pp_schedule(
    schedule,
    filename: str = "tmp/pp_schedule_debug.png",
) -> None:

    pp_order = schedule.pipeline_order
    # pipeline_order example:
    '''
    {
        0: [0F0, 0F1, 0F2, 0F3, ...],
        1: [None, 1F0, 1F1, 1F2, ...],
        2: [None, None, 2F0, 2F1, ...],
        3: [None, None, None, 3F0, ...],
    }
    '''
    import matplotlib.pyplot as plt

    # make sure the output directory exists
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    fig, ax = draw_pipeline_timeline(
        pp_order,
        title="Interleaved 1F1B (timeline)",
        outpath=filename
    )
    plt.close(fig)


class PipelineSchedule:
    """
    TODO: Do not need this class anymore. Consider move everything to base_schedule directly.

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
        # loss_fn: Optional[Callable[[Any, torch.Tensor], torch.Tensor]] = None,
        num_microbatches: Optional[int] = None,
    ):
        self.model_parts = model_parts
        self.stages = stages
        self.pp_mesh = pp_mesh
        # self.loss_fn = loss_fn



        if schedule_name == PipelineScheduleType.custom_1F1B:
            # Custom 1F1B schedule
            # from olmo_core.train.train_module.transformer.pipeline_schedule import (
            #     CustomSchedule1F1B,
            # )
            # schedule_class = CustomSchedule1F1B
            raise NotImplementedError("Custom 1F1B schedule is not implemented yet.")
        elif schedule_name == PipelineScheduleType.custom_interleaved_1F1B:
            # Custom interleaved 1F1B schedule
            from olmo_core.train.train_module.transformer.pipeline.pipeline_schedule import (
                CustomScheduleInterleaved1F1B,
            )
            schedule_class = CustomScheduleInterleaved1F1B
        else:
            raise RuntimeError(f"Unsupported schedule_name: {schedule_name}")


        if num_microbatches is None:
            num_microbatches = pp_mesh.size()


        schedule_impl = schedule_class(
            stages,  # type: ignore[arg-type]
            n_microbatches=num_microbatches,
            # loss_fn=self.loss_fn,
        )
        

        # torch.save(schedule.pipeline_order, 'tmp.pt')
        if torch.distributed.get_rank() == 0:
            debug_save_pp_schedule(schedule=schedule_impl)
        
        print(f'[PipelineSchedule] Using {schedule_name} with {num_microbatches} microbatches')


        self.schedule_impl = schedule_impl
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
        input_ids: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        forward_only: bool = False,
        **kwargs,
    ) -> List[List[Optional[LMOutputWithLoss]]]:
        """
        :param args: Only passed to first stage.
        :param kwargs: Passed to all stages.

        :return: A list with length of num stages. Each element of the list is an inner list with length of num microbatches. Each element in the inner list is either (1) the output of the corresponding microbatch if that stage is the last stage, or (2) None if that stage is not the last stage.
        """

        if self.has_last_stage:
            pass # keep target as is
        else:
            target = None

        if not self.has_first_stage:
            args = () # If there is no first stage, we need to provide an empty tuple for single_1F1B
        else:
            args = (input_ids,)

        # in inference mode, change to one seq per microbatch
        old_num_microbatches = None
        if forward_only:
            old_num_microbatches = self.schedule_impl._n_microbatches
            self.schedule_impl.reset_n_microbatches(input_ids.size(0) // 1) # one seq per microbatch

        self.schedule_impl.prepare_step(
            global_batch_size=input_ids.size(0),
            seqlen=input_ids.size(1),
        )
        step_output = self.schedule_impl.step(*args, target=target, forward_only=forward_only, **kwargs)

        self.schedule_impl.clear_step_info()

        # reset
        if forward_only:
            assert old_num_microbatches is not None
            self.schedule_impl.reset_n_microbatches(old_num_microbatches)

        return step_output
