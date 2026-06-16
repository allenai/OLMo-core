from dataclasses import dataclass
from collections import Counter
from functools import cached_property
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
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
    custom_1F1B_V = "Custom1F1BV"

    @property
    def is_single_stage(self) -> bool:
        if self in (self.single_1F1B, self.gpipe, self.custom_1F1B):
            return True
        elif self in (
            self.interleaved_1F1B,
            self.looped_bfs,
            self.interleaved_zero_bubble,
            self.zbv_zero_bubble,
            self.custom_interleaved_1F1B,
            self.custom_1F1B_V,
        ):
            return False
        else:
            raise OLMoConfigurationError(f"Invalid pipeline schedule '{self}'")

    @property
    def is_multi_stage(self) -> bool:
        return not self.is_single_stage

    @property
    def default_style(self) -> PipelineSplitStyle:
        if self in (self.zbv_zero_bubble, self.custom_1F1B_V):
            return PipelineSplitStyle.v
        else:
            return PipelineSplitStyle.loop


class PipelineP2PBackend(StrEnum):
    """
    Transport backend for custom pipeline activation/gradient P2P.
    """

    nccl = "nccl"
    nccl_rma = "nccl_rma"
    nccl_device = "nccl_device"


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

    p2p_use_separate_group: bool = False
    """
    Use a dedicated NCCL process group for pipeline stage activation/gradient P2P.
    This leaves the normal pipeline group untouched for bookkeeping collectives.
    """

    p2p_backend: PipelineP2PBackend = PipelineP2PBackend.nccl
    """
    Pipeline activation/gradient P2P backend. ``nccl`` is the existing
    two-sided torch.distributed P2P path. ``nccl_rma`` is the experimental
    NCCL 2.29 PutSignal/WaitSignal path. ``nccl_device`` is reserved for a
    future NCCL device API / GIN backend.
    """

    p2p_nccl_min_ctas: Optional[int] = None
    """
    Optional NCCL minCTAs override for the dedicated pipeline P2P group.
    Requires ``p2p_use_separate_group``.
    """

    p2p_nccl_max_ctas: Optional[int] = None
    """
    Optional NCCL maxCTAs override for the dedicated pipeline P2P group.
    Requires ``p2p_use_separate_group``.
    """

    forward_pull_ahead_extra_activations: Optional[List[int]] = None
    """
    Experimental 1F1B-V schedule knob. Allows the generator to pull forwards
    earlier if doing so costs at most this many extra live activations per rank.
    Set to ``None`` to disable, or provide one integer per PP rank.
    """

    def uses_separate_p2p_group(self) -> bool:
        return self.p2p_use_separate_group

    def _validate_p2p_nccl_ctas(self) -> None:
        if self.p2p_backend != PipelineP2PBackend.nccl and (
            self.p2p_nccl_min_ctas is not None or self.p2p_nccl_max_ctas is not None
        ):
            raise OLMoConfigurationError(
                "p2p_nccl_min_ctas/p2p_nccl_max_ctas only apply to p2p_backend='nccl'"
            )
        if self.p2p_nccl_min_ctas is not None and self.p2p_nccl_min_ctas <= 0:
            raise OLMoConfigurationError("p2p_nccl_min_ctas must be positive")
        if self.p2p_nccl_max_ctas is not None and self.p2p_nccl_max_ctas <= 0:
            raise OLMoConfigurationError("p2p_nccl_max_ctas must be positive")
        if (
            self.p2p_nccl_min_ctas is not None
            and self.p2p_nccl_max_ctas is not None
            and self.p2p_nccl_min_ctas > self.p2p_nccl_max_ctas
        ):
            raise OLMoConfigurationError("p2p_nccl_min_ctas cannot exceed p2p_nccl_max_ctas")

    def _p2p_nccl_options(self) -> Optional[Any]:
        self._validate_p2p_nccl_ctas()
        if self.p2p_nccl_min_ctas is None and self.p2p_nccl_max_ctas is None:
            return None

        if not hasattr(dist, "ProcessGroupNCCL"):
            raise OLMoConfigurationError("P2P NCCL CTA caps require the NCCL distributed backend")

        nccl_options = dist.ProcessGroupNCCL.Options()
        if not hasattr(nccl_options, "config"):
            raise OLMoConfigurationError(
                "P2P NCCL CTA caps require a PyTorch/NCCL build with ProcessGroupNCCL.Options.config"
            )

        if self.p2p_nccl_min_ctas is not None:
            nccl_options.config.min_ctas = self.p2p_nccl_min_ctas
        if self.p2p_nccl_max_ctas is not None:
            nccl_options.config.max_ctas = self.p2p_nccl_max_ctas
        return nccl_options

    @staticmethod
    def _pipeline_rank_groups(device_mesh: DeviceMesh) -> List[List[int]]:
        if device_mesh.mesh_dim_names is None:
            raise RuntimeError("could not build PP P2P groups without mesh dimension names")
        if "pp" not in device_mesh.mesh_dim_names:
            raise RuntimeError(
                f"could not build PP P2P groups from mesh with dimensions {device_mesh.mesh_dim_names}"
            )

        pp_dim = device_mesh.mesh_dim_names.index("pp")
        mesh = device_mesh.mesh.detach().cpu().to(torch.int64)
        mesh = mesh.movedim(pp_dim, 0).contiguous()
        if mesh.ndim == 1:
            return [mesh.tolist()]

        rank_groups: List[List[int]] = []
        for index in product(*(range(size) for size in mesh.shape[1:])):
            group = mesh[(slice(None),) + index].tolist()
            rank_groups.append([int(rank) for rank in group])
        return rank_groups

    def build_p2p_process_group(self, device_mesh: DeviceMesh) -> Optional[dist.ProcessGroup]:
        self._validate_p2p_nccl_ctas()
        if self.p2p_backend == PipelineP2PBackend.nccl_device:
            raise OLMoConfigurationError("p2p_backend='nccl_device' is reserved but not implemented yet")

        if not self.uses_separate_p2p_group():
            if self.p2p_nccl_min_ctas is not None or self.p2p_nccl_max_ctas is not None:
                raise OLMoConfigurationError(
                    "p2p_nccl_min_ctas/p2p_nccl_max_ctas require p2p_use_separate_group=True"
                )
            return None

        pg_options = self._p2p_nccl_options()
        current_group, _all_groups = dist.new_subgroups_by_enumeration(
            self._pipeline_rank_groups(device_mesh),
            backend="nccl",
            pg_options=pg_options,
            group_desc="pipeline_p2p",
        )
        return current_group

    def infer_style(self) -> PipelineSplitStyle:
        if self.style is not None:
            if (
                self.schedule == PipelineScheduleType.custom_1F1B_V
                and self.style != PipelineSplitStyle.v
            ):
                raise OLMoConfigurationError("custom_1F1B_V requires pipeline split style 'v'")
            if (
                self.schedule == PipelineScheduleType.custom_interleaved_1F1B
                and self.style != PipelineSplitStyle.loop
            ):
                raise OLMoConfigurationError(
                    "custom_interleaved_1F1B requires pipeline split style 'loop'; "
                    "use custom_1F1B_V for V-shaped placement"
                )
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
                           outpath: Optional[str] = None,
                           microbatch_index_offset: int = 0):

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

    def format_action(stage_id: int, kind: Any, mb: Optional[int]) -> str:
        if mb is None:
            return ""
        kind_name = getattr(kind, "name", "")
        if kind_name == "FORWARD":
            kind_label = "F"
        elif kind_name == "FULL_BACKWARD":
            kind_label = "B"
        else:
            kind_label = str(kind)
        return f"{stage_id}{kind_label}{mb + microbatch_index_offset}"

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
                color, width_rect, txt = C_FWD, 1, format_action(stage_id, kind, mb)
                # text_color = 'white'
                assert stage_id is not None
                if stage_id >= num_ranks:
                    color = C_FWD_2
                    text_color = 'black'
            elif kind.name == "FULL_BACKWARD":
                color, width_rect, txt = C_BWD, 2, format_action(stage_id, kind, mb)
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


def get_pipeline_bubble_stats(pp_order: Dict[int, List[Any]]) -> tuple[int, int, float]:
    ranks = sorted(pp_order.keys())
    if not ranks:
        return 0, 0, 0.0
    num_steps = max(len(pp_order[rank]) for rank in ranks)
    total_slots = len(ranks) * num_steps
    idle_slots = 0
    for rank in ranks:
        row = pp_order[rank]
        idle_slots += sum(1 for action in row if action is None)
        idle_slots += num_steps - len(row)
    bubble_rate = idle_slots / total_slots if total_slots > 0 else 0.0
    return idle_slots, total_slots, bubble_rate


def _schedule_action_parts(action: Any) -> tuple[int, Any, Optional[int]] | None:
    if action is None:
        return None
    stage_id, kind, mb, _need_offload, _need_reload = action
    if stage_id is None or mb is None:
        return None
    return int(stage_id), kind, int(mb)


def get_pipeline_activation_stats(pp_order: Dict[int, List[Any]]) -> dict[int, int]:
    """Return peak live activation count per rank from the symbolic table."""
    peaks: dict[int, int] = {}
    for rank, row in pp_order.items():
        live = 0
        peak = 0
        for action in row:
            parts = _schedule_action_parts(action)
            if parts is None:
                continue
            _stage_id, kind, _mb = parts
            kind_name = getattr(kind, "name", "")
            if kind_name == "FORWARD":
                live += 1
                peak = max(peak, live)
            elif kind_name == "FULL_BACKWARD_CONT":
                live -= 1
        peaks[rank] = peak
    return peaks


def get_pipeline_tick_exchange_stats(pp_order: Dict[int, List[Any]]) -> dict[str, Any]:
    """
    Count cross-rank dependencies where the consumer starts exactly one schedule
    tick after the producer. These edges are legal but can become profiler
    bubbles when runtime skew makes the producer finish late.
    """
    action_times: dict[tuple[int, str, int], tuple[int, int]] = {}
    max_stage = -1
    for rank, row in pp_order.items():
        for time_step, action in enumerate(row):
            parts = _schedule_action_parts(action)
            if parts is None:
                continue
            stage_id, kind, mb = parts
            kind_name = getattr(kind, "name", "")
            max_stage = max(max_stage, stage_id)
            if kind_name == "FORWARD":
                action_times[(stage_id, "F", mb)] = (rank, time_step)
            elif kind_name == "FULL_BACKWARD":
                action_times[(stage_id, "B", mb)] = (rank, time_step)
            elif kind_name == "FULL_BACKWARD_CONT":
                action_times[(stage_id, "B_", mb)] = (rank, time_step)

    tight_edges: list[tuple[int, int, int, str, int, int, int, int]] = []
    # (producer_time, producer_rank, consumer_rank, kind, producer_stage,
    #  consumer_stage, microbatch, consumer_time)
    for (stage_id, kind_label, mb), (consumer_rank, consumer_time) in action_times.items():
        if kind_label == "F" and stage_id > 0:
            producer_key = (stage_id - 1, "F", mb)
            edge_kind = "F"
            producer_stage = stage_id - 1
        elif kind_label == "B" and stage_id < max_stage:
            producer_key = (stage_id + 1, "B_", mb)
            edge_kind = "B"
            producer_stage = stage_id + 1
        else:
            continue

        producer = action_times.get(producer_key)
        if producer is None:
            continue
        producer_rank, producer_time = producer
        if producer_rank == consumer_rank:
            continue
        if consumer_time == producer_time + 1:
            tight_edges.append(
                (
                    producer_time,
                    producer_rank,
                    consumer_rank,
                    edge_kind,
                    producer_stage,
                    stage_id,
                    mb,
                    consumer_time,
                )
            )

    edges_by_tick_pair: Counter[tuple[int, int, int]] = Counter(
        (producer_time, producer_rank, consumer_rank)
        for producer_time, producer_rank, consumer_rank, *_rest in tight_edges
    )
    bidirectional_ticks = 0
    seen_pairs: set[tuple[int, int, int]] = set()
    for tick, src_rank, dst_rank in edges_by_tick_pair:
        reverse = (tick, dst_rank, src_rank)
        key = (tick, min(src_rank, dst_rank), max(src_rank, dst_rank))
        if key not in seen_pairs and reverse in edges_by_tick_pair:
            seen_pairs.add(key)
            bidirectional_ticks += 1

    samples = [
        (
            f"t{producer_time}->{consumer_time} "
            f"r{producer_rank}->{consumer_rank} "
            f"{producer_stage}{edge_kind}{mb}->{consumer_stage}{edge_kind}{mb}"
        )
        for (
            producer_time,
            producer_rank,
            consumer_rank,
            edge_kind,
            producer_stage,
            consumer_stage,
            mb,
            consumer_time,
        ) in tight_edges[:12]
    ]
    return {
        "tight_edges": len(tight_edges),
        "bidirectional_ticks": bidirectional_ticks,
        "samples": samples,
    }


def _safe_filename_component(value: Any) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return safe.strip("_") or "unknown"


def _default_pp_schedule_plot_dir() -> str:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "src" / "olmo_core").exists():
            return str(parent / "tmp" / "pp_schedules")
    return "tmp/pp_schedules"


def _format_schedule_action_label(action: Any, microbatch_index_offset: int = 0) -> str:
    if action is None:
        return ".."

    stage_id, kind, mb, need_offload, need_reload = action
    kind_name = getattr(kind, "name", "")
    if kind_name == "FORWARD":
        kind_label = "F"
    elif kind_name == "FULL_BACKWARD":
        kind_label = "B"
    elif kind_name == "FULL_BACKWARD_CONT":
        kind_label = "B_"
    else:
        kind_label = str(kind)

    if mb is None:
        label = f"{stage_id}{kind_label}"
    else:
        label = f"{stage_id}{kind_label}{mb + microbatch_index_offset}"
    if need_offload:
        label += "^offload"
    if need_reload:
        label = f"reload^{label}"
    return label


def format_pp_schedule_text(
    schedule: Any,
    *,
    microbatch_index_offset: int = 0,
) -> str:
    pp_order = schedule.pipeline_order
    idle_slots, total_slots, bubble_rate = get_pipeline_bubble_stats(pp_order)
    activation_peaks = get_pipeline_activation_stats(pp_order)
    exchange_stats = get_pipeline_tick_exchange_stats(pp_order)
    lines = [
        f"schedule_class: {schedule.__class__.__name__}",
        f"schedule_source: {getattr(schedule, 'pipeline_order_source', 'unknown')}",
        f"p2p_overlap: {getattr(schedule, 'enable_p2p_overlap', 'unknown')}",
        f"p2p_overlap_kinds: {sorted(getattr(schedule, 'p2p_overlap_kinds', []))}",
        f"p2p_overlap_ops: {sorted(getattr(schedule, 'p2p_overlap_ops', []))}",
        f"uses_separate_p2p_group: {getattr(schedule, 'uses_separate_p2p_group', 'unknown')}",
        f"activation_offload_schedule: {getattr(schedule, 'enable_activation_offload_schedule', 'unknown')}",
        f"bubble_rate: {bubble_rate:.6f} ({idle_slots}/{total_slots})",
        f"peak_live_activations_by_rank: {activation_peaks}",
        f"tick_exchange_edges: {exchange_stats['tight_edges']}",
        f"bidirectional_tick_exchanges: {exchange_stats['bidirectional_ticks']}",
        f"microbatch_index_offset: {microbatch_index_offset}",
        "",
    ]
    if exchange_stats["samples"]:
        lines.append("tick_exchange_samples:")
        for sample in exchange_stats["samples"]:
            lines.append(f"  {sample}")
        lines.append("")
    for rank in sorted(pp_order.keys()):
        labels = [
            _format_schedule_action_label(action, microbatch_index_offset)
            for action in pp_order[rank]
        ]
        lines.append(f"rank {rank}: " + " ".join(labels))
    return "\n".join(lines) + "\n"


def debug_save_pp_schedule(
    schedule,
    filename: str = "tmp/pp_schedule_debug.png",
    microbatch_index_offset: int = 0,
    text_filename: Optional[str] = None,
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

    idle_slots, total_slots, bubble_rate = get_pipeline_bubble_stats(pp_order)
    exchange_stats = get_pipeline_tick_exchange_stats(pp_order)
    source = getattr(schedule, "pipeline_order_source", "unknown")
    fig, ax = draw_pipeline_timeline(
        pp_order,
        title=(
            f"{schedule.__class__.__name__} | {source} | "
            f"bubble {bubble_rate:.1%} ({idle_slots}/{total_slots}) | "
            f"tick-ex {exchange_stats['tight_edges']}"
        ),
        outpath=filename,
        microbatch_index_offset=microbatch_index_offset,
    )
    plt.close(fig)
    if text_filename is not None:
        os.makedirs(os.path.dirname(text_filename) or ".", exist_ok=True)
        with open(text_filename, "w", encoding="utf-8") as f:
            f.write(
                format_pp_schedule_text(
                    schedule,
                    microbatch_index_offset=microbatch_index_offset,
                )
            )


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
        loss_fn: Optional[Callable[[Any, torch.Tensor], torch.Tensor]] = None,
        num_microbatches: Optional[int] = None,
        forward_pull_ahead_extra_activations: Optional[List[int]] = None,
    ):
        self.model_parts = model_parts
        self.stages = stages
        self.pp_mesh = pp_mesh
        self.loss_fn = loss_fn



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
        elif schedule_name == PipelineScheduleType.custom_1F1B_V:
            # Custom 1F1B-V schedule
            from olmo_core.train.train_module.transformer.pipeline.pipeline_schedule import (
                CustomSchedule1F1BV,
            )
            schedule_class = CustomSchedule1F1BV
        else:
            raise RuntimeError(f"Unsupported schedule_name: {schedule_name}")


        if num_microbatches is None:
            num_microbatches = pp_mesh.size()


        schedule_impl = schedule_class(
            stages,  # type: ignore[arg-type]
            n_microbatches=num_microbatches,
            # loss_fn=self.loss_fn,
            forward_pull_ahead_extra_activations=(
                0
                if forward_pull_ahead_extra_activations is None
                else forward_pull_ahead_extra_activations
            ),
        )
        

        if (
            torch.distributed.get_rank() == 0
            and os.environ.get("OLMO_PP_SCHEDULE_PLOT", "1").lower()
            not in {"0", "false", "no"}
        ):
            plot_dir = os.environ.get("OLMO_PP_SCHEDULE_PLOT_DIR", _default_pp_schedule_plot_dir())
            source = getattr(schedule_impl, "pipeline_order_source", "unknown")
            stem = "_".join(
                [
                    _safe_filename_component(schedule_name.value),
                    f"pp{pp_mesh.size()}",
                    f"mb{num_microbatches}",
                    _safe_filename_component(source),
                ]
            )
            plot_filename = os.path.join(plot_dir, f"{stem}.png")
            text_filename = os.path.join(plot_dir, f"{stem}.txt")
            debug_save_pp_schedule(
                schedule=schedule_impl,
                filename=plot_filename,
                text_filename=text_filename,
            )
            print(
                "[PipelineSchedule] Saved pipeline schedule artifacts to "
                f"{plot_filename} and {text_filename}"
            )
        
        print(f'[PipelineSchedule] Using {schedule_name} with {num_microbatches} microbatches') # TODO: no need to print on every rank


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
        num_microbatches: Optional[int] = None,
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

        # In inference mode, change to one seq per microbatch. Training dry
        # runs can also request a temporary smaller schedule.
        old_num_microbatches = None
        temporary_num_microbatches = input_ids.size(0) // 1 if forward_only else num_microbatches
        if temporary_num_microbatches is not None:
            old_num_microbatches = self.schedule_impl._n_microbatches
            if temporary_num_microbatches != old_num_microbatches:
                self.schedule_impl.reset_n_microbatches(temporary_num_microbatches)

        try:
            self.schedule_impl.prepare_step(
                global_batch_size=input_ids.size(0),
                seqlen=input_ids.size(1),
            )
            step_output = self.schedule_impl.step(*args, target=target, forward_only=forward_only, **kwargs)
        finally:
            self.schedule_impl.clear_step_info()

            if old_num_microbatches is not None:
                self.schedule_impl.reset_n_microbatches(old_num_microbatches)

        return step_output
