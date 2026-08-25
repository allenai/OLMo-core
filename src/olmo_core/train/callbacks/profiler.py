import json
import logging
import math
import os
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch

from olmo_core.distributed.parallel import (
    get_cp_mesh,
    get_dp_mesh,
    get_ep_mesh,
    get_pp_mesh,
    get_tp_mesh,
    get_world_mesh,
)
from olmo_core.distributed.utils import get_rank

from .callback import Callback

log = logging.getLogger(__name__)


_COLLECTIVE_EVENT_MARKERS = (
    "all_reduce",
    "allreduce",
    "all_gather",
    "allgather",
    "reduce_scatter",
    "reducescatter",
    "all_to_all",
    "alltoall",
    "broadcast",
    "c10d::",
    "nccl",
    "gloo",
)
_SYNC_EVENT_MARKERS = (
    "synchronize",
    "streamwait",
    "eventwait",
)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    position = (len(values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _summarize_distributed_events(events: Iterable[Any]) -> list[dict[str, Any]]:
    """Aggregate profiler events for distributed collectives and device synchronization."""
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"cpu_us": [], "device_us": []}
    )
    for event in events:
        name = str(event.name)
        normalized_name = name.lower().replace(" ", "")
        if any(marker in normalized_name for marker in _COLLECTIVE_EVENT_MARKERS):
            category = "collective"
        elif any(marker in normalized_name for marker in _SYNC_EVENT_MARKERS):
            category = "synchronization"
        else:
            continue

        input_shapes = repr(getattr(event, "input_shapes", None))
        timings = grouped[(category, name, input_shapes)]
        timings["cpu_us"].append(float(getattr(event, "cpu_time_total", 0.0) or 0.0))
        timings["device_us"].append(float(getattr(event, "device_time_total", 0.0) or 0.0))

    summary = []
    for (category, name, input_shapes), timings in sorted(grouped.items()):
        cpu_us = timings["cpu_us"]
        device_us = timings["device_us"]
        summary.append(
            {
                "category": category,
                "name": name,
                "input_shapes": input_shapes,
                "count": len(cpu_us),
                "cpu_us": {
                    "mean": sum(cpu_us) / len(cpu_us),
                    "p50": _percentile(cpu_us, 0.50),
                    "p95": _percentile(cpu_us, 0.95),
                    "total": sum(cpu_us),
                },
                "device_us": {
                    "mean": sum(device_us) / len(device_us),
                    "p50": _percentile(device_us, 0.50),
                    "p95": _percentile(device_us, 0.95),
                    "total": sum(device_us),
                },
            }
        )
    return summary


@dataclass
class ProfilerCallback(Callback):
    """
    Enables profiling/tracing of training steps using :mod:`torch.profiler`.
    Saved the results to a subdirectory of the save folder named "profiler".
    """

    skip_first: int = 0
    """
    Ignore this many steps before profiling cycles.
    """
    wait: int = 1
    """
    Idle for this many steps before activating.
    """
    warmup: int = 5
    """
    Start tracing, but discard the results, for this many steps.
    """
    active: int = 3
    """
    Actively trace this many steps.
    """
    repeat: int = 1
    """
    Repeat the cycle start at ``wait`` steps.
    """
    with_stack: bool = True
    """
    Whether to record source information (file and line number) for the ops.
    """
    profile_memory: bool = False
    """
    Whether to track tensor memory allocation/deallocation
    """
    enable_cuda_sync_events: bool = False
    """
    Whether to enable recording of CUDA sync events. Useful for critical-path analysis with
        https://hta.readthedocs.io/en/latest/source/features/lightweight_critical_path_analysis.html
    """
    export_distributed_event_summary: bool = False
    """
    Export a JSON summary of distributed collective and synchronization events for each trace.
    This is model-agnostic and includes event counts plus mean, p50, p95, and total CPU/device
    durations. Set :attr:`enable_cuda_sync_events` as well when synchronization attribution is
    important.
    """
    enabled: bool = True
    """
    Set to ``False`` to disable profiling.
    """
    ranks: str | None = None
    """
    Ranks to profile. Can be:

    - ``None``: Only rank 0 is profiled
    - String shortcuts:
      - ``"dp"``: Profile one rank (local rank 0) in each data parallel group
      - ``"tp"``: Profile one rank (local rank 0) in each tensor parallel group
      - ``"cp"``: Profile one rank (local rank 0) in each context parallel group
      - ``"pp"``: Profile one rank (local rank 0) in each pipeline parallel group
      - ``"ep"``: Profile one rank (local rank 0) in each expert parallel group
      - ``"all"``: Profile all ranks

    Useful in conjunction with https://github.com/facebookresearch/HolisticTraceAnalysis
    to analyze traces from a distributed training job.
    """

    _exit_stack = None
    _profiler = None
    _first_batch: bool = True

    def _should_profile_rank(self) -> bool:
        current_rank = get_rank()

        if self.ranks is None:
            return current_rank == 0
        elif isinstance(self.ranks, str):  # Handle string shortcuts for parallel groups
            world_mesh = get_world_mesh()
            if world_mesh is None:
                if self.ranks != "all":
                    log.warning("No world mesh available, falling back to rank 0 only")
                return current_rank == 0

            try:
                if self.ranks == "dp":
                    dp_mesh = get_dp_mesh(world_mesh)
                    return dp_mesh.get_local_rank() == 0
                elif self.ranks == "tp":
                    tp_mesh = get_tp_mesh(world_mesh)
                    return tp_mesh.get_local_rank() == 0
                elif self.ranks == "cp":
                    cp_mesh = get_cp_mesh(world_mesh)
                    return cp_mesh.get_local_rank() == 0
                elif self.ranks == "pp":
                    pp_mesh = get_pp_mesh(world_mesh)
                    return pp_mesh.get_local_rank() == 0
                elif self.ranks == "ep":
                    ep_mesh = get_ep_mesh(world_mesh)
                    return ep_mesh.get_local_rank() == 0
                elif self.ranks == "all":
                    return True
                else:
                    raise ValueError(f"Unknown rank shortcut '{self.ranks}'")
            except RuntimeError as e:
                log.warning(
                    f"Failed to determine parallel mesh for '{self.ranks}': {e}, falling back to rank 0 only"
                )
                return current_rank == 0
        else:
            raise TypeError(f"Invalid ranks specification: {self.ranks}")

    def pre_train(self):
        if not self.enabled or not self._should_profile_rank():
            return

        from torch.profiler import (
            ProfilerActivity,
            _ExperimentalConfig,
            profile,
            schedule,
        )

        profiling_schedule = schedule(
            wait=self.wait,
            warmup=self.warmup,
            active=self.active,
            repeat=self.repeat,
            skip_first=self.skip_first,
        )
        activities = [ProfilerActivity.CPU]
        if self.trainer.device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)

        experimental_config = None
        if self.enable_cuda_sync_events:
            experimental_config = _ExperimentalConfig(enable_cuda_sync_events=True)

        self._exit_stack = ExitStack()
        self._profiler = self._exit_stack.enter_context(
            profile(
                activities=activities,
                record_shapes=self.export_distributed_event_summary,
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
                schedule=profiling_schedule,
                on_trace_ready=self._on_trace_ready,
                experimental_config=experimental_config,
            )
        )
        self._first_batch = True

    def pre_load_batch(self):
        if not self.enabled or not self._should_profile_rank():
            return

        if self._first_batch:
            self._first_batch = False
        else:
            assert self._profiler is not None
            self._profiler.step()

    def _on_trace_ready(self, prof):
        assert self._profiler is not None
        output = self._profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=32)
        log.info(f"Profile by total GPU time at step {self._profiler.step_num}:\n{output}")
        output = self._profiler.key_averages().table(sort_by="self_cpu_time_total", row_limit=32)
        log.info(f"Profile by total CPU time at step {self._profiler.step_num}:\n{output}")

        log.info("Saving chrome trace from profiler...")
        output_dir = self.trainer.work_dir / "profiler"
        output_dir.mkdir(exist_ok=True, parents=True)
        trace_path = output_dir / f"rank-{get_rank()}-step-{prof.step_num}.chrome_trace.json.gz"
        prof.export_chrome_trace(str(trace_path))
        final_path = self.trainer.persist_working_file(trace_path)
        log.info(f"Chrome trace saved to '{final_path}'")

        if self.export_distributed_event_summary:
            summary_path = (
                output_dir / f"rank-{get_rank()}-step-{prof.step_num}.distributed-events.json"
            )
            summary = {
                "rank": get_rank(),
                "step": prof.step_num,
                "time_unit": "microseconds",
                "events": _summarize_distributed_events(prof.events()),
            }
            with summary_path.open("w") as f:
                json.dump(summary, f, indent=2)
            final_summary_path = self.trainer.persist_working_file(summary_path)
            log.info(f"Distributed event summary saved to '{final_summary_path}'")


@dataclass
class NvidiaProfilerCallback(Callback):
    """
    Wraps a window of training steps in the NVIDIA profiler (``cudaProfilerStart/Stop`` plus
    NVTX ranges), for use with Nsight Systems. Profiling runs from step :data:`start` to
    :data:`end` on the configured ranks.

    .. note::
        This only produces output when the job is launched under an external Nsight Systems
        session (e.g. ``nsys profile --capture-range=cudaProfilerApi ...``); on its own it just
        toggles a capture range with nothing recording.
    """

    start: int = 10
    """
    The step at which to start profiling.
    """
    end: int = 12
    """
    The step at which to stop profiling.
    """
    enabled: bool = True
    """
    Set to ``False`` to disable profiling.
    """
    profile_ranks: list[int] = field(default_factory=lambda: [0])
    """
    The ranks to profile.
    """

    _nvtx_ctx = None

    def pre_load_batch(self):
        # `pre_load_batch` runs before the trainer increments its step counter, so `self.step`
        # here is the previously completed step; compare against `start - 1` so the capture
        # window actually begins on the requested `start` step.
        if self.enabled and get_rank() in self.profile_ranks and self.step == self.start - 1:
            log.info(f"Starting NVIDIA profiler at rank={get_rank()} step={self.start}...")
            torch.cuda.cudart().cudaProfilerStart()
            self._nvtx_ctx = torch.autograd.profiler.emit_nvtx(record_shapes=True)
            self._nvtx_ctx.__enter__()

    def post_train_batch(self):
        if self.step == self.end:
            self._stop()

    def close(self):
        # Close the capture range even if training stops (cancel/error/short run) before `end`,
        # otherwise an external `nsys --capture-range=cudaProfilerApi` range is left open.
        self._stop()

    def _stop(self):
        if self._nvtx_ctx is not None:
            log.info(f"Stopping NVIDIA profiler at rank={get_rank()}...")
            self._nvtx_ctx.__exit__(None, None, None)
            self._nvtx_ctx = None
            torch.cuda.cudart().cudaProfilerStop()


@dataclass
class TorchMemoryHistoryCallback(Callback):
    """
    Records CUDA memory allocation history between steps :data:`start` and :data:`end` and
    dumps a snapshot pickle (viewable at https://pytorch.org/memory_viz) on the configured ranks.
    """

    start: int = 10
    """
    The step at which to start recording memory history.
    """
    end: int = 12
    """
    The step at which to stop recording and dump the snapshot.
    """
    enabled: bool = True
    """
    Set to ``False`` to disable profiling.
    """
    profile_ranks: list[int] = field(default_factory=lambda: [0])
    """
    The ranks to profile.
    """

    max_entries: int = 500000
    """
    The maximum number of memory-history entries to record.
    """

    output_dir: str = "."
    """
    Directory to write the snapshot pickle(s) to.
    """

    _recording: bool = False

    def pre_load_batch(self):
        # See `NvidiaProfilerCallback.pre_load_batch`: `self.step` here is the previously
        # completed step, so start recording at `start - 1` to include the requested `start` step.
        if self.enabled and get_rank() in self.profile_ranks and self.step == self.start - 1:
            log.info(f"Starting memory profiler at rank={get_rank()} step={self.start}...")
            torch.cuda.memory._record_memory_history(max_entries=self.max_entries)
            self._recording = True

    def post_train_batch(self):
        if self.step == self.end:
            self._dump_and_stop()

    def close(self):
        # Dump and disable on early exit (OOM/error/short run) so the failure window isn't lost
        # and recording doesn't stay on until process teardown.
        self._dump_and_stop()

    def _dump_and_stop(self):
        if not self._recording:
            return
        log.info(f"Dumping memory profiler at rank={get_rank()}...")
        os.makedirs(self.output_dir, exist_ok=True)
        torch.cuda.memory._dump_snapshot(
            os.path.join(self.output_dir, f"memsnapshot.{get_rank()}.pickle")
        )
        torch.cuda.memory._record_memory_history(enabled=None)
        self._recording = False
        log.info(f"Memory profiler stopped at rank={get_rank()}.")
