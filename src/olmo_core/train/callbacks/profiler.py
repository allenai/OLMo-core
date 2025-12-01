import logging
import os
from contextlib import ExitStack
from dataclasses import dataclass

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
                record_shapes=False,
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


@dataclass
class NvidiaProfilerCallback(Callback):
    """
    Enables the NVIDIA profiler for PyTorch training.
    It only needs to be called at `pre_load_batch` and `post_train_batch`.
    """

    start: int = 10
    """
    The step at which to start profiling.
    """
    end: int = 12
    """
    the step at which to stop profiling.
    """
    enabled: bool = True
    """
    Set to ``False`` to disable profiling.
    """
    profile_ranks: list[int] = None
    """
    The ranks to profile.
    """

    def pre_load_batch(self):
        if self.enabled and get_rank() in self.profile_ranks:
            if self.step == self.start:
                print(f"Starting NVIDIA profiler at rank={get_rank()} step={self.step}...")
                torch.cuda.cudart().cudaProfilerStart()
                torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

    def post_train_batch(self):
        if self.enabled and get_rank() in self.profile_ranks:
            if self.step == self.end:
                torch.cuda.cudart().cudaProfilerStop()
                print(f"Stopping NVIDIA profiler at rank={get_rank()} step={self.step}...")


@dataclass
class TorchMemoryHistoryCallback(Callback):
    """
    Enables the NVIDIA profiler for PyTorch training.
    It only needs to be called at `pre_load_batch` and `post_train_batch`.
    """

    start: int = 10
    """
    The step at which to start profiling.
    """
    end: int = 12
    """
    the step at which to stop profiling.
    """
    enabled: bool = True
    """
    Set to ``False`` to disable profiling.
    """
    profile_ranks: list[int] = None
    """
    The ranks to profile.
    """

    max_entries: int = 500000

    output_dir: str = "."

    def pre_load_batch(self):
        if self.enabled and get_rank() in self.profile_ranks:
            if self.step == self.start:
                print(f"Starting memory profiler at rank={get_rank()} step={self.step}...")
                torch.cuda.memory._record_memory_history(max_entries=self.max_entries)

    def post_train_batch(self):
        if self.enabled and get_rank() in self.profile_ranks:
            if self.step == self.end:
                print(f"Dumping memory profiler at rank={get_rank()} step={self.step}...")
                os.makedirs(self.output_dir, exist_ok=True)
                torch.cuda.memory._dump_snapshot(
                    os.path.join(self.output_dir, f"memsnapshot.{get_rank()}.pickle")
                )

                print(f"Stopping memory profiler at rank={get_rank()} step={self.step}...")
                torch.cuda.memory._record_memory_history(enabled=None)

                print(f"Memory profiler stopped at rank={get_rank()} step={self.step}.")
