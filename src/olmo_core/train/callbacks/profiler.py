import logging
from contextlib import ExitStack
from dataclasses import dataclass
from functools import cache

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

    def pre_train(self):
        if not self.enabled or not _should_profile_rank(self.ranks):
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
        if not self.enabled or not _should_profile_rank(self.ranks):
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

    start: int = 8
    """
    The step at which to start profiling.
    """
    end: int = 10
    """
    The step at which to stop profiling.
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
    """
    persist_nsys_traces: bool = True
    """
    If ``True``, automatically find and persist any nsys trace files (``*.nsys-rep``) from the
    work directory to the save folder after training completes. This is useful when using
    ``nsys profile`` with torchrun via ``BeakerLaunchConfig.nsys_profile``.
    """

    _nvtx_context = None
    """
    The NVTX context manager instance, stored so it can be properly exited.
    """

    def pre_load_batch(self):
        if self.enabled and _should_profile_rank(self.ranks):
            if self.step == self.start:
                log.info(f"Starting NVIDIA profiler at rank={get_rank()} step={self.step}")
                cudart = torch.cuda.cudart()
                if cudart is not None:
                    cudart.cudaProfilerStart()
                self._nvtx_context = torch.autograd.profiler.emit_nvtx(record_shapes=True)
                self._nvtx_context.__enter__()

    def post_train_batch(self):
        if self.enabled and _should_profile_rank(self.ranks):
            if self.step == self.end:
                cudart = torch.cuda.cudart()
                if cudart is not None:
                    cudart.cudaProfilerStop()
                if self._nvtx_context is not None:
                    self._nvtx_context.__exit__(None, None, None)
                    self._nvtx_context = None
                log.info(f"Stopped NVIDIA profiler at rank={get_rank()} step={self.step}")

    def close(self):
        if self._nvtx_context is not None:
            self._nvtx_context.__exit__(None, None, None)  # ensures the context is properly exited
            self._nvtx_context = None

    def post_train(self):
        """Persist nsys trace files after training completes."""
        if not self.persist_nsys_traces:
            return

        from pathlib import Path

        # Find all nsys trace files in work_dir
        work_dir = Path(self.trainer.work_dir)
        nsys_files = list(work_dir.glob("**/*.nsys-rep"))

        if not nsys_files:
            return

        # Only persist from rank 0 to avoid duplicate uploads
        if get_rank() != 0:
            return

        log.info(f"Found {len(nsys_files)} nsys trace file(s), persisting to save folder...")

        for trace_file in nsys_files:
            try:
                # Make path relative to work_dir for persist_working_file
                relative_path = trace_file.relative_to(work_dir)
                final_path = self.trainer.persist_working_file(relative_path)
                log.info(f"Persisted nsys trace: '{trace_file.name}' -> '{final_path}'")
            except Exception as e:
                log.warning(f"Failed to persist nsys trace '{trace_file}': {e}")


@cache
def _should_profile_rank(ranks: str | None) -> bool:
    current_rank = get_rank()

    if ranks is None:
        return current_rank == 0

    world_mesh = get_world_mesh()
    if world_mesh is None:
        if ranks != "all":
            log.warning("No world mesh available, falling back to rank 0 only")
        return current_rank == 0

    try:
        if ranks == "dp":
            dp_mesh = get_dp_mesh(world_mesh)
            return dp_mesh.get_local_rank() == 0
        elif ranks == "tp":
            tp_mesh = get_tp_mesh(world_mesh)
            return tp_mesh.get_local_rank() == 0
        elif ranks == "cp":
            cp_mesh = get_cp_mesh(world_mesh)
            return cp_mesh.get_local_rank() == 0
        elif ranks == "pp":
            pp_mesh = get_pp_mesh(world_mesh)
            return pp_mesh.get_local_rank() == 0
        elif ranks == "ep":
            ep_mesh = get_ep_mesh(world_mesh)
            return ep_mesh.get_local_rank() == 0
        elif ranks == "all":
            return True
        else:
            raise ValueError(f"Unknown rank shortcut '{ranks}'")
    except RuntimeError as e:
        log.warning(
            f"Failed to determine parallel mesh for '{ranks}': {e}, falling back to rank 0 only"
        )
        return current_rank == 0
