import logging
from contextlib import ExitStack
from dataclasses import dataclass

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
    enabled: bool = True
    """
    Set to ``False`` to disable profiling.
    """

    _exit_stack = None
    _profiler = None
    _first_batch: bool = True

    def pre_train(self):
        if not self.enabled or get_rank() != 0:
            return

        from torch.profiler import ProfilerActivity, profile, schedule

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

        self._exit_stack = ExitStack()
        self._profiler = self._exit_stack.enter_context(
            profile(
                activities=activities,
                record_shapes=False,
                profile_memory=False,
                with_stack=True,
                schedule=profiling_schedule,
                on_trace_ready=self._on_trace_ready,
            )
        )
        self._first_batch = True

    def pre_load_batch(self):
        if not self.enabled or get_rank() != 0:
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
        trace_path = output_dir / f"step-{prof.step_num}.chrome_trace.json.gz"
        prof.export_chrome_trace(str(trace_path))
        final_path = self.trainer.persist_working_file(trace_path)
        log.info(f"Chrome trace saved to '{final_path}'")
