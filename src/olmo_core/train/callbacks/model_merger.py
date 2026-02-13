"""
ModelMergeCallback for averaging model weights over a window of training steps.

This callback:
- Supports explicit merge_step configuration or interval-based merging
- Supports overlapping merge windows with per-window accumulators
- Does NOT save accumulator state (no state_dict/load_state_dict)
- On resume, recomputes accumulation from the last checkpoint
"""

import logging
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Set, Union

import torch

from olmo_core.distributed.checkpoint import save_state_dict
from olmo_core.distributed.utils import barrier, get_local_tensor, get_rank
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import clear_directory, dir_is_empty, file_exists, is_url, join_path

from .callback import Callback
from .evaluator_callback import EvaluatorCallback

log = logging.getLogger(__name__)


@dataclass
class ModelMergeCallback(Callback):
    """
    Averages model weights over the last ``merge_last_n_steps`` before each ``merge_step``
    and saves the result as a merged checkpoint.

    Ephemeral checkpoints are blocked during merge windows to ensure the full
    window is always re-accumulated on resume.
    """

    priority: ClassVar[int] = 2

    merge_step: Union[int, List[int]] = field(default_factory=list)
    """The step(s) at which to save merged checkpoint(s)."""

    merge_interval: Optional[int] = None
    """Merge every N steps. Alternative to explicit merge_step."""

    merge_last_n_steps: int = 100
    """Number of steps before each merge step to start accumulating the average."""

    output_suffix: str = "merged"
    """Suffix for merged checkpoint directory."""

    enabled: bool = False

    # Internal state (not checkpointed)
    _accumulators: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict, repr=False)
    _accumulator_counts: Dict[int, int] = field(default_factory=dict, repr=False)
    _merge_steps: List[int] = field(default_factory=list, repr=False)
    _completed_merges: Set[int] = field(default_factory=set, repr=False)

    def __post_init__(self):
        if self.merge_last_n_steps <= 0:
            raise OLMoConfigurationError(
                f"merge_last_n_steps must be positive, got {self.merge_last_n_steps}"
            )

        # Validate merge_interval
        if self.merge_interval is not None:
            if self.merge_interval <= 0:
                raise OLMoConfigurationError(
                    f"merge_interval must be positive, got {self.merge_interval}"
                )
            # Can't set both merge_step and merge_interval
            has_merge_step = (
                (isinstance(self.merge_step, int) and self.merge_step > 0)
                or (isinstance(self.merge_step, list) and len(self.merge_step) > 0)
            )
            if has_merge_step:
                raise OLMoConfigurationError(
                    "Cannot set both merge_step and merge_interval. Use one or the other."
                )
            # Defer step computation to pre_train (needs max_steps from trainer)
            return

        # Convert merge_step to list
        if isinstance(self.merge_step, int):
            self._merge_steps = [self.merge_step]
        else:
            self._merge_steps = sorted(self.merge_step)

        if not self._merge_steps:
            raise OLMoConfigurationError(
                "Either merge_step or merge_interval must be set."
            )

        # Validate
        invalid = [s for s in self._merge_steps if s <= 0]
        if invalid:
            raise OLMoConfigurationError(f"merge_step values must be positive, got: {invalid}")

    def _window_start(self, merge_step: int) -> int:
        return max(0, merge_step - self.merge_last_n_steps + 1)

    def _active_windows(self) -> List[int]:
        """Return merge steps whose windows include the current step."""
        current = self.step
        return [
            ms for ms in self._merge_steps
            if ms not in self._completed_merges
            and self._window_start(ms) <= current <= ms
        ]

    def _merged_checkpoint_path(self, step: int) -> str:
        return str(join_path(self.trainer.save_folder, f"step{step}-{self.output_suffix}"))

    def _merged_checkpoint_exists(self, step: int) -> bool:
        try:
            metadata_path = join_path(
                self._merged_checkpoint_path(step), "model_and_optim", ".metadata"
            )
            return file_exists(metadata_path)
        except Exception:
            return False

    def pre_train(self):
        if not self.enabled:
            return

        # Compute merge steps from interval if needed
        if self.merge_interval is not None:
            max_steps = self.trainer.max_steps
            if max_steps is None:
                raise OLMoConfigurationError(
                    "merge_interval requires max_steps to be known. "
                    "Set max_duration on the trainer."
                )
            self._merge_steps = list(range(self.merge_interval, max_steps + 1, self.merge_interval))
            if not self._merge_steps:
                log.warning(
                    f"No merge steps computed: merge_interval={self.merge_interval}, "
                    f"max_steps={max_steps}"
                )
                return

        log.info(f"ModelMergeCallback: merge_steps={self._merge_steps}")

        # Mark merge steps that are already past as completed
        current_step = self.step
        for ms in self._merge_steps:
            if ms < current_step:
                log.warning(
                    f"Current step {current_step} is past merge step {ms}. "
                    "This merge will be skipped."
                )
                self._completed_merges.add(ms)

        remaining = [ms for ms in self._merge_steps if ms not in self._completed_merges]
        log.info(f"Remaining merge steps: {remaining}")

    def post_train_batch(self):
        if not self.enabled:
            return

        active = self._active_windows()

        # Block ephemeral checkpoints during merge windows to prevent
        # mid-window resume points that would shorten the average.
        self.trainer.block_ephemeral_checkpoints = len(active) > 0

        if not active:
            return

        # Copy model weights to CPU once for all active windows
        model = self.trainer.train_module.model
        model_state = {
            k: get_local_tensor(p.data.detach()).to("cpu")
            for k, p in model.named_parameters()
        }

        for ms in active:
            self._accumulate_weights(ms, model_state)

        # Save any windows that just completed
        for ms in active:
            if self.step == ms:
                self._save_merged_checkpoint(ms)

    def _accumulate_weights(self, merge_step: int, model_state: Dict[str, torch.Tensor]):
        if merge_step not in self._accumulators:
            log.info(f"Starting weight accumulation for merge step {merge_step} at step {self.step}")
            self._accumulators[merge_step] = {
                k: torch.zeros_like(v, dtype=torch.float32, device="cpu")
                for k, v in model_state.items()
            }
            self._accumulator_counts[merge_step] = 0

        for key, value in model_state.items():
            self._accumulators[merge_step][key].add_(value.float())

        self._accumulator_counts[merge_step] += 1
        log.debug(
            f"Accumulated weights for merge step {merge_step} at step {self.step} "
            f"({self._accumulator_counts[merge_step]} total)"
        )

    def _save_merged_checkpoint(self, merge_step: int):
        accumulator = self._accumulators.get(merge_step)
        count = self._accumulator_counts.get(merge_step, 0)

        if accumulator is None or count == 0:
            log.warning(f"No weights accumulated for merge step {merge_step}, cannot save")
            return

        log.info(
            f"Saving merged checkpoint (average of {count} steps) "
            f"at step {merge_step}"
        )

        averaged_state: Dict[str, torch.Tensor] = {
            key: acc_val / count
            for key, acc_val in accumulator.items()
        }

        output_path = self._merged_checkpoint_path(merge_step)

        import os
        if get_rank() == 0:
            if not dir_is_empty(output_path):
                clear_directory(output_path)
            if not is_url(output_path):
                os.makedirs(output_path, exist_ok=True)

        barrier()

        save_state_dict(
            join_path(output_path, "model_and_optim"),
            {"model": averaged_state, "optim": {}},
            process_group=self.trainer.checkpointer.process_group,
        )

        barrier()
        log.info(f"Merged checkpoint saved to: {output_path}")

        self._evaluate_merged(averaged_state)

        # Clean up
        del self._accumulators[merge_step]
        del self._accumulator_counts[merge_step]
        self._completed_merges.add(merge_step)

    def _evaluate_merged(self, averaged_state: Dict[str, torch.Tensor]):
        evaluator_callbacks = [
            cb for cb in self.trainer.callbacks.values() if isinstance(cb, EvaluatorCallback)
        ]

        if not evaluator_callbacks:
            log.info("No EvaluatorCallback found, skipping merged model evaluation")
            return

        model = self.trainer.train_module.model
        params_dict = dict(model.named_parameters())

        # Store original weights on CPU to avoid doubling GPU memory
        log.info("Storing original model weights...")
        original_state = {
            k: get_local_tensor(p.data.detach()).to("cpu").clone()
            for k, p in params_dict.items()
        }

        barrier()

        log.info("Loading merged weights for evaluation...")
        with torch.no_grad():
            for name, param in params_dict.items():
                if name in averaged_state:
                    local_param = get_local_tensor(param.data)
                    local_param.copy_(averaged_state[name].to(local_param.device, local_param.dtype))

        barrier()

        try:
            for callback in evaluator_callbacks:
                log.info(f"Running merged model evaluation via {callback.__class__.__name__}...")
                callback._perform_eval(prefix="eval/merged")
        finally:
            log.info("Restoring original model weights...")
            with torch.no_grad():
                for name, param in params_dict.items():
                    if name in original_state:
                        local_param = get_local_tensor(param.data)
                        local_param.copy_(original_state[name].to(local_param.device, local_param.dtype))
            barrier()

    # NO state_dict or load_state_dict - state is not checkpointed!
    # On resume, accumulation starts fresh from the last checkpoint.
    

def compute_merge_steps_from_wsds(
    period_lengths: List[int],
    tokens_per_step: int,
    decay: Optional[int] = None,
    decay_fraction: Optional[float] = None,
) -> List[int]:
    """
    Compute merge steps from WSDS period configuration.
    """
    if decay is None and decay_fraction is None:
        raise ValueError("Either decay or decay_fraction must be set")

    merge_steps = []
    cumulative_tokens = 0

    for period_length in period_lengths:
        cumulative_tokens += period_length

        decay_tokens = (
            decay if decay is not None
            else int(round(decay_fraction * period_length))
        )

        pre_decay_tokens = cumulative_tokens - decay_tokens
        pre_decay_step = pre_decay_tokens // tokens_per_step
        merge_steps.append(pre_decay_step)

    return merge_steps


def compute_merge_window_starts(
    merge_steps: List[int],
    merge_last_n_steps: int,
) -> List[int]:
    """
    Compute the required checkpoint steps for merge windows.

    For overlapping windows, only the earliest start in each overlapping group
    is returned, since resuming from that checkpoint allows all overlapping
    windows to re-accumulate correctly.
    """
    if not merge_steps:
        return []

    required_starts: List[int] = []
    prev_merge_step = -1

    for ms in sorted(merge_steps):
        start = max(0, ms - merge_last_n_steps + 1)
        # If this window starts after the previous merge step completed,
        # it's a new group and needs its own checkpoint
        if start > prev_merge_step:
            required_starts.append(start)
        prev_merge_step = ms

    return required_starts
