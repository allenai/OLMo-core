"""
ModelMergeCallback that relies on external checkpoint coordination.

This callback:
- Supports explicit merge_step configuration or interval-based merging
- Does NOT save accumulator state (no state_dict/load_state_dict)
- Assumes checkpoints are saved at window start steps externally
- On resume, recomputes accumulation from scratch
"""

import logging
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Union

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

    IMPORTANT: This callback requires external checkpoint coordination.
    The checkpointer must be configured to save at each window start step
    (merge_step - merge_last_n_steps) to enable clean recovery on resume.

    Use ``compute_merge_window_starts()`` to get the required checkpoint steps.
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
    _accumulator: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)
    _n_accumulated: int = field(default=0, repr=False)
    _merge_steps: List[int] = field(default_factory=list, repr=False)
    _current_merge_idx: int = field(default=0, repr=False)

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

        self._check_for_overlapping_windows()

    def _check_for_overlapping_windows(self):
        # TODO: Allow overlapping windows. Track multiple accumulators and 
        # merge when each window completes.
        for i in range(1, len(self._merge_steps)):
            prev_step = self._merge_steps[i - 1]
            curr_step = self._merge_steps[i]
            gap = curr_step - prev_step

            if gap < self.merge_last_n_steps:
                raise OLMoConfigurationError(
                    f"Merge steps {prev_step} and {curr_step} are only {gap} steps apart, "
                    f"but merge_last_n_steps={self.merge_last_n_steps}. Windows would overlap."
                )

    @property
    def _current_merge_step(self) -> Optional[int]:
        if self._current_merge_idx >= len(self._merge_steps):
            return None
        return self._merge_steps[self._current_merge_idx]

    @property
    def _start_step(self) -> int:
        if self._current_merge_step is None:
            return -1
        return max(0, self._current_merge_step - self.merge_last_n_steps + 1)

    @property
    def _is_accumulating(self) -> bool:
        if self._current_merge_step is None:
            return False
        return self._start_step <= self.step <= self._current_merge_step

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
            self._check_for_overlapping_windows()

        log.info(f"ModelMergeCallback: merge_steps={self._merge_steps}")

        self._validate_checkpointer_config()

        # Skip past merge steps we can no longer accumulate for
        current_step = self.step
        while self._current_merge_idx < len(self._merge_steps):
            target = self._merge_steps[self._current_merge_idx]

            if target < current_step:
                log.warning(
                    f"Current step {current_step} is past merge step {target}. "
                    "This merge will be skipped."
                )
                self._current_merge_idx += 1
                continue

            break

        # Log window start steps for verification
        remaining = self._merge_steps[self._current_merge_idx:]
        window_starts = [max(0, s - self.merge_last_n_steps + 1) for s in remaining]
        log.info(
            f"Remaining merge steps: {remaining}. "
            f"Ensure checkpoints exist at window starts: {window_starts}"
        )

    def _validate_checkpointer_config(self):
        """
        Validate that the checkpointer is configured correctly for model merging.

        Requirements:
        1. Checkpointer's fixed_steps must include window start steps
        2. No ephemeral checkpoints should occur during merge windows

        This callback does NOT save its own state, so proper checkpoint coordination
        is required to ensure accumulation can be recomputed on resume.
        """
        from .checkpointer import CheckpointerCallback

        checkpointer = self.trainer.callbacks.get("checkpointer")
        if not isinstance(checkpointer, CheckpointerCallback):
            log.warning(
                "ModelMergeCallback: No CheckpointerCallback found. "
                "Cannot validate checkpoint configuration."
            )
            return

        # Compute required window start steps
        window_starts = compute_merge_window_starts(self._merge_steps, self.merge_last_n_steps)

        # Check that fixed_steps includes all window starts
        fixed_steps = set(checkpointer.fixed_steps or [])
        missing_starts = [s for s in window_starts if s not in fixed_steps]

        if missing_starts:
            raise OLMoConfigurationError(
                f"ModelMergeCallback: Checkpointer's fixed_steps is missing window start steps: "
                f"{missing_starts}. These checkpoints are required for proper resume behavior. "
                f"Add them using compute_merge_window_starts() in your ladder configuration."
            )

        # Check that ephemeral checkpoints won't occur during merge windows
        ephemeral_interval = checkpointer.ephemeral_save_interval
        if ephemeral_interval is not None:
            for merge_step in self._merge_steps:
                window_start = max(0, merge_step - self.merge_last_n_steps + 1)
                for step in range(window_start + 1, merge_step + 1):
                    if step % ephemeral_interval == 0:
                        raise OLMoConfigurationError(
                            f"ModelMergeCallback: Ephemeral checkpoint at step {step} "
                            f"would occur during merge window [{window_start}, {merge_step}]. "
                            f"This could cause partial accumulation on resume. Either:\n"
                            f"  - Increase ephemeral_save_interval to > {self.merge_last_n_steps}, or\n"
                            f"  - Disable ephemeral checkpoints (ephemeral_save_interval=None), or\n"
                            f"  - Adjust merge_last_n_steps to avoid overlap"
                        )

        log.info(
            f"ModelMergeCallback: Checkpointer configuration validated. "
            f"Window starts {window_starts} are in fixed_steps."
        )

    def post_train_batch(self):
        if not self.enabled or self._current_merge_step is None:
            return

        if self._is_accumulating:
            self._accumulate_weights()

        if self.step == self._current_merge_step:
            self._save_merged_checkpoint()

    def _accumulate_weights(self):
        model = self.trainer.train_module.model

        model_state = {
            k: get_local_tensor(p.data.detach()).to("cpu")
            for k, p in model.named_parameters()
        }

        if self._accumulator is None:
            log.info(f"Starting weight accumulation at step {self.step}")
            self._accumulator = {
                k: torch.zeros_like(v, dtype=torch.float32, device="cpu")
                for k, v in model_state.items()
            }

        for key, value in model_state.items():
            self._accumulator[key].add_(value.float())

        self._n_accumulated += 1
        log.debug(f"Accumulated weights at step {self.step} ({self._n_accumulated} total)")

    def _save_merged_checkpoint(self):
        if self._accumulator is None or self._n_accumulated == 0:
            log.warning("No weights accumulated, cannot save merged checkpoint")
            return

        log.info(
            f"Saving merged checkpoint (average of {self._n_accumulated} steps) "
            f"at step {self.step}"
        )

        averaged_state: Dict[str, torch.Tensor] = {
            key: acc_val / self._n_accumulated
            for key, acc_val in self._accumulator.items()
        }

        output_path = self._merged_checkpoint_path(self.step)

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

        # Clean up and advance
        self._accumulator = None
        self._n_accumulated = 0
        self._current_merge_idx += 1

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
    # On resume, accumulation starts fresh from the checkpoint at window start.


# ============================================================================
# Helper functions for ladder configuration
# ============================================================================

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
    Compute the window start steps where checkpoints should be saved.

    These are the steps at which accumulation begins. If training resumes
    from a checkpoint at a window start, accumulation can be recomputed.
    """
    return [max(0, step - merge_last_n_steps + 1) for step in merge_steps]
