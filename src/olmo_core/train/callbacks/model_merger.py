import logging
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed
import torch.distributed.checkpoint.state_dict as dist_cp_sd

from olmo_core.distributed.checkpoint import save_state_dict
from olmo_core.distributed.utils import barrier, get_rank
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import clear_directory, dir_is_empty, file_exists, is_url, join_path
from olmo_core.optim.scheduler import WSDS, Scheduler

from .callback import Callback
from .evaluator_callback import EvaluatorCallback

log = logging.getLogger(__name__)


@dataclass
class ModelMergeCallback(Callback):
    """
    Averages model weights over the last ``merge_last_n_steps`` before ``merge_step``
    and saves the result as a merged checkpoint (no optimizer state).
    """

    priority: ClassVar[int] = 2

    merge_step: Optional[Union[int, List[int]]] = None
    """
    The step(s) at which to save merged checkpoint(s). Can be a single int or a list of ints.
    If not set:
    - For WSDS schedulers: defaults to the steps right before each decay phase begins.
    - For other schedulers: defaults to every ``merge_interval`` steps.
    """

    merge_interval: int = 5000
    """Interval at which to save merged checkpoints (e.g., steps 5000, 10000, ...)."""

    merge_last_n_steps: int = 100
    """Number of steps before ``merge_step`` to start accumulating the average."""

    output_suffix: str = "merged"
    """
    Suffix for merged checkpoint directory. Checkpoint will be saved as "step{merge_step}-{output_suffix}".
    """

    enabled: bool = False
    _accumulator: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)
    _n_accumulated: int = field(default=0, repr=False)
    _merge_steps: List[int] = field(default_factory=list, repr=False)
    _current_merge_idx: int = field(default=0, repr=False)
    _restored: bool = field(default=False, repr=False)

    def __post_init__(self):
        if self.merge_last_n_steps <= 0:
            raise OLMoConfigurationError(
                f"ModelMergeCallback: merge_last_n_steps must be positive, got {self.merge_last_n_steps}"
            )

        # Validate merge steps
        if self.merge_step is not None:
            steps = [self.merge_step] if isinstance(self.merge_step, int) else self.merge_step
            invalid = [s for s in steps if s <= 0]
            if invalid:
                raise OLMoConfigurationError(
                    f"ModelMergeCallback: merge_step values must be positive, got invalid step(s): {invalid}"
                )

    def _get_decay_steps_from_scheduler(
        self, scheduler: Scheduler, max_steps: int
    ) -> Tuple[int, str]:
        assert max_steps >= 0, "max_steps must be >= 0"

        # Check for explicit decay attribute
        if hasattr(scheduler, "decay") and scheduler.decay is not None:
            return scheduler.decay, "scheduler.decay"

        if hasattr(scheduler, "decay_fraction") and scheduler.decay_fraction is not None:
            decay_steps = round(max_steps * scheduler.decay_fraction)
            return decay_steps, f"scheduler.decay_fraction ({scheduler.decay_fraction})"

        # No decay information found
        return 0, "no decay found"

    def _get_merge_steps_from_wsds(self, scheduler: WSDS, tokens_per_step: int) -> List[int]:
        """
        Extract merge steps from a WSDS scheduler (one per period, before each decay).
        NOTE: This is to handle the Olmo 3 ladder. We may need to expand on this to also include WSD.
        """
        assert tokens_per_step > 0, "tokens_per_step must be > 0"

        fixed_decay = scheduler.decay
        decay_fraction = scheduler.decay_fraction
        if fixed_decay is None:
            assert (
                decay_fraction is not None
            ), "Either scheduler.decay or scheduler.decay_fraction must be set"

        merge_steps = []
        cumulative_tokens = 0

        for i, period_length in enumerate(scheduler.period_lengths):
            cumulative_tokens += period_length  # end-of-period tokens

            decay_tokens = (
                fixed_decay
                if fixed_decay is not None
                else int(round(decay_fraction * period_length))
            )

            # Pre-decay merge (before decay starts)
            pre_decay_tokens = cumulative_tokens - decay_tokens
            pre_decay_step = pre_decay_tokens // tokens_per_step
            merge_steps.append(pre_decay_step)

            log.debug(
                "WSDS period %d: end=%d tokens, decay=%d tokens, merge_step=%d",
                i,
                cumulative_tokens,
                decay_tokens,
                pre_decay_step,
            )

        return merge_steps

    @property
    def _current_merge_step(self) -> Optional[int]:
        """The current merge step we're working towards."""
        if self._current_merge_idx >= len(self._merge_steps):
            return None
        return self._merge_steps[self._current_merge_idx]

    def _merged_checkpoint_path(self, step: int) -> str:
        """Returns the path for a merged checkpoint at a given step."""
        return str(join_path(self.trainer.save_folder, f"step{step}-{self.output_suffix}"))

    def _merged_checkpoint_exists(self, step: int) -> bool:
        """Check if a complete merged checkpoint already exists for a given step."""
        try:
            # Check for .metadata file which is written last by save_state_dict
            metadata_path = join_path(
                self._merged_checkpoint_path(step), "model_and_optim", ".metadata"
            )
            return file_exists(metadata_path)
        except Exception as e:
            log.warning(
                "ModelMergeCallback: failed to check if merged checkpoint exists for step %d: %s. "
                "Assuming checkpoint does not exist.",
                step,
                e,
            )
            return False

    def _compute_merge_steps(self) -> bool:
        """Compute merge steps from config or scheduler."""
        if self.merge_step is None:
            if self.trainer.max_steps is None:
                log.warning(
                    "ModelMergeCallback: merge_step not set and trainer has no max_steps. "
                    "Disabling merging."
                )
                self.enabled = False
                return False

            max_steps = self.trainer.max_steps
            scheduler = getattr(self.trainer.train_module, "scheduler", None)

            if isinstance(scheduler, WSDS):
                # NOTE: This assumes global_batch_size is in tokens
                tokens_per_step = self.trainer.data_loader.global_batch_size
                self._merge_steps = self._get_merge_steps_from_wsds(scheduler, tokens_per_step)
                log.info(
                    "ModelMergeCallback: detected WSDS scheduler with %d periods. "
                    "merge_steps set to %s (one before each decay phase, assuming %d tokens/step)",
                    len(scheduler.period_lengths),
                    self._merge_steps,
                    tokens_per_step,
                )
            else:
                # For non-WSDS schedulers, use fixed intervals
                self._merge_steps = list(
                    range(self.merge_interval, max_steps + 1, self.merge_interval)
                )
                if not self._merge_steps:
                    self._merge_steps = [max_steps]
                log.info(
                    "ModelMergeCallback: merge_steps set to %s (every %d steps)",
                    self._merge_steps,
                    self.merge_interval,
                )
        elif isinstance(self.merge_step, int):
            self._merge_steps = [self.merge_step]
            log.info("ModelMergeCallback: merge_step set to %d", self.merge_step)
        else:
            self._merge_steps = sorted(self.merge_step)
            log.info("ModelMergeCallback: merge_steps set to %s", self._merge_steps)

        self._merge_steps = sorted(set(self._merge_steps))
        return True

    def _warn_if_merge_in_decay_phase(self):
        """Warn if any merge steps fall within the decay phase."""
        if self.trainer.max_steps is None:
            return

        scheduler = getattr(self.trainer.train_module, "scheduler", None)
        if scheduler is None:
            return

        # Warn if user manually specified merge_step in decay phase of scheduler
        if isinstance(scheduler, WSDS) and self.merge_step is None:
            return

        decay_steps, decay_source = self._get_decay_steps_from_scheduler(
            scheduler, self.trainer.max_steps
        )
        if decay_steps <= 0:
            return

        decay_start = self.trainer.max_steps - decay_steps
        steps_in_decay = [s for s in self._merge_steps if s > decay_start]
        if steps_in_decay:
            log.warning(
                "ModelMergeCallback: merge step(s) %s fall within the decay phase "
                "(decay starts at step %d, from %s). "
                "Model weights during decay may not be ideal for merging.",
                steps_in_decay,
                decay_start + 1,
                decay_source,
            )

    def _warn_for_truncated_windows(self):
        """Warn if any merge steps would have truncated accumulation windows."""
        for step in self._merge_steps:
            if step < self.merge_last_n_steps:
                actual_window = step  # Can only accumulate from step 1 to step
                log.warning(
                    "ModelMergeCallback: merge step %d is less than merge_last_n_steps=%d. "
                    "The accumulation window will be truncated to %d steps (steps 1-%d).",
                    step,
                    self.merge_last_n_steps,
                    actual_window,
                    step,
                )

    def _check_for_overlapping_merge_windows(self):
        """
        Check if any merge windows overlap.
        TODO: Allow overlapping windows by tracking multiple accumulators, if desired.
        """
        if len(self._merge_steps) < 2:
            return

        for i in range(1, len(self._merge_steps)):
            prev_step = self._merge_steps[i - 1]
            curr_step = self._merge_steps[i]
            gap = curr_step - prev_step

            if gap < self.merge_last_n_steps:
                # NOTE: Could change to warning + skip instead of error, since merging is
                # complementary to training, not critical. For now, fail early to alert user.
                raise OLMoConfigurationError(
                    f"ModelMergeCallback: merge steps {prev_step} and {curr_step} are only {gap} steps apart, "
                    f"but merge_last_n_steps={self.merge_last_n_steps}. "
                    f"Merge windows would overlap. Either:\n"
                    f"  - Decrease merge_last_n_steps to <= {gap}, or\n"
                    f"  - Increase the gap between merge steps, or\n"
                    f"  - Remove one of the merge steps"
                )

    def pre_train(self):
        if not self.enabled:
            return

        # Restore or compute merge steps
        if self._restored and self._merge_steps:
            log.info(
                "ModelMergeCallback: restored state from checkpoint; using merge_steps=%s and current_merge_idx=%d",
                self._merge_steps,
                self._current_merge_idx,
            )
        elif not self._compute_merge_steps():
            return  # Could not determine merge steps (no merge_step set and no max_steps)

        self._warn_for_truncated_windows()
        self._warn_if_merge_in_decay_phase()
        self._check_for_overlapping_merge_windows()

        # Handle resume scenarios where we may have accumulated weights ready to save
        # or need to skip past completed merge steps
        current_step = self.step
        while self._current_merge_idx < len(self._merge_steps):
            target_merge_step = self._merge_steps[self._current_merge_idx]

            # If merged checkpoint already exists, skip it
            if self._merged_checkpoint_exists(target_merge_step):
                log.info(
                    "ModelMergeCallback: merged checkpoint for step %d already exists; skipping.",
                    target_merge_step,
                )
                self._accumulator = None
                self._n_accumulated = 0
                self._current_merge_idx += 1
                continue

            if target_merge_step < current_step:
                if self._n_accumulated > 0:
                    # We were accumulating for this merge step, but we're now past it.
                    # This can happen if the job was interrupted after accumulating but before saving.
                    log.warning(
                        "ModelMergeCallback: current step %d is past merge step %d, but we have %d accumulated "
                        "weight snapshots. Saving merged checkpoint now (may be missing final step(s)).",
                        current_step,
                        target_merge_step,
                        self._n_accumulated,
                    )
                    self._save_merged_checkpoint(target_step=target_merge_step)
                else:
                    # We're past this merge step and have nothing accumulated; skip it.
                    log.info(
                        "ModelMergeCallback: skipping past merge step %d (current step is %d)",
                        target_merge_step,
                        current_step,
                    )
                    self._accumulator = None
                    self._n_accumulated = 0
                    self._current_merge_idx += 1
                continue  # Re-check with updated index

            if target_merge_step == current_step and self._n_accumulated > 0:
                # We're exactly at the merge step with accumulated weights ready
                # This happens when resuming from a checkpoint saved before the merge completed
                log.info(
                    "ModelMergeCallback: resuming at merge step %d with %d accumulated weights, saving now...",
                    target_merge_step,
                    self._n_accumulated,
                )
                self._save_merged_checkpoint(target_step=target_merge_step)
            else:
                # Not past and not at merge step with weights ready - stop checking
                break

    @property
    def _start_step(self) -> int:
        """The step at which to start accumulating for the current merge step."""
        if self._current_merge_step is None:
            return -1
        # +1 so that merge_last_n_steps=100 gives exactly 100 steps (e.g., 901-1000 inclusive)
        return max(0, self._current_merge_step - self.merge_last_n_steps + 1)

    @property
    def _is_accumulating(self) -> bool:
        """Whether we are currently in the accumulation window for the current merge step."""
        if self._current_merge_step is None:
            return False
        return self._start_step <= self.step <= self._current_merge_step

    def post_train_batch(self):
        """
        After each training batch, potentially accumulate weights or save the merged model.
        """
        if not self.enabled or self._current_merge_step is None:
            return

        # Check if we should accumulate
        if self._is_accumulating:
            self._accumulate_weights()

        # Check if we should save
        if self.step == self._current_merge_step:
            self._save_merged_checkpoint()

    def post_train(self):
        """
        Warn if training ended before all merges could complete.
        """
        if not self.enabled or not self._merge_steps:
            return

        # Check if there are remaining merge steps that weren't reached
        remaining_steps = self._merge_steps[self._current_merge_idx :]
        if remaining_steps:
            unreached = [s for s in remaining_steps if self.step < s]
            if unreached:
                log.warning(
                    f"Training ended at step {self.step} before reaching merge_step(s) {unreached}. "
                    "These merges will not be performed."
                )

    def _accumulate_weights(self):
        """
        Adds current model weights to the accumulator (stored on CPU).

        Each rank accumulates its own shard of the weights (sharded approach).

        NOTE(large-scale): For larger models, calling state_dict() and moving to CPU at every
        step in the merge window may cause significant throughput overhead.
        Might want to consider sparse sampling (accumulate every Nth step instead of every step)
        or doing this one time (moving to CPU only at the end)
        """
        sd_options = dist_cp_sd.StateDictOptions(full_state_dict=False, cpu_offload=True)
        model_state = dist_cp_sd.get_model_state_dict(
            self.trainer.train_module.model, options=sd_options
        )

        if self._accumulator is None:
            log.info(f"Starting model weight averaging at step {self.step}")
            self._accumulator = {
                k: torch.zeros_like(v, dtype=torch.float32, device="cpu")
                for k, v in model_state.items()
            }

        for key, value in model_state.items():
            self._accumulator[key].add_(value.float())

        self._n_accumulated += 1
        log.debug(f"Accumulated weights at step {self.step} ({self._n_accumulated} total)")

    def _save_merged_checkpoint(self, target_step: Optional[int] = None):
        if self._accumulator is None or self._n_accumulated == 0:
            log.warning("No weights accumulated, cannot save merged checkpoint")
            return

        # Use target_step if provided (for recovery saves), otherwise use current step
        checkpoint_step = target_step if target_step is not None else self.step

        log.info(
            f"Saving merged checkpoint (average of {self._n_accumulated} steps) at step {checkpoint_step}"
        )

        # Each rank computes the average of its shard
        averaged_state: Dict[str, torch.Tensor] = {}
        for key, accumulated_value in self._accumulator.items():
            averaged_state[key] = accumulated_value / self._n_accumulated

        output_path = str(
            join_path(
                self.trainer.save_folder,
                f"step{checkpoint_step}-{self.output_suffix}",
            )
        )

        if get_rank() == 0:
            if not dir_is_empty(output_path):
                clear_directory(output_path)
            # For local paths, create the directory (remote paths are created implicitly)
            if not is_url(output_path):
                os.makedirs(output_path, exist_ok=True)

        # Wait for directory to be ready before all ranks save
        barrier()

        save_state_dict(
            join_path(output_path, "model_and_optim"),
            {"model": averaged_state, "optim": {}},
            process_group=self.trainer.checkpointer.process_group,
        )

        # Wait for all ranks to finish saving before proceeding
        barrier()

        log.info(f"Merged checkpoint saved to: {output_path}")

        self._evaluate_merged(averaged_state)

        # Clean up and advance to next merge step
        self._accumulator = None
        self._n_accumulated = 0
        self._current_merge_idx += 1

    def _evaluate_merged(self, averaged_state: Dict[str, torch.Tensor]):
        """
        Run evaluators with merged weights (metrics under 'eval/merged' prefix),
        then restore original weights.
        """

        evaluator_callbacks = [
            cb for cb in self.trainer.callbacks.values() if isinstance(cb, EvaluatorCallback)
        ]

        if not evaluator_callbacks:
            log.info("No EvaluatorCallback instances found, skipping merged model evaluation")
            return

        model = self.trainer.train_module.model

        # Store original weights using get_model_state_dict for proper FSDP support
        # Use cpu_offload=True to avoid doubling GPU memory usage
        log.info("Storing original model weights for merged model evaluation...")
        sd_options = dist_cp_sd.StateDictOptions(full_state_dict=False, cpu_offload=True)
        original_state = dist_cp_sd.get_model_state_dict(model, options=sd_options)

        # Ensure all ranks are ready before loading merged weights
        barrier()

        # Load merged weights using set_model_state_dict for proper FSDP support
        # Use full_state_dict=False since averaged_state is sharded (each rank has its shard)
        log.info("Loading merged weights for evaluation...")
        dist_cp_sd.set_model_state_dict(
            model,
            averaged_state,
            options=dist_cp_sd.StateDictOptions(full_state_dict=False, strict=True),
        )

        # Ensure all ranks have loaded merged weights before evaluation
        barrier()

        # Run evaluator callbacks with "eval/merged" prefix
        try:
            for callback in evaluator_callbacks:
                log.info(f"Running merged model evaluation via {callback.__class__.__name__}...")
                callback._perform_eval(prefix="eval/merged")
        finally:
            # Restore original weights using set_model_state_dict for proper FSDP support
            # Use full_state_dict=False since original_state is sharded
            log.info("Restoring original model weights...")
            dist_cp_sd.set_model_state_dict(
                model,
                original_state,
                options=dist_cp_sd.StateDictOptions(full_state_dict=False, strict=True),
            )
            # Ensure all ranks have restored before continuing training
            barrier()

    def state_dict(self) -> Dict[str, Any]:
        """
        Save callback state for checkpointing.

        NOTE: Each rank saves its own shard of the accumulator.
        This increases total checkpoint size by ~1x model sizeca.
        This might cause issues with scalability. Could consider:
            - Not checkpointing accumulator (would lose ability to resume mid-window)
            - Making accumulator checkpointing configurable
        """
        state = {
            "n_accumulated": self._n_accumulated,
            "current_merge_idx": self._current_merge_idx,
            "merge_steps": self._merge_steps,
        }
        if self._accumulator is not None:
            state["accumulator"] = self._accumulator
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Restore callback state from checkpoint.
        """
        self._n_accumulated = state_dict.get("n_accumulated", 0)
        self._current_merge_idx = state_dict.get("current_merge_idx", 0)
        self._merge_steps = state_dict.get("merge_steps", [])
        self._accumulator = state_dict.get("accumulator", None)
        self._restored = True
