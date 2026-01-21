import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from olmo_core.distributed.checkpoint import save_state_dict
from olmo_core.distributed.utils import get_rank
from olmo_core.io import join_path
from olmo_core.optim.scheduler import Scheduler

from .callback import Callback
from .evaluator_callback import EvaluatorCallback

log = logging.getLogger(__name__)


@dataclass
class ModelMergeCallback(Callback):
    """
    Averages model weights over the last ``merge_last_n_steps`` before ``merge_step``
    and saves the result as a merged checkpoint.
    """

    merge_step: Optional[Union[int, List[int]]] = None
    """
    The step(s) at which to save merged checkpoint(s). Can be a single int or a list of ints.
    If not set, defaults to the step right before the scheduler's decay phase begins.
    For schedulers without a decay phase, defaults to max_steps.
    """

    merge_last_n_steps: int = 100
    """
    Number of steps before ``merge_step`` to start accumulating the average.
    """

    output_suffix: str = "merged"
    """
    Suffix for the output checkpoint directory name.
    The merged checkpoint will be saved as "step{merge_step}-{output_suffix}".
    """

    validate: bool = False
    """
    If True, captures individual step weights during accumulation and validates
    that the merged weights are correctly averaged. Useful for testing.
    """

    enabled: bool = True
    _accumulator: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)
    _n_accumulated: int = field(default=0, repr=False)
    _merge_steps: List[int] = field(default_factory=list, repr=False)
    _current_merge_idx: int = field(default=0, repr=False)
    _captured_weights: List[Dict[str, torch.Tensor]] = field(default_factory=list, repr=False)

    def _get_decay_steps_from_scheduler(self, scheduler: Scheduler, max_steps: int) -> Tuple[int, str]:
        """
        Extract the number of decay steps from a scheduler.

        Returns:
            Tuple of (decay_steps, source_description) where source_description
            explains how the decay steps were determined.
        """
        # Check for explicit decay attribute
        if hasattr(scheduler, "decay") and scheduler.decay is not None:
            return scheduler.decay, "scheduler.decay"

        # Check for decay_fraction attribute
        if hasattr(scheduler, "decay_fraction") and scheduler.decay_fraction is not None:
            decay_steps = round(max_steps * scheduler.decay_fraction)
            return decay_steps, f"scheduler.decay_fraction ({scheduler.decay_fraction})"

        # No decay information found
        return 0, "no decay found"

    def _warn_if_steps_in_decay(self) -> None:
        """
        Check if any merge steps fall within the scheduler's decay phase and log a warning.
        """
        if self.trainer.max_steps is None:
            return

        max_steps = self.trainer.max_steps
        scheduler = getattr(self.trainer.train_module, "scheduler", None)
        if scheduler is None:
            return

        decay_steps, decay_source = self._get_decay_steps_from_scheduler(scheduler, max_steps)
        if decay_steps == 0:
            return

        decay_start = max_steps - decay_steps
        steps_in_decay = [s for s in self._merge_steps if s > decay_start]

        if steps_in_decay:
            log.warning(
                f"ModelMergeCallback: merge step(s) {steps_in_decay} fall within the decay phase "
                f"(decay starts at step {decay_start + 1}, from {decay_source}). "
                f"Model weights during decay may not be ideal for merging."
            )

    @property
    def _current_merge_step(self) -> Optional[int]:
        """The current merge step we're working towards, or None if all merges are done."""
        if self._current_merge_idx >= len(self._merge_steps):
            return None
        return self._merge_steps[self._current_merge_idx]

    def pre_train(self):
        """
        Resolve merge_step to max_steps minus decay steps if not explicitly set.

        For schedulers with decay phases (WSD, CosWithWarmupAndLinearDecay, etc.),
        merge_step will be set to the step right before decay begins.
        """
        if not self.enabled:
            return

        # Normalize merge_step to a sorted list
        if self.merge_step is None:
            if self.trainer.max_steps is None:
                log.warning(
                    "ModelMergeCallback: merge_step not set and trainer has no max_steps. "
                    "Disabling merging."
                )
                self.enabled = False
                return

            max_steps = self.trainer.max_steps
            decay_steps = 0
            decay_source = "no scheduler"

            # Try to get decay steps from the scheduler
            scheduler = getattr(self.trainer.train_module, "scheduler", None)
            if scheduler is not None:
                decay_steps, decay_source = self._get_decay_steps_from_scheduler(scheduler, max_steps)

            auto_merge_step = max_steps - decay_steps
            self._merge_steps = [auto_merge_step]

            if decay_steps > 0:
                log.info(
                    f"ModelMergeCallback: merge_step set to {auto_merge_step} "
                    f"(max_steps={max_steps} - decay_steps={decay_steps} from {decay_source})"
                )
            else:
                log.info(
                    f"ModelMergeCallback: merge_step set to {auto_merge_step} "
                    f"(max_steps, {decay_source})"
                )
        elif isinstance(self.merge_step, int):
            self._merge_steps = [self.merge_step]
            log.info(f"ModelMergeCallback: merge_step set to {self.merge_step}")
        else:
            self._merge_steps = sorted(self.merge_step)
            log.info(f"ModelMergeCallback: merge_steps set to {self._merge_steps}")

        # Warn if any merge steps fall within the decay phase
        self._warn_if_steps_in_decay()

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
        remaining_steps = self._merge_steps[self._current_merge_idx:]
        if remaining_steps:
            unreached = [s for s in remaining_steps if self.step < s]
            if unreached:
                log.warning(
                    f"Training ended at step {self.step} before reaching merge_step(s) {unreached}. "
                    "These merges will not be performed."
                )

    def _accumulate_weights(self):
        """
        Add current model weights to the accumulator.
        """
        model_state = self.trainer.train_module.model.state_dict()

        if self._accumulator is None:
            # Initialize accumulator with zeros
            log.info(f"Starting model weight averaging at step {self.step}")
            self._accumulator = {
                k: torch.zeros_like(v, dtype=torch.float32)
                for k, v in model_state.items()
            }

        # Add current weights to accumulator
        for key, value in model_state.items():
            self._accumulator[key].add_(value.float())

        # Capture individual weights for validation if enabled
        if self.validate:
            self._captured_weights.append({k: v.clone().float() for k, v in model_state.items()})

        self._n_accumulated += 1
        log.debug(f"Accumulated weights at step {self.step} ({self._n_accumulated} total)")

    def _save_merged_checkpoint(self):
        """
        Compute the average and save the merged checkpoint.
        """
        if self._accumulator is None or self._n_accumulated == 0:
            log.warning("No weights accumulated, cannot save merged checkpoint")
            return

        log.info(
            f"Saving merged checkpoint (average of {self._n_accumulated} steps) at step {self.step}"
        )

        # Compute the average
        averaged_state = {}
        for key, accumulated_value in self._accumulator.items():
            averaged_state[key] = accumulated_value / self._n_accumulated

        # Validate the average if enabled
        if self.validate:
            self._validate_average(averaged_state)

        # Only rank 0 saves
        if get_rank() == 0:
            output_path = str(join_path(
                self.trainer.save_folder,
                f"step{self.step}-{self.output_suffix}",
            ))

            # Create output directory
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path, exist_ok=True)

            # Save the averaged model state
            save_state_dict(
                join_path(output_path, "model_and_optim"),
                {"model": averaged_state, "optim": {}},
            )

            log.info(f"Merged checkpoint saved to: {output_path}")

        # Evaluate merged model (skips if no EvaluatorCallbacks found)
        self._evaluate_merged(averaged_state)

        # Clean up and advance to next merge step
        self._accumulator = None
        self._n_accumulated = 0
        self._current_merge_idx += 1
        self._captured_weights = []

    def _validate_average(self, averaged_state: Dict[str, torch.Tensor]):
        """
        Validate that the averaged weights are correctly computed from captured weights.

        Raises AssertionError if the weights don't match.
        """
        if not self._captured_weights:
            log.warning("No captured weights to validate against")
            return

        log.info(f"Validating merged weights against {len(self._captured_weights)} captured snapshots...")

        # Compute expected average from captured weights
        keys = list(self._captured_weights[0].keys())
        for key in keys:
            stacked = torch.stack([w[key] for w in self._captured_weights])
            expected = stacked.mean(dim=0)
            actual = averaged_state[key]

            if not torch.allclose(expected, actual, rtol=1e-4, atol=1e-6):
                raise AssertionError(
                    f"Validation failed for key '{key}': "
                    f"merged weights do not match expected average"
                )

        log.info("Validation passed: merged weights correctly average captured snapshots")

    def _evaluate_merged(self, averaged_state: Dict[str, torch.Tensor]):
        """
        Evaluate the merged model by temporarily swapping weights.

        This method:
        1. Stores the current model weights
        2. Loads the merged/averaged weights
        3. Runs all EvaluatorCallbacks with a "eval/merged" prefix
        4. Restores the original weights

        Metrics are recorded with an "eval/merged" prefix to distinguish them from
        regular evaluation metrics.
        """
        # Find EvaluatorCallback instances
        evaluator_callbacks = [
            cb for cb in self.trainer.callbacks.values()
            if isinstance(cb, EvaluatorCallback)
        ]

        if not evaluator_callbacks:
            log.info("No EvaluatorCallback instances found, skipping merged model evaluation")
            return

        model = self.trainer.train_module.model

        # Store original weights
        log.info("Storing original model weights for merged model evaluation...")
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Load merged weights (convert to model dtype)
        log.info("Loading merged weights for evaluation...")
        merged_state_converted = {}
        for key, value in averaged_state.items():
            if key in original_state:
                merged_state_converted[key] = value.to(original_state[key].dtype)
            else:
                merged_state_converted[key] = value
        model.load_state_dict(merged_state_converted)

        # Run evaluator callbacks with "eval/merged" prefix
        try:
            for callback in evaluator_callbacks:
                log.info(f"Running merged model evaluation via {callback.__class__.__name__}...")
                callback._perform_eval(prefix="eval/merged")
        finally:
            # Restore original weights
            log.info("Restoring original model weights...")
            model.load_state_dict(original_state)

    def state_dict(self) -> Dict[str, Any]:
        """
        Save callback state for checkpointing.
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
