import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed
import torch.distributed.checkpoint.state_dict as dist_cp_sd

from olmo_core.distributed.checkpoint import save_state_dict
from olmo_core.distributed.utils import barrier, get_rank, is_distributed
from olmo_core.io import join_path
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

    merge_step: Optional[Union[int, List[int]]] = None
    """
    The step(s) at which to save merged checkpoint(s). Can be a single int or a list of ints.
    If not set:
    - For WSDS schedulers: defaults to the steps right before each decay phase begins.
    - For other schedulers: defaults to every ``merge_interval`` steps.
    """

    merge_interval: int = 10000
    """Interval at which to save merged checkpoints (e.g., steps 10000, 20000, ...)."""

    merge_last_n_steps: int = 100
    """Number of steps before ``merge_step`` to start accumulating the average."""

    output_suffix: str = "merged"
    """
    Suffix for the output checkpoint directory name.
    The merged checkpoint will be saved as "step{merge_step}-{output_suffix}".
    """

    enabled: bool = True
    _accumulator: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)
    _n_accumulated: int = field(default=0, repr=False)
    _merge_steps: List[int] = field(default_factory=list, repr=False)
    _current_merge_idx: int = field(default=0, repr=False)

    def _get_decay_steps_from_scheduler(
        self, scheduler: Scheduler, max_steps: int
    ) -> Tuple[int, str]:
        # Check for explicit decay attribute
        if hasattr(scheduler, "decay") and scheduler.decay is not None:
            return scheduler.decay, "scheduler.decay"

        # Check for decay_fraction attribute
        if hasattr(scheduler, "decay_fraction") and scheduler.decay_fraction is not None:
            decay_steps = round(max_steps * scheduler.decay_fraction)
            return decay_steps, f"scheduler.decay_fraction ({scheduler.decay_fraction})"

        # No decay information found
        return 0, "no decay found"

    def _get_merge_steps_from_wsds(self, scheduler: WSDS, tokens_per_step: int) -> List[int]:
        """
        Extract merge steps from a WSDS scheduler (one per period, before each decay).

        :param scheduler: The WSDS scheduler.
        :param tokens_per_step: Global batch size (in tokens)
        :returns: List of step numbers where merges should occur.
        """
        merge_steps = []

        for i, period_length in enumerate(scheduler.period_lengths):
            # Compute cumulative tokens at end of this period
            if i == 0:
                period_end_tokens = period_length
            else:
                period_end_tokens = sum(scheduler.period_lengths[: i + 1])

            # Compute decay length for this period using public attributes
            if scheduler.decay is not None:
                decay_tokens = scheduler.decay
            else:
                assert scheduler.decay_fraction is not None
                decay_tokens = int(round(scheduler.decay_fraction * period_length))

            # Merge should happen right before decay starts
            pre_decay_tokens = period_end_tokens - decay_tokens

            # Convert tokens to steps
            merge_step = pre_decay_tokens // tokens_per_step
            merge_steps.append(merge_step)

            log.debug(
                f"WSDS period {i}: end={period_end_tokens} tokens, "
                f"decay={decay_tokens} tokens, merge_step={merge_step}"
            )

        return merge_steps

    @property
    def _current_merge_step(self) -> Optional[int]:
        """The current merge step we're working towards, or None if all merges are done."""
        if self._current_merge_idx >= len(self._merge_steps):
            return None
        return self._merge_steps[self._current_merge_idx]

    def pre_train(self):
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
            scheduler = getattr(self.trainer.train_module, "scheduler", None)

            # Check for WSDS scheduler (multiple anneals)
            if isinstance(scheduler, WSDS):
                # NOTE: This assumes global_batch_size is in tokens (true for TextDataLoaderBase)
                tokens_per_step = self.trainer.data_loader.global_batch_size
                self._merge_steps = self._get_merge_steps_from_wsds(scheduler, tokens_per_step)
                log.info(
                    f"ModelMergeCallback: detected WSDS scheduler with {len(scheduler.period_lengths)} periods. "
                    f"merge_steps set to {self._merge_steps} (one before each decay phase, "
                    f"assuming {tokens_per_step} tokens/step)"
                )
            else:
                # For non-WSDS schedulers, use fixed intervals
                self._merge_steps = list(
                    range(self.merge_interval, max_steps + 1, self.merge_interval)
                )
                # Ensure we have at least one merge step (at max_steps if interval > max_steps)
                if not self._merge_steps:
                    self._merge_steps = [max_steps]
                log.info(
                    f"ModelMergeCallback: merge_steps set to {self._merge_steps} "
                    f"(every {self.merge_interval} steps)"
                )
        elif isinstance(self.merge_step, int):
            self._merge_steps = [self.merge_step]
            log.info(f"ModelMergeCallback: merge_step set to {self.merge_step}")
        else:
            self._merge_steps = sorted(self.merge_step)
            log.info(f"ModelMergeCallback: merge_steps set to {self._merge_steps}")

        # Warn if any merge steps fall within the decay phase (only for non-WSDS schedulers)
        if self.trainer.max_steps is not None:
            scheduler = getattr(self.trainer.train_module, "scheduler", None)
            if scheduler is not None and not isinstance(scheduler, WSDS):
                decay_steps, decay_source = self._get_decay_steps_from_scheduler(
                    scheduler, self.trainer.max_steps
                )
                if decay_steps > 0:
                    decay_start = self.trainer.max_steps - decay_steps
                    steps_in_decay = [s for s in self._merge_steps if s > decay_start]
                    if steps_in_decay:
                        log.warning(
                            f"ModelMergeCallback: merge step(s) {steps_in_decay} fall within the "
                            f"decay phase (decay starts at step {decay_start + 1}, from {decay_source}). "
                            f"Model weights during decay may not be ideal for merging."
                        )

        # Handle resume scenarios where we may have accumulated weights ready to save
        # or need to skip past completed merge steps
        current_step = self.step
        while self._current_merge_idx < len(self._merge_steps):
            target_merge_step = self._merge_steps[self._current_merge_idx]

            if target_merge_step < current_step:
                # We're past this merge step - skip it
                log.info(
                    f"ModelMergeCallback: skipping past merge step {target_merge_step} "
                    f"(current step is {current_step})"
                )
                self._accumulator = None
                self._n_accumulated = 0
                self._current_merge_idx += 1
            elif target_merge_step == current_step and self._n_accumulated > 0:
                # We're exactly at the merge step with accumulated weights ready
                # This happens when resuming from a checkpoint saved before the merge completed
                log.info(
                    f"ModelMergeCallback: resuming at merge step {target_merge_step} "
                    f"with {self._n_accumulated} accumulated weights, saving now..."
                )
                self._save_merged_checkpoint()
                # _save_merged_checkpoint advances _current_merge_idx, so continue the loop
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
        Adds current model weights to the accumulator (stored on CPU, rank 0 only).

        TODO(large-scale): For larger models, calling state_dict() and moving to CPU at every
        step in the merge window may cause significant throughput overhead.
        Might want to consider sparse sampling (accumulate every Nth step instead of every step)
        or doing this one time (moving to CPU only at the end)
        """

        # All ranks must participate in get_model_state_dict for FSDP to gather weights,
        # but only rank 0 receives the actual state dict when full_state_dict=True.
        sd_options = dist_cp_sd.StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_state = dist_cp_sd.get_model_state_dict(
            self.trainer.train_module.model, options=sd_options
        )

        # Only rank 0 accumulates (other ranks get empty state dict with full_state_dict=True)
        if get_rank() == 0:
            if self._accumulator is None:
                # Initialize accumulator with zeros on CPU
                log.info(f"Starting model weight averaging at step {self.step}")
                self._accumulator = {
                    k: torch.zeros_like(v, dtype=torch.float32, device="cpu")
                    for k, v in model_state.items()
                }

            # Add current weights to accumulator (already on CPU from cpu_offload=True)
            for key, value in model_state.items():
                self._accumulator[key].add_(value.float())

        self._n_accumulated += 1
        log.debug(f"Accumulated weights at step {self.step} ({self._n_accumulated} total)")

    def _save_merged_checkpoint(self):
        if get_rank() == 0 and (self._accumulator is None or self._n_accumulated == 0):
            log.warning("No weights accumulated, cannot save merged checkpoint")
            return

        log.info(
            f"Saving merged checkpoint (average of {self._n_accumulated} steps) at step {self.step}"
        )

        # Compute the average on rank 0 (only rank 0 has the accumulator)
        averaged_state: Dict[str, torch.Tensor] = {}
        if get_rank() == 0:
            for key, accumulated_value in self._accumulator.items():  # type: ignore
                averaged_state[key] = accumulated_value / self._n_accumulated

        # Broadcast averaged_state from rank 0 to all ranks for distributed save and eval
        if is_distributed():
            object_list = [averaged_state]
            torch.distributed.broadcast_object_list(object_list, src=0)
            averaged_state = object_list[0]

        output_path = str(
            join_path(
                self.trainer.save_folder,
                f"step{self.step}-{self.output_suffix}",
            )
        )

        # Only rank 0 creates/clears the directory
        if get_rank() == 0:
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.makedirs(output_path, exist_ok=True)

        # Wait for directory to be ready before all ranks save
        barrier()

        # All ranks participate in distributed checkpoint saving
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
        # Find EvaluatorCallback instances
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

        # Load merged weights using set_model_state_dict for proper FSDP support
        # Use full_state_dict=True since averaged_state is a complete (non-sharded) state dict
        log.info("Loading merged weights for evaluation...")
        dist_cp_sd.set_model_state_dict(
            model, averaged_state, options=dist_cp_sd.StateDictOptions(full_state_dict=True, strict=True)
        )

        # Run evaluator callbacks with "eval/merged" prefix
        try:
            for callback in evaluator_callbacks:
                log.info(f"Running merged model evaluation via {callback.__class__.__name__}...")
                callback._perform_eval(prefix="eval/merged")
        finally:
            # Restore original weights using set_model_state_dict for proper FSDP support
            # Use full_state_dict=False since original_state is sharded (matches how we got it)
            log.info("Restoring original model weights...")
            dist_cp_sd.set_model_state_dict(
                model, original_state, options=dist_cp_sd.StateDictOptions(full_state_dict=False, strict=True)
            )

    def state_dict(self) -> Dict[str, Any]:
        """
        Save callback state for checkpointing.

        TODO(large-scale): Saving the accumulator during the merge window increases checkpoint
        size by ~1x model size. For larger models, need to think about:
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
