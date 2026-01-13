import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from olmo_core.distributed.checkpoint import save_state_dict
from olmo_core.distributed.utils import get_rank
from olmo_core.io import join_path

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class ModelMergeCallback(Callback):
    """
    A callback that maintains a running average of model weights during training.

    Starting at ``merge_step - merge_last_n_steps``, this callback begins accumulating
    model weights. At ``merge_step``, it saves the averaged model as a new checkpoint.

    .. important::
        This callback is not added automatically. You must explicitly add it to your
        trainer to enable merging.

    Example usage::

        # Merge at the last step (default), averaging the last 100 steps
        callback = ModelMergeCallback()

        # Or specify a custom merge step (averages steps 9901-10000)
        callback = ModelMergeCallback(
            merge_step=10000,
            merge_last_n_steps=100,
        )
    """

    merge_step: Optional[int] = None
    """
    The step at which to save the merged checkpoint. Defaults to the last step of training.
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

    enabled: bool = True
    _accumulator: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)
    _n_accumulated: int = field(default=0, repr=False)
    _merge_completed: bool = field(default=False, repr=False)

    def pre_train(self):
        """
        Resolve merge_step to max_steps if not explicitly set.
        """
        if not self.enabled:
            return

        if self.merge_step is None:
            if self.trainer.max_steps is None:
                log.warning(
                    "ModelMergeCallback: merge_step not set and trainer has no max_steps. "
                    "Disabling merging."
                )
                self.enabled = False
                return
            self.merge_step = self.trainer.max_steps
            log.info(f"ModelMergeCallback: merge_step set to {self.merge_step} (last step)")

    @property
    def _start_step(self) -> int:
        """The step at which to start accumulating."""
        if self.merge_step is None:
            return -1
        # +1 so that merge_last_n_steps=100 gives exactly 100 steps (e.g., 901-1000 inclusive)
        return max(0, self.merge_step - self.merge_last_n_steps + 1)

    @property
    def _is_accumulating(self) -> bool:
        """Whether we are currently in the accumulation window."""
        if self.merge_step is None:
            return False
        return self._start_step <= self.step <= self.merge_step

    def post_train_batch(self):
        """
        After each training batch, potentially accumulate weights or save the merged model.
        """
        if not self.enabled or self.merge_step is None:
            return

        if self._merge_completed:
            return

        # Check if we should accumulate
        if self._is_accumulating:
            self._accumulate_weights()

        # Check if we should save
        if self.step == self.merge_step:
            self._save_merged_checkpoint()

    def post_train(self):
        """
        Warn if training ended before merge could complete.
        """
        if not self.enabled or self.merge_step is None:
            return

        if self._merge_completed:
            return

        if self.step < self.merge_step:
            log.warning(
                f"Training ended at step {self.step} before reaching merge_step {self.merge_step}. "
                "Merge will not be performed."
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

        # Clean up
        self._accumulator = None
        self._n_accumulated = 0
        self._merge_completed = True

    def state_dict(self) -> Dict[str, Any]:
        """
        Save callback state for checkpointing.
        """
        state = {
            "n_accumulated": self._n_accumulated,
            "merge_completed": self._merge_completed,
        }
        if self._accumulator is not None:
            state["accumulator"] = self._accumulator
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Restore callback state from checkpoint.
        """
        self._n_accumulated = state_dict.get("n_accumulated", 0)
        self._merge_completed = state_dict.get("merge_completed", False)
        self._accumulator = state_dict.get("accumulator", None)
