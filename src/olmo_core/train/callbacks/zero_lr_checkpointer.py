import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

from .checkpointer import CheckpointerCallback

log = logging.getLogger(__name__)


@dataclass
class ZeroLRCheckpointerCallback(CheckpointerCallback):
    """
    Save checkpoints only at specific steps -- typically the period boundaries of WSD‑S.
    Intended for "checkpoint only when LR = 0".

    Pass 'save_steps' as a sorted list of *steps* (integers) at which to save.
    All other base behavior (async save, removal) is preserved.

    Example:
        save_steps = [S1, S2, S3, ...]  # cumulative period lengths in steps
    """

    # Disable the interval behavior in the base class by setting a huge interval
    # (we fully override 'post_train_batch' so this is purely defensive).
    save_interval: int = 1_000_000_000

    # user-provided exact steps to save at
    save_steps: Optional[List[int]] = None

    _save_steps_set: Set[int] = field(
        default_factory=set, init=False, repr=False, metadata={"omegaconf_ignore": True}
    )
    _last_saved_step: int = field(default=-1, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        if not self.save_steps:
            raise ValueError("'save_steps' must be provided (list of step indices to checkpoint).")
        # store as a set
        self._save_steps_set = {int(s) for s in self.save_steps}

    def post_train_batch(self):
        if not self.enabled:
            return

        self._await_last_checkpoint(blocking=False)
        if not self.checkpoint_pending:
            self._remove_old_checkpoints()

        step = int(self.step)
        if step in self._save_steps_set and step != self._last_saved_step:
            # save checkpoint at this exact step!
            path = self._save_checkpoint()
            self._last_saved_step = step
            log.info(f"Saved WSD‑S boundary checkpoint at step={step} -> {path}")
