import logging
from dataclasses import dataclass

from .callback import Callback

log = logging.getLogger(__name__)

__all__ = ["LayerDropCallback"]


@dataclass
class LayerDropCallback(Callback):
    """
    Drives train-time whole-block skipping (:mod:`olmo_core.nn.layer_drop`): before each step it
    hands the model's plan the global step, so every rank draws the same skip set and keeps it for
    all of that step's micro-batches. Records ``layer_drop/n_skipped`` and ``layer_drop/run_frac``
    (blocks that ran / all blocks) every step.
    """

    log_every: int = 10
    """Steps between plain-console summaries (0 disables; metrics still recorded)."""

    def _plan(self):
        return getattr(self.trainer.train_module.model, "_layer_drop", None)  # type: ignore[union-attr]

    def pre_step(self, batch):
        plan = self._plan()
        if plan is not None:
            plan.step = int(self.step)

    def post_step(self):
        plan = self._plan()
        if plan is None:
            return
        skipped = plan.skipped()
        self.trainer.record_metric("layer_drop/n_skipped", float(len(skipped)))
        self.trainer.record_metric("layer_drop/run_frac", 1.0 - len(skipped) / plan.n_layers)
        if self.log_every > 0 and self.step % self.log_every == 0:
            log.info("[layer-drop] step %d: skipped blocks %s", self.step, sorted(skipped))
