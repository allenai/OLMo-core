import logging
from dataclasses import dataclass

from .callback import Callback

log = logging.getLogger(__name__)

__all__ = ["FFNTokenDropCallback"]


@dataclass
class FFNTokenDropCallback(Callback):
    """
    Logs the realized FFN drop fraction of a drop-CPT run (:mod:`olmo_core.nn.ffn_token_drop`):
    ``ffn_drop/frac`` = fraction of (token, droppable layer) FFN calls skipped on this rank since
    the previous step. Recorded every step (see :class:`NestedFFNMoECallback` for why).
    """

    log_every: int = 10
    """Steps between plain-console summaries (0 disables; metrics still recorded)."""

    def post_step(self):
        cfg = getattr(self.trainer.train_module.model, "_ffn_token_drop", None)  # type: ignore[union-attr]
        if cfg is None:
            return
        metrics = cfg["holder"].pop_metrics()
        for name, value in metrics.items():
            self.trainer.record_metric(name, value)
        if self.log_every > 0 and self.step % self.log_every == 0 and metrics:
            log.info("[ffn-drop] step %d: dropped frac=%.4f", self.step, metrics["ffn_drop/frac"])
