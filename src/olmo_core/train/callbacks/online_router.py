import logging
from dataclasses import dataclass
from typing import Any, Dict

from olmo_core.distributed.utils import get_rank

from .callback import Callback

log = logging.getLogger(__name__)

__all__ = ["OnlineRouterCallback"]


@dataclass
class OnlineRouterCallback(Callback):
    """
    Drives :class:`~olmo_core.nn.online_token_router.OnlineTokenRouter`: before every optimizer
    step it refits the router on the targets the previous step's backward measured (one collective
    per step on every rank), and it logs the router's diagnostics (``online_router/*``: realised
    keep fraction, held-out rank correlation of the scores with the measured targets on the
    exploration tokens, gold-leak AUC, weight norms).
    """

    log_every: int = 1

    def _router(self):
        return getattr(self.trainer.train_module.model, "_online_token_router", None)  # type: ignore[union-attr]

    def pre_train(self):
        if self._router() is None:
            raise RuntimeError("OnlineRouterCallback attached but the model has no online token router")

    def pre_step(self, batch: Dict[str, Any]):
        router = self._router()
        router.begin_step(self.trainer.global_step, model=self.trainer.train_module.model)  # type: ignore[union-attr]

    def post_step(self):
        router = self._router()
        m = router.metrics()
        if self.log_every and self.step % self.log_every == 0:
            for k, v in m.items():
                self.trainer.record_metric(k, v)
        every = int(router.cfg.log_every)
        if get_rank() == 0 and every > 0 and (self.step <= 3 or self.step % every == 0):
            log.info(
                "[online-router] step %d: %s w=%s",
                self.step,
                " ".join(f"{k.split('/')[-1]}={v:.4g}" for k, v in sorted(m.items())),
                router.weights_digest(),
            )
