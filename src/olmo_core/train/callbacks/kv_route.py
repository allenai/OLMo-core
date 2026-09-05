import logging
from dataclasses import dataclass
from typing import Any, Optional

from .callback import Callback

log = logging.getLogger(__name__)

__all__ = ["KVRouteCallback"]


@dataclass
class KVRouteCallback(Callback):
    """
    Logs where the KV-cache router (:mod:`olmo_core.nn.attention.kv_route`) actually kept keys.

    Same rationale and structure as :class:`NestedFFNMoECallback`: every metric is the HARD
    (executed) decision of the step's last forward, recorded on every step, with the holder's
    schedule clock pinned to the global step so anneals survive a resume.

    - ``kv_route/mean_keep`` -- mean kept fraction over tokens and routed layers (= KV-cache size
      fraction, and the attention-score FLOP fraction on those layers).
    - ``kv_route/keep_L<i>`` -- kept fraction per routed layer.
    - ``kv_route/tier<k>_frac`` -- fraction of tokens kept in exactly ``k`` routed layers (the
      emergent "cache tiers").
    - ``kv_route/target`` / ``kv_route/explore`` -- the annealed schedules.
    """

    log_every: int = 10
    calls_per_step: int = 1

    def pre_step(self, batch):
        del batch
        holder = self._holder
        if holder is not None:
            holder.set_calls((self.step - 1) * self.calls_per_step)

    @property
    def _holder(self) -> Optional[Any]:
        cfg = getattr(self.trainer.train_module.model, "_kv_route", None)  # type: ignore[union-attr]
        return None if cfg is None else cfg["holder"]

    def post_attach(self):
        cfg = getattr(self.trainer.train_module.model, "_kv_route", None)  # type: ignore[union-attr]
        if cfg is None:
            log.warning(
                "KVRouteCallback attached, but the model has no _kv_route -- was enable_kv_route() "
                "called before the train module was built?"
            )
        else:
            log.info("KVRouteCallback tracking %d routed attention layers", len(cfg["routed"]))

    def post_step(self):
        holder = self._holder
        if holder is None:
            return
        # The step's last forward is complete; ``begin_forward`` of the NEXT forward snapshots it,
        # so read the live accumulators here.
        keep = holder.mean_keep(last_forward=False)
        self.trainer.record_metric("kv_route/mean_keep", keep)
        self.trainer.record_metric("kv_route/target", holder.current_target())
        self.trainer.record_metric("kv_route/explore", holder.current_explore())
        per_layer = {
            li: holder._hard_kept[li] / max(1, holder._n_tokens[li]) for li in holder._n_tokens
        }
        for li, kf in sorted(per_layer.items()):
            self.trainer.record_metric(f"kv_route/keep_L{li}", kf)
        if holder._tier_count is not None:
            import torch

            n_r = max(1, len(holder.routed_layers))
            hist = torch.bincount(holder._tier_count, minlength=n_r + 1).float()
            hist = (hist / hist.sum().clamp(min=1)).tolist()
            for t, frac in enumerate(hist):
                self.trainer.record_metric(f"kv_route/tier{t}_frac", frac)
        if self.log_every and self.step % self.log_every == 0:
            log.info(
                "[kv-route] step %d: keep %.3f (target %.3f) per-layer %s",
                self.step,
                keep,
                holder.current_target(),
                {li: round(v, 3) for li, v in sorted(per_layer.items())},
            )
