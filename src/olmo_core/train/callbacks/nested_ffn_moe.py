import logging
from dataclasses import dataclass
from typing import Any, Optional

from .callback import Callback

log = logging.getLogger(__name__)

__all__ = ["NestedFFNMoECallback"]


@dataclass
class NestedFFNMoECallback(Callback):
    """
    Logs where the nested-FFN mixture actually spent its FFN FLOPs
    (:mod:`olmo_core.nn.nested_ffn_moe`).

    Without this you cannot tell a genuine compute saving from a router that quietly collapsed --
    the CE curve looks the same either way, which is exactly how the role-gated FFN arms wasted
    runs. Every metric here is the HARD (executed) routing, not the router's probabilities:

    - ``ffn_moe/mean_cost`` -- mean per-token FFN cost on routed layers, as a fraction of dense.
    - ``ffn_moe/speedup`` -- ``1 / mean_cost``, the FFN-only speedup on routed layers.
    - ``ffn_moe/frac_rungK`` -- fraction of tokens that took rung ``K`` (0 = full, last = null).
    - ``ffn_moe/target`` / ``ffn_moe/explore`` -- the current annealed schedules.

    ``mean_cost`` is measured on the LAST microbatch of the step, which is representative because
    routing is per token over tens of thousands of tokens.

    .. note::
        Metrics are recorded on EVERY step, deliberately. The trainer only collects metrics on its
        own ``metrics_collect_interval``, so a callback that gates its own recording on a
        different interval silently records into steps that are never collected -- which is how
        the first version of this callback logged nothing at all for 111 steps of a live run.
    """

    log_every: int = 10
    """Steps between plain-console routing summaries (0 disables; metrics still recorded)."""

    calls_per_step: int = 1
    """
    Routed forwards per optimizer step on this rank (the gradient-accumulation factor). Used to
    pin the holder's schedule clock to ``(global_step - 1) * calls_per_step`` in ``pre_step``, so
    the target/exploration anneals are a function of the global step and survive a crash-resume.
    Before this the clock lived only in memory and every resume restarted the anneals: the first
    routed 4B arms ended 3000 steps with ``target=0.84`` instead of ``0.05``.
    """

    _warned: bool = False

    def pre_step(self, batch):
        del batch
        holder = self._holder
        if holder is not None:
            holder.set_calls((self.step - 1) * self.calls_per_step)

    @property
    def _holder(self) -> Optional[Any]:
        """Resolve the holder fresh every step rather than caching it in ``post_attach``.

        Caching looked right and logged "tracking 5 rungs" on attach, but recorded nothing for
        hundreds of steps of two live 4B runs: the instance that ``post_attach`` runs on is not
        the one that receives ``post_step``, so the cached attribute was always ``None`` at
        record time. Every callback in this package that works (e.g.
        :class:`GPUMemoryMonitorCallback`) recomputes its state in ``post_step`` instead.
        """
        cfg = getattr(self.trainer.train_module.model, "_nested_ffn_moe", None)  # type: ignore[union-attr]
        return None if cfg is None else cfg["holder"]

    def post_attach(self):
        cfg = getattr(self.trainer.train_module.model, "_nested_ffn_moe", None)  # type: ignore[union-attr]
        if cfg is None:
            # Be loud: a silently-inert monitor on a compute-saving run is worse than no monitor.
            log.warning(
                "NestedFFNMoECallback attached, but the train module's model has no "
                "_nested_ffn_moe -- routing metrics will NOT be logged. Was "
                "enable_nested_ffn_moe() called before the train module was built?"
            )
        else:
            log.info(
                "NestedFFNMoECallback: tracking %d rungs, costs=%s",
                len(cfg["costs"]),
                [round(c, 5) for c in cfg["costs"]],
            )

    def post_step(self):
        holder = self._holder
        if holder is None:
            if not self._warned:
                log.warning("NestedFFNMoECallback: no holder at post_step; metrics unavailable")
                self._warned = True
            return
        metrics = holder.metrics()
        for name, value in metrics.items():
            self.trainer.record_metric(name, value)
        # Also write a plain console line. record_metric feeds wandb and the metric block, but the
        # job log is the artifact that always exists (wandb is optional, and these runs are often
        # read after the fact from /data), and routing is the one thing you cannot reconstruct
        # from the CE curve. Cheap: a handful of floats every `log_every` steps.
        if self.log_every > 0 and self.step % self.log_every == 0:
            cost = metrics.get("ffn_moe/mean_cost")
            if cost is None:
                log.info("[ffn-moe] step %d: no routed forward recorded yet", self.step)
            else:
                fracs = " ".join(
                    f"{metrics[k]:.3f}"
                    for k in sorted(metrics)
                    if k.startswith("ffn_moe/frac_rung")
                )
                log.info(
                    "[ffn-moe] step %d: mean_cost=%.4f (%.1fx) target=%.3f explore=%.3f "
                    "min_layer=%d rungs=[%s]",
                    self.step,
                    cost,
                    metrics.get("ffn_moe/speedup", float("nan")),
                    metrics.get("ffn_moe/target", float("nan")),
                    metrics.get("ffn_moe/explore", float("nan")),
                    int(metrics.get("ffn_moe/min_layer", -1)),
                    fracs,
                )
            if self.step % (self.log_every * 10) == 0:
                per_layer = holder.per_layer_cost(last_forward=True)
                if per_layer:
                    log.info(
                        "[ffn-moe] step %d per-layer cost: %s",
                        self.step,
                        " ".join(f"L{k}:{v:.2f}" for k, v in per_layer.items()),
                    )
