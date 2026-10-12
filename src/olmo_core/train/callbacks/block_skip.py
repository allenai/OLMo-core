import logging
from dataclasses import dataclass
from typing import Any, Optional

from .callback import Callback

log = logging.getLogger(__name__)

__all__ = ["BlockSkipCallback"]


@dataclass
class BlockSkipCallback(Callback):
    """
    Logs where the block-skip router (:mod:`olmo_core.nn.block_skip`) actually ran blocks: the
    HARD per-step run fraction (``block_skip/mean_keep``), per routed block
    (``block_skip/keep_L<i>``), and the emergent depth histogram (``block_skip/depth<k>_frac`` =
    fraction of tokens that ran exactly ``k`` routed blocks). Same structure and caveats as
    :class:`NestedFFNMoECallback`.
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
        cfg = getattr(self.trainer.train_module.model, "_block_skip", None)  # type: ignore[union-attr]
        return None if cfg is None else cfg["holder"]

    def post_attach(self):
        cfg = getattr(self.trainer.train_module.model, "_block_skip", None)  # type: ignore[union-attr]
        if cfg is None:
            log.warning("BlockSkipCallback attached, but the model has no _block_skip")
        else:
            log.info("BlockSkipCallback tracking %d routed blocks", len(cfg["routed"]))

    def post_step(self):
        holder = self._holder
        if holder is None:
            return
        keep = holder.mean_keep(last_forward=False)
        self.trainer.record_metric("block_skip/mean_keep", keep)
        self.trainer.record_metric("block_skip/target", holder.current_target())
        per_layer = holder.per_layer_keep(last_forward=False)
        for li, kf in sorted(per_layer.items()):
            self.trainer.record_metric(f"block_skip/keep_L{li}", kf)
        if holder._depth_count is not None:
            import torch

            n_r = max(1, len(holder.routed_layers))
            hist = torch.bincount(holder._depth_count, minlength=n_r + 1).float()
            hist = (hist / hist.sum().clamp(min=1)).tolist()
            for k, frac in enumerate(hist):
                self.trainer.record_metric(f"block_skip/depth{k}_frac", frac)
        jb = getattr(self.trainer.train_module.model, "_joint_budget", None)  # type: ignore[union-attr]
        if jb is not None and "last_cost" in jb:
            self.trainer.record_metric("joint_budget/cost", jb["last_cost"])
            self.trainer.record_metric("joint_budget/target", jb["last_target"])
        if self.log_every and self.step % self.log_every == 0:
            log.info(
                "[block-skip] step %d: run %.3f (target %.3f) per-block %s%s",
                self.step,
                keep,
                holder.current_target(),
                {li: round(v, 2) for li, v in sorted(per_layer.items())},
                (
                    f" joint cost {jb['last_cost']:.3f}/{jb['last_target']:.3f}"
                    if jb and "last_cost" in jb
                    else ""
                ),
            )
