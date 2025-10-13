import logging
from dataclasses import dataclass
from typing import Dict

from olmo_core.distributed.utils import get_rank 

from .wandb import WandBCallback

log = logging.getLogger(__name__)


@dataclass
class WandBTokensCallback(WandBCallback):
    """
    Extends WandBCallback to also log metrics against 'global_tokens':
      - 'global_tokens' (monotone counter)
      - 'loss_by_tokens' with 'global_tokens' as its step metric
      - 'lr_by_tokens' with 'global_tokens' as its step metric

    This keeps the default step-based plots unchanged while adding token-axis figures.
    """

    enable_token_axis: bool = True
    loss_key: str = "loss"
    lr_key_fallback: str = "lr"  # if trainer metrics include it

    def pre_train(self):
        super().pre_train()
        if self.enabled and self.enable_token_axis and get_rank() == 0:
            try:
                self.wandb.define_metric("global_tokens", summary="max")
                self.wandb.define_metric("loss_by_tokens", step_metric="global_tokens")
                self.wandb.define_metric("lr_by_tokens", step_metric="global_tokens")
            except Exception as e:
                log.warning(f"Failed to define token-axis metrics in W&B: {e}")

    def _read_current_lr(self, metrics: Dict[str, float]) -> float:
        # 1) metric payload
        if self.lr_key_fallback in metrics:
            return float(metrics[self.lr_key_fallback])

        # 2) try to peek at optimizer param groups (best-effort)
        lr_val = None
        for attr_path in [
            "optimizer",
            "optim.optimizer",
            "train_module.optim.optimizer",
        ]:
            try:
                obj = self.trainer
                for name in attr_path.split("."):
                    obj = getattr(obj, name)
                # standard pytorch "lr" field
                if hasattr(obj, "param_groups") and obj.param_groups:
                    g0 = obj.param_groups[0]
                    if "lr" in g0:
                        lr_val = g0["lr"]
                        break
            except Exception:
                pass

        if lr_val is None:
            return float("nan")
        try:
            return float(lr_val)
        except Exception:
            return float("nan")

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        if not (self.enabled and get_rank() == 0):
            return

        # tokens from trainer (already used by the stock scheduler)
        tokens = getattr(self.trainer, "global_train_tokens_seen", None)
        metrics_aug = dict(metrics)  # keep the original
        if self.enable_token_axis and tokens is not None:
            metrics_aug["global_tokens"] = int(tokens)
            # loss-by-tokens (if present)
            if self.loss_key in metrics:
                metrics_aug["loss_by_tokens"] = float(metrics[self.loss_key])
            # lr-by-tokens (best-effort)
            metrics_aug["lr_by_tokens"] = self._read_current_lr(metrics)

        # log with both normal step axis and the token-axis series above
        self.wandb.log(metrics_aug, step=int(step))
