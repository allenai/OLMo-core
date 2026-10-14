"""
Callbacks for deliberate optimizer hyperparameter changes when branching a run from a checkpoint,
e.g. moving a repeated-data trunk to a neighboring learning rate / weight decay coordinate without
disturbing the loaded Adam moments.
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from torch.optim import Optimizer

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.optim.config import INITIAL_LR_FIELD

from .callback import Callback

log = logging.getLogger(__name__)

__all__ = ["transition_optimizer_hyperparameters", "OptimizerHyperparameterTransitionCallback"]


def transition_optimizer_hyperparameters(
    optim: Optimizer,
    *,
    source_lr: float,
    target_lr: float,
    source_weight_decay: float,
    target_weight_decay: float,
) -> int:
    """
    Validate loaded source hyperparameters and install a new LR/WD coordinate.

    Adam moments and step counters are intentionally untouched. Parameter groups with zero
    weight decay remain unregularized; every group using ``source_weight_decay`` transitions
    to ``target_weight_decay``.

    Optimizers that take their hyperparameters from the current config rather than the
    checkpoint (e.g. :class:`~olmo_core.optim.moe_optimizer.OLMoDDPOptimizer`) already hold the
    target values after loading. Those groups are accepted as-is, since the only thing the
    transition must preserve is the loaded optimizer state.

    :param optim: The optimizer whose parameter groups are updated in place.
    :param source_lr: The LR the source checkpoint was trained with.
    :param target_lr: The LR to install (also recorded as the group's initial LR so the
        scheduler continues from the new value).
    :param source_weight_decay: The weight decay regularized groups were trained with.
    :param target_weight_decay: The weight decay to install on those groups.

    :returns: The number of regularized parameter groups now at the target weight decay.

    :raises RuntimeError: If any group's loaded LR or weight decay matches neither the source
        nor the target values, or if no regularized group was found.
    """

    def close(a: float, b: float) -> bool:
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12)

    regularized_groups = 0
    for group_idx, group in enumerate(optim.param_groups):
        loaded_lr = float(group["lr"])
        if not (close(loaded_lr, source_lr) or close(loaded_lr, target_lr)):
            raise RuntimeError(
                f"optimizer group {group_idx} loaded LR {loaded_lr} matches neither the source "
                f"LR {source_lr} nor the target LR {target_lr}"
            )
        loaded_weight_decay = float(group.get("weight_decay", 0.0))
        if close(loaded_weight_decay, source_weight_decay) or close(
            loaded_weight_decay, target_weight_decay
        ):
            group["weight_decay"] = target_weight_decay
            regularized_groups += 1
        elif not close(loaded_weight_decay, 0.0):
            raise RuntimeError(
                f"optimizer group {group_idx} loaded weight decay {loaded_weight_decay} is "
                f"neither zero, the source weight decay {source_weight_decay}, nor the target "
                f"weight decay {target_weight_decay}"
            )
        group["lr"] = target_lr
        group[INITIAL_LR_FIELD] = target_lr
    if regularized_groups == 0:
        raise RuntimeError(
            "optimizer hyperparameter transition found no regularized parameter group"
        )
    return regularized_groups


@dataclass
class OptimizerHyperparameterTransitionCallback(Callback):
    """
    Apply an intentional once-only LR/WD transition after explicit checkpoint loading.

    The transition runs in :meth:`pre_train`, i.e. after the trainer has loaded its initial
    checkpoint, and is recorded in the callback state so a later resume does not re-apply it.
    Works with any train module exposing a single ``optim`` attribute (e.g. the transformer and
    OLMoDDP train modules).
    """

    source_lr: float
    target_lr: float
    source_weight_decay: float
    target_weight_decay: float
    expected_step: Optional[int] = None
    """
    If set, the global step the loaded checkpoint must be at. Leave ``None`` to accept whatever
    step the explicit ``load_path`` checkpoint resolved to.
    """
    _applied: bool = False

    def state_dict(self) -> Dict[str, Any]:
        return {"applied": self._applied}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._applied = bool(state_dict.get("applied", False))

    def pre_train(self):
        if self._applied:
            log.info("Optimizer hyperparameter transition was already applied; skipping")
            return
        if self.trainer.global_step == 0:
            raise RuntimeError(
                "Optimizer hyperparameter transition requires a loaded checkpoint, but the "
                "trainer is at step 0"
            )
        if self.expected_step is not None and self.trainer.global_step != self.expected_step:
            raise RuntimeError(
                "Optimizer hyperparameter transition expected checkpoint step "
                f"{self.expected_step}, got {self.trainer.global_step}"
            )
        optim = getattr(self.trainer.train_module, "optim", None)
        if optim is None or not hasattr(optim, "param_groups"):
            raise OLMoConfigurationError(
                "optimizer hyperparameter transition requires a train module with a single "
                f"'optim' attribute, got '{self.trainer.train_module.__class__.__name__}'"
            )
        adjusted = transition_optimizer_hyperparameters(
            optim,
            source_lr=self.source_lr,
            target_lr=self.target_lr,
            source_weight_decay=self.source_weight_decay,
            target_weight_decay=self.target_weight_decay,
        )
        self._applied = True
        log.info(
            "Transitioned optimizer hyperparameters at step %d: LR %.8g -> %.8g; WD %.8g "
            "-> %.8g across %d regularized groups; Adam moments and step counters unchanged",
            self.trainer.global_step,
            self.source_lr,
            self.target_lr,
            self.source_weight_decay,
            self.target_weight_decay,
            adjusted,
        )
