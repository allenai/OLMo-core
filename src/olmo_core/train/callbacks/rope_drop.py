import logging
from dataclasses import dataclass
from typing import Any, Dict

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.rope import RotaryEmbeddingBase

from ..train_module import TransformerTrainModule
from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class RoPEDropCallback(Callback):
    """
    A :class:`Callback` that disables RoPE (Rotary Position Embeddings) at a configurable
    fraction of training.

    This is useful for "NoPE" (No Positional Embeddings) experiments where RoPE is turned off
    partway through training to study the model's ability to generalize without positional
    information.

    :param drop_fraction: The fraction of ``max_steps`` at which to disable RoPE, in the
        range ``(0, 1]``.
    :param enabled: Whether this callback is enabled.
    """

    drop_fraction: float = 0.5
    """
    The fraction of ``max_steps`` at which to disable RoPE. Must be in the range ``(0, 1]``.
    For example, ``0.5`` means RoPE is disabled halfway through training.
    """

    enabled: bool = True
    """Whether this callback is enabled."""

    _rope_dropped: bool = False

    def pre_train(self):
        if not self.enabled:
            return

        if not (0.0 < self.drop_fraction <= 1.0):
            raise OLMoConfigurationError(
                f"'drop_fraction' must be in the range (0, 1], got {self.drop_fraction}"
            )

        if not isinstance(self.trainer.train_module, TransformerTrainModule):
            raise OLMoConfigurationError(
                "The RoPE drop callback requires a 'TransformerTrainModule', "
                f"got '{type(self.trainer.train_module)}' instead"
            )

        if self.trainer.max_steps is None:
            raise OLMoConfigurationError(
                "The RoPE drop callback requires 'max_steps' to be set on the trainer"
            )

        # Validate that the model has at least one block with RoPE.
        model = self.trainer.train_module.model
        has_rope = any(
            getattr(block.attention, "rope", None) is not None for block in model.blocks.values()
        )
        if not has_rope:
            raise OLMoConfigurationError(
                "The RoPE drop callback requires a model with RoPE, "
                "but no blocks have a RoPE module"
            )

        # If resuming from a checkpoint where RoPE was already dropped, re-apply the drop.
        if self._rope_dropped:
            self._disable_rope()

    def pre_step(self, batch: Dict[str, Any]):
        del batch

        if not self.enabled or self._rope_dropped:
            return

        assert self.trainer.max_steps is not None
        drop_step = int(self.trainer.max_steps * self.drop_fraction)

        if self.step >= drop_step:
            self._disable_rope()
            self._rope_dropped = True

    def _disable_rope(self):
        assert isinstance(self.trainer.train_module, TransformerTrainModule)
        model = self.trainer.train_module.model
        count = 0
        for block in model.blocks.values():
            rope: RotaryEmbeddingBase = getattr(block.attention, "rope", None)  # type: ignore
            if rope is not None:
                rope.disabled = True
                count += 1
        log.info(f"Disabled RoPE on {count} block(s) at step {self.step}")

    def state_dict(self) -> Dict[str, Any]:
        return {"rope_dropped": self._rope_dropped}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._rope_dropped = state_dict.get("rope_dropped", False)
