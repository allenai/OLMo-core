import logging
from typing import List, Optional

import torch

from olmo_core.utils import get_default_device, move_to_device

from .skip_step_optimizer import SkipStepOptimizer

log = logging.getLogger(__name__)


def _import_muon():
    try:
        from dion import Muon  # type: ignore
    except ImportError as e:
        raise ImportError(
            "The 'dion' package is required for SkipStepMuon. "
            "Install it with: pip install git+https://github.com/microsoft/dion.git"
        ) from e
    return Muon


class SkipStepMuon(_import_muon()):
    """
    A "skip step" version of :class:`Muon` that skips the entire optimizer step
    when a loss spike is detected.

    Unlike :class:`SkipStepAdamW` and :class:`SkipStepLion` which thread a
    ``step_factor`` through the update computation, this class skips the entire
    ``step()`` call. This avoids all distributed communication and Newton-Schulz
    compute on skip steps.

    This is safe because:

    - All ranks compute the same ``step_factor`` (loss is pre-synchronized).
    - Muon's ``step()`` is not ``torch.compile``'d, so branching is safe.
    - On skip: momentum, weights, and step counters are all untouched.

    .. important::
        ``latest_loss`` must be set to the **all-reduced** loss before calling
        :meth:`step()` so that all ranks make the same skip/proceed decision.
    """

    def __init__(
        self,
        params,
        *,
        rolling_interval_length: int = 128,
        sigma_factor: int = 6,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self.rolling_interval_length = rolling_interval_length
        self.sigma_factor = sigma_factor
        self._losses: List[torch.Tensor] = []
        self._grad_norms: List[torch.Tensor] = []
        self._device: Optional[torch.device] = None
        self._step_skipped: Optional[torch.Tensor] = None

    @property
    def device(self) -> torch.device:
        if self._device is None:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.numel() > 0:
                        self._device = p.device
                        break
            if self._device is None:
                self._device = get_default_device()
        return self._device

    @property
    def latest_loss(self) -> Optional[torch.Tensor]:
        if not self._losses:
            return None
        else:
            return self._losses[-1]

    @latest_loss.setter
    def latest_loss(self, loss: torch.Tensor):
        self._losses.append(loss)
        while len(self._losses) > self.rolling_interval_length + 1:
            self._losses.pop(0)

    @property
    def latest_grad_norm(self) -> Optional[torch.Tensor]:
        if not self._grad_norms:
            return None
        else:
            return self._grad_norms[-1]

    @latest_grad_norm.setter
    def latest_grad_norm(self, grad_norm: torch.Tensor):
        self._grad_norms.append(grad_norm)
        while len(self._grad_norms) > self.rolling_interval_length + 1:
            self._grad_norms.pop(0)

    @torch._dynamo.disable()
    def get_step_factor(self) -> torch.Tensor:
        """
        Returns a float tensor: ``1.0`` to proceed, ``0.0`` to skip.
        """
        if len(self._losses) < max(2, self.rolling_interval_length // 2):
            return move_to_device(torch.tensor(1.0), self.device)

        loss_std, loss_mean = torch.std_mean(torch.stack(self._losses[:-1]))
        assert self.latest_loss is not None
        if self._grad_norms:
            assert self.latest_grad_norm is not None
            grad_norm_std, grad_norm_mean = torch.std_mean(torch.stack(self._grad_norms[:-1]))
            step_factor = torch.logical_and(
                (self.latest_loss - loss_mean) <= self.sigma_factor * loss_std,
                (self.latest_grad_norm - grad_norm_mean) <= self.sigma_factor * grad_norm_std,
            )
        else:
            step_factor = (self.latest_loss - loss_mean) <= self.sigma_factor * loss_std

        return step_factor.float()

    @property
    def step_skipped(self) -> torch.Tensor:
        if self._step_skipped is not None:
            return self._step_skipped
        else:
            return torch.tensor(0.0)

    @torch.no_grad()
    def step(self, closure=None):
        step_factor = self.get_step_factor()
        self._step_skipped = 1 - step_factor
        # NOTE: `.item()` triggers a host-device sync. Unlike SkipStepAdamW, which avoids
        # this by threading a device-side step_factor through every arithmetic op, we can't
        # do that here because `Muon.step()` performs Newton-Schulz iterations and distributed
        # all-gathers that cannot be cheaply neutralized by multiplying by zero. The sync is
        # the lesser cost — skipping the entire step saves all of that compute and communication.
        if step_factor.item() == 0.0:
            return None
        return super().step(closure)


# Register as virtual subclass so isinstance(obj, SkipStepOptimizer) returns True.
# This is needed for TransformerTrainModule to automatically set latest_loss/latest_grad_norm.
SkipStepOptimizer.register(SkipStepMuon)
