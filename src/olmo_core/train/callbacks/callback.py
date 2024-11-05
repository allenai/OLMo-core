from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional

import torch

from olmo_core.aliases import PathOrStr
from olmo_core.config import Config

if TYPE_CHECKING:
    from ..trainer import Trainer


@dataclass
class Callback:
    """
    Trainer callback base class.

    Callbacks can be used to modify and extend the behavior of the trainer loop.
    This module contains a number of useful :class:`Callback` implementations, but you can
    always add your own.
    """

    priority: ClassVar[int] = 0
    """
    Priority of the callback. Determines the order in which callbacks run relative to each other.
    The higher the priority, the earlier a callback runs.
    """

    # NOTE: omegaconf can't use this annotation
    #  _trainer: Optional["Trainer"] = None
    _trainer = None

    @property
    def trainer(self) -> "Trainer":
        assert self._trainer is not None
        return self._trainer

    @trainer.setter
    def trainer(self, trainer: "Trainer"):
        self._trainer = trainer

    @property
    def step(self) -> int:
        return self.trainer.global_step

    #  def state_dict(self) -> Dict[str, Any]:
    #      return {}

    #  def load_state_dict(self, state_dict: Dict[str, Any]):
    #      del state_dict

    def post_attach(self):
        """
        Called right after the callback is attached to the :class:`~olmo_core.train.Trainer`.
        """
        pass

    def post_checkpoint_loaded(self, path: PathOrStr):
        """
        Called when a checkpoint is successfully loaded.

        :param path: The path/URL to the checkpoint.
        """
        del path

    def pre_train(self):
        """
        Runs before the training loop starts.
        """
        pass

    def pre_epoch(self):
        """
        Runs before the start of a new epoch.
        """
        pass

    def pre_load_batch(self):
        """
        Runs right before the next batch is fetched from the data loader.
        """
        pass

    def pre_step(self, batch: Dict[str, Any]):
        """
        Runs right before a training batch is processed.
        """
        del batch

    def pre_backward(
        self,
        *,
        batch: Dict[str, Any],
        micro_batch: Dict[str, Any],
        loss: torch.Tensor,
    ):
        """
        Runs right before the backward pass on a micro-batch. This can be used to modify the
        ``loss`` before ``loss.backward()`` is called.

        :param batch: The full batch.
        :param micro_batch: The micro-batch just used.
        :param loss: The combined loss from the micro-batch (``ce_loss`` plus the optional ``z_loss``).
        """
        del batch, micro_batch, loss

    def pre_optim_step(self):
        """
        Runs right after the forward-backward passes, right before the optimizer step.
        """
        pass

    def post_train_batch(self):
        """
        Runs after a training batch is processed.
        """
        pass

    def pre_eval_batch(self, batch: Dict[str, Any]):
        """
        Runs right before an eval batch is processed with :meth:`~olmo_core.train.Trainer.eval_batch()`.

        :param batch: The eval batch.
        """
        del batch

    def post_eval_batch(self):
        """
        Runs after after an eval batch is processed with :meth:`~olmo_core.train.Trainer.eval_batch()`.
        """
        pass

    def post_step(self):
        """
        Runs after a complete step (potentially including evals and checkpointing).
        """
        pass

    def post_checkpoint_saved(self, path: PathOrStr):
        """
        Called when a checkpoint is successfully saved.

        :param path: The path/URL to the checkpoint.
        """
        del path

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        """
        Called when metrics have been gathered for a given step (possibly a previous step).
        """
        del step, metrics

    def post_epoch(self):
        """
        Runs at the end of a complete epoch.
        """
        pass

    def post_train(self):
        """
        Runs after the training loop successfully completes.
        """
        pass

    def on_error(self, exc: BaseException):
        """
        Called when the training loop exits with an error.
        """
        del exc


@dataclass
class CallbackConfig(Callback, Config):
    """
    An alternative way to define callbacks when the callback class itself can't be serialized.
    """

    @abstractmethod
    def build(self, trainer: "Trainer") -> Optional[Callback]:
        """
        Build the actual :class:`Callback`.
        """
        raise NotImplementedError
