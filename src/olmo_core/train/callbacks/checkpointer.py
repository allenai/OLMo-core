import logging
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import List, Optional

from olmo_core.distributed.utils import get_fs_local_rank
from olmo_core.io import clear_directory

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class CheckpointerCallback(Callback):
    """
    Used to configure checkpointing during training.

    .. important::
        This callback gets added automatically if you don't explicitly configure it.
        If you want to override this callback you should subclass it.
    """

    save_interval: int = 250
    ephemeral_save_interval: Optional[int] = None
    pre_train_checkpoint: bool = True
    save_async: bool = False

    # Bookkeeping

    _future: Optional[Future] = None
    _latest_checkpoint: int = -1
    _checkpoints: List[str] = field(default_factory=list)
    _ephemeral_checkpoints: List[str] = field(default_factory=list)

    def _await_last_checkpoint(self, blocking: bool = True) -> Optional[Future]:
        if (fut := self._future) is not None:
            # Wait for last async checkpoint to finish.
            if blocking or fut.done():
                fut.result()
                self._future = None
                log.info(f"Checkpoint for step {self._latest_checkpoint:,d} saved successfully")
                return fut
        return None

    def _save_checkpoint(self) -> str:
        self._await_last_checkpoint()
        self._latest_checkpoint = self.step
        path = f"{self.trainer.save_folder}/step{self.step}"
        if self.save_async:
            self._future = self.trainer.checkpointer.save_async(
                path,
                self.trainer.model,
                self.trainer.optim,
                self.trainer.state_dict(),
            )
        else:
            self.trainer.checkpointer.save(
                path,
                self.trainer.model,
                self.trainer.optim,
                self.trainer.state_dict(),
            )
        return path

    def pre_train(self):
        if self.pre_train_checkpoint:
            self._checkpoints.append(self._save_checkpoint())

    def post_train_batch(self):
        self._await_last_checkpoint(blocking=False)
        if self.step % self.save_interval == 0:
            self._checkpoints.append(self._save_checkpoint())
        elif (
            self.ephemeral_save_interval is not None
            and self.step % self.ephemeral_save_interval == 0
        ):
            self._ephemeral_checkpoints.append(self._save_checkpoint())
            while len(self._ephemeral_checkpoints) > 1:
                oldest_path = self._ephemeral_checkpoints.pop(0)
                if get_fs_local_rank() == 0:
                    clear_directory(oldest_path)

    def post_train(self):
        if self.step > self._latest_checkpoint:
            self._checkpoints.append(self._save_checkpoint())
        self._await_last_checkpoint()