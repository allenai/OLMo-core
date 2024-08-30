import logging
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch.distributed as dist

from olmo_core.distributed.utils import (
    backend_supports_cpu,
    get_fs_local_rank,
    get_rank,
    is_distributed,
    scatter_object,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.io import clear_directory

from ..checkpoint import Checkpointer
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
    """
    The interval, in steps, with which to save permanent checkoints.
    """

    ephemeral_save_interval: Optional[int] = None
    """
    The interval, in steps, with which to save temporary checkpoints. It's useful to set this to
    a frequent interval for preemptible jobs.
    """

    pre_train_checkpoint: Optional[bool] = None
    """
    Save a pretrain checkpoint. Defaults to ``True`` unless the trainer resumes from a checkpoint.
    """

    save_async: bool = False
    """
    Save checkpoints asynchronously. Requires a backend that supports CPU.
    """

    # Bookkeeping

    # NOTE: can't use type annotation here, omegaconf doesn't like it
    #  _future: Optional[Future] = None
    _future = None
    _latest_checkpoint: int = -1
    _checkpoints: List[str] = field(default_factory=list)
    _ephemeral_checkpoints: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.save_interval < 1:
            raise OLMoConfigurationError("'save_interval' must be at least 1")
        if self.ephemeral_save_interval is not None:
            if self.ephemeral_save_interval < 1:
                raise OLMoConfigurationError("'ephemeral_save_interval' must be at least 1")
            if self.ephemeral_save_interval >= self.save_interval:
                raise OLMoConfigurationError(
                    "'ephemeral_save_interval' must be less than 'save_interval'"
                )

    @property
    def checkpointer(self) -> Checkpointer:
        return self.trainer.checkpointer

    @property
    def save_folder(self) -> str:
        return self.trainer.save_folder

    def _await_last_checkpoint(self, blocking: bool = True) -> Optional[Future]:
        if (fut := self._future) is not None:
            # Wait for last async checkpoint to finish.
            if blocking or fut.done():
                fut.result()
                self._future = None
                log.info(f"Checkpoint for step {self._latest_checkpoint:,d} saved successfully")
                return fut
        return None

    def _save_checkpoint(self, save_async: Optional[bool] = None) -> str:
        save_async = save_async if save_async is not None else self.save_async
        self._await_last_checkpoint()
        self._latest_checkpoint = self.step
        dirname = self.checkpointer.checkpoint_dirname(self.step)
        path = f"{self.save_folder}/{dirname}"
        if save_async:
            log.info(f"Saving checkpoint for step {self.step} to '{path}' asynchronously...")
            self._future = self.checkpointer.save_async(
                path,
                self.trainer.model,
                self.trainer.optim,
                self.trainer.state_dict(),
            )
        else:
            log.info(f"Saving checkpoint for step {self.step} to '{path}'...")
            self.checkpointer.save(
                path,
                self.trainer.model,
                self.trainer.optim,
                self.trainer.state_dict(),
            )
            log.info("Checkpoint saved")
        return path

    def _remove_checkpoint(self, path: str):
        if get_fs_local_rank() == 0:
            self.trainer.thread_pool.submit(clear_directory, path)

    def pre_train(self):
        # Maybe create a new process group for async checkpointing.
        if is_distributed() and self.save_async and self.checkpointer.process_group is None:
            if not backend_supports_cpu():
                raise RuntimeError("a CPU-capable backend is required for async checkpointing")

            log.info(
                "Creating new process group for checkpointing (needed for async checkpointing)"
            )
            self.checkpointer.process_group = dist.new_group()

        # Maybe save a pre-train checkpoint.
        if self.step == 0 and (
            self.pre_train_checkpoint
            or (self.pre_train_checkpoint is None and not self.trainer.checkpoint_loaded)
        ):
            self._checkpoints.append(self._save_checkpoint())

        # Collect existing ephemeral checkpoints from previous runs.
        if self.ephemeral_save_interval is not None:
            ephemeral_checkpoints: List[Tuple[int, str]] = []

            # Only search from rank 0 to avoid hammering remote file stores with requests.
            if get_rank() == 0:
                for step_num, path in self.checkpointer.find_checkpoints(self.save_folder):
                    if step_num == 0 and step_num % self.save_interval == 0:
                        continue
                    elif step_num % self.ephemeral_save_interval == 0:
                        ephemeral_checkpoints.append((step_num, path))
            ephemeral_checkpoints = scatter_object(ephemeral_checkpoints)

            # TODO: handle this if we ever restore callback state.
            assert not self._ephemeral_checkpoints

            self._ephemeral_checkpoints = [
                path for _, path in sorted(ephemeral_checkpoints, key=lambda x: x[0])
            ]
            for path in self._ephemeral_checkpoints:
                log.info(f"Collected existing ephemeral checkpoint at '{path}'")

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
                log.info(f"Removing old ephemeral checkpoint at '{oldest_path}'...")
                self._remove_checkpoint(oldest_path)

    def post_train(self):
        if self.step > self._latest_checkpoint:
            self._checkpoints.append(self._save_checkpoint(save_async=False))
        self._await_last_checkpoint()
