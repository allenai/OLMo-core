import json
import os
import re
import tempfile
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Generator, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from cached_path import cached_path
from torch.optim import Optimizer

from ..aliases import PathOrStr
from ..config import Config
from ..distributed.checkpoint import (
    async_save_model_and_optim_state,
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from ..distributed.utils import barrier, get_fs_local_rank, get_rank, is_distributed
from ..exceptions import OLMoConfigurationError
from ..io import (
    clear_directory,
    dir_is_empty,
    file_exists,
    is_url,
    list_directory,
    normalize_path,
    upload,
)
from ..utils import wait_for
from ..version import VERSION


@dataclass
class CheckpointMetadata(Config):
    version: str = VERSION


@dataclass
class Checkpointer:
    """
    Trainer checkpointer.
    """

    METADATA_FNAME: ClassVar[str] = ".metadata.json"
    CHECKPOINT_DIR: ClassVar[str] = "step{step}"

    save_overwrite: bool = False
    process_group: Optional[dist.ProcessGroup] = None

    def save(self, dir: PathOrStr, model: nn.Module, optim: Optimizer, train_state: Dict[str, Any]):
        """
        Save model, optim, and other training state to a local or remote directory.
        """
        dir = normalize_path(dir)
        with self._temporary_wd(dir) as wd:
            # Save trainer state.
            self._save_train_state(dir, wd, train_state)

            # Save model and optim state.
            model_and_optim_dir = (
                f"{dir}/model_and_optim" if is_url(dir) else wd / "model_and_optim"
            )
            save_model_and_optim_state(
                model_and_optim_dir,
                model,
                optim,
                process_group=self.process_group,
                save_overwrite=self.save_overwrite,
            )

        self._save_metadata(dir, CheckpointMetadata())

    def save_async(
        self, dir: PathOrStr, model: nn.Module, optim: Optimizer, train_state: Dict[str, Any]
    ) -> Future[None]:
        """
        An async version of :meth:`save()`.
        """
        if is_distributed() and self.process_group is None:
            raise OLMoConfigurationError(
                "a checkointer process group is required for async checkointing!"
            )

        dir = normalize_path(dir)

        with self._temporary_wd(dir) as wd:
            # Save trainer state.
            self._save_train_state(dir, wd, train_state)

        # Save model and optim state.
        model_and_optim_dir = f"{dir}/model_and_optim"
        future = async_save_model_and_optim_state(
            model_and_optim_dir,
            model,
            optim,
            process_group=self.process_group,
            save_overwrite=self.save_overwrite,
        )

        def done_callback(fut: Future):
            del fut
            self._save_metadata(dir, CheckpointMetadata())

        # Upload metadata when everything else is done.
        future.add_done_callback(done_callback)

        return future

    def load(
        self,
        dir: PathOrStr,
        model: nn.Module,
        optim: Optimizer,
        *,
        load_optimizer_state: bool = True,
        load_trainer_state: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Load model, optim, and other training state from a local or remote checkpoint directory
        created via :meth:`save()` or :meth:`save_async()`.
        """
        dir = normalize_path(dir)

        # Maybe load trainer state.
        trainer_state: Optional[Dict[str, Any]] = None
        if load_trainer_state:
            try:
                trainer_state = torch.load(
                    cached_path(f"{dir}/train/rank{get_rank()}.pt", quiet=True)
                )
            except FileNotFoundError:
                # Fall back to rank 0 train state.
                # This can happen when we're restoring a checkpoint with a different world size.
                trainer_state = torch.load(cached_path(f"{dir}/train/rank0.pt", quiet=True))

        # Load model and optimizer state.
        load_model_and_optim_state(
            f"{dir}/model_and_optim",
            model,
            optim if load_optimizer_state else None,
            process_group=self.process_group,
        )

        return trainer_state

    def write_file(self, dir: PathOrStr, fname: str, contents: Union[str, bytes]) -> PathOrStr:
        """
        Write something to a file in a local or remote directory.

        :param dir: The path/URL of the directory to write the file to.
        :param fname: The name of the file to write, relative to ``dir``.
        :param contents: The contents of the file to write.

        :returns: The path/URL of the file.
        """
        dir = normalize_path(dir)
        fname = normalize_path(fname)

        if not is_url(dir):
            Path(dir).mkdir(exist_ok=True, parents=True)

        mode = "wb" if isinstance(contents, bytes) else "wt"
        tmp_file = tempfile.NamedTemporaryFile(
            mode=mode, delete=False, dir=None if is_url(dir) else dir
        )
        tmp_path = Path(tmp_file.name)
        try:
            tmp_file.write(contents)
            tmp_file.flush()

            target: PathOrStr
            if is_url(dir):
                target = f"{dir}/{fname}"
                upload(tmp_path, target, save_overwrite=self.save_overwrite)
            else:
                target = Path(dir) / fname
                if target.is_file() and not self.save_overwrite:
                    raise FileExistsError(target)
                target.parent.mkdir(exist_ok=True, parents=True)
                tmp_path.rename(target)

            return target
        finally:
            tmp_path.unlink(missing_ok=True)

    @classmethod
    def checkpoint_dirname(cls, step: int) -> str:
        return cls.CHECKPOINT_DIR.format(step=step)

    @classmethod
    def dir_is_checkpoint(cls, dir: PathOrStr) -> bool:
        """
        Check if a directory is a checkpoint directory.
        """
        dir = normalize_path(dir)
        paths_to_check = [
            f"{dir}/train/rank0.pt",
            f"{dir}/model_and_optim/.metadata",
            f"{dir}/{cls.METADATA_FNAME}",
        ]
        for path in paths_to_check:
            if not file_exists(path):
                return False
        return True

    @classmethod
    def find_checkpoints(cls, dir: PathOrStr) -> Generator[Tuple[int, str], None, None]:
        """
        Find checkpoints within a directory.
        """
        dir = normalize_path(dir)
        for path in list_directory(dir):
            name = os.path.basename(path)
            if (m := re.match("^" + cls.CHECKPOINT_DIR.format(step=r"(\d+)$"), name)) is not None:
                step = int(m.group(1))

                # Make sure the directory is a valid checkpoint dir.
                if not cls.dir_is_checkpoint(path):
                    continue

                yield step, path

    @classmethod
    def contains_checkpoint(cls, dir: PathOrStr) -> bool:
        """
        Check if a directory is a checkpoint directory or contains a child checkpoint directory.
        """
        if cls.dir_is_checkpoint(dir):
            return True

        try:
            next(cls.find_checkpoints(dir))
            return True
        except (StopIteration, FileNotFoundError):
            return False

    @classmethod
    def latest_checkpoint(cls, dir: PathOrStr) -> str:
        """
        Find the latest checkpoint in a directory of checkpoints.

        :raises FileNotFoundError: If no checkpoints are found.
        """
        dir = normalize_path(dir)
        latest_step: Optional[int] = None
        latest_checkpoint: Optional[str] = None
        for step, path in cls.find_checkpoints(dir):
            if latest_step is None or step > latest_step:
                latest_step = step
                latest_checkpoint = path

        if latest_checkpoint is None:
            raise FileNotFoundError(f"No checkpoints found in '{dir}'")
        else:
            return latest_checkpoint

    def _save_train_state(self, dir: PathOrStr, wd: Path, train_state: Dict[str, Any]):
        train_dir = wd / "train"
        # NOTE: if 'dir' is a URL, the 'wd' will be a different temp dir for each rank.
        if is_url(dir) or get_fs_local_rank() == 0:
            train_dir.mkdir(exist_ok=True, parents=True)
        wait_for(train_dir.exists, description=f"Waiting on '{train_dir}' to be created...")
        torch.save(train_state, train_dir / f"rank{get_rank()}.pt")

    def _save_metadata(self, dir: PathOrStr, metadata: CheckpointMetadata):
        if get_rank() == 0:
            self.write_file(dir, self.METADATA_FNAME, json.dumps(metadata.as_dict(json_safe=True)))

    def _prepare_dir(self, dir: PathOrStr, ensure_exists: bool = True) -> str:
        dir = normalize_path(dir)

        # Make sure checkpoint directory doesn't exist unless it's okay to overwrite it.
        if not dir_is_empty(dir):
            if self.save_overwrite:
                if get_fs_local_rank() == 0:
                    clear_directory(dir)
            else:
                raise FileExistsError(dir)

        if ensure_exists and not is_url(dir):
            if get_fs_local_rank() == 0:
                Path(dir).mkdir(exist_ok=True, parents=True)
            # Ensure the dir exists for all ranks before continuing. This might take a second if we're
            # saving to an NFS drive or something like that.
            wait_for(Path(dir).exists, description=f"Waiting on '{dir}' to be created...")

        barrier()
        return dir

    def _get_tmp_dir(self, dir: PathOrStr) -> Path:
        # Prepare temporary directory.
        tmp_dir: Path
        if is_url(dir):
            tmp_dir = Path(tempfile.mkdtemp())
        else:
            tmp_dir = Path(dir).with_name(Path(dir).name + "-tmp")
            if get_fs_local_rank() == 0:
                clear_directory(tmp_dir)
                tmp_dir.mkdir(exist_ok=True, parents=True)

        # In the cases where we're using a shared NFS drive between ranks to save checkpoints,
        # creating the temp directory from rank 0 might not be immediately
        # realized in the file systems of the other ranks.
        # So we wait here across all ranks until that tmp checkpoint directory is visible.
        wait_for(lambda: tmp_dir.exists(), "Waiting for checkpoint directory", timeout=10.0)
        barrier()
        return tmp_dir

    def _teardown_tmp_dir(self, dir: PathOrStr, tmp_dir: Path):
        if not is_url(dir):
            # Replace the temporary directory with the actual checkpoint directory.
            if get_fs_local_rank() == 0:
                # Replace temp directory with target checkpoint directory.
                try:
                    tmp_dir.replace(str(dir))
                except FileNotFoundError:
                    # Caught when another (file-system) local rank 0 has already replaced the tmp directory.
                    # This can happen when nodes are saving to a common NFS drive but otherwise have distinct
                    # file-systems.
                    if not Path(dir).exists():
                        raise

            # In the cases where we're using a shared NFS drive between ranks to save checkpoints,
            # replacing the temp directory with the final directory from rank 0 might not be immediately
            # realized in the file systems of the other ranks.
            # So we wait here across all ranks until that final checkpoint directory is visible.
            wait_for(lambda: Path(dir).exists(), "Waiting for checkpoint directory", timeout=10.0)
        else:
            # NOTE: each rank will have its own tmp dir
            # Upload files to final location.
            for path in tmp_dir.glob("**/*"):
                if not path.is_file():
                    continue
                upload(
                    path,
                    f"{dir}/{path.relative_to(tmp_dir)}",
                    save_overwrite=self.save_overwrite,
                )

            # Then remove the temp dir.
            clear_directory(tmp_dir)

        barrier()

    @contextmanager
    def _temporary_wd(self, dir: PathOrStr) -> Generator[Path, None, None]:
        # No need to mkdir here since we'll directly replace the temporary directory with
        # this directory below.
        dir = self._prepare_dir(dir, ensure_exists=False)

        tmp_dir = self._get_tmp_dir(dir)

        yield tmp_dir

        barrier()

        self._teardown_tmp_dir(dir, tmp_dir)
