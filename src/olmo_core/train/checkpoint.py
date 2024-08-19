import tempfile
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from cached_path import cached_path
from torch.optim import Optimizer

from olmo_core.distributed.checkpoint import (
    async_save_model_and_optim_state,
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.utils import barrier, get_fs_local_rank, get_rank
from olmo_core.io import (
    PathOrStr,
    clear_directory,
    dir_is_empty,
    is_url,
    normalize_path,
    upload,
)
from olmo_core.utils import wait_for


@dataclass
class Checkpointer:
    """
    Trainer checkpointer.
    """

    save_overwrite: bool = False
    process_group: Optional[dist.ProcessGroup] = None

    def save(self, dir: PathOrStr, model: nn.Module, optim: Optimizer, train_state: Dict[str, Any]):
        """
        Save model, optim, and other training state to a local or remote directory.
        """
        dir = normalize_path(dir)
        with self._temporary_wd(dir) as wd:
            # Save trainer state.
            train_dir = wd / "train"
            train_dir.mkdir(exist_ok=True, parents=True)
            torch.save(train_state, train_dir / f"rank{get_rank()}.pt")

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

    def save_async(
        self, dir: PathOrStr, model: nn.Module, optim: Optimizer, train_state: Dict[str, Any]
    ) -> Future[None]:
        """
        An async version of :meth:`save()`.
        """
        dir = normalize_path(dir)
        with self._temporary_wd(dir) as wd:
            # Save trainer state.
            train_dir = wd / "train"
            train_dir.mkdir(exist_ok=True, parents=True)
            torch.save(train_state, train_dir / f"rank{get_rank()}.pt")

        # Save model and optim state.
        model_and_optim_dir = f"{dir}/model_and_optim"
        return async_save_model_and_optim_state(
            model_and_optim_dir,
            model,
            optim,
            process_group=self.process_group,
            save_overwrite=self.save_overwrite,
        )

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
                trainer_state = torch.load(cached_path(f"{dir}/train/rank{get_rank()}.pt"))
            except FileNotFoundError:
                # Fall back to rank 0 train state.
                # This can happen when we're restoring a checkpoint with a different world size.
                trainer_state = torch.load(cached_path(f"{dir}/train/rank0.pt"))

        # Load model and optimizer state.
        load_model_and_optim_state(
            f"{dir}/model_and_optim",
            model,
            optim if load_optimizer_state else None,
            process_group=self.process_group,
        )

        return trainer_state

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
            if get_fs_local_rank() == 0:
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
                tmp_dir.unlink(missing_ok=True)

        barrier()

    @contextmanager
    def _temporary_wd(self, dir: PathOrStr) -> Generator[Path, None, None]:
        # No need to mkdir here since we'll directly replace the temporary directory with
        # this directory below.
        dir = self._prepare_dir(dir, ensure_exists=False)

        tmp_dir = self._get_tmp_dir(dir)

        yield tmp_dir

        self._teardown_tmp_dir(dir, tmp_dir)
