"""
A high-level distributed checkpointing module with a unified API for saving and
loading both local and remote checkpoints.

Features
--------

- Save with one distributed topology, seamlessly load with a different one. For example,
  with FSDP/FSDP2 you can save/load checkpoints with different world sizes or sharding strategies.
- Save/load directly to/from a remote object store like S3 or GCS. When loading from a remote object
  store each rank only downloads the fraction of the data it needs for its local
  (potentially sharded) tensors.

Overview
--------

Use :func:`save_model_and_optim_state()` to write a checkpoint with your model and optimizer's state,
then use :func:`load_model_and_optim_state()` to load the checkpoint in-place.

API Reference
-------------
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from olmo_core.distributed.utils import barrier, get_fs_local_rank
from olmo_core.io import (
    PathOrStr,
    clear_directory,
    dir_is_empty,
    is_url,
    normalize_path,
)
from olmo_core.utils import gc_cuda, wait_for

from .filesystem import RemoteFileSystemReader, RemoteFileSystemWriter

__all__ = [
    "save_model_and_optim_state",
    "async_save_model_and_optim_state",
    "load_model_and_optim_state",
    "RemoteFileSystemWriter",
    "RemoteFileSystemReader",
]

log = logging.getLogger(__name__)


@torch.no_grad()
def save_model_and_optim_state(
    dir: PathOrStr,
    model: nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    save_overwrite: bool = False,
) -> None:
    """
    Save model and optimizer state dictionaries. The model state can be a sharded model, in which
    case this method will correctly handle the optimizer state to ensure it can be loaded again with
    a different distributed topology through :func:`load_model_and_optim_state()`.

    .. seealso::
        - :func:`load_model_and_optim_state()`
        - :func:`async_save_model_and_optim_state()`

    .. tip::
        With :class:`~torch.distributed.fsdp.FullyShardedDataParallel` models it's not necessary
        to set the state dict type before calling this (or :func:`load_model_and_optim_state()`) via
        :meth:`~torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type()` or other methods.
        This function handles that internally.

    :param dir: Path/URL to save to.
    :param model: The model to save state from.
    :param optim: The optimizer to save state from.
    :param process_group: The process group to use for distributed collectives.
    :param save_overwrite: Overwrite existing files.

    :raises FileExistsError: If the checkpoint dir exists and is non-empty unless ``save_overwrite=True``.
    """
    dir = _prepare_env_for_save(dir, process_group=process_group, save_overwrite=save_overwrite)

    with _checkpoint_context(model):
        state_dict = _prepare_state_dict(model, optim=optim, process_group=process_group)
        dist_cp.state_dict_saver.save(
            state_dict,
            storage_writer=RemoteFileSystemWriter(dir),
            process_group=process_group,
        )


@torch.no_grad()
def async_save_model_and_optim_state(
    dir: PathOrStr,
    model: nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    save_overwrite: bool = False,
) -> Future[None]:
    """
    An async version of :func:`save_model_and_optim_state()`.

    This code first de-stages the state dict on the CPU, then writes it in a separate thread.
    """
    dir = _prepare_env_for_save(dir, process_group=process_group, save_overwrite=save_overwrite)

    with _checkpoint_context(model):
        state_dict = _prepare_state_dict(model, optim=optim, process_group=process_group)
        return dist_cp.state_dict_saver.async_save(
            state_dict,
            storage_writer=RemoteFileSystemWriter(dir),
            process_group=process_group,
        )


@torch.no_grad()
def load_model_and_optim_state(
    dir: PathOrStr,
    model: nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    process_group: Optional[dist.ProcessGroup] = None,
):
    """
    Load model and optimizer state in-place from a checkpoint saved via :func:`save_model_and_optim_state()`.
    This method is agnostic to the distributed topology in that it can load checkpoints saved with a different
    distributed topology (e.g. FSDP/FSDP2, DDP).

    .. seealso::
        - :func:`save_model_and_optim_state()`

    .. tip::
        With :class:`~torch.distributed.fsdp.FullyShardedDataParallel` models it's not necessary
        to set the state dict type before calling this (or :func:`load_model_and_optim_state()`) via
        :meth:`~torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type()` or other methods.
        This function handles that internally.

    :param dir: Path/URL to the checkpoint saved via :func:`save_model_and_optim_state()`.
    :param model: The model to load the state into.
    :param optim: The optimizer to load the state into.
    :param process_group: The process group to use for distributed collectives.
    """
    dir = normalize_path(dir)

    with _checkpoint_context(model):
        state_dict = _prepare_state_dict(model, optim, process_group=process_group)

        dist_cp.load(
            state_dict,
            checkpoint_id=dir,
            storage_reader=RemoteFileSystemReader(dir),
            process_group=process_group,
        )

        if optim is not None:
            set_state_dict(
                model,
                optim,
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict["optim"],
                options=StateDictOptions(strict=True),
            )
        else:
            set_model_state_dict(model, state_dict["model"], options=StateDictOptions(strict=True))

        #  # Load model state.
        #  model_state_dict = _load_model_state(dir, model, process_group=process_group)

        #  # Maybe load optim state.
        #  if optim is not None:
        #      _load_optim_state(dir, model, optim, model_state_dict, process_group=process_group)


def _prepare_env_for_save(
    dir: PathOrStr, process_group: Optional[dist.ProcessGroup] = None, save_overwrite: bool = False
) -> str:
    dir = normalize_path(dir)

    # Prepare checkpoint folder.
    if save_overwrite:
        if get_fs_local_rank(process_group) == 0:
            clear_directory(dir)
    elif not dir_is_empty(dir):
        raise FileExistsError(dir)

    barrier(process_group)

    if not is_url(dir):
        if get_fs_local_rank(process_group) == 0:
            Path(dir).mkdir(exist_ok=True, parents=True)
        # Ensure the dir exists for all ranks before continuing. This might take a second if we're
        # saving to an NFS drive or something like that.
        wait_for(Path(dir).exists, description=f"Waiting on '{dir}' to be created...")
        barrier(process_group)

    return dir


def _prepare_state_dict(
    model: nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    del process_group  # I feel like these torch functions should take a process group argument.
    sd_options = StateDictOptions(full_state_dict=False, cpu_offload=True)

    if optim is not None:
        model_state, optim_state = get_state_dict(model, optim, options=sd_options)
        return {"model": model_state, "optim": optim_state}
    else:
        return {"model": get_model_state_dict(model, options=sd_options)}

    #  state_dict: Dict[str, Any]
    #  if isinstance(model, FSDP):
    #      state_dict = {"model": model.state_dict()}
    #  else:
    #      state_dict = {"model": model}

    #  if optim is not None:
    #      state_dict["optim"] = _prepare_optimizer_state_to_save(
    #          model, optim, process_group=process_group
    #      )

    #  return state_dict


#  def _prepare_optimizer_state_to_save(
#      model: nn.Module,
#      optim: torch.optim.Optimizer,
#      process_group: Optional[dist.ProcessGroup] = None,
#  ) -> Any:
#      _init_optimizer_state(optim)
#      if isinstance(model, FSDP):
#          return FSDP.optim_state_dict(model, optim, group=process_group)
#      else:
#          return optim


@torch.no_grad()
def _init_optimizer_state(optim: torch.optim.Optimizer):
    """
    Ensure optimizer state is initialized for checkpointing.
    """
    if optim.state:
        return
    for group in optim.param_groups:
        for p in group["params"]:
            # Some parameters may be empty for sharded models, in which case the state does not need
            # to be initialized.
            if p.numel() > 0:
                p.grad = torch.zeros_like(p, memory_format=torch.preserve_format)
    optim.step()
    optim.zero_grad()


def _checkpoint_context(model: nn.Module):
    if isinstance(model, FSDP):
        from torch.distributed.fsdp.api import StateDictType

        return FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT)
    else:
        from contextlib import nullcontext

        return nullcontext()


def _prepare_model_state_dict_to_load(model: nn.Module) -> Dict[str, Any]:
    if isinstance(model, FSDP):
        return {"model": model.state_dict()}
    else:
        return {"model": model}


def _load_model_state(
    dir: str, model: nn.Module, process_group: Optional[dist.ProcessGroup] = None
) -> Dict[str, Any]:
    model_state = _prepare_model_state_dict_to_load(model)
    dist_cp.load(
        model_state,
        checkpoint_id=dir,
        storage_reader=RemoteFileSystemReader(dir),
        process_group=process_group,
    )
    if isinstance(model_state["model"], dict):
        model.load_state_dict(model_state["model"])
    return model_state


def _load_optim_state(
    dir: str,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    model_state_dict: Dict[str, Any],
    process_group: Optional[dist.ProcessGroup] = None,
):
    _init_optimizer_state(optim)
    if isinstance(model, FSDP):
        from torch.distributed.checkpoint.optimizer import (
            load_sharded_optimizer_state_dict,
        )

        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=model_state_dict["model"],
            optimizer_key="optim",
            storage_reader=RemoteFileSystemReader(dir),
        )
        del model_state_dict
        # Make sure tensors are on CPU! PyTorch puts them on GPU even if we have `offload_to_cpu=True`.
        _move_optim_state_to_cpu(optim_state["optim"])

        flattened_osd = FSDP.optim_state_dict_to_load(model, optim, optim_state["optim"])
        del optim_state
        # Put flattened optim state on CPU since `Optimizer.load_state_dict()` will create a deepcopy
        # of the whole state dict which takes up unnecessary GPU memory.
        _move_optim_state_to_cpu(flattened_osd)

        optim.load_state_dict(flattened_osd)
    else:
        del model_state_dict
        dist_cp.load(
            {"optim": optim},
            checkpoint_id=dir,
            storage_reader=RemoteFileSystemReader(dir),
            process_group=process_group,
        )


def _move_optim_state_to_cpu(optim_state: Dict[str, Any]):
    for state in optim_state["state"].values():
        for k in state.keys():
            state[k] = state[k].cpu()
    gc_cuda()
