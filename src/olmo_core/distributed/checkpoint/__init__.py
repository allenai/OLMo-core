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

You can unshard a checkpoint saved this way with :func:`unshard_checkpoint()`.

API Reference
-------------
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.metadata import Metadata

from olmo_core.aliases import PathOrStr
from olmo_core.io import clear_directory, dir_is_empty, is_url, normalize_path
from olmo_core.utils import gc_cuda, wait_for

from ..utils import barrier, get_fs_local_rank, is_distributed
from .filesystem import RemoteFileSystemReader, RemoteFileSystemWriter

__all__ = [
    "save_state_dict",
    "save_model_and_optim_state",
    "async_save_model_and_optim_state",
    "load_model_and_optim_state",
    "unshard_checkpoint",
    "get_checkpoint_metadata",
]

log = logging.getLogger(__name__)


@torch.no_grad()
def save_state_dict(
    dir: PathOrStr,
    state_dict: Dict[str, Any],
    process_group: Optional[dist.ProcessGroup] = None,
    save_overwrite: bool = False,
):
    """
    Save an arbitrary state dictionary to a distributed format that can loaded again with
    a different distributed topology.

    .. important::
        Please use :func:`save_model_and_optim_state` to save model/optimizer state dicts instead
        unless you know what you're doing.

    :param dir: Path/URL to save to.
    :param state_dict: The state dict to save.
    :param process_group: The process group to use for distributed collectives.
    :param save_overwrite: Overwrite existing files.
    """
    dir = _prepare_env_for_save(dir, process_group=process_group, save_overwrite=save_overwrite)
    dist_cp.state_dict_saver.save(
        state_dict,
        storage_writer=RemoteFileSystemWriter(dir),
        process_group=process_group,
    )


@torch.no_grad()
def save_model_and_optim_state(
    dir: PathOrStr,
    model: nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    *,
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
        - :func:`unshard_checkpoint()`

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
    state_dict = _prepare_state_dict(model, optim=optim, process_group=process_group)
    planner = DefaultSavePlanner(dedup_save_to_lowest_rank=True)
    dist_cp.state_dict_saver.save(
        state_dict,
        storage_writer=RemoteFileSystemWriter(dir),
        process_group=process_group,
        planner=planner,
    )


@torch.no_grad()
def async_save_model_and_optim_state(
    dir: PathOrStr,
    model: nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    save_overwrite: bool = False,
) -> Future[None]:
    """
    An async version of :func:`save_model_and_optim_state()`.

    This code first de-stages the state dict on the CPU, then writes it in a separate thread.
    """
    dir = _prepare_env_for_save(dir, process_group=process_group, save_overwrite=save_overwrite)
    state_dict = _prepare_state_dict(model, optim=optim, process_group=process_group)
    planner = DefaultSavePlanner(dedup_save_to_lowest_rank=True)
    return dist_cp.state_dict_saver.async_save(
        state_dict,
        storage_writer=RemoteFileSystemWriter(dir),
        process_group=process_group,
        planner=planner,
    )


@torch.no_grad()
def load_model_and_optim_state(
    dir: PathOrStr,
    model: nn.Module,
    optim: Optional[torch.optim.Optimizer] = None,
    *,
    process_group: Optional[dist.ProcessGroup] = None,
    key_mapping: Optional[Dict[str, str]] = None,
    pre_download: bool = False,
    work_dir: Optional[PathOrStr] = None,
):
    """
    Load model and optimizer state in-place from a checkpoint saved via :func:`save_model_and_optim_state()`.
    This method is agnostic to the distributed topology in that it can load checkpoints saved with a different
    distributed topology (e.g. FSDP/FSDP2, DDP).

    .. seealso::
        - :func:`save_model_and_optim_state()`
        - :func:`unshard_checkpoint()`

    .. tip::
        With :class:`~torch.distributed.fsdp.FullyShardedDataParallel` models it's not necessary
        to set the state dict type before calling this (or :func:`save_model_and_optim_state()`) via
        :meth:`~torch.distributed.fsdp.FullyShardedDataParallel.state_dict_type()` or other methods.
        This function handles that internally.

    .. warning::
        Due to the way :mod:`torch.distributed.checkpoint` works, if you have keys in the checkpoint
        dict that are not present in the current state of the model or optimizer, those keys won't
        be loaded.

        For example, if you added a custom field to one of your optimizer's param groups
        before saving the checkpoint, but don't have that field in the param group of the optimizer
        you're loading into, it won't be added.

        This can cause unexpected behavior if you're not careful. In this case the best thing to do
        is to ensure all keys are in present param groups when you initialize the optimizer, before saving
        or loading a checkpoint.

    :param dir: Path/URL to the checkpoint saved via :func:`save_model_and_optim_state()`.
    :param model: The model to load the state into.
    :param optim: The optimizer to load the state into.
    :param process_group: The process group to use for distributed collectives.
    :param key_mapping: Can be used to load a checkpoint where certain parameter have different names.
        This dictionary should map current keys to keys in the checkpoint to be loaded.
    :param pre_download: Download and cache relevant remote checkpoint files before trying to read from them.
    :param work_dir: A working directory for caching files/directories.
    """
    dir = normalize_path(dir)
    state_dict = _prepare_state_dict(model, optim, process_group=process_group)
    reader = RemoteFileSystemReader(dir, pre_download=pre_download, work_dir=work_dir)

    if key_mapping is not None:
        metadata = reader.read_metadata()
        for current_key, original_key in key_mapping.items():
            if f"model.{original_key}" not in metadata.state_dict_metadata:
                continue

            log.info(f"Mapping current param '{current_key}' to '{original_key}' in checkpoint")
            state_dict["model"][original_key] = state_dict["model"].pop(current_key)

            if optim is None:
                continue

            state_dict["optim"]["state"][original_key] = state_dict["optim"]["state"].pop(
                current_key
            )
            for group in state_dict["optim"]["param_groups"]:
                if current_key in group["params"]:
                    idx = group["params"].index(current_key)
                    group["params"][idx] = original_key
                    break

    dist_cp.load(
        state_dict,
        checkpoint_id=dir,
        storage_reader=reader,
        process_group=process_group,
    )

    if key_mapping is not None:
        metadata = reader.read_metadata()
        for current_key, original_key in key_mapping.items():
            if f"model.{original_key}" not in metadata.state_dict_metadata:
                continue

            state_dict["model"][current_key] = state_dict["model"].pop(original_key)

            if optim is None:
                continue

            state_dict["optim"]["state"][current_key] = state_dict["optim"]["state"].pop(
                original_key
            )
            for group in state_dict["optim"]["param_groups"]:
                if original_key in group["params"]:
                    idx = group["params"].index(original_key)
                    group["params"][idx] = current_key
                    break

    dist_cp_sd.set_model_state_dict(
        model, state_dict["model"], options=dist_cp_sd.StateDictOptions(strict=True)
    )
    gc_cuda()

    if optim is not None:
        dist_cp_sd.set_optimizer_state_dict(
            model, optim, state_dict["optim"], options=dist_cp_sd.StateDictOptions(strict=True)
        )
        gc_cuda()


def unshard_checkpoint(
    dir: PathOrStr,
    target_dir: PathOrStr,
    *,
    optim: Optional[bool] = None,
    save_overwrite: bool = False,
    use_safetensors: bool = False,
    pre_download: bool = False,
    work_dir: Optional[PathOrStr] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    Convert a checkpoint saved via :func:`save_model_and_optim_state()` into unsharded
    model and optimizer checkpoint files that can be loaded directly with :func:`torch.load()`
    or `safetensors <https://github.com/huggingface/safetensors>`_ if ``use_safetensors=True``.

    .. warning::
        The safetensors format cannot be used to save optimizer state, since optimizer state
        can contain arbitrary Python objects that need to be pickled.
        Therefore ``optim=True`` and ``use_safetensors=True`` is incompatible.

    .. warning::
        This should only be called in a non-distributed context. Otherwise a :class:`RuntimeError` is raised.

    :param dir: The path/URL to the original checkpoint created via :func:`save_model_and_optim_state()`.
    :param target_dir: The directory to save the unsharded model/optimizer checkpoint files to.
        This must be a local directory. URLs are not supported.
    :param optim: Whether to unshard the optimizer state. This defaults to ``True`` as long as
        ``use_safetensors=False``.
    :param save_overwrite: Overwrite any existing files in ``target_dir``.
    :param use_safetensors: Save the unsharded files with :func:`safetensors.torch.save_file()` instead
        of :func:`torch.save()`.
    :param pre_download: Download and cache relevant remote checkpoint files before trying to read from them.
    :param work_dir: A working directory for caching files/directories.

    :return: The path to the unsharded model checkpoint and the path to the unsharded
        optimizer checkpoint if ``optim=True``.

    :raises FileExistsError: If the ``target_dir`` is non-empty and ``save_overwrite=False``.
    """
    # Adapted from `torch.distributed.checkpoint.format_utils.dcp_to_torch_save()`.

    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    if optim is None:
        optim = not use_safetensors
    elif optim and use_safetensors:
        raise NotImplementedError("`optim=True` is incompatible with `use_safetensors=True`")

    if is_distributed():
        raise RuntimeError("'unshard_checkpoint' cannot be called in a distributed context")

    def save(state_dict: Dict[str, Any], path: Path):
        if path.is_file() and not save_overwrite:
            raise FileExistsError(
                f"'{path}' already exists, use `save_overwrite=True` to overwrite it"
            )

        if use_safetensors:
            from safetensors.torch import save_file

            save_file(state_dict, path)
        else:
            torch.save(state_dict, path)

    dir = normalize_path(dir)
    if is_url(target_dir):
        raise ValueError("'target_dir' must be a local directory")
    target_dir = Path(normalize_path(target_dir))
    target_dir.mkdir(exist_ok=True, parents=True)

    ext = "pt" if not use_safetensors else "safetensors"
    model_path = target_dir / f"model.{ext}"
    optim_path = target_dir / f"optim.{ext}" if optim else None

    model_sd: Dict[str, Any] = {}
    _load_state_dict(
        model_sd,
        storage_reader=RemoteFileSystemReader(dir, pre_download=pre_download, work_dir=work_dir),
        planner=_EmptyStateDictLoadPlanner(keys=["model"]),
        no_dist=True,
    )
    if not model_sd:
        raise RuntimeError("no model state found in checkpoint")
    save(model_sd["model"], model_path)
    del model_sd
    gc_cuda()

    if optim_path is not None:
        optim_sd: Dict[str, Any] = {}
        _load_state_dict(
            optim_sd,
            storage_reader=RemoteFileSystemReader(
                dir, pre_download=pre_download, work_dir=work_dir
            ),
            planner=_EmptyStateDictLoadPlanner(keys=["optim"]),
            no_dist=True,
        )
        if not optim_sd:
            raise RuntimeError("no optimizer state found in checkpoint")
        save(optim_sd["optim"], optim_path)
        del optim_sd
        gc_cuda()

    return model_path, optim_path


def get_checkpoint_metadata(dir: PathOrStr) -> Metadata:
    """
    Load the metadata from a checkpoint.

    :param dir: The path/URL to the checkpoint.
    """
    dir = normalize_path(dir)
    storage_reader = RemoteFileSystemReader(dir)
    return storage_reader.read_metadata()


def _prepare_env_for_save(
    dir: PathOrStr,
    process_group: Optional[dist.ProcessGroup] = None,
    save_overwrite: bool = False,
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
    sd_options = dist_cp_sd.StateDictOptions(full_state_dict=False, cpu_offload=True)

    state_dict: Dict[str, Any] = {
        "model": dist_cp_sd.get_model_state_dict(model, options=sd_options)
    }
    if optim is not None:
        state_dict["optim"] = dist_cp_sd.get_optimizer_state_dict(model, optim, options=sd_options)

    return state_dict


def _get_key(state_dict: Dict[str, Any], key: str, pop: bool = False) -> Any:
    if key in state_dict:
        if pop:
            return state_dict.pop(key)
        else:
            return state_dict[key]

    if "." not in key:
        raise KeyError(key)

    root, key = key.split(".", 1)
    if root not in state_dict:
        raise KeyError(root)

    return _get_key(state_dict[root], key, pop=pop)


def _set_key(state_dict: Dict[str, Any], key: str, value: Any):
    if "." not in key or all(["." in k for k in state_dict.keys()]):
        state_dict[key] = value
        return

    root, key = key.split(".", 1)
    if root not in state_dict:
        raise KeyError(root)

    return _set_key(state_dict[root], key, value=value)
