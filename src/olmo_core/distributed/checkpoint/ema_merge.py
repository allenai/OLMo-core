"""
Exponential Moving Average (EMA) checkpoint merging utilities.

This module provides functionality to load multiple checkpoints, merge them using
Exponential Moving Average, and save the result in the same distributed format.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from olmo_core.aliases import PathOrStr
from olmo_core.utils import gc_cuda

import torch.distributed.checkpoint.state_dict as dist_cp_sd

from . import (
    get_checkpoint_metadata,
    load_model_and_optim_state,
    save_model_and_optim_state,
)

__all__ = [
    "ema_merge_checkpoints",
    "ema_merge_model_states",
]

log = logging.getLogger(__name__)


def ema_merge_checkpoints(
    checkpoint_dirs: List[PathOrStr],
    output_dir: PathOrStr,
    model: nn.Module,
    *,
    ema_decay: float = 0.999,
    optim: Optional[torch.optim.Optimizer] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    save_overwrite: bool = False,
    pre_download: bool = False,
    work_dir: Optional[PathOrStr] = None,
    thread_count: Optional[int] = None,
    throttle_uploads: bool = False,
    quiet: bool = False,
) -> None:
    """
    Load multiple checkpoints, merge them using Exponential Moving Average, and save the result.
    
    The EMA is computed as: merged = decay * current + (1 - decay) * new_checkpoint
    where checkpoints are processed in the order provided.
    
    :param checkpoint_dirs: List of paths/URLs to checkpoints to merge, in order of processing.
    :param output_dir: Path/URL to save the merged checkpoint.
    :param model: The model instance to use for loading/saving (defines structure).
    :param ema_decay: EMA decay factor (0 < decay < 1). Higher values give more weight to earlier checkpoints.
    :param optim: Optional optimizer to save state for. If provided, only the optimizer state 
        from the first checkpoint will be saved.
    :param process_group: The process group to use for distributed collectives.
    :param save_overwrite: Overwrite existing files in output_dir.
    :param pre_download: Download and cache remote checkpoint files before reading.
    :param work_dir: Working directory for caching files.
    :param thread_count: Number of threads for I/O operations.
    :param throttle_uploads: Throttle uploads to prevent overwhelming the storage backend.
    :param quiet: Suppress progress messages.
    
    :raises ValueError: If checkpoint_dirs is empty or ema_decay is not in valid range.
    :raises RuntimeError: If checkpoints have incompatible structures.
    """
    if not checkpoint_dirs:
        raise ValueError("checkpoint_dirs cannot be empty")
    
    if not (0 < ema_decay < 1):
        raise ValueError(f"ema_decay must be between 0 and 1, got {ema_decay}")
    
    if not quiet:
        log.info("Starting EMA merge of %d checkpoints with decay=%f", len(checkpoint_dirs), ema_decay)
    
    metadatas = []
    for i, checkpoint_dir in enumerate(checkpoint_dirs):
        try:
            metadata = get_checkpoint_metadata(checkpoint_dir)
            metadatas.append(metadata)
            if not quiet:
                log.info("Validated checkpoint %d/%d: %s", i+1, len(checkpoint_dirs), checkpoint_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to read metadata from checkpoint {checkpoint_dir}: {e}") from e
    
    first_model_keys = {k for k in metadatas[0].state_dict_metadata.keys() if k.startswith("model.")}
    for i, metadata in enumerate(metadatas[1:], 1):
        model_keys = {k for k in metadata.state_dict_metadata.keys() if k.startswith("model.")}
        if model_keys != first_model_keys:
            missing = first_model_keys - model_keys
            extra = model_keys - first_model_keys
            raise RuntimeError(
                f"Checkpoint {i} has incompatible model structure. "
                f"Missing keys: {missing}, Extra keys: {extra}"
            )
    
    merged_state_dict: Optional[Dict[str, Any]] = None
    
    for i, checkpoint_dir in enumerate(checkpoint_dirs):
        if not quiet:
            log.info("Loading checkpoint %d/%d: %s", i+1, len(checkpoint_dirs), checkpoint_dir)
        
        if i == 0:
            load_model_and_optim_state(
                checkpoint_dir,
                model,
                optim=optim,
                process_group=process_group,
                pre_download=pre_download,
                work_dir=work_dir,
                thread_count=thread_count,
            )
            merged_state_dict = _get_model_state_dict_from_model(model)
            if not quiet:
                log.info("Initialized EMA with checkpoint 1")
        else:
            temp_model = _create_empty_model_like(model)
            load_model_and_optim_state(
                checkpoint_dir,
                temp_model,
                process_group=process_group,
                pre_download=pre_download,
                work_dir=work_dir,
                thread_count=thread_count,
            )
            current_state_dict = _get_model_state_dict_from_model(temp_model)
            if merged_state_dict is not None:
                merged_state_dict = _ema_merge_state_dicts(
                    merged_state_dict, current_state_dict, ema_decay
                )
            
            if not quiet:
                log.info("Merged checkpoint %d with EMA decay=%f", i+1, ema_decay)

            del temp_model, current_state_dict
            gc_cuda()

    if merged_state_dict is not None:
        _set_model_state_dict_to_model(model, merged_state_dict)
    
    if not quiet:
        log.info("Saving merged checkpoint to: %s", output_dir)

    save_model_and_optim_state(
        output_dir,
        model,
        optim=optim,
        process_group=process_group,
        save_overwrite=save_overwrite,
        thread_count=thread_count,
        throttle_uploads=throttle_uploads,
    )
    
    if not quiet:
        log.info("EMA checkpoint merge completed successfully")


def ema_merge_model_states(
    model_state_dicts: List[Dict[str, Any]],
    ema_decay: float = 0.999,
) -> Dict[str, Any]:
    """
    Merge multiple model state dictionaries using Exponential Moving Average.
    
    :param model_state_dicts: List of model state dictionaries to merge.
    :param ema_decay: EMA decay factor (0 < decay < 1).
    
    :return: Merged model state dictionary.
    
    :raises ValueError: If model_state_dicts is empty or ema_decay is not in valid range.
    """
    if not model_state_dicts:
        raise ValueError("model_state_dicts cannot be empty")
    
    if not (0 < ema_decay < 1):
        raise ValueError(f"ema_decay must be between 0 and 1, got {ema_decay}")
    
    if len(model_state_dicts) == 1:
        return model_state_dicts[0].copy()
    
    merged = model_state_dicts[0].copy()
    
    for state_dict in model_state_dicts[1:]:
        merged = _ema_merge_state_dicts(merged, state_dict, ema_decay)
    
    return merged


def _ema_merge_state_dicts(
    target: Dict[str, Any],
    source: Dict[str, Any], 
    decay: float,
) -> Dict[str, Any]:
    """
    Apply EMA merge: target = decay * target + (1 - decay) * source
    
    :param target: Target state dict (will be modified in-place).
    :param source: Source state dict to merge in.
    :param decay: EMA decay factor.
    
    :return: The modified target state dict.
    """
    for key in target.keys():
        if key not in source:
            log.warning("Key '%s' found in target but not in source, skipping", key)
            continue
        
        target_param = target[key]
        source_param = source[key]
        
        if not isinstance(target_param, torch.Tensor) or not isinstance(source_param, torch.Tensor):
            continue
        
        if target_param.shape != source_param.shape:
            raise RuntimeError(
                f"Shape mismatch for parameter '{key}': "
                f"target {target_param.shape} vs source {source_param.shape}"
            )
        
        if target_param.dtype != source_param.dtype:
            log.warning(
                "Dtype mismatch for parameter '%s': target %s vs source %s, converting source to target dtype",
                key, target_param.dtype, source_param.dtype
            )
            source_param = source_param.to(target_param.dtype)

        target[key] = decay * target_param + (1 - decay) * source_param
    
    for key in source.keys():
        if key not in target:
            log.warning("Key '%s' found in source but not in target, ignoring", key)
    
    return target


def _get_model_state_dict_from_model(model: nn.Module) -> Dict[str, Any]:
    
    
    state_dict_options = dist_cp_sd.StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
    )
    return dist_cp_sd.get_model_state_dict(model, options=state_dict_options)


def _set_model_state_dict_to_model(model: nn.Module, state_dict: Dict[str, Any]) -> None:
    
    dist_cp_sd.set_model_state_dict(
        model, 
        state_dict, 
        options=dist_cp_sd.StateDictOptions(strict=True)
    )


def _create_empty_model_like(model: nn.Module) -> nn.Module:
    import copy
    model_copy = copy.deepcopy(model)
    model_copy = model_copy.cpu()
    
    return model_copy


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python ema_merge.py <checkpoint1> <checkpoint2> [checkpoint3...] <output_dir>")
        sys.exit(1)
    
    checkpoint_dirs = sys.argv[1:-1]
    output_dir = sys.argv[-1]
    
    # You need to initialize your actual model here
    # model = YourModel.from_config(config)
    print(f"Would merge {len(checkpoint_dirs)} checkpoints to {output_dir}")
    print("Note: You need to provide an actual model instance")
