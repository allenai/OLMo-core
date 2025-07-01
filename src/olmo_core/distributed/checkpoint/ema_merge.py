"""
Exponential Moving Average (EMA) checkpoint merging utilities.

This module provides functionality to load multiple checkpoints, merge them using
Exponential Moving Average, and save the result in the same distributed format.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn as nn

from olmo_core.aliases import PathOrStr
from olmo_core.utils import gc_cuda

from olmo_core.distributed.checkpoint import (
    get_checkpoint_metadata,
    load_model_and_optim_state,
    save_model_and_optim_state,
)

__all__ = [
    "ema_merge_checkpoints",
    "ema_merge_model_states",
]

log = logging.getLogger(__name__)


def ema_merge_shards_generic(
    checkpoint_dirs: List[PathOrStr],
    output_dir: PathOrStr,
    file_pattern: str = "*.distcp",
    *,
    ema_decay: float = 0.999,
    save_overwrite: bool = False,
    quiet: bool = False,
) -> None:
    """
    Merge sharded files at the file level, preserving exact file structure.
    
    Works with both .distcp files and .pt files in train directories.
    """
    import shutil
    from pathlib import Path
    
    if not checkpoint_dirs:
        raise ValueError("checkpoint_dirs cannot be empty")
    
    if not (0 < ema_decay < 1):
        raise ValueError(f"ema_decay must be between 0 and 1, got {ema_decay}")
    
    checkpoint_paths = [Path(d) for d in checkpoint_dirs]
    output_path = Path(output_dir)
    
    if not quiet:
        print(f"Starting shard-level EMA merge of {len(checkpoint_dirs)} checkpoints")
    
    # Get all shard files from first checkpoint
    first_checkpoint = checkpoint_paths[0]
    shard_files = list(first_checkpoint.glob(file_pattern))
    
    if not shard_files:
        raise RuntimeError(f"No {file_pattern} files found in {first_checkpoint}")
    
    if not quiet:
        print(f"Found {len(shard_files)} shard files to merge")
        
        # Check file sizes for debugging
        for shard_file in shard_files[:3]:  # Show first 3 for debugging
            file_size = shard_file.stat().st_size
            print(f"  {shard_file.name}: {file_size} bytes")
    
    # Verify all checkpoints have the same shard files
    for i, checkpoint_path in enumerate(checkpoint_paths[1:], 1):
        other_shard_files = set(f.name for f in checkpoint_path.glob(file_pattern))
        first_shard_files = set(f.name for f in shard_files)
        
        if other_shard_files != first_shard_files:
            missing = first_shard_files - other_shard_files
            extra = other_shard_files - first_shard_files
            raise RuntimeError(
                f"Checkpoint {i} has different shard files. "
                f"Missing: {missing}, Extra: {extra}"
            )
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy metadata files first (only for .distcp directories)
    if file_pattern == "*.distcp":
        metadata_files = [".metadata", "meta.pt"]
        for meta_file in metadata_files:
            src = first_checkpoint / meta_file
            if src.exists():
                if save_overwrite or not (output_path / meta_file).exists():
                    shutil.copy2(src, output_path / meta_file)
    
    # Merge each shard file
    successful_merges = 0
    skipped_shards = []
    
    for shard_file in shard_files:
        if not quiet:
            print(f"Merging shard: {shard_file.name}")
        
        try:
            # Validate file integrity first
            for i, checkpoint_path in enumerate(checkpoint_paths):
                shard_path = checkpoint_path / shard_file.name
                if not shard_path.exists():
                    raise FileNotFoundError(f"Shard file missing: {shard_path}")
                
                # Check if file is readable and not empty
                file_size = shard_path.stat().st_size
                if file_size == 0:
                    raise RuntimeError(f"Shard file is empty: {shard_path}")
                
                # Try to load just the header to validate integrity
                try:
                    torch.load(shard_path, weights_only=False)
                except Exception as e:
                    raise RuntimeError(f"Shard file corrupted at checkpoint {i}: {shard_path} - {e}")
            
            # Load first shard as base
            base_shard = torch.load(first_checkpoint / shard_file.name, weights_only=False)
            
            # Apply EMA with subsequent shards
            for checkpoint_path in checkpoint_paths[1:]:
                current_shard_path = checkpoint_path / shard_file.name
                current_shard = torch.load(current_shard_path, weights_only=False)
                base_shard = _ema_merge_shard_data(base_shard, current_shard, ema_decay)
            
            # Save merged shard
            output_shard_path = output_path / shard_file.name
            torch.save(base_shard, output_shard_path)
            successful_merges += 1
            
        except Exception as e:
            print(f"❌ Error processing shard {shard_file.name}: {e}")
            print(f"   Skipping this shard and continuing...")
            skipped_shards.append(shard_file.name)
            continue
    
    if not quiet:
        if skipped_shards:
            print(f"Shard-level EMA merge completed! {successful_merges}/{len(shard_files)} shards merged successfully.")
            print(f"⚠️  Skipped {len(skipped_shards)} corrupted shards: {skipped_shards[:5]}{'...' if len(skipped_shards) > 5 else ''}")
        else:
            print(f"Shard-level EMA merge completed! {successful_merges} shards merged successfully.")
    
    return successful_merges, len(skipped_shards)


def ema_merge_checkpoints_shard_level(
    checkpoint_dirs: List[PathOrStr],
    output_dir: PathOrStr,
    *,
    ema_decay: float = 0.999,
    save_overwrite: bool = False,
    quiet: bool = False,
) -> tuple[int, int]:
    """
    Merge checkpoints at the shard level, preserving exact file structure.
    
    This merges individual .distcp files while maintaining the same sharding layout.
    
    Returns:
        tuple[int, int]: (successful_merges, skipped_shards)
    """
    return ema_merge_shards_generic(
        checkpoint_dirs, output_dir, "*.distcp",
        ema_decay=ema_decay, save_overwrite=save_overwrite, quiet=quiet
    )


def _ema_merge_shard_data(target_shard: Any, source_shard: Any, decay: float) -> Any:
    """
    Apply EMA merge to shard data while preserving structure.
    """
    if isinstance(target_shard, torch.Tensor) and isinstance(source_shard, torch.Tensor):
        if target_shard.shape != source_shard.shape:
            raise RuntimeError(f"Shape mismatch: {target_shard.shape} vs {source_shard.shape}")
        return decay * target_shard + (1 - decay) * source_shard
    
    elif isinstance(target_shard, dict) and isinstance(source_shard, dict):
        merged = {}
        for key in target_shard:
            if key in source_shard:
                merged[key] = _ema_merge_shard_data(target_shard[key], source_shard[key], decay)
            else:
                merged[key] = target_shard[key]
        return merged
    
    elif isinstance(target_shard, (list, tuple)) and isinstance(source_shard, (list, tuple)):
        if len(target_shard) != len(source_shard):
            raise RuntimeError(f"Length mismatch: {len(target_shard)} vs {len(source_shard)}")
        merged = []
        for t_item, s_item in zip(target_shard, source_shard):
            merged.append(_ema_merge_shard_data(t_item, s_item, decay))
        return type(target_shard)(merged)
    
    else:
        # For non-tensor data (metadata, etc.), just keep the target
        return target_shard


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
        log.info(
            "Starting EMA merge of %d checkpoints with decay=%f", len(checkpoint_dirs), ema_decay
        )

    metadatas = []
    for i, checkpoint_dir in enumerate(checkpoint_dirs):
        try:
            metadata = get_checkpoint_metadata(checkpoint_dir)
            metadatas.append(metadata)
            if not quiet:
                log.info(
                    "Validated checkpoint %d/%d: %s", i + 1, len(checkpoint_dirs), checkpoint_dir
                )
        except Exception as e:
            raise RuntimeError(
                f"Failed to read metadata from checkpoint {checkpoint_dir}: {e}"
            ) from e

    first_model_keys = {
        k for k in metadatas[0].state_dict_metadata.keys() if k.startswith("model.")
    }
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
            log.info("Loading checkpoint %d/%d: %s", i + 1, len(checkpoint_dirs), checkpoint_dir)

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
                log.info("Merged checkpoint %d with EMA decay=%f", i + 1, ema_decay)

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
                key,
                target_param.dtype,
                source_param.dtype,
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
        model, state_dict, options=dist_cp_sd.StateDictOptions(strict=True)
    )


def _create_empty_model_like(model: nn.Module) -> nn.Module:
    import copy

    model_copy = copy.deepcopy(model)
    model_copy = model_copy.cpu()

    return model_copy


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python ema_merge.py <checkpoint1> <checkpoint2> [checkpoint3...] <output_dir>"
        )
        sys.exit(1)

    checkpoint_dirs = sys.argv[1:-1]
    output_dir = sys.argv[-1]

    # Load config from first checkpoint directory  
    config_path = Path(checkpoint_dirs[0]) / "config.json"
    if not config_path.exists():
        print(f"Error: config.json not found in {checkpoint_dirs[0]}")
        sys.exit(1)

    print(f"Merging {len(checkpoint_dirs)} checkpoints to {output_dir}")
    
    # Create output directory structure
    import shutil
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy config files from first checkpoint
    for filename in ["config.json", "config.yaml", "data_paths.txt"]:
        src_path = Path(checkpoint_dirs[0]) / filename
        if src_path.exists():
            shutil.copy2(src_path, output_path / filename)
            print(f"Copied {filename}")
    
    # Merge model_and_optim directories (shard-level)
    model_and_optim_dirs = [str(Path(d) / "model_and_optim") for d in checkpoint_dirs]
    output_model_and_optim = str(output_path / "model_and_optim")
    model_successful, model_skipped = 0, 0
    
    # Debug: Check what files are in the first model_and_optim directory
    first_model_optim_dir = Path(model_and_optim_dirs[0])
    if first_model_optim_dir.exists():
        distcp_files = list(first_model_optim_dir.glob("*.distcp"))
        pt_files = list(first_model_optim_dir.glob("*.pt"))
        print(f"Debug: Found {len(distcp_files)} .distcp files and {len(pt_files)} .pt files in {first_model_optim_dir}")
        if distcp_files:
            print(f"  First few .distcp files: {[f.name for f in distcp_files[:3]]}")
        if pt_files:
            print(f"  First few .pt files: {[f.name for f in pt_files[:3]]}")
    else:
        print(f"❌ model_and_optim directory not found: {first_model_optim_dir}")
    
    print("Merging model_and_optim...")
    
    # Check if we have .distcp files (distributed checkpoints)
    if distcp_files:
        print("⚠️  Detected .distcp files - these are distributed checkpoints that cannot be merged at shard level")
        print("   .distcp files require PyTorch's distributed checkpoint APIs for proper loading")
        print("   For EMA merging of distributed checkpoints, use ema_merge_checkpoints() with a model instance")
        print("   Copying model_and_optim directory from first checkpoint as fallback...")
        import shutil
        output_model_and_optim_path = output_path / "model_and_optim"
        if output_model_and_optim_path.exists():
            shutil.rmtree(output_model_and_optim_path)
        shutil.copytree(Path(model_and_optim_dirs[0]), output_model_and_optim_path)
        print("   model_and_optim directory copied (distributed checkpoints require full merge)")
        model_successful, model_skipped = 0, 0
    elif pt_files:
        print("Merging .pt files in model_and_optim...")
        try:
            model_successful, model_skipped = ema_merge_shards_generic(
                checkpoint_dirs=model_and_optim_dirs,
                output_dir=output_model_and_optim,
                file_pattern="*.pt",
                ema_decay=0.999,
                save_overwrite=True
            )
        except Exception as e:
            print(f"❌ Error merging model_and_optim .pt files: {e}")
            print("Copying model_and_optim directory instead...")
            import shutil
            output_model_and_optim_path = output_path / "model_and_optim"
            if output_model_and_optim_path.exists():
                shutil.rmtree(output_model_and_optim_path)
            shutil.copytree(Path(model_and_optim_dirs[0]), output_model_and_optim_path)
            print("Copied model_and_optim directory (merge failed)")
            model_successful, model_skipped = 0, 0
    else:
        print("No mergeable files found in model_and_optim, copying directory...")
        import shutil
        output_model_and_optim_path = output_path / "model_and_optim"
        if output_model_and_optim_path.exists():
            shutil.rmtree(output_model_and_optim_path)
        shutil.copytree(Path(model_and_optim_dirs[0]), output_model_and_optim_path)
        print("Copied model_and_optim directory")
        model_successful, model_skipped = 0, 0
    
    # Merge train directories (if they exist and have mergeable files)
    train_dirs = [str(Path(d) / "train") for d in checkpoint_dirs]
    first_train_dir = Path(train_dirs[0])
    train_successful, train_skipped = 0, 0
    
    if first_train_dir.exists():
        output_train_dir = str(output_path / "train")
        
        # Check for .distcp files first (distributed checkpoints)
        distcp_files = list(first_train_dir.glob("*.distcp"))
        if distcp_files:
            print("Merging train directory distcp shards...")
            train_distcp_successful, train_distcp_skipped = ema_merge_shards_generic(
                checkpoint_dirs=train_dirs,
                output_dir=output_train_dir,
                file_pattern="*.distcp",
                ema_decay=0.999,
                save_overwrite=True
            )
            train_successful += train_distcp_successful
            train_skipped += train_distcp_skipped
        
        # Check for .pt files (rank files)
        pt_files = list(first_train_dir.glob("*.pt"))
        if pt_files:
            print("Merging train directory pt shards...")
            train_pt_successful, train_pt_skipped = ema_merge_shards_generic(
                checkpoint_dirs=train_dirs,
                output_dir=output_train_dir,
                file_pattern="*.pt",
                ema_decay=0.999,
                save_overwrite=True
            )
            train_successful += train_pt_successful
            train_skipped += train_pt_skipped
        
        # If no mergeable files found, just copy the directory
        if not distcp_files and not pt_files:
            output_train_dir_path = output_path / "train"
            if output_train_dir_path.exists():
                shutil.rmtree(output_train_dir_path)
            shutil.copytree(first_train_dir, output_train_dir_path)
            print("Copied train directory (no shards to merge)")
    
    print(f"✅ EMA merge completed! Output saved to {output_dir}")
    print(f"   - Config files copied")
    
    # Report model_and_optim results
    if model_successful > 0:
        if model_skipped > 0:
            print(f"   - model_and_optim: {model_successful} shards merged, {model_skipped} skipped")
        else:
            print(f"   - model_and_optim: {model_successful} shards merged")
    else:
        # Check if it was because of .distcp files
        first_model_optim_dir = Path(model_and_optim_dirs[0])
        if first_model_optim_dir.exists() and list(first_model_optim_dir.glob("*.distcp")):
            print(f"   - model_and_optim: directory copied (distributed checkpoints require full merge)")
        else:
            print(f"   - model_and_optim: directory copied (no mergeable shards found)")
    
    if (output_path / "train").exists():
        if train_successful > 0:
            if train_skipped > 0:
                print(f"   - train: {train_successful} shards merged, {train_skipped} skipped (corrupted)")
            else:
                print(f"   - train: {train_successful} shards merged")
        else:
            print(f"   - train: directory copied")
