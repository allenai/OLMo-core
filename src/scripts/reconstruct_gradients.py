#!/usr/bin/env python3
"""
Script to reconstruct full gradients from FSDP/HSDP distributed gradient dumps.
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import click
import torch

from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def load_shard(filepath: Path) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load gradient shard and extract DTensor metadata (shard_dim, full_shape, placements)."""
    grad = torch.load(filepath, map_location="cpu", weights_only=False)

    metadata = {
        "is_dtensor": False,
        "shard_dim": None,
        "full_shape": None,
        "placements": None,
    }

    # Check if it's a DTensor (FSDP/HSDP saves with _local_tensor attribute)
    if hasattr(grad, "_local_tensor"):
        local_tensor = grad._local_tensor
        metadata["is_dtensor"] = True

        # Extract placements to get shard dimension
        if hasattr(grad, "placements") and grad.placements:
            metadata["placements"] = grad.placements
            for placement in grad.placements:
                if placement.is_shard():
                    metadata["shard_dim"] = placement.dim
                    break

        # Extract full shape of unsharded parameter from spec
        if hasattr(grad, "_spec") and hasattr(grad._spec, "shape"):
            metadata["full_shape"] = tuple(grad._spec.shape)  # type: ignore[assignment]

        return local_tensor, metadata

    # Not a DTensor
    return grad, metadata


def load_config(grad_dir: Path) -> Dict[str, Any]:
    """Load gradient dumper config (world_size, shard_degree, num_replicas)."""
    config_path = grad_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {grad_dir}")

    with open(config_path) as f:
        metadata = json.load(f)

    return metadata


def collect_gradient_files(grad_dir: Path, step: int) -> Dict[str, Dict[int, Path]]:
    """Collect and group gradient files by parameter: {param_name: {rank: filepath}}."""
    grad_files = list(grad_dir.glob(f"rank*_step{step}_*.pt"))
    if not grad_files:
        raise ValueError(f"No gradient files found for step {step} in {grad_dir}")

    params: Dict[str, Dict[int, Path]] = defaultdict(dict)
    for filepath in grad_files:
        # Parse filename: rank{N}_step{S}_{param_name}.pt
        parts = filepath.stem.split("_", 2)
        if len(parts) != 3:
            log.warning(f"Skipping malformed filename: {filepath.name}")
            continue

        try:
            rank = int(parts[0].replace("rank", ""))
            file_step = int(parts[1].replace("step", ""))
            param_name = parts[2]

            if file_step == step:
                params[param_name][rank] = filepath
        except (ValueError, IndexError) as e:
            log.warning(f"Error parsing filename {filepath.name}: {e}")
            continue

    return dict(params)


def reconstruct_gradients(
    param_to_files: Dict[str, Dict[int, Path]],
    shard_degree: int,
    num_replicas: int,
    verify: bool = False,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """Reconstruct full gradients by concatenating shards from all ranks."""
    gradients = {}

    log.info(f"Reconstructing gradients from ranks 0-{shard_degree-1}...")
    for i, (param_name, rank_files) in enumerate(sorted(param_to_files.items()), 1):
        required_ranks = list(range(shard_degree))
        missing = [r for r in required_ranks if r not in rank_files]
        if missing:
            log.warning(
                f"  [{i}/{len(param_to_files)}] {param_name}: SKIP (missing ranks {missing})"
            )
            continue

        # Load shards from first replica only (ranks 0 to shard_degree-1)
        shards = []
        shard_metadata = []
        for rank in required_ranks:
            grad, metadata = load_shard(rank_files[rank])
            shards.append(grad)
            shard_metadata.append(metadata)

        shard_dim = None
        expected_shape = None
        for metadata in shard_metadata:
            if metadata["is_dtensor"] and metadata["shard_dim"] is not None:
                shard_dim = metadata["shard_dim"]
                expected_shape = metadata["full_shape"]
                break

        # Fallback to dim=0 if no DTensor metadata found (FSDP default)
        if shard_dim is None:
            if verbose:
                log.warning(
                    f"  No shard dimension found for {param_name}, defaulting to dim=0 (FSDP default)"
                )
            shard_dim = 0

        # Verify all shards have consistent shapes along all dimensions except the shard dimension
        shapes = [s.shape for s in shards]
        base_shape = list(shapes[0])
        for j, shape in enumerate(shapes[1:], 1):
            for dim_idx, (dim_size, base_size) in enumerate(zip(shape, base_shape)):
                if dim_idx != shard_dim and dim_size != base_size:
                    log.error(
                        f"  [{i}/{len(param_to_files)}] {param_name}: ERROR - "
                        f"shard {j} has inconsistent shape at dim {dim_idx}: "
                        f"{dim_size} vs {base_size}"
                    )

        full_grad = torch.cat(shards, dim=shard_dim)

        if expected_shape is not None and tuple(full_grad.shape) != expected_shape:
            log.warning(
                f"  [{i}/{len(param_to_files)}] {param_name}: "
                f"reconstructed shape {tuple(full_grad.shape)} doesn't match "
                f"expected {expected_shape}"
            )

        # Verify against other replicas
        if verify and num_replicas > 1:
            for replica_id in range(1, num_replicas):
                replica_shards = []
                replica_start_rank = replica_id * shard_degree
                for offset in range(shard_degree):
                    rank = replica_start_rank + offset
                    if rank in rank_files:
                        replica_grad, _ = load_shard(rank_files[rank])
                        replica_shards.append(replica_grad)

                if len(replica_shards) == shard_degree:
                    replica_full_grad = torch.cat(replica_shards, dim=shard_dim)
                    if not torch.allclose(full_grad, replica_full_grad, rtol=1e-5, atol=1e-8):
                        log.warning(f"  Replica {replica_id} gradients differ for {param_name}!")

        gradients[param_name] = full_grad

        if verbose:
            log.info(
                f"  [{i}/{len(param_to_files)}] {param_name}: "
                f"shape={tuple(full_grad.shape)}, "
                f"norm={full_grad.norm().item():.6e}, "
                f"mean={full_grad.mean().item():.6e}"
            )
        else:
            log.info(f"  [{i}/{len(param_to_files)}] {param_name}: {tuple(full_grad.shape)}")

    return gradients


@click.command()
@click.argument(
    "grad_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.argument(
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--step",
    type=int,
    required=True,
    help="Training step to reconstruct gradients for",
)
@click.option(
    "--verify",
    is_flag=True,
    default=False,
    help="Verify reconstruction across replicas (for HSDP)",
)
@click.option(
    "--skip-individual",
    is_flag=True,
    default=False,
    help="Skip saving individual parameter files (only save combined file)",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print detailed information including norms and means",
)
@click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="Suppress progress messages",
)
def main(
    grad_dir: Path,
    output_dir: Path,
    step: int,
    verify: bool,
    skip_individual: bool,
    verbose: bool,
    quiet: bool,
):
    """
    Reconstruct full gradients from FSDP/HSDP sharded gradient dumps.

    GRAD_DIR: Directory containing gradient dumps (with config.json)

    OUTPUT_DIR: Directory where reconstructed gradients will be saved

    Example:

        python src/scripts/reconstruct_gradients.py \\
            /tmp/tutorial-run-4gpus/grad_dumper \\
            /tmp/full_gradients \\
            --step 2
    """
    # Load configuration
    try:
        config = load_config(grad_dir)
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    shard_degree = config["shard_degree"]
    num_replicas = config["num_replicas"]

    if not quiet:
        log.info(f"Configuration: {config['parallel_type']}")
        log.info(f"World size: {config['world_size']}")
        log.info(f"Shard degree: {shard_degree}")
        log.info(f"Num replicas: {num_replicas}")

    param_to_files = collect_gradient_files(grad_dir, step)
    if not quiet:
        log.info(f"\nFound {len(param_to_files)} parameters with gradient files")

    gradients = reconstruct_gradients(
        param_to_files,
        shard_degree,
        num_replicas,
        verify=verify,
        verbose=verbose,
    )

    if not gradients:
        log.error("No gradients were successfully reconstructed!")
        sys.exit(1)

    output_dir.mkdir(exist_ok=True, parents=True)

    if not skip_individual:
        for param_name, grad in gradients.items():
            output_file = output_dir / f"step{step}_{param_name}.pt"
            torch.save(grad, output_file)

    all_grads_file = output_dir / f"step{step}_all_gradients.pt"
    torch.save(gradients, all_grads_file)

    summary = {
        "step": step,
        "source_metadata": config,
        "num_parameters": len(gradients),
        "parameter_names": list(gradients.keys()),
        "parameter_shapes": {name: list(grad.shape) for name, grad in gradients.items()},
        "parameter_norms": {name: float(grad.norm().item()) for name, grad in gradients.items()},
    }

    summary_file = output_dir / f"step{step}_metadata.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    if not quiet:
        log.info(f"Reconstructed gradients saved to: {output_dir}")
        if not skip_individual:
            log.info(f"Individual files: step{step}_{{param_name}}.pt")
        log.info(f"All gradients: {all_grads_file.name}")
        log.info(f"Metadata: {summary_file.name}")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
