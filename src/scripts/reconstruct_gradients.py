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
from safetensors import safe_open
from safetensors.torch import save_file

from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def load_shard(filepath: Path) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Load gradient shard from safetensors and extract metadata."""
    with safe_open(str(filepath), framework="pt", device="cpu") as f:
        grad = f.get_tensor("gradient")
        file_metadata = f.metadata() or {}

    metadata = {
        "shard_dim": int(file_metadata["shard_dim"]) if "shard_dim" in file_metadata else 0,
        "full_shape": tuple(json.loads(file_metadata["full_shape"]))
        if "full_shape" in file_metadata
        else None,
    }

    return grad, metadata


def infer_config_from_files(param_to_files: Dict[str, Dict[int, Path]]) -> Dict[str, Any]:
    """Infer distributed configuration from the gradient files themselves."""
    if not param_to_files:
        raise ValueError("No gradient files found")

    # Get any parameter's file dict to infer config
    sample_param_files = next(iter(param_to_files.values()))

    # shard_degree = number of ranks that have shards
    shard_degree = len(sample_param_files)

    # world_size = max rank + 1
    world_size = max(sample_param_files.keys()) + 1

    # num_replicas = world_size / shard_degree (for HSDP)
    num_replicas = world_size // shard_degree if shard_degree > 0 else 1

    return {
        "world_size": world_size,
        "shard_degree": shard_degree,
        "num_replicas": num_replicas,
        "parallel_type": "hsdp" if num_replicas > 1 else "fsdp",
    }


def collect_gradient_files(grad_dir: Path, step: int) -> Dict[str, Dict[int, Path]]:
    """Collect and group gradient files by parameter: {param_name: {rank: filepath}}."""
    step_dir = grad_dir / f"step{step}"
    if not step_dir.exists():
        raise ValueError(f"Step directory not found: {step_dir}")

    grad_files = list(step_dir.glob("rank*.safetensors"))
    if not grad_files:
        raise ValueError(f"No gradient files found for step {step} in {step_dir}")

    params: Dict[str, Dict[int, Path]] = defaultdict(dict)
    for filepath in grad_files:
        # Parse filename: rank{N}_{param_name}.safetensors
        parts = filepath.stem.split("_", 1)
        if len(parts) != 2:
            log.warning(f"Skipping malformed filename: {filepath.name}")
            continue

        try:
            rank = int(parts[0].replace("rank", ""))
            param_name = parts[1]
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
        for rank in required_ranks:
            grad, metadata = load_shard(rank_files[rank])
            shards.append(grad)

        # Get metadata from first shard (all shards should have same metadata)
        _, metadata = load_shard(rank_files[required_ranks[0]])
        shard_dim = metadata["shard_dim"]
        expected_shape = metadata["full_shape"]

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

        # Verify against other replicas (only for HSDP)
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
@click.option(
    "--grad-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory containing gradient dumps",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory where reconstructed gradients will be saved",
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

    Reads from: {grad_dir}/step{N}/rank*.safetensors
    Saves to:   {output_dir}/step{N}/*.safetensors

    Example:

        python src/scripts/reconstruct_gradients.py \\
            --grad-dir /tmp/tutorial-run-4gpus/grad_dumper \\
            --output-dir /tmp/full_gradients \\
            --step 2
    """
    param_to_files = collect_gradient_files(grad_dir, step)

    config = infer_config_from_files(param_to_files)
    shard_degree = config["shard_degree"]
    num_replicas = config["num_replicas"]

    if not quiet:
        log.info("Inferred configuration from gradient files:")
        log.info(f"  Parallel type: {config['parallel_type']}")
        log.info(f"  World size: {config['world_size']}")
        log.info(f"  Shard degree: {shard_degree}")
        log.info(f"  Num replicas: {num_replicas}")
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

    # Create step-specific subdirectory
    step_output_dir = output_dir / f"step{step}"
    step_output_dir.mkdir(exist_ok=True, parents=True)

    if not skip_individual:
        for param_name, grad in gradients.items():
            output_file = step_output_dir / f"{param_name}.safetensors"
            save_file({"gradient": grad}, str(output_file))

    all_grads_file = step_output_dir / "all_gradients.safetensors"
    save_file(gradients, str(all_grads_file))

    summary = {
        "step": step,
        "source_metadata": config,
        "num_parameters": len(gradients),
        "parameter_names": list(gradients.keys()),
        "parameter_shapes": {name: list(grad.shape) for name, grad in gradients.items()},
        "parameter_norms": {name: float(grad.norm().item()) for name, grad in gradients.items()},
    }

    summary_file = step_output_dir / "metadata.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    if not quiet:
        log.info(f"Reconstructed gradients saved to: {step_output_dir}")
        if not skip_individual:
            log.info("Individual files: {param_name}.safetensors")
        log.info(f"All gradients: {all_grads_file.name}")
        log.info(f"Metadata: {summary_file.name}")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
