#!/usr/bin/env python3
"""
Script to unshard distributed checkpoints into regular PyTorch or safetensors format.

This converts checkpoints saved with save_model_and_optim_state() into unsharded
checkpoint files that can be loaded directly with torch.load() or safetensors.
"""

import logging
from pathlib import Path

import click

from olmo_core.distributed.checkpoint import UnshardStrategy, unshard_checkpoint
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


@click.command()
@click.argument(
    "checkpoint_dir",
    type=str,
)
@click.argument(
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True),
)
@click.option(
    "--optim/--no-optim",
    default=None,
    help="Whether to unshard optimizer state (default: True for PyTorch format, False for safetensors)",
)
@click.option(
    "--safetensors",
    is_flag=True,
    default=False,
    help="Save with safetensors format instead of PyTorch format",
)
@click.option(
    "--strategy",
    type=click.Choice(["one_file", "one_file_per_tensor", "chunks"]),
    default="one_file",
    help="Unsharding strategy to use",
)
@click.option(
    "--chunk-size",
    type=int,
    default=None,
    help="Chunk size in bytes for 'chunks' strategy (required when using chunks strategy)",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Overwrite existing files in output directory",
)
@click.option(
    "--pre-download",
    is_flag=True,
    default=False,
    help="Pre-download remote checkpoint files before reading (useful for cloud storage)",
)
@click.option(
    "--work-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="Working directory for caching files",
)
@click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="Do not show progress messages",
)
def main(
    checkpoint_dir: str,
    output_dir: str,
    optim: bool | None,
    safetensors: bool,
    strategy: str,
    chunk_size: int | None,
    overwrite: bool,
    pre_download: bool,
    work_dir: str | None,
    quiet: bool,
):
    """
    Unshard a distributed checkpoint into regular PyTorch or safetensors format.

    CHECKPOINT_DIR: Path or URL to the sharded checkpoint directory

    OUTPUT_DIR: Local directory where unsharded checkpoint will be saved
    """
    # Build unshard strategy
    unshard_strategy: UnshardStrategy
    if strategy == "one_file":
        unshard_strategy = UnshardStrategy.one_file()
    elif strategy == "one_file_per_tensor":
        unshard_strategy = UnshardStrategy.one_file_per_tensor()
    elif strategy == "chunks":
        if chunk_size is None:
            raise click.UsageError("--chunk-size is required when using 'chunks' strategy")
        unshard_strategy = UnshardStrategy.chunks(chunk_size)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if not quiet:
        log.info(f"Unsharding checkpoint from: {checkpoint_dir}")
        log.info(f"Output directory: {output_dir}")
        log.info(f"Strategy: {strategy}")
        log.info(f"Format: {'safetensors' if safetensors else 'PyTorch'}")
        if optim is not None:
            log.info(f"Include optimizer: {optim}")

    # Unshard the checkpoint
    model_path, optim_path = unshard_checkpoint(
        dir=checkpoint_dir,
        target_dir=output_dir,
        optim=optim,
        save_overwrite=overwrite,
        use_safetensors=safetensors,
        unshard_strategy=unshard_strategy,
        pre_download=pre_download,
        work_dir=work_dir,
        quiet=quiet,
    )

    if not quiet:
        log.info("\nUnsharding complete!")
        log.info(f"Model checkpoint: {model_path}")
        if optim_path is not None:
            log.info(f"Optimizer checkpoint: {optim_path}")

        # Display file sizes if they're files
        if isinstance(model_path, Path) and model_path.is_file():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            log.info(f"Model size: {size_mb:.2f} MB")
        elif isinstance(model_path, Path) and model_path.is_dir():
            total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
            size_mb = total_size / (1024 * 1024)
            log.info(
                f"Total model size: {size_mb:.2f} MB across {len(list(model_path.rglob('*')))} files"
            )

        if optim_path is not None and isinstance(optim_path, Path) and optim_path.is_file():
            size_mb = optim_path.stat().st_size / (1024 * 1024)
            log.info(f"Optimizer size: {size_mb:.2f} MB")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
