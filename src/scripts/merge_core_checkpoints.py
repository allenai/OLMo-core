import gc
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict, List

import click
import torch
from torch.distributed.checkpoint import TensorStorageMetadata

from olmo_core.data.utils import chunked
from olmo_core.distributed.checkpoint import (
    get_checkpoint_metadata,
    load_state_dict,
    save_state_dict,
)
from olmo_core.io import join_path
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def process_checkpoint_key(
    path: str,
    key: str,
    meta: Any,
    merged_state_dict: Dict[str, Any],
    checkpoint_index: int,
    total_checkpoints: int,
    lock: Lock,
) -> None:
    """Process a single key from a checkpoint, loading and merging it into the merged state dict."""
    if not isinstance(meta, TensorStorageMetadata):
        with lock:
            if key in merged_state_dict:
                log.info(
                    f"Skipping non-tensor '{key}' from checkpoint {checkpoint_index} because we already have a value..."
                )
            else:
                log.info(f"Loading non-tensor '{key}' from checkpoint {checkpoint_index}...")
                mini_state_dict = {key: None}
                load_state_dict(path, mini_state_dict)
                merged_state_dict[key] = mini_state_dict[key]
    else:
        # Initialize the tensor in merged_state_dict if it doesn't exist yet
        with lock:
            if key not in merged_state_dict:
                merged_state_dict[key] = torch.zeros(meta.size)

        log.info(f"Loading '{key}' from checkpoint {checkpoint_index}...")
        # Load into a temporary tensor (no lock needed for local operation)
        tensor = torch.empty_like(merged_state_dict[key])
        load_state_dict(path, {key: tensor})

        # Add to merged tensor with lock protection
        with lock:
            merged_state_dict[key].add_(tensor, alpha=1 / total_checkpoints)


def merge_checkpoints(
    model_paths: List[str],
    output_path: str,
) -> None:
    # sanity check
    if any(p.rstrip("/").endswith("model_and_optim") for p in model_paths):
        raise ValueError("Checkpoint paths must not end in 'model_and_optim'")
    if output_path.rstrip("/").endswith("model_and_optim"):
        raise ValueError("Output path must not end in 'model_and_optim'")

    # merge checkpoints
    checkpoint_paths = [join_path(p, "model_and_optim") for p in model_paths]
    checkpoint_metadata = [get_checkpoint_metadata(path) for path in checkpoint_paths]
    merged_state_dict: Dict[str, Any] = {}
    lock = Lock()
    total_checkpoints = len(checkpoint_paths)

    for i, (path, metadata) in enumerate(zip(checkpoint_paths, checkpoint_metadata)):
        # Process all keys in this checkpoint in parallel using a thread pool
        max_workers = min(os.cpu_count() or 1, 32)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    process_checkpoint_key,
                    path,
                    key,
                    meta,
                    merged_state_dict,
                    i,
                    total_checkpoints,
                    lock,
                )
                for key, meta in metadata.state_dict_metadata.items()
            ]
            # Wait for all tasks to complete and propagate any exceptions
            for future in futures:
                future.result()
        gc.collect()

    # create the output dir
    log.info(f"Saving merged checkpoint to {output_path}...")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    def ignore_subdir(path: str, files):
        if os.path.basename(path) == "model_and_optim":
            return files
        else:
            return []

    shutil.copytree(model_paths[0], output_path, ignore=ignore_subdir)
    save_state_dict(join_path(output_path, "model_and_optim"), merged_state_dict)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--model",
    "-m",
    "model_paths",
    multiple=True,
    required=True,
    help="Model checkpoint path. Should be specified multiple times for different checkpoints",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    help="Output directory for the merged checkpoint",
)
def main(model_paths: tuple, output_path: str):
    """
    Merge OLMo-core model checkpoints by averaging their weights.

    Weights are accumulated in float32 for numerical stability, then converted
    back to the original dtype of the checkpoints.

    Optimizer state is taken from the first model in the list.

    Examples:

    \b
    # Merge different local model checkpoints
    python merge_core_checkpoints.py \\
        --model gs://ai2-llm/checkpoints/stego32-highlr-filter3/step656000 \\
        --model gs://ai2-llm/checkpoints/stego32-highlr-filter3/step655000 \\
        --output ./merged_checkpoint
    """
    merge_checkpoints(model_paths=list(model_paths), output_path=output_path)


if __name__ == "__main__":
    prepare_cli_environment()
    main()
