import gc
import logging
import os
import shutil
import time
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
    for i, (path, metadata) in enumerate(zip(checkpoint_paths, checkpoint_metadata)):
        # Separate non-tensor and tensor keys upfront
        non_tensor_keys = []
        tensor_keys = []

        for key, meta in metadata.state_dict_metadata.items():
            if not isinstance(meta, TensorStorageMetadata):
                non_tensor_keys.append((key, meta))
            else:
                tensor_keys.append((key, meta))

        # Load all non-tensor keys in one chunk
        if non_tensor_keys:
            non_tensor_batch_dict = {}
            for key, meta in non_tensor_keys:
                if key in merged_state_dict:
                    log.info(
                        f"Skipping non-tensor '{key}' from checkpoint {i} because we already have a value..."
                    )
                else:
                    non_tensor_batch_dict[key] = None

            if non_tensor_batch_dict:
                log.info(
                    f"Loading {len(non_tensor_batch_dict)} non-tensor keys from checkpoint {i}..."
                )
                load_state_dict(path, non_tensor_batch_dict)
                for key in non_tensor_batch_dict:
                    merged_state_dict[key] = non_tensor_batch_dict[key]

        # Process tensor keys in batches of 128
        total_tensor_keys = len(tensor_keys)
        keys_processed = 0
        start_time = time.time()

        for batch in chunked(tensor_keys, 128):
            # Initialize tensors in merged_state_dict
            for key, meta in batch:
                if key not in merged_state_dict:
                    merged_state_dict[key] = torch.zeros(meta.size)

            # Build batch dict with temporary tensors
            tensor_batch_dict = {}
            for key, meta in batch:
                tensor_batch_dict[key] = torch.empty_like(merged_state_dict[key])

            # Load all tensors in one call
            load_state_dict(path, tensor_batch_dict)

            # Add to merged_state_dict
            for key, meta in batch:
                merged_state_dict[key].add_(tensor_batch_dict[key], alpha=1 / len(checkpoint_paths))

            # Update progress and log
            keys_processed += len(batch)
            elapsed_time = time.time() - start_time
            keys_remaining = total_tensor_keys - keys_processed

            if keys_processed > 0 and elapsed_time > 0:
                time_per_key = elapsed_time / keys_processed
                eta_seconds = time_per_key * keys_remaining
                log.info(
                    f"Checkpoint {i}: processed {keys_processed}/{total_tensor_keys} tensor keys, "
                    f"ETA: {eta_seconds:.1f}s"
                )

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
