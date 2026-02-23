#!/usr/bin/env python3
"""
Download the exact training batch for a given checkpoint and step number.

This script reconstructs the dataset and data loader state from a training checkpoint,
then extracts the exact batch that we trained on at the specified step.
"""

import argparse
import json
import logging
import pickle
import sys
from typing import Any, Dict, List

import torch
from cached_path import cached_path

from olmo_core.data import (
    DataCollator,
    NumpyFSLDataLoader,
    NumpyFSLDatasetBase,
    NumpyFSLDatasetConfig,
    NumpyPackedFSLDatasetConfig,
)
from olmo_core.io import normalize_path

log = logging.getLogger(__name__)


def load_checkpoint_config(checkpoint_dir: str) -> Dict[str, Any]:
    """Load config.json from checkpoint directory."""
    config_path = f"{checkpoint_dir}/config.json"
    with open(cached_path(config_path)) as f:
        return json.load(f)


def load_data_paths(checkpoint_dir: str) -> List[str]:
    """Load data_paths.txt from checkpoint directory."""
    data_paths_file = f"{checkpoint_dir}/data_paths.txt"
    with open(cached_path(data_paths_file)) as f:
        return [line.strip() for line in f if line.strip()]


def load_trainer_state(checkpoint_dir: str) -> Dict[str, Any]:
    """Load train/rank0.pt from checkpoint directory."""
    trainer_state_path = f"{checkpoint_dir}/train/rank0.pt"
    return torch.load(cached_path(trainer_state_path), weights_only=False)


def verify_paths_match(
    data_paths: List[str], dataset_paths: List[str]
) -> bool:
    """Verify that data_paths.txt matches the paths from the reconstructed dataset."""
    if len(data_paths) != len(dataset_paths):
        log.error(
            f"Path count mismatch: data_paths.txt has {len(data_paths)} paths, "
            f"dataset has {len(dataset_paths)}"
        )
        return False

    for i, (actual, expected) in enumerate(zip(data_paths, dataset_paths)):
        if normalize_path(actual) != normalize_path(expected):
            log.error(f"Path mismatch at index {i}:\n  Actual:   {actual}\n  Expected: {expected}")
            return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download exact training batch for a specific training step. "
            "The checkpoint is only used to get the dataset configuration (paths, tokenizer, seed, etc.)."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path or URL to checkpoint directory (e.g., gs://ai2-llm/checkpoints/OLMo25-1B/step231200/)."
        ),
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Step number to extract tokens for",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        help="Output path. Will write a pickle file. If not given, it writes the token ids to stdout.",
        default=None,
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="/tmp/download_training_tokens",
        help="Working directory for dataset cache",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    checkpoint_dir = normalize_path(args.checkpoint)

    # Load checkpoint files
    log.info(f"Loading checkpoint config from {checkpoint_dir}")
    config = load_checkpoint_config(checkpoint_dir)

    log.info("Loading data paths from checkpoint")
    data_paths = load_data_paths(checkpoint_dir)

    log.info("Loading trainer state from checkpoint")
    trainer_state = load_trainer_state(checkpoint_dir)

    # Get data loader state
    data_loader_state = trainer_state["data_loader"]

    if args.step < trainer_state["global_step"]:
        log.warning(
            f"Requested step {args.step} is before checkpoint step {trainer_state['global_step']}. "
            f"Make sure this checkpoint has the correct dataset configuration for that earlier step."
        )

    # Extract configuration
    dataset_config_dict = config["dataset"]
    data_loader_config_dict = config["data_loader"]

    # Verify FSL dataset
    if data_loader_state["dataset_type"] != "fsl":
        log.error(f"Only FSL datasets are supported, got: {data_loader_state['dataset_type']}")
        sys.exit(1)

    # Reconstruct dataset
    dataset_class_name = dataset_config_dict["_CLASS_"]
    if dataset_class_name == "olmo_core.data.numpy_dataset.NumpyDatasetConfig":
        log.warning(
            "Dataset config class is 'NumpyDatasetConfig' (base class). Assuming 'NumpyFSLDatasetConfig'."
        )
        config_class = NumpyFSLDatasetConfig
    elif dataset_class_name == "olmo_core.data.numpy_dataset.NumpyFSLDatasetConfig":
        config_class = NumpyFSLDatasetConfig
    elif dataset_class_name == "olmo_core.data.numpy_dataset.NumpyPackedFSLDatasetConfig":
        config_class = NumpyPackedFSLDatasetConfig
    else:
        log.error(f"Unsupported dataset config class: {dataset_class_name}")
        sys.exit(1)

    log.info(f"Using dataset config class: {config_class.__name__}")

    # Start with the full dataset config and only modify what we need
    config_fields = dict(dataset_config_dict)

    # Override work_dir with the one from command line args
    config_fields["work_dir"] = args.work_dir

    # Clean up tokenizer dict (remove _CLASS_ field)
    if "tokenizer" in config_fields and isinstance(config_fields["tokenizer"], dict):
        config_fields["tokenizer"] = {
            k: v for k, v in config_fields["tokenizer"].items() if k != "_CLASS_"
        }

    # Remove fields that shouldn't be passed to from_dict
    config_fields.pop("name", None)
    config_fields.pop("_CLASS_", None)  # Remove _CLASS_ field, we already determined the class

    log.info("Building dataset config from checkpoint fields")
    dataset_config = config_class.from_dict(config_fields)

    log.info("Building dataset (resolving paths from mix)")
    dataset = dataset_config.build()
    assert isinstance(dataset, NumpyFSLDatasetBase), f"Expected FSL dataset, got {type(dataset)}"
    log.info(f"Dataset has {len(dataset.paths)} source paths")

    # Verify that data_paths.txt matches the reconstructed dataset paths.
    # This comparison happens after build(), so source_permutation_seed has already been applied
    # to dataset.paths, matching the order saved in data_paths.txt during training.
    log.info("Verifying data paths match")
    if not verify_paths_match(data_paths, [str(p) for p in dataset.paths]):
        log.error(
            "Path verification failed! data_paths.txt does not match the reconstructed dataset"
        )
        sys.exit(1)

    # Prepare the dataset.
    # NOTE: For NumpyPackedFSLDataset, this runs the full packing algorithm, which requires
    # downloading all source files and is very expensive. If you have access to the cached
    # packing results from the training cluster, point --work-dir to that cache directory.
    log.info(
        f"Preparing dataset (work_dir={args.work_dir}). "
        "For packed datasets this runs the packing algorithm, which can be very slow..."
    )
    dataset.prepare()
    log.info(f"Dataset prepared: {len(dataset)} instances")

    # Pre-compute source_sizes from the already-cached file_sizes to avoid a second round of
    # ~960 concurrent HEAD requests to R2 that overwhelms the connection pool. The file_sizes
    # property was already populated during prepare(), and source_sizes is just file_sizes
    # divided by item_size, but it's a separate property that would re-query every path.
    from olmo_core.data.numpy_dataset import NumpyPackedFSLDataset

    if isinstance(dataset, NumpyPackedFSLDataset):
        item_size = dataset.dtype(0).itemsize
        dataset._source_sizes = [s // item_size for s in dataset.file_sizes]

    # Verify fingerprint
    if dataset.fingerprint != data_loader_state["dataset_fingerprint"]:
        log.error(
            f"Dataset fingerprint mismatch!\n"
            f"  Checkpoint: {data_loader_state['dataset_fingerprint']}\n"
            f"  Computed:   {dataset.fingerprint}"
        )
        log.error(
            "This may indicate the dataset has changed in the code since the checkpoint was created."
        )
        sys.exit(1)

    collator = DataCollator(pad_token_id=dataset.pad_token_id)

    log.info("Building data loader")
    data_loader = NumpyFSLDataLoader(
        dataset,
        collator=collator,
        global_batch_size=data_loader_config_dict["global_batch_size"],
        work_dir=args.work_dir,
        seed=data_loader_state["seed"],
        shuffle=True,
        dp_world_size=1,  # We're extracting for a single "rank"
        dp_rank=0,
        num_threads=10,  # 10 is the default connection pool size
    )

    # Reshuffle to regenerate the same global indices as during training
    log.info(f"Reshuffling data loader (epoch={data_loader_state['epoch']})")
    data_loader.reshuffle(epoch=data_loader_state["epoch"], in_memory=True)

    log.info(f"Loading batch for step {args.step}")
    batch = data_loader[args.step]

    # Trace each instance index back to its source file(s).
    if isinstance(dataset, NumpyPackedFSLDataset):
        source_files = []
        for idx in batch["index"].tolist():
            for i, (start, end) in enumerate(dataset.source_instance_offsets):
                if start <= idx < end:
                    source_files.append([str(p) for p in dataset._source_path_groups[i]])
                    break
            else:
                source_files.append(None)
        batch["source_files"] = source_files

    if args.output is not None:
        with open(args.output, "wb") as f:
            pickle.dump(batch, f)
        log.info(f"Batch saved to {args.output}")
    else:
        for instance in batch["input_ids"]:
            print(", ".join(str(token_id.item()) for token_id in instance))


if __name__ == "__main__":
    main()
