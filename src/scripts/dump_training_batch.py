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

import bettermap
import torch
from cached_path import cached_path

from olmo_core.data import (
    DataCollator,
    NumpyFSLDataLoader,
    NumpyFSLDatasetBase,
    NumpyFSLDatasetConfig,
)
from olmo_core.data.mixes import DataMix
from olmo_core.data.tokenizer import TokenizerConfig
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


def verify_paths_match_mix(
    data_paths: List[str], mix_name: str, tokenizer: TokenizerConfig, mix_base_dir: str
) -> bool:
    """Verify that data_paths.txt matches the paths from the mix in config.json."""
    mix = DataMix(mix_name)
    assert tokenizer.identifier is not None
    mix_paths, _ = mix.build(mix_base_dir, tokenizer.identifier)

    if len(data_paths) != len(mix_paths):
        log.error(
            f"Path count mismatch: data_paths.txt has {len(data_paths)} paths, mix has {len(mix_paths)}"
        )
        return False

    for i, (actual, expected) in enumerate(zip(data_paths, mix_paths)):
        if normalize_path(actual) != normalize_path(expected):
            log.error(f"Path mismatch at index {i}:\n  Actual:   {actual}\n  Expected: {expected}")
            return False
    return True


def NumpyFSLDataLoader_getitem_monkeypatch(self: NumpyFSLDataLoader, index: int) -> Dict[str, Any]:
    """
    This is a monkey patch for the data loader that loads all instances in parallel, much faster
    than doing it sequentially would be.

    During training, we don't need this, because we have many workers. But we don't have workers
    here.
    """

    # NOTE: Make sure the logic here matches that in '_get_local_instance_indices()'

    # NOTE: 'indices' are global instance indices.
    indices = self.get_global_indices()[: self.total_size]

    # Slice up by batch.
    assert isinstance(self.dataset, NumpyFSLDatasetBase)
    instances_per_batch = self.global_batch_size // self.dataset.sequence_length
    # shape: (global num batches, global num instances per batch)
    indices = indices.reshape(-1, instances_per_batch)

    # Slice batches into micro batches for the local DP rank.
    if self.dp_world_size > 1:
        indices = indices[:, self.dp_rank :: self.dp_world_size]

    # Get instances for the batch.
    # This is the change from the original implementation.
    instances = list(
        bettermap.ordered_map_per_thread(
            lambda idx: self._get_dataset_item(int(idx)),
            indices[index],
            parallelism=10,  # default connection pool size
        )
    )

    return self.collator(instances)


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
        "--monkeypatch",
        default=False,
        action="store_true",
        help="Use the monkeypatch to make loading faster. Use at your own risk. Verify at least once or twice that it produces the same output before relying on it!",
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
    log.info(f"Loading checkpoint from {checkpoint_dir}")
    config = load_checkpoint_config(checkpoint_dir)
    data_paths = load_data_paths(checkpoint_dir)
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

    # Verify that data_paths.txt matches the mix
    tokenizer_config_dict = {
        k: v for k, v in dataset_config_dict["tokenizer"].items() if k != "_CLASS_"
    }
    tokenizer_config = TokenizerConfig(**tokenizer_config_dict)

    if not verify_paths_match_mix(
        data_paths,
        dataset_config_dict["mix"],
        tokenizer_config,
        dataset_config_dict.get("mix_base_dir", "gs://ai2-llm"),
    ):
        log.error("Mix verification failed! data_paths.txt does not match the mix from config.json")
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
    else:
        log.error(f"Unsupported dataset config class: {dataset_class_name}")
        sys.exit(1)

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
    # Keep mix and mix_base_dir - the dataset will build paths from the mix

    dataset_config = config_class.from_dict(config_fields)
    dataset = dataset_config.build()
    assert isinstance(dataset, NumpyFSLDatasetBase), f"Expected FSL dataset, got {type(dataset)}"

    # Prepare the dataset
    dataset.prepare()

    # Verify fingerprint
    if dataset.fingerprint != data_loader_state["dataset_fingerprint"]:
        log.warning(
            f"Dataset fingerprint mismatch!\n"
            f"  Checkpoint: {data_loader_state['dataset_fingerprint']}\n"
            f"  Computed:   {dataset.fingerprint}"
        )
        log.warning("This may indicate the dataset has changed since the checkpoint was created.")

    collator = DataCollator(pad_token_id=dataset.pad_token_id)

    data_loader = NumpyFSLDataLoader(
        dataset,
        collator=collator,
        global_batch_size=data_loader_config_dict["global_batch_size"],
        work_dir=args.work_dir,
        seed=data_loader_state["seed"],
        shuffle=True,
        dp_world_size=1,  # We're extracting for a single "rank"
        dp_rank=0,
    )

    if args.monkeypatch:
        if isinstance(data_loader, NumpyFSLDataLoader):
            NumpyFSLDataLoader.__getitem__ = NumpyFSLDataLoader_getitem_monkeypatch  # type: ignore
            log.info("Monkeypatch is active.")
        else:
            raise ValueError("Monkey patch is only supported for NumpyFSLDataLoader.")

    # Reshuffle to regenerate the same global indices as during training
    data_loader.reshuffle(epoch=data_loader_state["epoch"], in_memory=True)

    batch = data_loader[args.step]

    if args.output is not None:
        with open(args.output, "wb") as f:
            pickle.dump(batch, f)
    else:
        for instance in batch["input_ids"]:
            print(", ".join(str(token_id.item()) for token_id in instance))


if __name__ == "__main__":
    main()
