#!/usr/bin/env python3
"""
Inspect the first batch produced by the dataset and data loader configuration.

This script builds the dataset and data loader from the OLMo3-32B-long-context.py
configuration and prints information about the first batch.
"""

import argparse
import tempfile
from pathlib import Path
from typing import Any, Dict

import torch

from olmo_core.data import (
    DataCollator,
    NumpyDataLoaderConfig,
    NumpyPackedFSLDatasetConfig,
)
from olmo_core.data.numpy_dataset import InstanceFilterConfig
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.internal.experiment import CommonComponents, DataComponents

# Constants from OLMo3-32B-long-context.py
SEQUENCE_LENGTH = 64 * 1024  # 64k seq len
GLOBAL_BATCH_SIZE = 8 * 1024 * 1024  # ~8M tokens


def build_data_components(
    common: CommonComponents,
    intra_document_masking: bool = True,
    include_instance_filter: bool = False,
) -> DataComponents:
    """
    Default dataset and data loader configurations. Constructs a simple FSL dataset and data loader
    configuration with default settings.
    """
    dataset_config = NumpyPackedFSLDatasetConfig.glob(
        "gs://ai2-llm/preprocessed/tylerr/lc-reshard-final-cleaned/v0.1/allenai/dolma2-tokenizer/*.npy",
        tokenizer=common.tokenizer,
        work_dir=common.work_dir,
        sequence_length=common.max_sequence_length,
        generate_doc_lengths=intra_document_masking,  # enables intra-document masking
        source_group_size=8,
        source_permutation_seed=123,
        instance_filter_config=None
        if not include_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=common.global_batch_size, seed=34521, num_workers=8
    )

    return DataComponents(dataset=dataset_config, data_loader=data_loader_config)


def create_mock_common_components(work_dir: str) -> CommonComponents:
    """Create a minimal CommonComponents object for building the dataset."""
    return CommonComponents(
        run_name="inspect-batch",
        root_dir="gs://ai2-llm",  # Not used for this dataset config
        work_dir=work_dir,
        save_folder="",  # Not used
        launch=None,  # Not used
        tokenizer=TokenizerConfig.dolma2(),
        max_sequence_length=SEQUENCE_LENGTH,
        global_batch_size=GLOBAL_BATCH_SIZE,
    )


def inspect_batch(
    work_dir: str,
    intra_document_masking: bool = True,
    include_instance_filter: bool = False,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """
    Build the dataset and data loader, then get and inspect the first batch.

    Args:
        work_dir: Working directory for dataset caching
        intra_document_masking: Whether to enable intra-document masking
        include_instance_filter: Whether to include instance filtering
        num_workers: Number of data loader workers (0 for single-threaded)

    Returns:
        The first batch dictionary
    """
    # Create mock common components
    common = create_mock_common_components(work_dir)

    # Build data components using the same function as the training script
    data_components = build_data_components(
        common,
        intra_document_masking=intra_document_masking,
        include_instance_filter=include_instance_filter,
    )

    # Override num_workers if specified
    if num_workers is not None:
        data_components.data_loader.num_workers = num_workers

    # Build the dataset
    print("Building dataset...")
    dataset = data_components.dataset.build()
    print(f"Dataset type: {type(dataset).__name__}")

    # Prepare the dataset (must be called before accessing len() or other properties)
    print("Preparing dataset...")
    dataset.prepare()
    print(f"Dataset fingerprint: {dataset.fingerprint}")
    print(f"Dataset length: {len(dataset):,} instances")

    # Build the data loader
    print("Building data loader...")
    collator = DataCollator(pad_token_id=dataset.pad_token_id)
    data_loader = data_components.data_loader.build(
        dataset=dataset,
        collator=collator,
        dp_process_group=None,  # Single rank
    )
    print(f"Data loader type: {type(data_loader).__name__}")
    print(f"Total batches: {data_loader.total_batches:,}")

    # Reshuffle to prepare for iteration
    print("Reshuffling data loader...")
    data_loader.reshuffle(epoch=0)

    # Get the first batch
    print("\nGetting first batch...")
    batch = data_loader[0]

    # Print batch information
    print("\n" + "=" * 80)
    print("BATCH INFORMATION")
    print("=" * 80)
    print(f"\nBatch keys: {list(batch.keys())}")

    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"\n{key}:")
            print(f"  Shape: {value.shape}")
            print(f"  Dtype: {value.dtype}")
            print(f"  Device: {value.device}")
            if value.numel() > 0:
                print(f"  Min: {value.min().item()}")
                print(f"  Max: {value.max().item()}")
                if value.dtype in (torch.int32, torch.int64, torch.long):
                    print(f"  Mean: {value.float().mean().item():.2f}")
                else:
                    print(f"  Mean: {value.mean().item():.2f}")
            # Show a small sample if it's not too large
            if value.numel() <= 100:
                print(f"  Values: {value}")
            elif value.dim() == 1:
                print(f"  First 10 values: {value[:10].tolist()}")
            elif value.dim() == 2:
                print(f"  First row (first 10): {value[0, :10].tolist()}")
        else:
            print(f"\n{key}: {type(value).__name__} = {value}")

    # Reset the data loader
    data_loader.reset()

    return batch


def main():
    parser = argparse.ArgumentParser(
        description="Inspect the first batch from the OLMo3-32B-long-context dataset configuration"
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=None,
        help="Working directory for dataset caching (default: temporary directory)",
    )
    parser.add_argument(
        "--no-intra-document-masking",
        action="store_true",
        help="Disable intra-document masking",
    )
    parser.add_argument(
        "--include-instance-filter",
        action="store_true",
        help="Include instance filtering",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loader workers (default: 0 for single-threaded)",
    )
    parser.add_argument(
        "--save-batch",
        type=str,
        default=None,
        help="Path to save the batch as a pickle file (optional)",
    )

    args = parser.parse_args()

    # Use temporary directory if not specified
    if args.work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="olmo_batch_inspect_")
        print(f"Using temporary work directory: {work_dir}")
    else:
        work_dir = args.work_dir
        Path(work_dir).mkdir(parents=True, exist_ok=True)

    try:
        batch = inspect_batch(
            work_dir=work_dir,
            intra_document_masking=not args.no_intra_document_masking,
            include_instance_filter=args.include_instance_filter,
            num_workers=args.num_workers,
        )

        if args.save_batch:
            import pickle

            print(f"\nSaving batch to {args.save_batch}...")
            with open(args.save_batch, "wb") as f:
                pickle.dump(batch, f)
            print("Batch saved successfully!")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
