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
from typing import Any, Dict

import torch
from cached_path import cached_path

from olmo_core.data import (
    DataCollator,
    InstanceFilterConfig,
    NumpyDataLoaderBase,
    NumpyFSLDatasetBase,
    NumpyPackedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.utils import get_cumulative_document_lengths
from olmo_core.io import normalize_path

log = logging.getLogger(__name__)


# ============================================================================
# Dataset configuration from OLMo3.1-7B-hybrid-long-context.py
# ============================================================================

SEQUENCE_LENGTH = 65536
GLOBAL_BATCH_SIZE = 4 * 1024 * 1024  # ~4M tokens
DATA_LOADER_SEED = 34521

TOKENIZER = TokenizerConfig(
    vocab_size=128256,
    eos_token_id=128001,
    pad_token_id=128002,
    bos_token_id=128000,
    identifier="allenai/dolma2-tokenizer",
)

DATA_GLOB = (
    "gs://ai2-llm/preprocessed/tylerr/lc-reshard-final-cleaned/v0.1/allenai/dolma2-tokenizer/*.npy"
)


def build_dataset_config(
    work_dir: str,
    intra_document_masking: bool = True,
    include_instance_filter: bool = True,
) -> NumpyPackedFSLDatasetConfig:
    """Build the dataset config matching the training script."""
    return NumpyPackedFSLDatasetConfig.glob(
        DATA_GLOB,
        tokenizer=TOKENIZER,
        work_dir=work_dir,
        sequence_length=SEQUENCE_LENGTH,
        generate_doc_lengths=intra_document_masking,
        source_group_size=8,
        source_permutation_seed=123,
        instance_filter_config=None
        if not include_instance_filter
        else InstanceFilterConfig(
            repetition_max_period=13, repetition_min_period=1, repetition_max_count=32
        ),
    )


def load_checkpoint_config(checkpoint_dir: str) -> Dict[str, Any]:
    """Load config.json from checkpoint directory."""
    config_path = f"{checkpoint_dir}/config.json"
    with open(cached_path(config_path)) as f:
        return json.load(f)


def load_trainer_state(checkpoint_dir: str) -> Dict[str, Any]:
    """Load train/rank0.pt from checkpoint directory."""
    trainer_state_path = f"{checkpoint_dir}/train/rank0.pt"
    return torch.load(cached_path(trainer_state_path), weights_only=False)


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
    parser.add_argument(
        "--dp-rank",
        type=int,
        default=0,
        help="DP rank to extract batch for (default: 0)",
    )
    parser.add_argument(
        "--dp-world-size",
        type=int,
        default=None,
        help="Total DP world size. If not specified, uses the value from the checkpoint.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch number. If not specified, uses the value from the checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Data loader seed. If not specified, uses the value from the checkpoint or default.",
    )
    parser.add_argument(
        "--no-instance-filter",
        action="store_true",
        help="Disable instance filtering",
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
    trainer_state = load_trainer_state(checkpoint_dir)

    # Get data loader state from checkpoint
    data_loader_state = trainer_state["data_loader"]
    data_loader_config_dict = config["data_loader"]

    if args.step < trainer_state["global_step"]:
        log.warning(
            f"Requested step {args.step} is before checkpoint step {trainer_state['global_step']}. "
            f"Make sure this checkpoint has the correct dataset configuration for that earlier step."
        )

    # Build dataset using hardcoded config from training script
    log.info("Building dataset from hardcoded config...")
    dataset_config = build_dataset_config(
        work_dir=args.work_dir,
        include_instance_filter=not args.no_instance_filter,
    )
    dataset = dataset_config.build()
    assert isinstance(dataset, NumpyFSLDatasetBase), f"Expected FSL dataset, got {type(dataset)}"

    # Prepare the dataset
    log.info("Preparing dataset...")
    dataset.prepare()

    log.info(f"Dataset ready: {len(dataset)} instances, sequence_length={dataset.sequence_length}")

    # Optionally verify fingerprint
    checkpoint_fingerprint = data_loader_state.get("dataset_fingerprint")
    if checkpoint_fingerprint and dataset.fingerprint != checkpoint_fingerprint:
        log.warning(
            f"Dataset fingerprint mismatch (this may be expected if config differs):\n"
            f"  Checkpoint: {checkpoint_fingerprint}\n"
            f"  Computed:   {dataset.fingerprint}"
        )

    collator = DataCollator(pad_token_id=dataset.pad_token_id)

    # Determine DP world size
    if args.dp_world_size is not None:
        dp_world_size = args.dp_world_size
    else:
        # Try to get from checkpoint state, fall back to 1
        dp_world_size = data_loader_state.get("dp_world_size", 1)
        log.info(f"Using dp_world_size={dp_world_size} from checkpoint")

    if args.dp_rank >= dp_world_size:
        log.error(f"dp_rank ({args.dp_rank}) must be < dp_world_size ({dp_world_size})")
        sys.exit(1)

    # Determine seed
    if args.seed is not None:
        seed = args.seed
    else:
        seed = data_loader_state.get("seed", DATA_LOADER_SEED)
        log.info(f"Using seed={seed} from checkpoint")

    # Determine epoch
    if args.epoch is not None:
        epoch = args.epoch
    else:
        epoch = data_loader_state.get("epoch", 0)
        log.info(f"Using epoch={epoch} from checkpoint")

    log.info(f"Extracting batch for dp_rank={args.dp_rank}/{dp_world_size}")

    data_loader = NumpyDataLoaderBase.wrap_numpy_dataset(
        dataset,
        collator=collator,
        global_batch_size=data_loader_config_dict.get("global_batch_size", GLOBAL_BATCH_SIZE),
        work_dir=args.work_dir,
        seed=seed,
        shuffle=True,
        dp_world_size=dp_world_size,
        dp_rank=args.dp_rank,
        num_threads=16,
    )

    # Reshuffle to regenerate the same global indices as during training
    log.info(f"Reshuffling for epoch {epoch}...")
    data_loader.reshuffle(epoch=epoch, in_memory=True)

    log.info(f"Total batches in epoch: {data_loader.total_batches}")

    if args.step >= data_loader.total_batches:
        log.error(f"Step {args.step} out of range (total batches: {data_loader.total_batches})")
        sys.exit(1)

    batch = data_loader[args.step]

    # Print detailed batch info
    print("\n" + "=" * 80)
    print(f"BATCH INFO: Step={args.step}, DP Rank={args.dp_rank}/{dp_world_size}, Epoch={epoch}")
    print("=" * 80)

    input_ids = batch["input_ids"]
    print("\n[Shapes]")
    print(f"  input_ids:    {tuple(input_ids.shape)} (dtype={input_ids.dtype})")

    if "label_mask" in batch:
        label_mask = batch["label_mask"]
        print(f"  label_mask:   {tuple(label_mask.shape)} (dtype={label_mask.dtype})")
        # Count trainable tokens per instance
        trainable_per_instance = label_mask.sum(dim=-1).tolist()
        print(f"  trainable tokens per instance: {trainable_per_instance[:5]}... (first 5)")
        print(f"  total trainable tokens: {label_mask.sum().item()}")

    if "doc_lens" in batch:
        doc_lens = batch["doc_lens"]
        print(f"  doc_lens:     {tuple(doc_lens.shape)} (dtype={doc_lens.dtype})")

        # Compute cu_seqlens for the batch
        cu_seqlens = get_cumulative_document_lengths(doc_lens)
        print(f"  cu_seqlens:   {tuple(cu_seqlens.shape)} (dtype={cu_seqlens.dtype})")
        print(f"  cu_seqlens values: {cu_seqlens[:20].tolist()}... (first 20)")

        # Show doc counts per instance
        docs_per_instance = (doc_lens > 0).sum(dim=-1).tolist()
        print(f"  docs per instance: {docs_per_instance[:10]}... (first 10)")

    if "max_doc_lens" in batch:
        print(f"  max_doc_lens: {batch['max_doc_lens'][:10]}... (first 10)")

    if "index" in batch:
        indices = batch["index"]
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        print("\n[Instance Indices]")
        print(f"  {indices[:10]}... (first 10 of {len(indices)})")

    # Instance filter info
    if "instance_mask" in batch:
        instance_mask = batch["instance_mask"]
        print("\n[Instance Filter]")
        print(f"  instance_mask shape: {tuple(instance_mask.shape)}")

        num_passed = instance_mask.sum().item()
        num_failed = (~instance_mask).sum().item()
        total = instance_mask.numel()

        print(f"  Passed filter: {num_passed}/{total} ({100 * num_passed / total:.1f}%)")
        print(f"  FAILED filter: {num_failed}/{total} ({100 * num_failed / total:.1f}%)")

        if num_failed > 0:
            failed_indices = torch.where(~instance_mask)[0].tolist()
            print(f"  Failed instance indices (within batch): {failed_indices}")
    else:
        print("\n[Instance Filter]")
        print("  No instance_mask in batch (instance filter not enabled or all passed)")

    # Metadata if present
    if "metadata" in batch:
        print("\n[Metadata]")
        print(f"  {batch['metadata'][:3]}... (first 3)")

    print("\n" + "=" * 80)

    # Check for EOS tokens (important for document boundary detection)
    print("\n[EOS Token Analysis]")
    eos_token_id = TOKENIZER.eos_token_id
    print(f"  EOS token ID: {eos_token_id}")
    eos_counts = (input_ids == eos_token_id).sum(dim=-1).tolist()
    print(f"  EOS tokens per instance: {eos_counts}")

    # Find EOS positions in first instance
    eos_positions = (input_ids[0] == eos_token_id).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        print(f"  EOS positions in first instance: {eos_positions[:20].tolist()}... (first 20)")
        print(f"  Total EOS in first instance: {len(eos_positions)}")
    else:
        print("  WARNING: No EOS tokens found in first instance!")

    # Check for potential issues with cu_seqlens format
    if "doc_lens" in batch:
        print("\n[Potential Mechanical Issues]")
        doc_lens = batch["doc_lens"]
        cu_seqlens = get_cumulative_document_lengths(doc_lens)

        # Check 1: cu_seqlens should match batch * seq_len
        expected_total = input_ids.shape[0] * input_ids.shape[1]
        actual_total = cu_seqlens[-1].item()
        if actual_total != expected_total:
            print(f"  WARNING: cu_seqlens[-1]={actual_total} != batch*seq_len={expected_total}")
        else:
            print(f"  OK: cu_seqlens[-1]={actual_total} matches batch*seq_len")

        # Check 2: With CP=2, the model expects batch_size=1 in GatedDeltaNet
        # But data loader provides batch_size=2 for this DP rank
        batch_size = input_ids.shape[0]
        print(f"  Batch size for this DP rank: {batch_size}")
        print("  NOTE: GatedDeltaNet with CP requires batch_size=1 per GPU")
        print(
            f"        With CP=2, each GPU processes batch_size={batch_size // 2} (if split by CP)"
        )

        # Check 3: Number of documents
        n_docs = cu_seqlens.numel() - 1
        print(f"  Number of documents (cu_seqlens entries - 1): {n_docs}")

        # Check 4: Doc lens sum check per instance
        for i in range(min(batch_size, 3)):
            instance_doc_lens = doc_lens[i][doc_lens[i] > 0]
            doc_sum = instance_doc_lens.sum().item()
            print(
                f"  Instance {i}: {len(instance_doc_lens)} docs, sum={doc_sum}, seq_len={input_ids.shape[1]}"
            )
            if doc_sum != input_ids.shape[1]:
                print(f"    WARNING: doc_lens sum ({doc_sum}) != seq_len ({input_ids.shape[1]})")

    # Save to file if requested
    if args.output is not None:
        with open(args.output, "wb") as f:
            pickle.dump(batch, f)
        print(f"\nBatch saved to: {args.output}")
    else:
        print("\n[Token IDs (first instance, first 100 tokens)]")
        first_instance = input_ids[0].tolist()
        print(f"  {first_instance[:100]}")


if __name__ == "__main__":
    main()
