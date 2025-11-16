"""
Precompute entropy and cross-entropy for dataset shards.

This script processes data shards and computes per-token entropy and cross-entropy
using an entropy model. The computed values are stored as raw binary files
(compatible with get_bytes_range) that can be loaded alongside the original data.

Usage:
    python src/examples/blt/compute_entropies.py \
        --model-checkpoint /path/to/checkpoint \
        --output-dir /path/to/entropy/output \
        --batch-size 32 \
        --sequence-length 2048

Environment variables:
    DATA_SOURCE: One of 'dclm', 'dolmino', 'dolma2_code_string', 'dolmino_code_string', 'tulu3'
    HAS_WEKA: Set to '1' if running on Weka filesystem

For each input shard like "part-0-00000.npy", this will create:
    - output_dir/entropy/path/to/part-0-00000.npy (entropy values as raw binary float16)
    - output_dir/cross_entropy/path/to/part-0-00000.npy (cross-entropy values as raw binary float16)

Note: Output files are raw binary (not .npy format) compatible with get_bytes_range for efficient loading.
"""

import argparse
import glob
import logging
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from olmo_core.config import DType
from olmo_core.io import get_bytes_range
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.data import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Data source configuration
DATA_SOURCE = os.environ.get("DATA_SOURCE", "dclm")

if DATA_SOURCE == "dclm":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources.txt").read().strip().splitlines()
elif DATA_SOURCE == "dolmino":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources_dolmino.txt").read().strip().splitlines()
elif DATA_SOURCE == "dolma2_code_string":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources_dolma2_code_string.txt").read().strip().splitlines()
elif DATA_SOURCE == "dolma2_150b_code_string":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources_dolma2_150b_code_string.txt").read().strip().splitlines()
elif DATA_SOURCE == "dolmino_code_string":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources_dolmino_code_string.txt").read().strip().splitlines()
elif DATA_SOURCE == "tulu3":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources_tulu3.txt").read().strip().splitlines()
elif DATA_SOURCE == "fineweb2_thai_sample":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources_fineweb2_thai_sample.txt").read().strip().splitlines()
elif DATA_SOURCE == "fineweb2_thai_sample_typhoon_tokenized":
    _DATA_SOURCES = open(Path(__file__).parent / "data_sources_fineweb2_thai_sample_typhoon_tokenized.txt").read().strip().splitlines()
else:
    raise ValueError(f"Unknown DATA_SOURCE: {DATA_SOURCE}. Must be one of 'dclm', 'dolmino', 'dolma2_code_string', 'dolmino_code_string'.")

DATA_PATHS = ["/weka/oe-training-default/" + x for x in _DATA_SOURCES]

if not os.environ.get("HAS_WEKA"):
    DATA_PATHS = [x.replace("/weka/oe-training-default/", "gs://") for x in DATA_PATHS]


def load_array_slice(path: str, start_idx: int, end_idx: int, dtype=np.uint16) -> np.ndarray:
    """Load a slice of a numpy array from disk without loading the entire file."""
    item_size = dtype(0).itemsize
    bytes_start = start_idx * item_size
    num_bytes = (end_idx - start_idx) * item_size

    buffer = get_bytes_range(path, bytes_start, num_bytes)
    return np.frombuffer(buffer, dtype=dtype)


def get_array_length(path: str, dtype=np.uint16) -> int:
    """Get the length of a numpy array file without loading it."""
    from olmo_core.io import get_file_size
    file_size = get_file_size(path)
    item_size = dtype(0).itemsize
    return file_size // item_size


@torch.compile
def _postprocess_logits(logits: torch.Tensor, input_ids: torch.Tensor, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    logprobs = F.log_softmax(logits, dim=-1)[:, :-1]  # (batch, seq_len-1, vocab_size)

    # Compute entropy (negative entropy, so higher = more certain)
    # H = -sum(p * log(p))
    # We return -H so that lower values = more uncertain
    entropy = torch.sum(torch.exp(logprobs) * logprobs, dim=-1)  # (batch, seq_len-1)

    # Compute cross-entropy (negative log probability of next token)
    # This is the main path logprobs
    cross_entropy = torch.gather(
        logprobs,
        dim=-1,
        index=input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)  # (batch, seq_len-1)

    # Mask out padding positions
    mask = input_ids[:, 1:] != pad_token_id
    entropy = torch.where(mask, entropy, torch.tensor(0.0, dtype=entropy.dtype))
    cross_entropy = torch.where(mask, cross_entropy, torch.tensor(0.0, dtype=cross_entropy.dtype))

    return entropy, cross_entropy  # (batch, seq_len-1), (batch, seq_len-1)


def compute_entropies_for_batch(
    input_ids: torch.Tensor,
    model: torch.nn.Module,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute entropy and cross-entropy for a batch of sequences.

    Args:
        input_ids: Token IDs of shape (batch_size, seq_len)
        model: PyTorch transformer model
        pad_token_id: Token ID used for padding
        device: Device to run inference on

    Returns:
        entropy: Negative entropy values of shape (batch_size, seq_len-1)
        cross_entropy: Negative log probability values of shape (batch_size, seq_len-1)
    """
    # Move input to device
    input_ids = input_ids

    # Get logits from model
    with torch.no_grad():
        logits = model(input_ids)  # (batch_size, seq_len, vocab_size)

    return _postprocess_logits(logits, input_ids, pad_token_id)


def process_shard(
    input_path: str,
    output_entropy_path: str,
    output_cross_entropy_path: str,
    model: torch.nn.Module,
    sequence_length: int,
    stride: int,
    batch_size: int,
    pad_token_id: int = 0,
    dtype: np.dtype = np.uint32,
    device: str = "cuda",
) -> None:
    """
    Process a single data shard and compute entropy/cross-entropy values.

    Args:
        input_path: Path to input .npy file
        output_entropy_path: Path to output entropy .npy file
        output_cross_entropy_path: Path to output cross-entropy .npy file
        model: PyTorch transformer model
        sequence_length: Length of each sequence window
        stride: Stride between consecutive windows (for overlapping)
        batch_size: Number of sequences to process at once
        pad_token_id: Token ID used for padding
        dtype: Data type of input array
        device: Device to run inference on
    """
    log.info(f"Processing {input_path}")

    # Get total length of input array
    total_length = get_array_length(input_path, dtype=dtype)

    # Prepare output arrays - same length as input
    # Use float32 for accumulation to avoid precision issues
    entropy_sum = np.zeros(total_length, dtype=np.float32)
    cross_entropy_sum = np.zeros(total_length, dtype=np.float32)
    count = np.zeros(total_length, dtype=np.int32)

    # Calculate number of windows with stride
    # Include a final window that extends to the end, even if it overlaps more
    num_windows = (total_length - sequence_length) // stride + 1

    # Add one more window if there are remaining tokens at the end
    last_window_end = (num_windows - 1) * stride + sequence_length
    if last_window_end < total_length:
        num_windows += 1

    log.info(f"Total length: {total_length}, Sequence length: {sequence_length}, Stride: {stride}")
    log.info(f"Processing {num_windows} overlapping windows")

    # Process windows in batches
    num_batches = (num_windows + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"Processing {Path(input_path).name}"):
        start_window = batch_idx * batch_size
        end_window = min(start_window + batch_size, num_windows)

        # Load batch of windows
        batch_data_list = []
        window_starts = []
        window_ends = []

        for window_idx in range(start_window, end_window):
            window_start = window_idx * stride
            window_end = min(window_start + sequence_length, total_length)
            window_starts.append(window_start)
            window_ends.append(window_end)

            window_data = load_array_slice(input_path, window_start, window_end, dtype=dtype)

            # Pad if necessary to maintain sequence_length
            if len(window_data) < sequence_length:
                window_data = np.pad(
                    window_data,
                    (0, sequence_length - len(window_data)),
                    mode='constant',
                    constant_values=pad_token_id
                )

            batch_data_list.append(window_data)

        # Stack into batch
        batch_data = np.stack(batch_data_list, axis=0)

        # Convert to tensor
        input_ids = torch.from_numpy(batch_data).long()

        # Compute entropies
        entropy, cross_entropy = compute_entropies_for_batch(
            input_ids.to(device), model, pad_token_id=pad_token_id
        )

        # Accumulate results in overlapping positions
        entropy_np = entropy.cpu().to(torch.float32).numpy()
        cross_entropy_np = cross_entropy.cpu().to(torch.float32).numpy()

        for i, (window_start, window_end) in enumerate(zip(window_starts, window_ends)):
            # Only accumulate values for the actual (non-padded) part of the window
            # The entropy/cross_entropy output is seq_len-1, so we need to account for that
            actual_length = window_end - window_start - 1  # -1 because entropy is computed on seq_len-1
            output_start = window_start
            output_end = window_start + actual_length

            entropy_sum[output_start:output_end] += entropy_np[i, :actual_length]
            cross_entropy_sum[output_start:output_end] += cross_entropy_np[i, :actual_length]
            count[output_start:output_end] += 1

    # Average where there are overlaps, keep zeros where no coverage
    mask = count > 0
    entropy_output = np.zeros(total_length, dtype=np.float16)
    cross_entropy_output = np.zeros(total_length, dtype=np.float16)

    entropy_output[mask] = (entropy_sum[mask] / count[mask]).astype(np.float16)
    cross_entropy_output[mask] = (cross_entropy_sum[mask] / count[mask]).astype(np.float16)

    # Log coverage statistics
    coverage = (count > 0).sum() / total_length * 100
    max_overlap = count.max()
    log.info(f"Coverage: {coverage:.2f}%, Max overlap: {max_overlap}")

    # Save outputs as raw binary files (compatible with get_bytes_range)
    log.info(f"Saving entropy to {output_entropy_path}")
    os.makedirs(os.path.dirname(output_entropy_path), exist_ok=True)
    entropy_output.tofile(output_entropy_path)

    log.info(f"Saving cross-entropy to {output_cross_entropy_path}")
    os.makedirs(os.path.dirname(output_cross_entropy_path), exist_ok=True)
    cross_entropy_output.tofile(output_cross_entropy_path)

    log.info(f"Completed processing {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute entropy and cross-entropy for dataset shards"
    )
    parser.add_argument(
        "run_name",
        type=str,
        help="Name of the run",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for entropy and cross-entropy shards",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="Sequence length for each window (default: 2048)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride between windows. If None, uses sequence_length (no overlap). Use smaller values for overlapping windows that will be averaged.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint32",
        choices=["uint8", "uint16", "uint32", "uint64"],
        help="Data type of input arrays (default: uint32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--model-arch",
        type=str,
        default="olmo2_190M",
        help="Model architecture name (default: olmo2_190M)",
    )
    parser.add_argument(
        "--process_every",
        type=str,
        default="1/1",
        help="Process only a fraction of shards, in the form 'i/n' to process the i-th out of n equal parts (default: 1/1 = all shards)",
    )

    args = parser.parse_args()

    # Convert dtype string to numpy dtype
    dtype = getattr(np, args.dtype)

    # Set stride to sequence_length if not provided (no overlap)
    stride = args.stride if args.stride is not None else args.sequence_length

    if stride > args.sequence_length:
        raise ValueError(f"Stride ({stride}) cannot be greater than sequence_length ({args.sequence_length})")

    log.info(f"Using sequence_length={args.sequence_length}, stride={stride}")

    # Get input files from DATA_PATHS
    log.info(f"Using DATA_SOURCE: {DATA_SOURCE}")
    log.info(f"Found {len(DATA_PATHS)} data paths")

    # Expand glob patterns in DATA_PATHS
    input_files = []
    for pattern in DATA_PATHS:
        if "*" in pattern or "?" in pattern:
            # Glob pattern
            matched = glob.glob(pattern)
            if not matched:
                log.warning(f"No files matched pattern: {pattern}")
            input_files.extend(matched)
        else:
            # Single file or directory
            if os.path.isfile(pattern):
                input_files.append(pattern)
            elif os.path.isdir(pattern):
                # List all .npy files in directory
                dir_files = glob.glob(os.path.join(pattern, "*.npy"))
                input_files.extend(dir_files)
            else:
                log.warning(f"Path does not exist: {pattern}")

    if not input_files:
        log.error("No input files found!")
        return

    input_files = sorted(input_files)

    log.info(f"Found {len(input_files)} input files to process")

    # Initialize model
    log.info(f"Loading model from {args.model_checkpoint}")
    log.info(f"Using architecture: {args.model_arch}")

    # Build tokenizer config
    tokenizer_config = TokenizerConfig.dolma2()
    pad_token_id = tokenizer_config.pad_token_id

    # Build model config
    model_config = getattr(TransformerConfig, args.model_arch)(
        vocab_size=tokenizer_config.padded_vocab_size(),
        dtype=DType.bfloat16,
    )

    # Build model
    log.info("Building model...")
    model = model_config.build(init_device=args.device)

    # Load checkpoint
    log.info("Loading checkpoint weights...")
    load_model_and_optim_state(
        args.model_checkpoint,
        model,
    )

    # Move model to device and set to eval mode
    model = model.to(args.device)
    model.eval()
    model.apply_compile()

    log.info(f"Model loaded successfully on {args.device}")

    process_every_nom, process_every_denom = map(int, args.process_every.split("/"))
    input_files_to_process = []

    for idx, input_file in enumerate(input_files):
        if (idx % process_every_denom) == (process_every_nom - 1):
            input_files_to_process.append(input_file)

    # Process each shard
    for input_path in tqdm(input_files_to_process, desc="Processing shards"):
        # Determine output paths
        # Preserve directory structure relative to data root
        if "/weka/oe-training-default/" in input_path:
            input_path_no_prefix = input_path.split("/weka/oe-training-default/")[-1]
        elif "gs://" in input_path:
            input_path_no_prefix = input_path.split("gs://")[-1].split("/", 1)[-1]
        else:
            raise ValueError(f"Input path {input_path} does not contain expected prefix")

        output_entropy_path = os.path.join(args.output_dir, "entropy", input_path_no_prefix)
        output_cross_entropy_path = os.path.join(args.output_dir, "cross_entropy", input_path_no_prefix)

        # Skip if already processed
        if os.path.exists(output_entropy_path) and os.path.exists(output_cross_entropy_path):
            log.info(f"Skipping {input_path} (already processed)")
            continue

        try:
            process_shard(
                input_path=input_path,
                output_entropy_path=output_entropy_path,
                output_cross_entropy_path=output_cross_entropy_path,
                model=model,
                sequence_length=args.sequence_length,
                stride=stride,
                batch_size=args.batch_size,
                pad_token_id=pad_token_id,
                dtype=dtype,
                device=args.device,
            )
        except Exception as e:
            log.error(f"Failed to process {input_path}: {e}")
            raise

    log.info("All shards processed successfully!")


if __name__ == "__main__":
    main()
