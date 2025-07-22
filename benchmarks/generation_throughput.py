import argparse
import logging
import time
from pathlib import Path
from typing import List

import torch
from rich import print

from olmo_core.generate.generation import TransformerGenerationModule
from olmo_core.utils import get_default_device, seed_all

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")


@torch.inference_mode()
def measure_throughput(
    generation_module: TransformerGenerationModule,
    batch_size: int,
    prompt_length: int,
    max_new_tokens: int,
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.long,
    warmup: bool = True,
) -> float:
    """Measure generation throughput (tokens/s) for a single batch size.

    Args:
        generation_module: Loaded TransformerGenerationModule.
        batch_size: Number of sequences in the batch.
        prompt_length: Length of the prompt passed to the model.
        max_new_tokens: Number of tokens to generate (per sequence).
        vocab_size: Size of the vocabulary (upper-bound for random token generation).
        device: Target device for tensors.
        dtype: Data type of the generated token IDs tensor.
        warmup: If True, run one warm-up generation (not timed) to exclude compilation / cache building.

    Returns:
        Throughput in tokens / second (including all sequences in the batch).
    """
    # Build random prompt
    prompt = torch.randint(
        1, vocab_size - 1, (batch_size, prompt_length), dtype=dtype, device=device
    )

    # Warm-up run to trigger compilation / KV-cache allocation
    if warmup:
        generation_module.generate_batch(
            prompt,
            max_length=prompt_length + max_new_tokens,
            use_cache=True,
            log_timing=False,
        )
        torch.cuda.synchronize(device) if device.type == "cuda" else None

    start = time.perf_counter()
    generation_module.generate_batch(
        prompt,
        max_length=prompt_length + max_new_tokens,
        use_cache=True,
        log_timing=False,
    )
    # Ensure all kernels have completed
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    elapsed = time.perf_counter() - start

    total_tokens_generated = batch_size * max_new_tokens
    return total_tokens_generated / elapsed if elapsed > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure generation throughput for an OLMo checkpoint."
    )
    parser.add_argument(
        "checkpoint_dir", type=Path, help="Path to the checkpoint directory (local or remote)."
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[4, 16, 64, 256],
        help="List of batch sizes to benchmark.",
    )
    parser.add_argument("--prompt-length", type=int, default=32, help="Length of the input prompt.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Number of tokens to generate for each sequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g., 'cuda', 'cpu', 'cuda:0'). Defaults to the first available GPU, else CPU.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model with torch.compile() before benchmarking.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility of the prompts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    # Resolve device
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = get_default_device()

    log.info(f"Using device: {device}")

    # Load generation module from checkpoint
    log.info("Loading model from checkpoint ... this may take a while on first run.")
    generation_module = TransformerGenerationModule.from_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        work_dir=Path("/tmp/olmo_generation_bench"),
        device=device,
        compile_model=args.compile,
        load_thread_count=args.threads,
    )

    vocab_size = generation_module.model.vocab_size  # type: ignore[attr-defined]

    results: List[str] = []
    for bs in args.batch_sizes:
        throughput = measure_throughput(
            generation_module,
            batch_size=bs,
            prompt_length=args.prompt_length,
            max_new_tokens=args.max_new_tokens,
            vocab_size=vocab_size,
            device=device,
        )
        results.append(f"[bold cyan]Batch {bs:>4}[/]: {throughput:>8.1f} tokens/s")

    print("\n[bold green]=== Generation Throughput Results ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
