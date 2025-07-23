import argparse
import logging
import time
from pathlib import Path
from typing import List

import torch
from rich import print

from olmo_core.config import DType
from olmo_core.generate.config import GenerationConfig, TransformerGenerationModuleConfig
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
    total_sequences: int,
    vocab_size: int,
    use_cache: bool,
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
        total_sequences: Total number of sequences to generate.
        vocab_size: Size of the vocabulary (upper-bound for random token generation).
        device: Target device for tensors.
        dtype: Data type of the generated token IDs tensor.
        warmup: If True, run one warm-up generation (not timed) to exclude compilation / cache building.

    Returns:
        Throughput in tokens generated / second (including all sequences in the batch).
    """
    # Warm-up to trigger compilation / graph building. Run at most one batch.
    if warmup:
        warm_bs = min(batch_size, total_sequences)
        warm_prompt = torch.randint(
            1, vocab_size - 1, (warm_bs, prompt_length), dtype=dtype, device=device
        )
        generation_module.generate_batch(
            warm_prompt,
            max_length=prompt_length + max_new_tokens,
            use_cache=use_cache,
            log_timing=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # Benchmark loop – process exactly `total_sequences` sequences (possibly >1 batches).
    seqs_remaining = total_sequences
    tokens_generated_total = 0

    start = time.perf_counter()

    while seqs_remaining > 0:
        cur_bs = min(batch_size, seqs_remaining)
        prompt = torch.randint(
            1, vocab_size - 1, (cur_bs, prompt_length), dtype=dtype, device=device
        )
        generation_module.generate_batch(
            prompt,
            max_length=prompt_length + max_new_tokens,
            use_cache=use_cache,
            log_timing=False,
        )

        tokens_generated_total += cur_bs * max_new_tokens
        seqs_remaining -= cur_bs

    # Ensure all kernels have completed before timing stops
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start

    return tokens_generated_total / elapsed if elapsed > 0 else 0.0


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
    parser.add_argument(
        "--total-sequences",
        type=int,
        default=128,
        help="Total number of sequences to generate.",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=32,
        help="Length of the input prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
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
        "--use-cache",
        action="store_true",
        help="Use the KV-cache for generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility of the prompts.",
    )
    return parser.parse_args()


def main():
    print("\n[bold blue]🚀 OLMo Generation Throughput Benchmark 🚀[/]")
    print("[dim]Measuring autoregressive generation performance...[/]\n")

    args = parse_args()
    seed_all(args.seed)

    # Print benchmark conditions
    print("[bold green]=== Benchmark Configuration ===")
    print(f"[cyan]Checkpoint:[/] {args.checkpoint_dir}")
    print(f"[cyan]Device:[/] {args.device or 'auto-detect'}")
    print(f"[cyan]Batch sizes:[/] {args.batch_sizes}")
    print(f"[cyan]Total sequences:[/] {args.total_sequences}")
    print(f"[cyan]Prompt length:[/] {args.prompt_length}")
    print(f"[cyan]Max new tokens:[/] {args.max_new_tokens}")
    print(f"[cyan]Use cache:[/] {args.use_cache}")
    print(f"[cyan]Compile model:[/] {args.compile}")
    print(f"[cyan]Random seed:[/] {args.seed}")
    print()

    # Resolve device
    if args.device is not None:
        device = torch.device(args.device)
        if device.type == "cuda":
            assert torch.cuda.is_available(), "CUDA is not available"
            torch.cuda.init()
            _ = torch.empty(0, device="cuda")
            assert torch.cuda.is_initialized(), "CUDA is not initialized"
    else:
        device = get_default_device()
    log.info(f"Using device: {device}")

    # Load generation module from checkpoint
    log.info("Loading model from checkpoint ...")
    print("[bold green]Loading model from checkpoint ...[/]")
    generation_module = TransformerGenerationModule.from_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        work_dir=Path("/tmp/olmo_generation_bench"),
        dtype=DType.bfloat16,
        device=device,
        compile_model=args.compile,
    )
    print("[bold green]Model loaded successfully![/]")

    # Print available memory
    if device.type == "cuda":
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        memory_available = memory_total - memory_reserved
        print(
            f"[cyan]GPU Memory:[/] {memory_allocated:.1f}GB allocated, {memory_available:.1f}GB available, {memory_total:.1f}GB total"
        )

    print("[bold green]Running benchmark...[/]")
    results: List[str] = []
    for bs in args.batch_sizes:
        throughput = measure_throughput(
            generation_module=generation_module,
            batch_size=bs,
            prompt_length=args.prompt_length,
            max_new_tokens=args.max_new_tokens,
            total_sequences=args.total_sequences,
            vocab_size=generation_module.model.vocab_size,
            use_cache=args.use_cache,
            device=device,
        )
        results.append(f"[bold cyan]Batch {bs:>4}[/]: {throughput:>8.1f} tokens/s")

    print("\n[bold green]=== Generation Throughput Results ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
