"""
Train a mock FSDP transformer model. Launch this script via `torchrun`:
    torchrun --nproc-per-node=8 -m benchmarks.fsdp.train
"""

import argparse
import contextlib
import logging
import time
from collections import deque
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)

from .common import TransformerConfig, build_components, compute_loss, print_rank0

log = logging.getLogger(__name__)


def main(
    config: TransformerConfig,
    batch_size: int,
    num_batches: int = 100,
    fsdp_wrapper: Literal["torch", "olmo_core", "ddp"] = "olmo_core",
    dry_run: bool = False,
    save_path: Optional[str] = None,
    load_path: Optional[str] = None,
    mixed_precision: bool = True,
    profile: bool = False,
    trace_output: str = "/tmp/traces/olmo_core.chrome_trace.json.gz",
    max_grad_norm: Optional[float] = None,
    **kwargs,
):
    model, optim, dataloader = build_components(
        config,
        batch_size,
        num_batches=num_batches,
        fsdp_wrapper=fsdp_wrapper,
        mixed_precision=mixed_precision,
        **kwargs,
    )

    if load_path is not None:
        print_rank0(f"Loading checkpoint from {load_path}...")
        load_model_and_optim_state(load_path, model, optim)

    if dry_run:
        print_rank0("Dry run complete")
        return

    if save_path is not None:
        checkpoint_dir = Path(save_path) / "pretrain"
        print_rank0(f"Saving checkpoint to {checkpoint_dir}...")
        save_model_and_optim_state(checkpoint_dir, model, optim)

    profiler = contextlib.nullcontext()
    if profile:
        from torch.profiler import ProfilerActivity, schedule

        def on_trace_ready(p):
            trace_path = Path(trace_output).expanduser()
            trace_path.parent.mkdir(exist_ok=True, parents=True)
            p.export_chrome_trace(str(trace_path))
            print_rank0(f"Tracing complete, saved to '{trace_path}'")

        profiler = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
            schedule=schedule(wait=1, warmup=5, active=3, repeat=1),
            on_trace_ready=on_trace_ready,
        )

        print_rank0(torch.cuda.memory_summary())

    print_rank0("Starting training...")
    batch_times: deque[float] = deque([], 50)
    with profiler as p:
        for i, batch in enumerate(iter(dataloader)):
            log.debug("Batch: %s", batch)
            batch_start = time.monotonic()

            # Zero-gradients.
            optim.zero_grad(set_to_none=True)

            # Run forward pass.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed_precision):
                loss = compute_loss(model, batch)

            # Trigger backward pass.
            loss.backward()

            # Clip gradient norms.
            norm: Optional[torch.Tensor] = None
            if max_grad_norm is not None:
                if hasattr(model, "clip_grad_norm_"):
                    norm = model.clip_grad_norm_(max_grad_norm)
                else:
                    norm = clip_grad_norm_(model.parameters(), max_grad_norm)

            # Take optimizer step.
            optim.step()

            batch_time = time.monotonic() - batch_start
            if i > 0:
                batch_times.append(batch_time)
            norm_str = f"{norm.item():.3f}" if norm is not None else "n/a"
            print_rank0(
                f"Batch [{i+1}/{num_batches}]:\n"
                f"  loss={loss.item():.3f}\n"
                f"  throughput/seconds_per_batch={batch_time:.3f}\n"
                f"  grad/total_norm={norm_str}"
            )

            if profile and i == 2:
                print_rank0(torch.cuda.memory_summary())

            if p is not None:
                p.step()

    if batch_times:
        time_per_batch = sum(batch_times) / len(batch_times)
        print_rank0(f"Average throughput: {time_per_batch:.3f}s/b")

    if save_path is not None:
        checkpoint_dir = Path(save_path) / "final"
        print_rank0(f"Saving checkpoint to {checkpoint_dir}...")
        save_model_and_optim_state(checkpoint_dir, model, optim)

    print_rank0("Training complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train.py", description="Train an FSDP model")
    parser.add_argument(
        "--fsdp",
        choices=["torch", "olmo_core", "ddp"],
        default="olmo_core",
        help="""The FSDP implementation.""",
    )
    parser.add_argument(
        "--model-size",
        choices=["tiny", "small", "medium"],
        default="tiny",
        help="""The model size.""",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="""The per-device batch size.""",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=100,
        help="""The number of batches to train for.""",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
    )
    parser.add_argument(
        "--trace-output",
        type=str,
        default="/tmp/traces/olmo_core.chrome_trace.json.gz",
    )
    parser.add_argument(
        "--save-path",
        type=str,
    )
    parser.add_argument(
        "--load-path",
        type=str,
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
    )
    parser.add_argument(
        "--max-prefetch-count",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=234523,
    )
    args = parser.parse_args()

    mixed_precision = not args.no_mixed_precision

    config: TransformerConfig
    if args.model_size == "tiny":
        config = TransformerConfig.tiny()
    elif args.model_size == "small":
        config = TransformerConfig.small()
    elif args.model_size == "medium":
        config = TransformerConfig.medium()
    else:
        raise NotImplementedError(args.model_size)

    if args.debug:
        #  os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        config.debug = True

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

    if args.debug and dist.get_rank() == 0:
        logging.basicConfig(level=logging.DEBUG)

    main(
        config,
        args.batch_size,
        num_batches=args.num_batches,
        fsdp_wrapper=args.fsdp,
        dry_run=args.dry_run,
        save_path=args.save_path,
        load_path=args.load_path,
        profile=args.profile,
        trace_output=args.trace_output,
        mixed_precision=mixed_precision,
        max_prefetch_count=args.max_prefetch_count,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
    )
