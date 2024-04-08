"""
Test a mock FSDP transformer model against PyTorch FSDP. Launch this script via `torchrun`:
    torchrun --nproc-per-node=8 -m benchmarks.fsdp.test
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)

from .common import TransformerConfig, build_components, print_rank0

log = logging.getLogger(__name__)


def main(
    config: TransformerConfig,
    batch_size: int,
    num_batches: int = 100,
    dry_run: bool = False,
    save_path: Optional[str] = None,
):
    torch_model, torch_optim, dataloader = build_components(
        config, batch_size, num_batches=num_batches, fsdp_wrapper="torch"
    )
    olmo_model, olmo_optim, _ = build_components(
        config, batch_size, num_batches=num_batches, fsdp_wrapper="olmo_core"
    )

    checkpoint_dir = Path(save_path or "/tmp/olmo-core-fsdp-benchmark-test")
    print_rank0(f"Saving torch FSDP checkpoint to {checkpoint_dir}...")
    save_model_and_optim_state(checkpoint_dir, torch_model, torch_optim)

    print_rank0(f"Loading OLMo-core FSDP checkpoint from {checkpoint_dir}...")
    load_model_and_optim_state(checkpoint_dir, olmo_model, olmo_optim)

    if dry_run:
        print_rank0("Dry run complete")
        return

    print_rank0("Test complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train.py", description="Train an FSDP model")
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
        "--save-path",
        type=str,
    )
    args = parser.parse_args()

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
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

    if args.debug and dist.get_rank() == 0:
        logging.basicConfig(level=logging.DEBUG)

    main(
        config,
        args.batch_size,
        num_batches=args.num_batches,
        dry_run=args.dry_run,
        save_path=args.save_path,
    )
