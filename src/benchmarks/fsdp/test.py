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
from torch.distributed.fsdp import FullyShardedDataParallel as TorchFSDP

from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
    save_model_and_optim_state,
)
from olmo_core.distributed.fsdp import FSDP

from .common import TransformerConfig, build_components, compute_loss, print_rank0

log = logging.getLogger(__name__)


def main(
    config: TransformerConfig,
    batch_size: int,
    num_batches: int = 100,
    dry_run: bool = False,
    save_path: Optional[str] = None,
    wrap_blocks: bool = True,
    mixed_precision: bool = True,
):
    torch_model, torch_optim, dataloader = build_components(
        config,
        batch_size,
        num_batches=num_batches,
        fsdp_wrapper="torch",
        wrap_blocks=wrap_blocks,
        mixed_precision=mixed_precision,
    )
    assert isinstance(torch_model, TorchFSDP)

    olmo_model, olmo_optim, _ = build_components(
        config,
        batch_size,
        num_batches=num_batches,
        fsdp_wrapper="olmo_core",
        wrap_blocks=wrap_blocks,
        mixed_precision=mixed_precision,
    )
    assert isinstance(olmo_model, FSDP)

    checkpoint_dir = Path(save_path or "/tmp/olmo-core-fsdp-benchmark-test")
    print_rank0(f"Saving torch FSDP checkpoint to {checkpoint_dir}...")
    save_model_and_optim_state(checkpoint_dir, torch_model, torch_optim, save_overwrite=True)

    print_rank0(f"Loading OLMo-core FSDP checkpoint from {checkpoint_dir}...")
    load_model_and_optim_state(checkpoint_dir, olmo_model, olmo_optim)

    print_rank0("Checking state dict...")
    with TorchFSDP.summon_full_params(torch_model), olmo_model.summon_full_params():
        torch_state_dict = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in torch_model.state_dict().items()}
        olmo_state_dict = olmo_model.state_dict()
        assert torch_state_dict.keys() == olmo_state_dict.keys()
        for key in torch_state_dict:
            torch.testing.assert_close(
                torch_state_dict[key], olmo_state_dict[key], msg=lambda msg: f"Failure for {key}: {msg}"
            )

    if dry_run:
        print_rank0("Dry run complete")
        return

    batches = iter(dataloader)

    print_rank0("Running first batch...")
    batch1 = next(batches)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed_precision):
        torch_logits = torch_model(batch1)
        olmo_logits = olmo_model(batch1)
        torch_loss = compute_loss(torch_model, batch1, logits=torch_logits)
        olmo_loss = compute_loss(olmo_model, batch1, logits=olmo_logits)
    torch.testing.assert_close(olmo_logits, torch_logits)
    torch.testing.assert_close(olmo_loss, torch_loss)

    torch_loss.backward()
    olmo_loss.backward()
    for (param_name, olmo_param), torch_param in zip(olmo_model.named_parameters(), torch_model.parameters()):
        if olmo_param.numel() > 0:
            assert olmo_param.grad is not None, f"Gradient for {param_name} is None!"
        if torch_param.numel() > 0:
            assert torch_param.grad is not None, f"Gradient for {param_name} is None!"

    print_rank0("Test complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train.py", description="Train an FSDP model")
    parser.add_argument(
        "--model-size",
        choices=["tiniest", "tiny", "small", "medium"],
        default="tiniest",
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
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
    )
    args = parser.parse_args()

    mixed_precision = not args.no_mixed_precision

    config: TransformerConfig
    wrap_blocks: bool = True
    if args.model_size == "tiniest":
        config = TransformerConfig.tiniest()
        wrap_blocks = False
    elif args.model_size == "tiny":
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
        wrap_blocks=wrap_blocks,
        mixed_precision=mixed_precision,
    )
