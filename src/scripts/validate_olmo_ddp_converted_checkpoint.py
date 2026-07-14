#!/usr/bin/env python3
"""Validate a converted OLMoDDP checkpoint on CUDA with expert parallelism."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist

from olmo_core.distributed.utils import get_rank, get_world_size, init_distributed
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
from olmo_core.nn.transformer import OLMoDDPModelConfig
from olmo_core.train.checkpoint import Checkpointer
from olmo_core.train.train_module.transformer import (
    OLMoDDPTrainModuleConfig,
    TransformerExpertParallelConfig,
)
from olmo_core.utils import prepare_cli_environment


log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--ep-degree", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--rowwise-nblocks", type=int, default=128)
    parser.add_argument("--seed", type=int, default=6198)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/olmo-ddp-smoke"))
    parser.add_argument(
        "--hard-exit",
        action="store_true",
        help="Exit without interpreter teardown after all checks pass (for NVSHMEM teardown bugs).",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def run_forward(train_module, input_ids: torch.Tensor, labels: torch.Tensor, *, training: bool):
    for model_part in train_module.model_parts:
        model_part.train(training)

    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        output = train_module.model_forward_no_pipeline(
            input_ids,
            labels=labels,
            loss_reduction="mean",
            return_logits=False,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    if not isinstance(output, LMOutputWithLoss):
        raise TypeError(f"Expected LMOutputWithLoss, got {type(output)}")
    if output.loss is None or output.ce_loss is None:
        raise RuntimeError("Forward did not return loss and CE loss")
    return output.loss.detach().float(), output.ce_loss.detach().float(), elapsed


def gather_scalars(*values: torch.Tensor) -> list[list[float]]:
    local = torch.stack([value.reshape(()) for value in values])
    gathered = [torch.empty_like(local) for _ in range(get_world_size())]
    dist.all_gather(gathered, local)
    return [[float(value) for value in rank_values.cpu()] for rank_values in gathered]


def main() -> None:
    args = parse_args()
    prepare_cli_environment()
    init_distributed("nccl", shared_filesytem=True)

    rank = get_rank()
    world_size = get_world_size()
    if world_size != args.ep_degree:
        raise ValueError(
            f"This smoke expects world_size == ep_degree, got {world_size} and {args.ep_degree}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    checkpoint = args.checkpoint.resolve()
    config = load_config(checkpoint / "config.json")
    model_config = OLMoDDPModelConfig.from_dict(config["model"])
    model_config.recompute_all_blocks_by_chunk = False
    model_config.recompute_each_block = False
    model_config.recompute_block_keys = None
    model_config.validate()

    train_config = OLMoDDPTrainModuleConfig.from_dict(config["train_module"])
    train_config.rank_microbatch_size = args.sequence_length
    train_config.max_sequence_length = args.sequence_length
    train_config.compile_model = False
    train_config.pp_config = None
    train_config.tp_config = None
    train_config.cp_config = None
    train_config.ep_config = TransformerExpertParallelConfig(degree=args.ep_degree)

    device = torch.device("cuda", int(torch.cuda.current_device()))
    model = model_config.build(init_device="meta")
    routed_blocks = list(model.named_routed_blocks())
    if not routed_blocks:
        raise RuntimeError("Converted model has no routed blocks")
    for _, block in routed_blocks:
        block.ep.rowwise_nblocks = args.rowwise_nblocks
    bad_paths = {
        str(block.ep.path)
        for _, block in routed_blocks
        if block.ep.path != ExpertParallelPath.rowwise_nvshmem
    }
    if bad_paths:
        raise RuntimeError(f"Expected rowwise_nvshmem routed blocks, got {sorted(bad_paths)}")

    train_module = train_config.build(model, device=device, eval_only=True)
    checkpointer = Checkpointer(
        work_dir=args.work_dir,
        load_thread_count=4,
    )
    checkpointer.load(
        checkpoint,
        train_module,
        load_trainer_state=False,
        load_optim_state=False,
    )
    dist.barrier()

    first_parameter = next(train_module.model_parts[0].parameters())
    parameter_sample = first_parameter.detach().reshape(-1)[:4096].float()
    if not torch.isfinite(parameter_sample).all():
        raise RuntimeError("Loaded parameter sample contains non-finite values")
    parameter_sample_sum = parameter_sample.sum()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    input_ids = torch.randint(
        0,
        model_config.vocab_size,
        (1, args.sequence_length),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    labels = input_ids.clone()
    labels[:, 0] = -100

    torch.cuda.reset_peak_memory_stats(device)
    rowwise_loss, rowwise_ce, rowwise_seconds = run_forward(
        train_module, input_ids, labels, training=True
    )
    sync_loss, sync_ce, sync_seconds = run_forward(
        train_module, input_ids, labels, training=False
    )
    peak_memory_bytes = torch.cuda.max_memory_allocated(device)

    all_values = gather_scalars(
        rowwise_loss,
        rowwise_ce,
        sync_loss,
        sync_ce,
        parameter_sample_sum,
    )
    for rank_values in all_values:
        if not all(math.isfinite(value) for value in rank_values):
            raise RuntimeError(f"Rank results contain non-finite values: {all_values}")

    if not torch.allclose(rowwise_ce, sync_ce, rtol=args.rtol, atol=args.atol):
        raise RuntimeError(
            "Rowwise and synchronized EP CE losses disagree: "
            f"rowwise={rowwise_ce.item()}, sync={sync_ce.item()}"
        )

    if rank == 0:
        ce_values = [rank_values[1] for rank_values in all_values]
        result = {
            "status": "SMOKE_PASS",
            "checkpoint": str(checkpoint),
            "world_size": world_size,
            "ep_degree": args.ep_degree,
            "sequence_length": args.sequence_length,
            "rowwise_nblocks": args.rowwise_nblocks,
            "routed_block_count": len(routed_blocks),
            "rowwise_loss": float(rowwise_loss),
            "rowwise_ce_loss": float(rowwise_ce),
            "sync_loss": float(sync_loss),
            "sync_ce_loss": float(sync_ce),
            "ce_abs_diff": float((rowwise_ce - sync_ce).abs()),
            "rowwise_seconds": rowwise_seconds,
            "sync_seconds": sync_seconds,
            "peak_memory_bytes": peak_memory_bytes,
            "rank_values": all_values,
            "rank_rowwise_ce_spread": max(ce_values) - min(ce_values),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        }
        print(json.dumps(result, indent=2), flush=True)

    dist.barrier()
    if args.hard_exit:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
