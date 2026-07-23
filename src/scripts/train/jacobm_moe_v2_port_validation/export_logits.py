#!/usr/bin/env python3
"""Strict-load a ported checkpoint and export deterministic logits/routing state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from olmo_core.distributed.checkpoint import RemoteFileSystemReader
from olmo_core.distributed.utils import get_rank, get_world_size, init_distributed
from olmo_core.train.checkpoint import Checkpointer
from olmo_core.train.train_module.transformer import OLMoDDPTrainModuleConfig
from olmo_core.utils import prepare_cli_environment
from scripts.train.jacobm_moe_v2_port_validation.config_adapter import (
    adapt_train_module_payload,
    build_model_config,
    load_recorded_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input-artifact", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/moe-v2-port-logits"))
    return parser.parse_args()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def register_hooks(model) -> tuple[dict[str, torch.Tensor], list[Any]]:
    captured: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def store(name: str, value: Any) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            captured[name] = tensor.float() if tensor.is_floating_point() else tensor

    def block_hook(name: str):
        def hook(_module, _inputs, output) -> None:
            store(f"blocks.{name}.output", output)

        return hook

    def router_hook(name: str):
        def hook(_module, _inputs, output) -> None:
            if not isinstance(output, tuple) or len(output) != 4:
                raise TypeError(f"Unexpected router output for block {name}: {type(output)}")
            expert_weights, expert_indices, batch_size_per_expert, aux_info = output
            store(f"blocks.{name}.router.expert_weights", expert_weights)
            store(f"blocks.{name}.router.expert_indices", expert_indices)
            store(f"blocks.{name}.router.batch_size_per_expert", batch_size_per_expert)
            if isinstance(aux_info, tuple) and len(aux_info) >= 2:
                store(f"blocks.{name}.router.scores", aux_info[0])
                store(f"blocks.{name}.router.logits", aux_info[1])

        return hook

    for name, block in model.blocks.items():
        handles.append(block.register_forward_hook(block_hook(name)))
        router = getattr(block, "routed_experts_router", None)
        if router is not None:
            handles.append(router.register_forward_hook(router_hook(name)))
    return captured, handles


def audit_checkpoint_schema(train_module, checkpoint: Path) -> dict[str, int]:
    checkpoint_dir = checkpoint / "model_and_optim"
    metadata = RemoteFileSystemReader(checkpoint_dir, thread_count=4).read_metadata()
    checkpoint_main = {
        key
        for key, value in metadata.state_dict_metadata.items()
        if key.endswith(".main") and isinstance(value, TensorStorageMetadata)
    }
    mapped: set[str] = set()
    parameter_count = 0
    parameter_numel = 0
    for model_part in train_module.model_parts:
        for name, parameter in model_part.named_parameters():
            key = train_module._resolve_model_checkpoint_key(  # noqa: SLF001
                name, set(metadata.state_dict_metadata)
            )
            if key is None:
                raise RuntimeError(f"No checkpoint tensor resolves to model parameter {name!r}")
            tensor_meta = metadata.state_dict_metadata[key]
            if not isinstance(tensor_meta, TensorStorageMetadata):
                raise TypeError(f"Expected tensor metadata for {key}")
            if tensor_meta.size.numel() != parameter.numel():
                raise RuntimeError(
                    f"Shape/numel mismatch for {name}: checkpoint {tuple(tensor_meta.size)} "
                    f"vs model {tuple(parameter.shape)}"
                )
            if key in mapped:
                raise RuntimeError(f"Checkpoint tensor {key!r} maps to multiple model parameters")
            mapped.add(key)
            parameter_count += 1
            parameter_numel += parameter.numel()
    if mapped != checkpoint_main:
        raise RuntimeError(
            "Strict model-main schema mismatch: "
            f"missing={sorted(checkpoint_main - mapped)[:20]}, "
            f"unexpected={sorted(mapped - checkpoint_main)[:20]}"
        )
    return {
        "parameter_count": parameter_count,
        "parameter_numel": parameter_numel,
        "checkpoint_main_tensor_count": len(checkpoint_main),
    }


def main() -> None:
    args = parse_args()
    prepare_cli_environment()
    init_distributed("nccl", shared_filesytem=True)
    if get_world_size() != 1:
        raise ValueError("Strict parity intentionally requires world_size=1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    checkpoint = args.checkpoint.resolve()
    config_path = args.config.resolve() if args.config else checkpoint / "config.json"
    recorded = load_recorded_config(config_path)
    model_config = build_model_config(config_path)
    for name in ("recompute_all_blocks_by_chunk", "recompute_each_block"):
        if hasattr(model_config, name):
            setattr(model_config, name, False)
    if hasattr(model_config, "recompute_block_keys"):
        model_config.recompute_block_keys = None

    train_config = OLMoDDPTrainModuleConfig.from_dict(
        adapt_train_module_payload(recorded["train_module"])
    )
    train_config.rank_microbatch_size = args.sequence_length
    train_config.max_sequence_length = args.sequence_length
    train_config.compile_model = False
    for name in ("pp_config", "tp_config", "cp_config", "ep_config", "ac_config"):
        setattr(train_config, name, None)

    device = torch.device("cuda", torch.cuda.current_device())
    model = model_config.build(init_device="meta")
    train_module = train_config.build(model, device=device, eval_only=True)
    schema = audit_checkpoint_schema(train_module, checkpoint)
    Checkpointer(work_dir=args.work_dir, load_thread_count=4).load(
        checkpoint,
        train_module,
        load_trainer_state=False,
        load_optim_state=False,
    )
    dist.barrier()
    for model_part in train_module.model_parts:
        model_part.eval()

    reference = torch.load(args.input_artifact, map_location="cpu", weights_only=True)
    input_ids_cpu = reference["input_ids"].to(dtype=torch.long, device="cpu")
    if tuple(input_ids_cpu.shape) != (1, args.sequence_length):
        raise ValueError(f"Unexpected fixed input shape {tuple(input_ids_cpu.shape)}")
    input_ids = input_ids_cpu.to(device)
    intermediates, handles = register_hooks(train_module.model_parts[0])
    with torch.inference_mode():
        output = train_module.model_forward_no_pipeline(input_ids)
    for handle in handles:
        handle.remove()
    if not isinstance(output, torch.Tensor):
        raise TypeError(f"Expected logits tensor, got {type(output)}")
    logits = output.detach().float().cpu().contiguous()
    expected_shape = (1, args.sequence_length, model_config.vocab_size)
    if tuple(logits.shape) != expected_shape:
        raise ValueError(f"Expected logits shape {expected_shape}, got {tuple(logits.shape)}")
    if not torch.isfinite(logits).all():
        raise RuntimeError("Exported logits contain non-finite values")

    metadata = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "world_size": get_world_size(),
        "sequence_length": args.sequence_length,
        "vocab_size": model_config.vocab_size,
        "input_ids_sha256": tensor_sha256(input_ids_cpu),
        "logits_sha256": tensor_sha256(logits),
        "source_logits_dtype": str(output.dtype),
        "stored_logits_dtype": str(logits.dtype),
        "captured_intermediate_count": len(intermediates),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        **schema,
    }
    if get_rank() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "input_ids": input_ids_cpu,
                "logits": logits,
                "metadata": metadata,
                "intermediates": intermediates,
            },
            args.output,
        )
        print(json.dumps({"status": "EXPORT_COMPLETE", **metadata}, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
