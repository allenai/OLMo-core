#!/usr/bin/env python3
"""Export deterministic full-vocabulary logits from a legacy or OLMoDDP checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist

from olmo_core.distributed.utils import get_rank, get_world_size, init_distributed
from olmo_core.train.checkpoint import Checkpointer
from olmo_core.utils import prepare_cli_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        help="Config to use; defaults to CHECKPOINT/config.json.",
    )
    parser.add_argument(
        "--model-kind",
        choices=("legacy", "olmo-ddp"),
        required=True,
        help="Select the config/train-module classes provided by the active checkout.",
    )
    parser.add_argument(
        "--input-artifact",
        type=Path,
        help="Reuse input_ids from a previously exported artifact.",
    )
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=6198)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/moe-logits-export"))
    parser.add_argument(
        "--capture-intermediates",
        action="store_true",
        help="Store every block output and routed-expert router output for localization.",
    )
    parser.add_argument(
        "--enable-grad",
        action="store_true",
        help="Keep autograd enabled to exercise the training-path kernels without backward.",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_configs(kind: str, config: Dict[str, Any]):
    if kind == "legacy":
        from olmo_core.nn.transformer import MoEFusedV2TransformerConfig
        from olmo_core.train.train_module.transformer import (
            MoEV2TransformerTrainModuleConfig,
        )

        model_config = MoEFusedV2TransformerConfig.from_dict(config["model"])
        train_config = MoEV2TransformerTrainModuleConfig.from_dict(config["train_module"])
    else:
        from olmo_core.nn.transformer import OLMoDDPModelConfig
        from olmo_core.train.train_module.transformer import OLMoDDPTrainModuleConfig

        model_config = OLMoDDPModelConfig.from_dict(config["model"])
        train_config = OLMoDDPTrainModuleConfig.from_dict(config["train_module"])

    return model_config, train_config


def load_or_create_input_ids(
    input_artifact: Path | None,
    *,
    sequence_length: int,
    seed: int,
    vocab_size: int,
) -> torch.Tensor:
    if input_artifact is not None:
        artifact = torch.load(input_artifact, map_location="cpu", weights_only=True)
        input_ids = artifact["input_ids"]
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids in the reference artifact is not a tensor")
        if tuple(input_ids.shape) != (1, sequence_length):
            raise ValueError(
                f"Expected input shape (1, {sequence_length}), got {tuple(input_ids.shape)}"
            )
        return input_ids.to(dtype=torch.long, device="cpu")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randint(
        0,
        vocab_size,
        (1, sequence_length),
        generator=generator,
        dtype=torch.long,
    )


def register_intermediate_hooks(model) -> tuple[Dict[str, torch.Tensor], list[Any]]:
    captured: Dict[str, torch.Tensor] = {}
    handles = []

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


def main() -> None:
    args = parse_args()
    prepare_cli_environment()
    init_distributed("nccl", shared_filesytem=True)

    if get_world_size() != 1:
        raise ValueError(
            "Cross-branch logit export intentionally runs with world_size=1 to avoid "
            "expert-parallel collective ordering differences"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    checkpoint = args.checkpoint.resolve()
    config_path = args.config.resolve() if args.config is not None else checkpoint / "config.json"
    config = load_config(config_path)
    model_config, train_config = build_configs(args.model_kind, config)

    for name in ("recompute_all_blocks_by_chunk", "recompute_each_block"):
        if hasattr(model_config, name):
            setattr(model_config, name, False)
    if hasattr(model_config, "recompute_block_keys"):
        model_config.recompute_block_keys = None
    model_config.validate()

    train_config.rank_microbatch_size = args.sequence_length
    train_config.max_sequence_length = args.sequence_length
    train_config.compile_model = False
    for name in ("pp_config", "tp_config", "cp_config", "ep_config", "ac_config"):
        if hasattr(train_config, name):
            setattr(train_config, name, None)

    device = torch.device("cuda", torch.cuda.current_device())
    model = model_config.build(init_device="meta")
    train_module = train_config.build(model, device=device, eval_only=True)
    Checkpointer(work_dir=args.work_dir, load_thread_count=4).load(
        checkpoint,
        train_module,
        load_trainer_state=False,
        load_optim_state=False,
    )
    dist.barrier()

    for model_part in train_module.model_parts:
        model_part.eval()

    input_ids_cpu = load_or_create_input_ids(
        args.input_artifact,
        sequence_length=args.sequence_length,
        seed=args.seed,
        vocab_size=model_config.vocab_size,
    )
    input_ids = input_ids_cpu.to(device)

    intermediates: Dict[str, torch.Tensor] = {}
    hook_handles: list[Any] = []
    if args.capture_intermediates:
        intermediates, hook_handles = register_intermediate_hooks(train_module.model_parts[0])

    torch.cuda.synchronize()
    forward_context = torch.enable_grad() if args.enable_grad else torch.inference_mode()
    with forward_context:
        output = train_module.model_forward_no_pipeline(input_ids)
    torch.cuda.synchronize()
    for handle in hook_handles:
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
        "config_sha256": file_sha256(config_path),
        "model_kind": args.model_kind,
        "world_size": get_world_size(),
        "sequence_length": args.sequence_length,
        "vocab_size": model_config.vocab_size,
        "seed": args.seed,
        "source_logits_dtype": str(output.dtype),
        "stored_logits_dtype": str(logits.dtype),
        "input_ids_sha256": tensor_sha256(input_ids_cpu),
        "logits_sha256": tensor_sha256(logits),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "captured_intermediate_count": len(intermediates),
        "grad_enabled": args.enable_grad,
    }
    artifact = {
        "input_ids": input_ids_cpu,
        "logits": logits,
        "metadata": metadata,
        "intermediates": intermediates,
    }
    if get_rank() == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, args.output)
        print(json.dumps({"status": "EXPORT_COMPLETE", **metadata}, indent=2), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
