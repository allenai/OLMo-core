"""Compare Qwen 3.5 MoE execution with and without Ulysses CP=2."""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoConfig

from olmo_core.config import DType
from olmo_core.distributed.parallel.context_parallel import ContextParallelConfig
from olmo_core.distributed.parallel.data_parallel import DataParallelConfig, DataParallelType
from olmo_core.distributed.parallel.expert_parallel import ExpertParallelConfig as MeshEPConfig
from olmo_core.distributed.utils import get_local_tensor
from olmo_core.nn.attention import AttentionBackendName, UlyssesContextParallelStyle
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig, ExpertParallelPath
from olmo_core.nn.moe.v2.qwen import build_qwen3_moe_config_from_hf_config
from olmo_core.nn.transformer import TransformerActivationCheckpointingMode
from olmo_core.train import prepare_training_environment
from olmo_core.train.train_module.transformer.moe_train_module import MoEV2TransformerTrainModule

log = logging.getLogger(__name__)


DENSE_GRADIENTS = (
    "embeddings.weight",
    "blocks.0.attention.w_q.weight",
    "blocks.3.attention.w_q.weight",
    "blocks.0.routed_experts_router.weight",
    "blocks.0.shared_experts.w_up_gate",
    "lm_head.w_out.weight",
)
EXPERT_GRADIENTS = (
    "blocks.0.routed_experts.w_up_gate",
    "blocks.3.routed_experts.w_down",
)


def _mesh_owner() -> MoEV2TransformerTrainModule:
    owner = MoEV2TransformerTrainModule.__new__(MoEV2TransformerTrainModule)
    owner.world_mesh = {}
    owner.pp_group = None
    owner.dp_group = None
    owner.tp_group = None
    owner.cp_group = None
    owner.ep_dp_group = None
    owner.ep_mp_group = None
    owner.dense_dp_cp_group = None
    owner.expert_param_group = None
    return owner


def _build_mesh(*, cp_degree: int | None, ep_degree: int) -> MoEV2TransformerTrainModule:
    owner = _mesh_owner()
    owner._build_world_mesh(
        dp=DataParallelConfig(name=DataParallelType.ddp),
        cp=ContextParallelConfig(degree=cp_degree) if cp_degree is not None else None,
        ep=MeshEPConfig(degree=ep_degree),
        device_type="cuda",
    )
    return owner


def _checkpoint_model_dir(checkpoint: Path) -> Path:
    model_dir = checkpoint / "model_and_optim"
    return model_dir if model_dir.is_dir() else checkpoint


def _build_model(
    *,
    hf_config: dict[str, Any],
    checkpoint: Path,
    cp_degree: int | None,
    ep_degree: int,
    compile_model: bool,
    activation_memory_budget: float,
) -> tuple[torch.nn.Module, MoEV2TransformerTrainModule]:
    mesh = _build_mesh(cp_degree=cp_degree, ep_degree=ep_degree)
    ep = ExpertParallelConfig(
        path=ExpertParallelPath.sync_1d,
        capacity_factor=1.25,
    )
    config = build_qwen3_moe_config_from_hf_config(
        hf_config,
        dtype=DType.bfloat16,
        attention_backend=AttentionBackendName.flash_4,
        compile_friendly_recompute=True,
        ep=ep,
    )
    config.recompute_each_block = False
    model = config.build(init_device="meta")
    model.to_empty(device=torch.device("cuda"))
    if cp_degree is not None:
        model.apply_cp(
            mesh.world_mesh["dense"]["cp"],
            uly=UlyssesContextParallelStyle(),
        )
    model.apply_ep(
        dp_mesh=mesh.world_mesh["dense"]["dp"],
        ep_mesh=mesh.world_mesh["moe"],
    )
    model.apply_activation_checkpointing(
        TransformerActivationCheckpointingMode.budget,
        activation_memory_budget=activation_memory_budget,
    )
    # Use the DDP train module's topology-aware loader. Expert parameters are
    # rank-local tensors after EP is applied, so the generic loader sees a shape
    # mismatch against the checkpoint's global expert tensors.
    mesh.model_parts = [model]
    mesh.eval_only = True
    mesh.load_state_dict_direct(
        _checkpoint_model_dir(checkpoint),
        process_group=dist.group.WORLD,
        thread_count=32,
        load_optim_state=False,
    )
    if compile_model:
        model.apply_compile()
    return model, mesh


def _release_cuda_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    dist.barrier()


def _load_tokens(path: Path, *, offset: int, length: int) -> torch.Tensor:
    # Pretokenized OLMo datasets use headerless uint32 arrays despite the .npy suffix.
    tokens = np.memmap(path, mode="r", dtype=np.uint32)
    if offset + length > tokens.size:
        raise ValueError(
            f"Token file only has {tokens.size:,} tokens, need {offset + length:,}"
        )
    values = np.asarray(tokens[offset : offset + length], dtype=np.int64).copy()
    return torch.from_numpy(values)


def _load_label_mask(path: Path, *, offset: int, length: int) -> torch.Tensor:
    mask = np.memmap(path, mode="r", dtype=np.bool_)
    if offset + length > mask.size:
        raise ValueError(
            f"Label-mask file only has {mask.size:,} items, need {offset + length:,}"
        )
    return torch.from_numpy(np.asarray(mask[offset : offset + length]).copy())


def _load_document_offsets(path: Path, *, count: int, min_length: int) -> list[int]:
    offsets: list[int] = []
    with gzip.open(path, mode="rt", newline="") as metadata_file:
        for row in csv.reader(metadata_file):
            start, end = (int(value) for value in row)
            if end - start >= min_length:
                offsets.append(start)
                if len(offsets) == count:
                    return offsets
    raise ValueError(
        f"Found only {len(offsets)} documents of at least {min_length:,} tokens in {path}"
    )


def _capture_final_hidden(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    with_grad: bool,
    capture_layers: bool = False,
) -> tuple[torch.Tensor, Any, dict[str, torch.Tensor]]:
    captured: dict[str, torch.Tensor] = {}
    layer_outputs: dict[str, torch.Tensor] = {}

    def capture(_module, args) -> None:
        captured["hidden"] = args[0] if with_grad else args[0].detach()

    handles = [model.lm_head.register_forward_pre_hook(capture)]
    if capture_layers:
        for name, block in model.blocks.items():
            handles.append(
                block.register_forward_hook(
                    lambda _module, _args, output, layer_name=name: layer_outputs.__setitem__(
                        layer_name, output.detach().cpu()
                    )
                )
            )
    try:
        output = model(
            input_ids,
            logits_to_keep=1,
        )
    finally:
        for handle in handles:
            handle.remove()
    return captured["hidden"], output, layer_outputs


def _exchange_reference_shards(
    reference: torch.Tensor,
    *,
    local_length: int,
    cp_group: dist.ProcessGroup,
) -> torch.Tensor:
    reference_send = (
        reference.view(2, local_length, reference.shape[-1]).contiguous().to("cuda")
    )
    reference_local = torch.empty_like(reference_send)
    dist.all_to_all_single(reference_local, reference_send, group=cp_group)
    return reference_local


def _distributed_tensor_metrics(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    reference = reference.float()
    actual = actual.float()
    if reference.shape != actual.shape:
        raise RuntimeError(f"Shape mismatch: {tuple(reference.shape)} != {tuple(actual.shape)}")
    diff = actual - reference
    stats = torch.stack(
        (
            diff.abs().sum(),
            diff.square().sum(),
            reference.square().sum(),
            actual.square().sum(),
            (reference * actual).sum(),
            torch.tensor(reference.numel(), device=reference.device),
        )
    ).to(
        device="cuda",
        dtype=torch.float64,
    )
    max_abs = diff.abs().max().to(device="cuda", dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX)
    count = stats[5].item()
    reference_l2 = stats[2].item() ** 0.5
    actual_l2 = stats[3].item() ** 0.5
    diff_l2 = stats[1].item() ** 0.5
    return {
        "max_abs_diff": max_abs.item(),
        "mean_abs_diff": stats[0].item() / count,
        "rms_diff": (stats[1].item() / count) ** 0.5,
        "reference_rms": (stats[2].item() / count) ** 0.5,
        "actual_rms": (stats[3].item() / count) ** 0.5,
        "relative_l2_error": diff_l2 / max(reference_l2, 1e-30),
        "cosine_similarity": stats[4].item()
        / max(reference_l2 * actual_l2, 1e-30),
    }


def _sample_flat(tensor: torch.Tensor, max_values: int) -> torch.Tensor:
    flat = tensor.detach().float().reshape(-1)
    if flat.numel() <= max_values:
        return flat.cpu()
    indices = torch.linspace(
        0,
        flat.numel() - 1,
        steps=max_values,
        device=flat.device,
        dtype=torch.float64,
    ).long()
    return flat.index_select(0, indices).cpu()


def _gradient_samples(
    model: torch.nn.Module,
    *,
    sample_values: int,
) -> dict[str, torch.Tensor]:
    parameters = dict(model.named_parameters())
    samples: dict[str, torch.Tensor] = {}
    for name in DENSE_GRADIENTS + EXPERT_GRADIENTS:
        parameter = parameters[name]
        if parameter.grad is None:
            raise RuntimeError(f"Missing gradient for {name}")
        gradient = get_local_tensor(parameter.grad).detach().float()
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"Nonfinite gradient for {name}")
        sample = _sample_flat(gradient, sample_values).to("cuda")
        if name in DENSE_GRADIENTS:
            # Dense parameters see a different DP/CP shard on each rank. Sample
            # first, then sum those same coordinates across the world.
            dist.all_reduce(sample, op=dist.ReduceOp.SUM)
        samples[name] = sample.cpu()
    return samples


def _write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    if dist.get_rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2) + "\n")
        log.info("Parity metrics:\n%s", json.dumps(metrics, indent=2))


@torch.no_grad()
def verify_forward(args: argparse.Namespace, hf_config: dict[str, Any]) -> dict[str, Any]:
    rank = dist.get_rank()
    reference_input_ids = _load_tokens(
        args.token_ids,
        offset=rank * args.sequence_length,
        length=args.sequence_length,
    ).unsqueeze(0)

    log.info("Running no-CP reference forward at sequence length %d", args.sequence_length)
    reference_model, reference_mesh = _build_model(
        hf_config=hf_config,
        checkpoint=args.checkpoint,
        cp_degree=None,
        ep_degree=args.ep_degree,
        compile_model=args.compile,
        activation_memory_budget=args.activation_memory_budget,
    )
    reference_model.eval()
    reference_hidden, _, reference_layers = _capture_final_hidden(
        reference_model,
        reference_input_ids.to("cuda"),
        with_grad=False,
        capture_layers=args.layerwise,
    )
    reference_hidden_cpu = reference_hidden.to("cpu")
    del reference_hidden, reference_model, reference_mesh
    _release_cuda_memory()

    log.info("Running CP=2 forward at sequence length %d", args.sequence_length)
    cp_model, cp_mesh = _build_model(
        hf_config=hf_config,
        checkpoint=args.checkpoint,
        cp_degree=2,
        ep_degree=args.ep_degree,
        compile_model=args.compile,
        activation_memory_budget=args.activation_memory_budget,
    )
    cp_model.eval()
    cp_global_ranks = dist.get_process_group_ranks(cp_mesh.cp_group)
    cp_input_ids = torch.stack(
        [
            _load_tokens(
                args.token_ids,
                offset=source_rank * args.sequence_length,
                length=args.sequence_length,
            )
            for source_rank in cp_global_ranks
        ]
    )
    cp_hidden, _, cp_layers = _capture_final_hidden(
        cp_model,
        cp_input_ids.to("cuda"),
        with_grad=False,
        capture_layers=args.layerwise,
    )
    local_length = cp_hidden.shape[1]
    reference_local = _exchange_reference_shards(
        reference_hidden_cpu,
        local_length=local_length,
        cp_group=cp_mesh.cp_group,
    )
    if reference_local.shape != cp_hidden.shape:
        raise RuntimeError(
            f"Reference exchange produced {tuple(reference_local.shape)}, "
            f"CP output has {tuple(cp_hidden.shape)}"
        )
    hidden_metrics = _distributed_tensor_metrics(reference_local, cp_hidden)

    k = min(args.logit_positions_per_cp_rank, local_length)
    reference_logits = cp_model.lm_head(reference_local[:, -k:])
    cp_logits = cp_model.lm_head(cp_hidden[:, -k:])
    logit_metrics = _distributed_tensor_metrics(reference_logits, cp_logits)
    top1 = torch.stack(
        (
            (reference_logits.argmax(-1) == cp_logits.argmax(-1)).sum(),
            torch.tensor(
                reference_logits.shape[0] * reference_logits.shape[1],
                device=reference_logits.device,
            ),
        )
    ).to(dtype=torch.float64)
    dist.all_reduce(top1, op=dist.ReduceOp.SUM)
    logit_metrics["top1_agreement"] = (top1[0] / top1[1]).item()

    layer_metrics: dict[str, dict[str, Any]] = {}
    if args.layerwise:
        if reference_layers.keys() != cp_layers.keys():
            raise RuntimeError(
                f"Layer capture mismatch: {reference_layers.keys()} != {cp_layers.keys()}"
            )
        for name in reference_layers:
            reference_layer_local = _exchange_reference_shards(
                reference_layers[name],
                local_length=cp_layers[name].shape[1],
                cp_group=cp_mesh.cp_group,
            )
            actual_layer = cp_layers[name].to("cuda")
            block = cp_model.blocks[name]
            layer_metrics[name] = {
                "mixer": type(block.attention).__name__,
                **_distributed_tensor_metrics(reference_layer_local, actual_layer),
            }
            del reference_layer_local, actual_layer

    metrics = {
        "mode": "forward",
        "sequence_length": args.sequence_length,
        "cp_degree": 2,
        "ep_degree": args.ep_degree,
        "compile": args.compile,
        "global_sequences": dist.get_world_size(),
        "hidden_state": hidden_metrics,
        "sampled_logits": logit_metrics,
        "layers": layer_metrics,
    }
    _write_metrics(args.output_json, metrics)
    if hidden_metrics["cosine_similarity"] < args.min_hidden_cosine:
        raise AssertionError(f"Hidden-state parity failed: {hidden_metrics}")
    if logit_metrics["cosine_similarity"] < args.min_logit_cosine:
        raise AssertionError(f"Logit parity failed: {logit_metrics}")
    if logit_metrics["top1_agreement"] < args.min_top1_agreement:
        raise AssertionError(f"Top-1 parity failed: {logit_metrics}")
    del cp_model
    _release_cuda_memory()
    return metrics


def _backward_inputs(
    token_path: Path,
    label_mask_path: Path,
    document_metadata_path: Path,
    *,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rank = dist.get_rank()
    document_offsets = _load_document_offsets(
        document_metadata_path,
        count=dist.get_world_size() // 2,
        min_length=sequence_length,
    )
    offset = document_offsets[rank // 2]
    input_ids = _load_tokens(
        token_path,
        offset=offset,
        length=sequence_length,
    ).unsqueeze(0)
    label_mask = _load_label_mask(
        label_mask_path,
        offset=offset,
        length=sequence_length,
    ).unsqueeze(0)
    labels = input_ids.roll(shifts=-1, dims=1)
    labels.masked_fill_(~label_mask.roll(shifts=-1, dims=1), -100)
    labels[:, -1] = -100
    return input_ids, labels


def _run_backward(
    args: argparse.Namespace,
    hf_config: dict[str, Any],
    *,
    cp: bool,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    model, mesh = _build_model(
        hf_config=hf_config,
        checkpoint=args.checkpoint,
        cp_degree=2 if cp else None,
        ep_degree=args.ep_degree,
        compile_model=args.compile,
        activation_memory_budget=args.activation_memory_budget,
    )
    model.train()
    input_ids, labels = _backward_inputs(
        args.token_ids,
        args.label_mask,
        args.document_metadata,
        sequence_length=args.sequence_length,
    )
    batch_num_tokens_for_loss = (labels != -100).sum().to("cuda")
    dist.all_reduce(batch_num_tokens_for_loss, group=mesh.dp_group)
    batch_num_tokens_for_loss = batch_num_tokens_for_loss / dist.get_world_size(mesh.dp_group)
    output = model(
        input_ids.to("cuda"),
        labels=labels.to("cuda"),
        loss_reduction="sum",
        loss_div_factor=batch_num_tokens_for_loss,
        return_logits=False,
        doc_lens=torch.tensor([[args.sequence_length]], dtype=torch.int32),
        max_doc_lens=[args.sequence_length],
    )
    output.loss.backward()
    loss_stats = torch.stack([output.loss.detach().float(), output.ce_loss.detach().float()])
    dist.all_reduce(loss_stats, op=dist.ReduceOp.SUM)
    samples = _gradient_samples(model, sample_values=args.gradient_sample_values)
    metrics = {
        "loss_sum": loss_stats[0].item(),
        "ce_loss_sum": loss_stats[1].item(),
    }
    del output, model
    _release_cuda_memory()
    return metrics, samples


def verify_backward(args: argparse.Namespace, hf_config: dict[str, Any]) -> dict[str, Any]:
    log.info("Running no-CP backward reference")
    reference_loss, reference_gradients = _run_backward(args, hf_config, cp=False)
    compare_cp = args.backward_comparison == "cp2"
    comparison_name = "cp2" if compare_cp else "no_cp_repeat"
    log.info("Running %s backward comparison", "CP=2" if compare_cp else "no-CP repeat")
    comparison_loss, comparison_gradients = _run_backward(args, hf_config, cp=compare_cp)

    gradient_metrics = {
        name: _distributed_tensor_metrics(reference_gradients[name].to("cuda"), value.to("cuda"))
        for name, value in comparison_gradients.items()
    }
    loss_relative_diff = abs(comparison_loss["loss_sum"] - reference_loss["loss_sum"]) / max(
        abs(reference_loss["loss_sum"]), 1e-30
    )
    ce_relative_diff = abs(comparison_loss["ce_loss_sum"] - reference_loss["ce_loss_sum"]) / max(
        abs(reference_loss["ce_loss_sum"]), 1e-30
    )
    metrics = {
        "mode": "backward",
        "sequence_length": args.sequence_length,
        "global_sequences": dist.get_world_size(),
        "comparison_topology": comparison_name,
        "cp_degree": 2 if compare_cp else None,
        "ep_degree": args.ep_degree,
        "compile": args.compile,
        "reference": reference_loss,
        comparison_name: comparison_loss,
        "loss_relative_diff": loss_relative_diff,
        "ce_loss_relative_diff": ce_relative_diff,
        "gradients": gradient_metrics,
    }
    _write_metrics(args.output_json, metrics)
    if ce_relative_diff > args.max_loss_relative_diff:
        raise AssertionError(f"CE-loss parity failed: {metrics}")
    failures = {
        name: value
        for name, value in gradient_metrics.items()
        if value["cosine_similarity"] < args.min_gradient_cosine
    }
    if failures:
        raise AssertionError(f"Gradient parity failed: {failures}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("forward", "backward"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--token-ids", type=Path, required=True)
    parser.add_argument("--label-mask", type=Path)
    parser.add_argument("--document-metadata", type=Path)
    parser.add_argument("--hf-model", default="Qwen/Qwen3.5-35B-A3B-Base")
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--ep-degree", type=int, default=8)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--layerwise", action="store_true")
    parser.add_argument("--activation-memory-budget", type=float, default=0.3)
    parser.add_argument("--logit-positions-per-cp-rank", type=int, default=4)
    parser.add_argument("--gradient-sample-values", type=int, default=65_536)
    parser.add_argument("--min-hidden-cosine", type=float, default=0.999)
    parser.add_argument("--min-logit-cosine", type=float, default=0.999)
    parser.add_argument("--min-top1-agreement", type=float, default=0.99)
    parser.add_argument("--min-gradient-cosine", type=float, default=0.995)
    parser.add_argument("--max-loss-relative-diff", type=float, default=0.01)
    parser.add_argument("--backward-comparison", choices=("cp2", "no-cp"), default="cp2")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    args.sequence_length = args.sequence_length or (65_536 if args.mode == "forward" else 4_096)
    args.label_mask = args.label_mask or args.token_ids.with_name(
        args.token_ids.name.replace("token_ids_", "labels_mask_")
    )
    args.document_metadata = args.document_metadata or args.token_ids.with_suffix(".csv.gz")

    prepare_training_environment(seed=123, shared_filesystem=True)
    if dist.get_world_size() != args.ep_degree:
        raise RuntimeError(
            f"This diagnostic expects world size == EP degree, got {dist.get_world_size()} and {args.ep_degree}"
        )
    hf_config = AutoConfig.from_pretrained(args.hf_model, trust_remote_code=False).to_dict()
    if args.mode == "forward":
        verify_forward(args, hf_config)
    else:
        verify_backward(args, hf_config)


if __name__ == "__main__":
    main()
