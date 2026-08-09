"""Replay one exact Stage-1 batch and measure s002 MoE routing behavior.

The data batch is restored from a saved trainer/data-loader checkpoint independently of the
model checkpoint being evaluated. By default the next batch is replayed, or a later exact batch
can be selected with ``--replay-global-step``. This makes routing comparisons across Stage-1
checkpoints and rank microbatch sizes use the same packed examples, image augmentations, padding,
and token order. Run this on one complete EP group, normally eight GPUs for s002::

    torchrun --standalone --nproc-per-node=8 src/scripts/eval/s002_routing_probe.py \
        --checkpoint /path/to/model-checkpoint \
        --data-state-checkpoint /path/to/step6500 \
        --microbatch-instances 4 8
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.distributed as dist

from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.utils import move_to_device, seed_all
from scripts.eval.s002_downstream import (
    DEFAULT_OUTPUT_ROOT,
    _build_model_and_module_config,
    _checkpoint_state_dir,
    _config_path,
    _git_revision,
)

log = logging.getLogger(__name__)

DEFAULT_DATA_STATE_CHECKPOINT = (
    "/weka/oe-training-default/rustin/experiments/vision-moe/checkpoints/"
    "s002-stage1-corrected-clean-32k-b300-20260807/step2000"
)

_MODALITIES = ("all", "text", "image_patch", "image_structural")
_STAT_NAMES = (
    "normalized_entropy_sum",
    "top1_probability_sum",
    "top4_probability_mass_sum",
    "top4_to_top5_logit_margin_sum",
    "top4_to_top5_probability_margin_sum",
    "logit_margin_below_0.001",
    "logit_margin_below_0.01",
    "logit_margin_below_0.05",
    "token_count",
)


@dataclass
class _LayerChunkRecord:
    expert_counts: torch.Tensor
    modality_expert_counts: torch.Tensor
    modality_stats: torch.Tensor
    expert_indices: torch.Tensor
    valid_routes: torch.Tensor
    physical_routes: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--initial-stage1",
        action="store_true",
        help=(
            "Treat CHECKPOINT as the base LM checkpoint and reconstruct the exact initial "
            "Stage-1 composite model from DATA_STATE_CHECKPOINT/config.json."
        ),
    )
    parser.add_argument(
        "--config",
        help="Model config JSON (defaults to CHECKPOINT/config.json).",
    )
    parser.add_argument(
        "--data-state-checkpoint",
        default=DEFAULT_DATA_STATE_CHECKPOINT,
        help="Trainer checkpoint whose next packed batch is replayed.",
    )
    parser.add_argument(
        "--data-rank-offset",
        type=int,
        default=0,
        help="First original data rank to replay; use 0 or 8 for the two s002 EP groups.",
    )
    parser.add_argument(
        "--replay-global-step",
        type=int,
        help="Exact later training step to replay (defaults to the first step after the save).",
    )
    parser.add_argument("--ep-degree", type=int, default=8)
    parser.add_argument(
        "--microbatch-instances",
        type=int,
        nargs="+",
        default=[4, 8],
        help="Rank-local sequence counts to compare on the same eight-sequence data batch.",
    )
    parser.add_argument("--checkpoint-load-threads", type=int, default=8)
    parser.add_argument(
        "--data-prefetch-workers",
        type=int,
        default=0,
        help="Loader workers used while reconstructing the saved batch; order is unaffected.",
    )
    parser.add_argument("--output")
    return parser.parse_args()


def _load_stage1_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "train" / "Molmo2-Stage1.py"
    name = "_olmo_core_molmo2_stage1_routing_probe"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Stage-1 definitions from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _modality_masks(
    input_ids: torch.Tensor,
    token_type_ids: torch.Tensor,
    valid_tokens: torch.Tensor,
    *,
    image_patch_token_id: int,
) -> Dict[str, torch.Tensor]:
    if input_ids.shape != token_type_ids.shape or input_ids.shape != valid_tokens.shape:
        raise ValueError("input_ids, token_type_ids, and valid_tokens must have identical shapes")
    valid_tokens = valid_tokens.to(dtype=torch.bool)
    image_patch = valid_tokens & (input_ids == image_patch_token_id)
    image_structural = valid_tokens & (token_type_ids != 0) & ~image_patch
    text = valid_tokens & (token_type_ids == 0)
    if not torch.equal(text | image_patch | image_structural, valid_tokens):
        raise ValueError("The routing modality masks do not partition all valid tokens")
    if bool((text & image_patch).any()) or bool((text & image_structural).any()):
        raise ValueError("The routing modality masks overlap")
    return {
        "all": valid_tokens,
        "text": text,
        "image_patch": image_patch,
        "image_structural": image_structural,
    }


def _capacity_stats(
    global_expert_counts: torch.Tensor,
    *,
    ep_world_size: int,
    physical_routes_per_source: int,
    capacity_factor: float,
    global_valid_routes: int,
) -> Dict[str, Any]:
    if global_expert_counts.ndim != 1:
        raise ValueError("global_expert_counts must be one-dimensional")
    if global_expert_counts.numel() % ep_world_size != 0:
        raise ValueError("The expert count must be divisible by ep_world_size")
    if physical_routes_per_source <= 0 or global_valid_routes < 0:
        raise ValueError("Route counts must be positive, with non-negative valid routes")
    experts_per_rank = global_expert_counts.numel() // ep_world_size
    destination_counts = global_expert_counts.reshape(ep_world_size, experts_per_rank).sum(dim=1)
    rank_capacity = max(1, math.ceil(capacity_factor * physical_routes_per_source))
    dropped = (destination_counts - rank_capacity).clamp_min(0)
    accepted_counts = destination_counts.clamp_max(rank_capacity)
    return {
        "rank_capacity": rank_capacity,
        "destination_route_counts": [int(value) for value in destination_counts.tolist()],
        "destination_utilization": [
            float(value) / rank_capacity for value in accepted_counts.tolist()
        ],
        "max_destination_utilization": float(accepted_counts.max()) / rank_capacity,
        "requested_destination_pressure": [
            float(value) / rank_capacity for value in destination_counts.tolist()
        ],
        "max_requested_destination_pressure": float(destination_counts.max()) / rank_capacity,
        "dropped_routes": int(dropped.sum()),
        "global_valid_routes": global_valid_routes,
        "global_drop_rate": (
            float(dropped.sum()) / global_valid_routes if global_valid_routes else 0.0
        ),
    }


def _slice_batch(batch: Mapping[str, Any], start: int, end: int) -> Dict[str, Any]:
    batch_size = int(batch["input_ids"].shape[0])
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch_size:
            out[key] = value[start:end]
        elif isinstance(value, list) and len(value) == batch_size:
            out[key] = value[start:end]
        else:
            out[key] = value
    return out


def _batch_fingerprint(batch: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key]
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(memoryview(tensor.numpy()).cast("B"))
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _resolve_replay_step(saved_global_step: int, replay_global_step: int | None) -> tuple[int, int]:
    target = saved_global_step + 1 if replay_global_step is None else replay_global_step
    if target <= saved_global_step:
        raise ValueError(f"Replay step {target} must be later than saved step {saved_global_step}")
    return target, target - saved_global_step


class _RoutingRecorder:
    def __init__(self, routers: Mapping[str, torch.nn.Module], image_patch_token_id: int):
        self.routers = dict(routers)
        self.image_patch_token_id = image_patch_token_id
        self.records: Dict[str, List[_LayerChunkRecord]] = {layer: [] for layer in self.routers}
        self._masks: Dict[str, torch.Tensor] | None = None
        self._handles = [
            router.register_forward_hook(self._make_hook(layer), with_kwargs=True)
            for layer, router in self.routers.items()
        ]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def reset(self) -> None:
        self.records = {layer: [] for layer in self.routers}
        self._masks = None

    def set_batch(self, batch: Mapping[str, Any]) -> None:
        self._masks = _modality_masks(
            batch["input_ids"],
            batch["token_type_ids"],
            batch["router_token_mask"],
            image_patch_token_id=self.image_patch_token_id,
        )

    def clear_batch(self) -> None:
        self._masks = None

    def _make_hook(self, layer: str):
        def hook(module, args, kwargs, output):
            del args
            if self._masks is None:
                raise RuntimeError("Routing recorder has no active batch masks")
            expert_weights, expert_indices, batch_size_per_expert, aux_info = output
            del expert_weights, batch_size_per_expert
            if expert_indices is None or aux_info is None:
                raise RuntimeError(f"Routed layer {layer} returned no routing assignments")
            expert_indices = expert_indices.detach()
            scores, logits = (value.detach() for value in aux_info[:2])
            token_mask = kwargs.get("token_mask")
            if token_mask is not None and not torch.equal(
                token_mask.to(dtype=torch.bool), self._masks["all"]
            ):
                raise RuntimeError(f"Routed layer {layer} received an unexpected token mask")

            num_experts = int(module.num_experts)
            counts = []
            for modality in _MODALITIES:
                selected = expert_indices[self._masks[modality]].reshape(-1)
                counts.append(torch.bincount(selected, minlength=num_experts))
            modality_expert_counts = torch.stack(counts)

            entropy = -(scores * scores.clamp_min(torch.finfo(scores.dtype).tiny).log()).sum(
                dim=-1
            ) / math.log(num_experts)
            top_logits, top_indices = logits.topk(5, dim=-1)
            top_probabilities = scores.gather(-1, top_indices)
            logit_margin = top_logits[..., 3] - top_logits[..., 4]
            probability_margin = top_probabilities[..., 3] - top_probabilities[..., 4]

            stats = []
            for modality in _MODALITIES:
                mask = self._masks[modality]
                stats.append(
                    torch.stack(
                        [
                            entropy[mask].sum(),
                            top_probabilities[..., 0][mask].sum(),
                            top_probabilities[..., :4].sum(dim=-1)[mask].sum(),
                            logit_margin[mask].sum(),
                            probability_margin[mask].sum(),
                            (logit_margin[mask] < 0.001).sum().to(dtype=torch.float32),
                            (logit_margin[mask] < 0.01).sum().to(dtype=torch.float32),
                            (logit_margin[mask] < 0.05).sum().to(dtype=torch.float32),
                            mask.sum().to(dtype=torch.float32),
                        ]
                    )
                )

            valid_tokens = self._masks["all"].sum(dtype=torch.long)
            self.records[layer].append(
                _LayerChunkRecord(
                    expert_counts=modality_expert_counts[0],
                    modality_expert_counts=modality_expert_counts,
                    modality_stats=torch.stack(stats),
                    expert_indices=expert_indices[self._masks["all"]].detach().cpu(),
                    valid_routes=valid_tokens * int(module.top_k),
                    physical_routes=expert_indices.numel(),
                )
            )

        return hook


def _collective_sum(value: torch.Tensor) -> torch.Tensor:
    out = value.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def _layer_mode_summary(
    records: Sequence[_LayerChunkRecord],
    *,
    capacity_factor: float,
    ep_world_size: int,
) -> Dict[str, Any]:
    global_counts_by_chunk = []
    capacity_by_chunk = []
    for record in records:
        global_counts = _collective_sum(record.expert_counts)
        global_valid_routes = int(_collective_sum(record.valid_routes).item())
        global_counts_by_chunk.append(global_counts)
        capacity_by_chunk.append(
            _capacity_stats(
                global_counts.cpu(),
                ep_world_size=ep_world_size,
                physical_routes_per_source=record.physical_routes,
                capacity_factor=capacity_factor,
                global_valid_routes=global_valid_routes,
            )
        )

    local_total_counts = torch.stack([record.expert_counts for record in records]).sum(dim=0)
    local_imbalance = (
        local_total_counts.max().float() / local_total_counts.float().mean().clamp_min(1)
    )
    dist.all_reduce(local_imbalance, op=dist.ReduceOp.MAX)

    global_counts = torch.stack(global_counts_by_chunk).sum(dim=0)
    modality_counts = _collective_sum(
        torch.stack([record.modality_expert_counts for record in records]).sum(dim=0)
    )
    modality_stats = _collective_sum(
        torch.stack([record.modality_stats for record in records]).sum(dim=0)
    )

    modality_summaries: Dict[str, Any] = {}
    for modality_idx, modality in enumerate(_MODALITIES):
        counts = modality_counts[modality_idx]
        values = modality_stats[modality_idx]
        token_count = float(values[-1])
        means: Dict[str, Any] = {
            name.removesuffix("_sum"): float(values[idx]) / max(token_count, 1.0)
            for idx, name in enumerate(_STAT_NAMES[:5])
        }
        means.update(
            {
                "fraction_logit_margin_below_0.001": float(values[5]) / max(token_count, 1.0),
                "fraction_logit_margin_below_0.01": float(values[6]) / max(token_count, 1.0),
                "fraction_logit_margin_below_0.05": float(values[7]) / max(token_count, 1.0),
                "token_count": int(token_count),
                "requested_expert_route_counts": [int(value) for value in counts.cpu().tolist()],
                "requested_expert_load_imbalance": float(
                    counts.max().float() / counts.float().mean().clamp_min(1)
                ),
                "requested_expert_count_cv": float(
                    counts.float().std(unbiased=False) / counts.float().mean().clamp_min(1)
                ),
            }
        )
        modality_summaries[modality] = means

    total_dropped = sum(chunk["dropped_routes"] for chunk in capacity_by_chunk)
    total_valid = sum(chunk["global_valid_routes"] for chunk in capacity_by_chunk)
    return {
        "capacity_factor": capacity_factor,
        "physical_forwards": len(records),
        "max_symm_buffer_utilization": max(
            chunk["max_destination_utilization"] for chunk in capacity_by_chunk
        ),
        "max_requested_destination_pressure": max(
            chunk["max_requested_destination_pressure"] for chunk in capacity_by_chunk
        ),
        "predicted_dropped_routes": total_dropped,
        "predicted_global_drop_rate": total_dropped / max(total_valid, 1),
        "valid_routes": total_valid,
        "requested_local_expert_load_imbalance_max": float(local_imbalance),
        "requested_global_expert_load_imbalance": float(
            global_counts.max().float() / global_counts.float().mean().clamp_min(1)
        ),
        "requested_global_expert_count_cv": float(
            global_counts.float().std(unbiased=False) / global_counts.float().mean().clamp_min(1)
        ),
        "requested_global_expert_route_counts": [
            int(value) for value in global_counts.cpu().tolist()
        ],
        "forwards": capacity_by_chunk,
        "modalities": modality_summaries,
    }


def _run_mode(
    train_module,
    recorder: _RoutingRecorder,
    batch: Mapping[str, Any],
    *,
    microbatch_instances: int,
    capacity_factors: Mapping[str, float],
    ep_world_size: int,
) -> tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    recorder.reset()
    batch_instances = int(batch["input_ids"].shape[0])
    if batch_instances % microbatch_instances != 0:
        raise ValueError(
            f"Batch size {batch_instances} is not divisible by microbatch size "
            f"{microbatch_instances}"
        )

    started = time.monotonic()
    for start in range(0, batch_instances, microbatch_instances):
        cpu_microbatch = _slice_batch(batch, start, start + microbatch_instances)
        gpu_microbatch = move_to_device(cpu_microbatch, train_module.device)
        recorder.set_batch(gpu_microbatch)
        output = train_module.eval_batch(gpu_microbatch)
        recorder.clear_batch()
        del output, gpu_microbatch, cpu_microbatch
    torch.cuda.synchronize()

    layers = {}
    assignments = {}
    for layer, records in recorder.records.items():
        expected_forwards = batch_instances // microbatch_instances
        if len(records) != expected_forwards:
            raise RuntimeError(
                f"Layer {layer} produced {len(records)} hook records, expected {expected_forwards}"
            )
        layers[layer] = _layer_mode_summary(
            records,
            capacity_factor=capacity_factors[layer],
            ep_world_size=ep_world_size,
        )
        assignments[layer] = torch.cat([record.expert_indices for record in records], dim=0)

    return (
        {
            "microbatch_instances": microbatch_instances,
            "microbatch_tokens": microbatch_instances * int(batch["input_ids"].shape[1]),
            "elapsed_seconds": time.monotonic() - started,
            "layers": layers,
        },
        assignments,
    )


def _compare_assignments(
    assignments: Mapping[int, Mapping[str, torch.Tensor]],
) -> Dict[str, Any]:
    modes = sorted(assignments)
    if len(modes) < 2:
        return {}
    baseline = modes[0]
    out: Dict[str, Any] = {}
    for candidate in modes[1:]:
        layers = {}
        for layer, expected in assignments[baseline].items():
            actual = assignments[candidate][layer]
            if expected.shape != actual.shape:
                raise RuntimeError(
                    f"Layer {layer} assignment shapes differ between modes: "
                    f"{tuple(expected.shape)} != {tuple(actual.shape)}"
                )
            local = torch.tensor(
                [(expected != actual).sum(), expected.numel()],
                device=torch.cuda.current_device(),
                dtype=torch.long,
            )
            global_counts = _collective_sum(local)
            layers[layer] = {
                "different_routes": int(global_counts[0]),
                "total_routes": int(global_counts[1]),
                "different_route_fraction": float(global_counts[0]) / max(int(global_counts[1]), 1),
            }
        out[f"microbatch_{baseline}_vs_{candidate}"] = layers
    return out


def _restore_batch(args: argparse.Namespace, stage1: ModuleType):
    data_checkpoint = Path(args.data_state_checkpoint).resolve()
    with (data_checkpoint / "config.json").open() as f:
        config = stage1.ExperimentConfig.from_dict(json.load(f))
    tokenizer, token_ids = stage1._load_tokenizer(config.tokenizer_id, config.hf_cache_dir)
    datasets, weights, names = stage1._build_mixture_sources(tokenizer, config)

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    original_rank = args.data_rank_offset + rank
    state_path = data_checkpoint / "train" / f"rank{original_rank}.pt"
    trainer_state = torch.load(state_path, map_location="cpu", weights_only=False)
    data_state = trainer_state["data_loader"]
    saved_global_step = int(trainer_state["global_step"])
    saved_batches_processed = int(data_state["batches_processed"])
    if saved_batches_processed != saved_global_step:
        raise RuntimeError(
            "Cannot map trainer steps to packed batches: "
            f"global_step={saved_global_step}, batches_processed={saved_batches_processed}"
        )
    replayed_global_step, batches_to_replay = _resolve_replay_step(
        saved_global_step, args.replay_global_step
    )
    packing_state = data_state.get("packing_state") or {}
    original_world_size = int(packing_state.get("dp_world_size", world_size))
    if args.data_rank_offset < 0 or args.data_rank_offset + world_size > original_world_size:
        raise ValueError(
            f"Ranks [{args.data_rank_offset}, {args.data_rank_offset + world_size}) do not fit "
            f"the saved data world size {original_world_size}"
        )
    if packing_state.get("dp_rank", original_rank) != original_rank:
        raise ValueError(f"{state_path} does not contain data rank {original_rank}")

    work_dir = Path(DEFAULT_OUTPUT_ROOT) / "_routing_probe_loader_work"
    loader = stage1.MixtureDataLoader(
        datasets,
        weights,
        config.collator.build(),
        work_dir=work_dir,
        global_batch_size=config.global_batch_size,
        seed=config.data_seed,
        pack=config.pack_sequences,
        pack_max_crops=config.pack_max_crops if config.pack_sequences else None,
        pack_buffer_size=config.pack_buffer_size if config.pack_sequences else 0,
        prefetch_workers=args.data_prefetch_workers,
        dataset_names=names,
        dp_world_size=original_world_size,
        dp_rank=original_rank,
    )
    loader.load_state_dict(data_state)
    loader.reshuffle(epoch=data_state.get("epoch"))
    iterator = iter(loader)
    batch = None
    for _ in range(batches_to_replay):
        batch = next(iterator)
    assert batch is not None
    expected_batch = saved_batches_processed + batches_to_replay
    if loader.batches_processed != expected_batch:
        raise RuntimeError(
            f"Replayed data batch {loader.batches_processed}, expected {expected_batch}"
        )
    return (
        batch,
        config,
        token_ids,
        trainer_state,
        original_rank,
        state_path,
        replayed_global_step,
    )


def _default_output(checkpoint: Path, data_checkpoint: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    checkpoint_name = (
        checkpoint.parent.name if checkpoint.name == "model_and_optim" else checkpoint.name
    )
    return (
        Path(DEFAULT_OUTPUT_ROOT)
        / checkpoint_name
        / f"routing-probe-data-{data_checkpoint.name}-{stamp}.json"
    )


@torch.no_grad()
def _reset_image_token_rows_for_eval(train_module, token_ids: Sequence[int], *, seed: int) -> None:
    """Initialize the s002 input-only image rows without requiring an eval-only optimizer."""
    if not token_ids or len(set(token_ids)) != len(token_ids):
        raise ValueError("token_ids must be a non-empty sequence of unique IDs")

    lm = train_module.multimodal_model.lm
    if lm.embeddings is None or lm.lm_head is None:
        raise RuntimeError("Image-token initialization requires LM embeddings and an LM head")
    if min(token_ids) < 0 or max(token_ids) >= lm.vocab_size:
        raise ValueError(f"Image token IDs must be within [0, {lm.vocab_size}), got {token_ids}")

    generator = torch.Generator(device=train_module.device).manual_seed(seed)
    row_index = torch.tensor(token_ids, device=train_module.device, dtype=torch.long)
    embedding_rows = torch.nn.Embedding(
        len(token_ids),
        lm.d_model,
        device=train_module.device,
        dtype=lm.embeddings.weight.dtype,
    )
    lm.init_method.init_embeddings(
        embedding_rows,
        d_model=lm.d_model,
        embed_scale=lm.embed_scale,
        std=lm.embedding_init_std if lm.embedding_init_std is not None else lm.init_std,
        generator=generator,
    )
    lm.embeddings.weight.index_copy_(0, row_index, embedding_rows.weight)


def _load_initial_stage1_weights(
    train_module,
    config,
    token_ids,
    *,
    base_checkpoint: Path,
    checkpoint_load_threads: int,
) -> Path:
    """Load the exact model state used immediately before the first Stage-1 optimizer step."""
    base_state_dir = _checkpoint_state_dir(base_checkpoint)
    log.info("Loading initial s002 language-model weights from %s", base_state_dir)
    train_module.load_state_dict_direct(
        base_state_dir,
        process_group=dist.group.WORLD,
        thread_count=checkpoint_load_threads,
        load_optim_state=False,
    )

    from olmo_core.nn.vision import (
        load_siglip_hf_vision_state_dict,
        siglip_hf_state_dict_to_vision,
        vision_state_fingerprint,
    )

    hf_vision_state = load_siglip_hf_vision_state_dict(
        config.vision_model_id,
        revision=config.vision_revision,
        cache_dir=config.hf_cache_dir,
    )
    vision_state = siglip_hf_state_dict_to_vision(
        hf_vision_state, train_module.multimodal_model.cfg.vision
    )
    fingerprint = vision_state_fingerprint(vision_state)
    if fingerprint != config.vision_fingerprint:
        raise ValueError(
            "SigLIP2 vision checkpoint fingerprint mismatch: "
            f"expected {config.vision_fingerprint}, got {fingerprint}"
        )
    train_module.load_vision_state_dict(vision_state)

    _reset_image_token_rows_for_eval(
        train_module,
        [
            token_ids.im_start_id,
            token_ids.im_end_id,
            token_ids.im_patch_id,
            token_ids.im_col_id,
            token_ids.low_res_im_start_id,
            token_ids.image_placeholder_id,
        ],
        seed=config.init_seed,
    )
    return base_state_dir


def main() -> None:
    args = _parse_args()
    if args.ep_degree <= 0:
        raise ValueError("--ep-degree must be positive")
    if len(set(args.microbatch_instances)) != len(args.microbatch_instances):
        raise ValueError("--microbatch-instances values must be unique")
    if any(value <= 0 for value in args.microbatch_instances):
        raise ValueError("--microbatch-instances values must be positive")
    if args.data_prefetch_workers < 0:
        raise ValueError("--data-prefetch-workers must be non-negative")

    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prepare_training_environment()

    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        if world_size != args.ep_degree:
            raise ValueError(
                "The fixed-batch probe must run on exactly one complete EP group: "
                f"WORLD_SIZE={world_size}, ep_degree={args.ep_degree}"
            )

        stage1 = _load_stage1_module()
        (
            batch,
            data_config,
            token_ids,
            trainer_state,
            original_rank,
            state_path,
            replayed_global_step,
        ) = _restore_batch(args, stage1)
        batch_instances = int(batch["input_ids"].shape[0])
        if any(batch_instances % value for value in args.microbatch_instances):
            raise ValueError(
                f"Saved rank batch has {batch_instances} instances and cannot be evenly split "
                f"by every requested microbatch size {args.microbatch_instances}"
            )
        local_fingerprint = _batch_fingerprint(batch)
        fingerprints: List[str] = [""] * world_size
        dist.all_gather_object(fingerprints, local_fingerprint)

        checkpoint = Path(args.checkpoint).resolve()
        if args.initial_stage1:
            config_path = (
                Path(args.config).resolve()
                if args.config
                else Path(args.data_state_checkpoint).resolve() / "config.json"
            )
            seed_all(data_config.init_seed)
        else:
            config_path = _config_path(checkpoint, args.config)
        with config_path.open() as f:
            raw_model_config = json.load(f)
        model, module_config, checkpoint_kind = _build_model_and_module_config(
            raw_model_config,
            ep_degree=args.ep_degree,
            max_sequence_length=int(batch["input_ids"].shape[1]),
            rank_batch_size=max(args.microbatch_instances) * int(batch["input_ids"].shape[1]),
            ep_path=ExpertParallelPath.rowwise_nvshmem,
        )
        if checkpoint_kind != "multimodal_stage1":
            raise ValueError("The routing probe requires a multimodal Stage-1 checkpoint")
        module_config.response_logits_only = True
        train_module = module_config.build(model, eval_only=True)
        if args.initial_stage1:
            state_dir = _load_initial_stage1_weights(
                train_module,
                data_config,
                token_ids,
                base_checkpoint=checkpoint,
                checkpoint_load_threads=args.checkpoint_load_threads,
            )
            checkpoint_kind = "multimodal_stage1_initial_from_base"
        else:
            state_dir = _checkpoint_state_dir(checkpoint)
            log.info("Loading model checkpoint from %s", state_dir)
            train_module.load_state_dict_direct(
                state_dir,
                process_group=dist.group.WORLD,
                thread_count=args.checkpoint_load_threads,
                load_optim_state=False,
            )

        blocks = dict(train_module.multimodal_model.lm.named_routed_blocks())
        routers = {layer: block.routed_experts_router for layer, block in blocks.items()}
        if any(router is None for router in routers.values()):
            raise RuntimeError("A routed block is missing its router")
        capacity_factors = {
            layer: float(block.ep.capacity_factor) for layer, block in blocks.items()
        }
        recorder = _RoutingRecorder(routers, token_ids.im_patch_id)
        try:
            modes: Dict[str, Any] = {}
            assignments: Dict[int, Dict[str, torch.Tensor]] = {}
            for microbatch_instances in args.microbatch_instances:
                log.info(
                    "Replaying data step %d with %d sequences per rank microbatch",
                    replayed_global_step,
                    microbatch_instances,
                )
                summary, mode_assignments = _run_mode(
                    train_module,
                    recorder,
                    batch,
                    microbatch_instances=microbatch_instances,
                    capacity_factors=capacity_factors,
                    ep_world_size=args.ep_degree,
                )
                modes[str(microbatch_instances)] = summary
                assignments[microbatch_instances] = mode_assignments
        finally:
            recorder.close()

        payload = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint),
            "checkpoint_state_dir": str(state_dir),
            "checkpoint_kind": checkpoint_kind,
            "config": str(config_path),
            "git": _git_revision(),
            "protocol": {
                "world_size": world_size,
                "ep_degree": args.ep_degree,
                "attention_backend": "flex",
                "ep_path": ExpertParallelPath.rowwise_nvshmem.value,
                "route_count_semantics": "requested_pre_drop",
                "initial_stage1": args.initial_stage1,
                "microbatch_instances": args.microbatch_instances,
                "sequence_length": int(batch["input_ids"].shape[1]),
                "data_prefetch_workers": args.data_prefetch_workers,
            },
            "fixed_batch": {
                "data_state_checkpoint": str(Path(args.data_state_checkpoint).resolve()),
                "saved_global_step": int(trainer_state["global_step"]),
                "replayed_global_step": replayed_global_step,
                "original_data_world_size": int(
                    trainer_state["data_loader"]["packing_state"]["dp_world_size"]
                ),
                "original_data_ranks": list(
                    range(args.data_rank_offset, args.data_rank_offset + world_size)
                ),
                "rank_state_path": str(state_path),
                "local_original_rank": original_rank,
                "rank_batch_instances": batch_instances,
                "rank_batch_tokens": batch_instances * int(batch["input_ids"].shape[1]),
                "rank_fingerprints": fingerprints,
                "local_valid_tokens": int(batch["router_token_mask"].sum()),
                "local_packing_fill": float(batch["router_token_mask"].float().mean()),
                "local_source_names": batch.get("pack_source_names"),
                "data_config": {
                    "global_batch_size": data_config.global_batch_size,
                    "data_seed": data_config.data_seed,
                    "pack_sequences": data_config.pack_sequences,
                    "pack_buffer_size": data_config.pack_buffer_size,
                    "pack_max_crops": data_config.pack_max_crops,
                },
            },
            "modes": modes,
            "assignment_comparisons": _compare_assignments(assignments),
        }

        output = (
            Path(args.output)
            if args.output
            else _default_output(checkpoint, Path(args.data_state_checkpoint).resolve())
        )
        if rank == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            temporary.replace(output)
            log.info("Wrote routing probe to %s", output)
        dist.barrier()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
