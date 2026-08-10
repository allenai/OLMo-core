"""Evaluate native s002 multimodal checkpoints on fixed PixMo validation samples.

This evaluator keeps checkpoints in OLMo-core's distributed format, runs the model with
EP8, and reports response-token-weighted CE/PPL for matched caption, count-only, and
basic-pointing samples. The same explicit sample indices and OLMo 3 chat serializer are
used for every checkpoint in a comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch.distributed as dist
from s002_downstream import (
    DEFAULT_OUTPUT_ROOT,
    _build_model_and_module_config,
    _checkpoint_state_dir,
    _config_path,
    _git_revision,
)

from olmo_core.data.multimodal import (
    PIXMO_DATASETS,
    MultimodalCollator,
    MultimodalDataLoader,
    PixMoCapDatasetConfig,
    PixMoCountDatasetConfig,
    PixMoPointsDatasetConfig,
)
from olmo_core.data.multimodal.sft_common import (
    SFT_MESSAGE_FORMATS,
    MaxSequenceLengthDataset,
    SftMessageFormat,
    validate_sft_message_format,
)
from olmo_core.distributed.utils import all_reduce_value, get_rank, get_world_size
from olmo_core.eval import MultimodalLMEvaluator
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.vision import prepare_molmo2_tokenizer
from olmo_core.nn.vision.molmo2_tokens import Molmo2TokenIds
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.utils import gc_cuda, move_to_device

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = (
    "/weka/oe-training-default/rustin/experiments/vision-moe/checkpoints/"
    "s002-stage2-v9-pilot-bounded-errors-5a81c40c/step200"
)
DEFAULT_TOKENIZER = (
    "/weka/oe-training-default/robertb/olmo3moe-post-training/checkpoints/"
    "s002-olmo3moe-instruct-sft-resume-to1000-fused-20260727-hf"
)
DEFAULT_HF_CACHE = "/weka/oe-training-default/rustin/hf-cache/hub"
DEFAULT_EXAMPLES = 512
DEFAULT_SAMPLE_SEED = 6198
DEFAULT_MAX_SEQUENCE_LENGTH = 16384
DEFAULT_MAX_CROPS = 8
DEFAULT_RANK_BATCH_INSTANCES = 2
TASK_NAMES = ("caption", "count", "points")
TASK_SEED_OFFSETS = {"caption": 0, "count": 1, "points": 2}


@dataclass(frozen=True)
class TaskSpec:
    """A named validation dataset and its deterministic source indices."""

    name: str
    dataset: Any
    indices: Sequence[int]


class IndexedDataset:
    """Expose a fixed source-index selection while preserving source-epoch semantics."""

    def __init__(self, dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = tuple(int(index) for index in indices)
        self.config = getattr(dataset, "config", None)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.get(index, 0)

    def get(self, index: int, epoch: int = 0):
        source_index = self.indices[index]
        getter = getattr(self.dataset, "get", None)
        return getter(source_index, epoch) if getter is not None else self.dataset[source_index]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", help="Config JSON (defaults to CHECKPOINT/config.json).")
    parser.add_argument("--tasks", nargs="+", choices=TASK_NAMES, default=list(TASK_NAMES))
    parser.add_argument("--examples", type=int, default=DEFAULT_EXAMPLES)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--ep-degree", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    parser.add_argument("--rank-batch-instances", type=int, default=DEFAULT_RANK_BATCH_INSTANCES)
    parser.add_argument("--max-crops", type=int, default=DEFAULT_MAX_CROPS)
    parser.add_argument(
        "--message-format",
        choices=SFT_MESSAGE_FORMATS,
        default="olmo3_chat",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--hf-cache", default=DEFAULT_HF_CACHE)
    parser.add_argument("--work-dir", help="Local evaluator work directory.")
    parser.add_argument("--output", help="Result JSON path under Rustin's eval root by default.")
    parser.add_argument("--checkpoint-load-threads", type=int, default=8)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace, world_size: int) -> None:
    if args.examples <= 0:
        raise ValueError("--examples must be positive")
    if args.ep_degree <= 0 or world_size % args.ep_degree != 0:
        raise ValueError(
            f"WORLD_SIZE ({world_size}) must be divisible by positive --ep-degree "
            f"({args.ep_degree})"
        )
    if args.max_sequence_length <= 0:
        raise ValueError("--max-sequence-length must be positive")
    if args.rank_batch_instances <= 0:
        raise ValueError("--rank-batch-instances must be positive")
    if args.max_crops <= 0:
        raise ValueError("--max-crops must be positive")
    if args.checkpoint_load_threads <= 0:
        raise ValueError("--checkpoint-load-threads must be positive")
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError("--tasks must not contain duplicates")


def _representative_indices(size: int, examples: int, *, seed: int) -> List[int]:
    """Select a deterministic, without-replacement permutation prefix."""
    if size <= 0:
        raise ValueError("dataset size must be positive")
    if examples > size:
        raise ValueError(f"Requested {examples} examples from a dataset of size {size}")
    return np.random.RandomState(seed).permutation(size)[:examples].astype(int).tolist()


def _indices_sha256(indices: Sequence[int]) -> str:
    encoded = json.dumps(list(indices), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_task_datasets(
    tokenizer,
    token_ids: Molmo2TokenIds,
    *,
    message_format: SftMessageFormat,
    max_sequence_length: int,
    max_crops: int,
    sample_seed: int,
) -> Dict[str, Any]:
    common: Dict[str, Any] = {
        "max_crops": max_crops,
        "loss_token_weighting": "none",
        "token_ids": token_ids,
        "message_format": message_format,
        "seed": sample_seed,
    }
    return {
        "caption": PixMoCapDatasetConfig(
            dataset_path=f"{PIXMO_DATASETS}/cap",
            split="validation",
            mode="caption",
            max_sequence_length=max_sequence_length,
            **common,
        ).build(tokenizer),
        "count": PixMoCountDatasetConfig(
            split="validation",
            counting=True,
            **common,
        ).build(tokenizer),
        "points": PixMoPointsDatasetConfig(
            split="validation",
            kind="basic",
            counting=False,
            **common,
        ).build(tokenizer),
    }


def _build_task_specs(
    datasets: Dict[str, Any],
    tasks: Sequence[str],
    *,
    examples: int,
    sample_seed: int,
) -> List[TaskSpec]:
    specs = []
    for name in tasks:
        dataset = datasets[name]
        indices = _representative_indices(
            len(dataset),
            examples,
            seed=sample_seed + TASK_SEED_OFFSETS[name],
        )
        specs.append(TaskSpec(name=name, dataset=dataset, indices=indices))
    return specs


def _default_output(checkpoint: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    checkpoint_name = (
        checkpoint.parent.name if checkpoint.name == "model_and_optim" else checkpoint.name
    )
    return Path(DEFAULT_OUTPUT_ROOT) / checkpoint_name / f"fast-vision-complete-{stamp}.json"


def _evaluate_task(
    train_module,
    task: TaskSpec,
    collator: MultimodalCollator,
    *,
    work_dir: Path,
    max_sequence_length: int,
    rank_batch_instances: int,
    sample_seed: int,
    token_ids: Molmo2TokenIds,
    dp_world_size: int,
    dp_rank: int,
) -> Dict[str, Any]:
    global_batch_instances = rank_batch_instances * dp_world_size
    if len(task.indices) % global_batch_instances != 0:
        raise ValueError(
            f"Task {task.name!r} has {len(task.indices)} examples, which is not divisible by "
            f"the global data-parallel batch of {global_batch_instances}"
        )

    selected = IndexedDataset(task.dataset, task.indices)
    bounded = MaxSequenceLengthDataset(
        selected,
        max_sequence_length,
        token_ids=token_ids,
    )
    loader = MultimodalDataLoader(
        bounded,
        collator,
        work_dir=work_dir / task.name,
        global_batch_size=global_batch_instances * max_sequence_length,
        seed=sample_seed,
        shuffle=False,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
    )
    evaluator = MultimodalLMEvaluator(
        name=f"pixmo-{task.name}-validation",
        batches=loader,
        device=train_module.device,
        process_group=train_module.dp_process_group,
        deterministic=True,
    )
    evaluator.reset_metrics()

    started = time.monotonic()
    batches = 0
    local_examples = 0
    for batch in evaluator:
        batches += 1
        local_examples += int(batch["input_ids"].shape[0])
        batch = move_to_device(batch, train_module.device)
        output = train_module.eval_batch(dict(batch))
        if not isinstance(output, LMOutputWithLoss):
            raise TypeError(f"Expected LMOutputWithLoss, got {type(output).__name__}")
        evaluator.update_metrics(batch, output.ce_loss, output.logits)
        if get_rank() == 0 and (batches == 1 or batches % 20 == 0):
            log.info("[%s] batch %d/%d", task.name, batches, len(loader))

    expected_local_examples = len(task.indices) // dp_world_size
    if local_examples != expected_local_examples:
        raise RuntimeError(
            f"Task {task.name!r} evaluated {local_examples} local examples, expected "
            f"{expected_local_examples}"
        )

    # MeanMetric.compute() reduces its internal tensors in place, so preserve the local
    # weight before computing the globally reduced CE/PPL.
    local_response_token_weight = evaluator.ce_loss.weight.detach().clone()
    metrics = {
        name: float(value.detach().cpu().item())
        for name, value in evaluator.compute_metrics().items()
    }
    response_token_weight = all_reduce_value(
        local_response_token_weight,
        train_module.device,
        group=train_module.dp_process_group,
    )
    result = {
        "metrics": metrics,
        "dataset_size": len(task.dataset),
        "examples": len(task.indices),
        "sample_indices": list(task.indices),
        "sample_indices_sha256": _indices_sha256(task.indices),
        "batches_per_dp_rank": batches,
        "examples_per_dp_rank": local_examples,
        "response_token_weight": float(response_token_weight.detach().cpu().item()),
        "elapsed_seconds": time.monotonic() - started,
    }
    if get_rank() == 0:
        log.info("Finished %s: %s", task.name, metrics)
    del evaluator, loader, bounded, selected
    gc_cuda()
    return result


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _validate_args(args, world_size)

    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prepare_training_environment()

    try:
        from transformers import GPT2Tokenizer

        checkpoint = Path(args.checkpoint).resolve()
        config_path = _config_path(checkpoint, args.config)
        with config_path.open() as config_file:
            raw_config = json.load(config_file)

        model, module_config, detected_checkpoint_kind = _build_model_and_module_config(
            raw_config,
            ep_degree=args.ep_degree,
            max_sequence_length=args.max_sequence_length,
            rank_batch_size=args.rank_batch_instances * args.max_sequence_length,
        )
        if detected_checkpoint_kind != "multimodal_stage1":
            raise ValueError("Fast vision evaluation requires a multimodal checkpoint")
        train_module = module_config.build(model, eval_only=True)
        state_dir = _checkpoint_state_dir(checkpoint)
        log.info("Loading multimodal checkpoint from %s", state_dir)
        train_module.load_state_dict_direct(
            state_dir,
            process_group=dist.group.WORLD,
            thread_count=args.checkpoint_load_threads,
            load_optim_state=False,
        )

        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer, cache_dir=args.hf_cache)
        token_ids = prepare_molmo2_tokenizer(tokenizer, model_vocab_size=100352)
        message_format = validate_sft_message_format(
            args.message_format,
            tokenizer=tokenizer,
            token_ids=token_ids,
        )
        if raw_config["model"]["image_patch_token_id"] != token_ids.im_patch_id:
            raise ValueError(
                "Tokenizer image-patch ID does not match checkpoint model config: "
                f"{token_ids.im_patch_id} != {raw_config['model']['image_patch_token_id']}"
            )
        if tokenizer.pad_token_id is None:
            raise ValueError(f"Tokenizer {args.tokenizer!r} does not define a pad token")

        collator = MultimodalCollator(
            pad_token_id=int(tokenizer.pad_token_id),
            label_ignore_index=-100,
            pad_sequence_length=args.max_sequence_length,
        )
        datasets = _build_task_datasets(
            tokenizer,
            token_ids,
            message_format=message_format,
            max_sequence_length=args.max_sequence_length,
            max_crops=args.max_crops,
            sample_seed=args.sample_seed,
        )
        tasks = _build_task_specs(
            datasets,
            args.tasks,
            examples=args.examples,
            sample_seed=args.sample_seed,
        )

        dp_process_group = train_module.dp_process_group
        dp_world_size = get_world_size(dp_process_group)
        dp_rank = get_rank(dp_process_group)
        global_batch_instances = args.rank_batch_instances * dp_world_size
        if args.examples % global_batch_instances != 0:
            raise ValueError(
                f"--examples ({args.examples}) must be divisible by the global data-parallel batch "
                f"({global_batch_instances})"
            )

        output = Path(args.output) if args.output else _default_output(checkpoint)
        work_dir = (
            Path(args.work_dir)
            if args.work_dir
            else Path(os.environ.get("RESULTS_DIR", "/tmp")) / "s002-fast-vision"
        )
        results = {
            task.name: _evaluate_task(
                train_module,
                task,
                collator,
                work_dir=work_dir,
                max_sequence_length=args.max_sequence_length,
                rank_batch_instances=args.rank_batch_instances,
                sample_seed=args.sample_seed,
                token_ids=token_ids,
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
            )
            for task in tasks
        }

        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint),
            "checkpoint_state_dir": str(state_dir),
            # The shared detector labels every native multimodal s002 config
            # ``multimodal_stage1`` because both training stages have the same tensor
            # architecture. Expose that detector result without mislabeling Stage 2.
            "model_family": "multimodal_s002",
            "detected_checkpoint_kind": detected_checkpoint_kind,
            "config": str(config_path),
            "git": _git_revision(),
            "protocol": {
                "harness": "native-olmo-core-fast-vision",
                "tasks": list(args.tasks),
                "dataset_split": "validation",
                "examples_per_task": args.examples,
                "sample_seed": args.sample_seed,
                "task_seed_offsets": TASK_SEED_OFFSETS,
                "sample_selection": (
                    "numpy.RandomState(sample_seed + task_seed_offset)."
                    "permutation(dataset_size)[:examples]"
                ),
                "source_epoch": 0,
                "message_format": message_format,
                "tokenizer": args.tokenizer,
                "token_ids": token_ids.as_config_dict(),
                "max_sequence_length": args.max_sequence_length,
                "max_high_resolution_crops": args.max_crops,
                "image_tensor_includes_global_crop": True,
                "loss_token_weighting": "none",
                "metric_aggregation": "response-token-weighted mean cross entropy",
                "rank_batch_instances": args.rank_batch_instances,
                "global_batch_instances": global_batch_instances,
                "world_size": world_size,
                "ep_degree": args.ep_degree,
                "ep_dp_degree": world_size // args.ep_degree,
                "dp_process_group_size": dp_world_size,
                "attention_backend": "flex",
            },
            "results": results,
        }
        if get_rank() == 0:
            _write_json_atomic(output, payload)
            log.info("Wrote results to %s", output)
        dist.barrier()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
