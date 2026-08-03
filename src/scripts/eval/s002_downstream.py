"""Evaluate native s002-family OLMo3 MoE checkpoints with ai2-olmo-eval.

This deliberately uses the checkpoint's native OLMo-core model and EP sharding instead
of an HF conversion. Run it with ``torchrun`` and a world size divisible by ``--ep-degree``.
Both the pure pretrained s002 checkpoint and multimodal Stage-1 checkpoints are supported;
multimodal checkpoints are evaluated through their language-model path with no images.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, cast

import torch
import torch.distributed as dist

from olmo_core.data.utils import get_labels
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank
from olmo_core.eval.task_groups import TASK_GROUPS
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.attention.backend import AttentionBackendName
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath
from olmo_core.nn.transformer import OLMoDDPModelConfig
from olmo_core.nn.vision import MultimodalLMConfig
from olmo_core.optim import OLMoDDPOptimizerConfig
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.callbacks.evaluator_callback import DownstreamEvaluator
from olmo_core.train.train_module import (
    EvalBatchSpec,
    MultimodalOLMoDDPTrainModuleConfig,
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)
from olmo_core.utils import gc_cuda, move_to_device

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = "/weka/oe-training-default/robertb/s002-step125500"
DEFAULT_OUTPUT_ROOT = "/weka/oe-training-default/rustin/experiments/vision-moe/evals"
DEFAULT_TOKENIZER = "allenai/dolma2-tokenizer"
DEFAULT_PAD_TOKEN_ID = 100277
DEFAULT_EOS_TOKEN_ID = 100257


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--config",
        help="Config JSON to use (defaults to CHECKPOINT/config.json).",
    )
    parser.add_argument("--task-group", choices=sorted(TASK_GROUPS), default="fast")
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Explicit task names; when supplied these replace --task-group.",
    )
    parser.add_argument("--ep-degree", type=int, default=8)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument(
        "--rank-batch-size",
        type=int,
        default=8192,
        help="Maximum tokens per evaluator batch on each EP-DP rank.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Stop each task after this many batches (smoke testing only).",
    )
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--pad-token-id", type=int, default=DEFAULT_PAD_TOKEN_ID)
    parser.add_argument("--eos-token-id", type=int, default=DEFAULT_EOS_TOKEN_ID)
    parser.add_argument(
        "--output", help="Result JSON path; generated under Rustin's eval root by default."
    )
    parser.add_argument("--checkpoint-load-threads", type=int, default=8)
    return parser.parse_args()


def _checkpoint_state_dir(checkpoint: Path) -> Path:
    if checkpoint.name == "model_and_optim":
        return checkpoint
    nested = checkpoint / "model_and_optim"
    return nested if nested.is_dir() else checkpoint


def _config_path(checkpoint: Path, explicit: str | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    root = checkpoint.parent if checkpoint.name == "model_and_optim" else checkpoint
    return root / "config.json"


def _configure_lm_for_eval(lm_config: OLMoDDPModelConfig) -> None:
    """Select kernels available in the experiment image and remove training-only recompute."""
    blocks = [lm_config.block, *(lm_config.block_overrides or {}).values()]
    for block in blocks:
        if isinstance(block.sequence_mixer, AttentionConfig):
            block.sequence_mixer.backend = AttentionBackendName.flex
        if block.ep is not None:
            block.ep.path = ExpertParallelPath.rowwise_nvshmem
    lm_config.recompute_each_block = False
    lm_config.recompute_all_blocks_by_chunk = False
    lm_config.two_batch_overlap = False


def _build_model_and_module_config(
    raw_config: Dict[str, Any],
    *,
    ep_degree: int,
    max_sequence_length: int,
    rank_batch_size: int,
) -> Tuple[torch.nn.Module, OLMoDDPTrainModuleConfig, str]:
    model_dict = raw_config["model"]
    common = dict(
        rank_microbatch_size=rank_batch_size,
        max_sequence_length=max_sequence_length,
        optim=OLMoDDPOptimizerConfig(),  # Required by the config; not built in eval-only mode.
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            only_allreduce_last_microbatch=True,
        ),
        ep_config=TransformerExpertParallelConfig(degree=ep_degree),
    )

    if "lm" in model_dict and "vision" in model_dict:
        model_config = MultimodalLMConfig.from_dict(model_dict)
        if not isinstance(model_config.lm, OLMoDDPModelConfig):
            raise TypeError(
                "The multimodal checkpoint does not contain an OLMoDDP language-model config"
            )
        _configure_lm_for_eval(model_config.lm)
        model = model_config.build(init_device="meta")
        module_config = MultimodalOLMoDDPTrainModuleConfig(
            freeze_params=["vision.*"], response_logits_only=False, **common
        )
        return model, module_config, "multimodal_stage1"

    model_config = OLMoDDPModelConfig.from_dict(model_dict)
    _configure_lm_for_eval(model_config)
    model = model_config.build(init_device="meta")
    return model, OLMoDDPTrainModuleConfig(**common), "pretrained_lm"


def _git_revision() -> Dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


def _default_output(checkpoint: Path, task_group: str, partial: bool) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "partial" if partial else "complete"
    checkpoint_name = (
        checkpoint.parent.name if checkpoint.name == "model_and_optim" else checkpoint.name
    )
    return Path(DEFAULT_OUTPUT_ROOT) / checkpoint_name / f"olmes-{task_group}-{suffix}-{stamp}.json"


def _float_metrics(metrics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {name: float(value.detach().cpu().item()) for name, value in metrics.items()}


def _evaluate_tasks(
    train_module,
    tasks: Iterable[str],
    tokenizer,
    *,
    max_batches: int | None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    device = train_module.device
    for task in tasks:
        evaluator = DownstreamEvaluator(
            name="downstream",
            task=task,
            batch_spec=cast(EvalBatchSpec, train_module.eval_batch_spec),
            tokenizer=tokenizer,
            device=device,
            dp_process_group=train_module.dp_process_group,
        )
        evaluator.reset_metrics()
        started = time.monotonic()
        batches = 0
        instances = 0
        for batch in evaluator:
            batches += 1
            instances += int(batch["input_ids"].shape[0])
            batch = move_to_device(batch, device)
            labels = get_labels(batch)
            with torch.no_grad():
                output = train_module.eval_batch(dict(batch), labels=labels)
            if not isinstance(output, LMOutputWithLoss):
                raise TypeError(f"Expected LMOutputWithLoss, got {type(output).__name__}")
            logits, _, ce_loss, _ = output
            evaluator.update_metrics(batch, ce_loss, logits)
            if get_rank() == 0 and (batches == 1 or batches % 20 == 0):
                log.info(
                    "[%s] batch %d/%s",
                    task,
                    batches,
                    evaluator.total_batches if evaluator.total_batches is not None else "?",
                )
            if max_batches is not None and batches >= max_batches:
                break

        metrics = _float_metrics(evaluator.compute_metrics())
        results[task] = {
            "metrics": metrics,
            "batches_per_ep_dp_rank": batches,
            "instances_per_ep_dp_rank": instances,
            "total_batches_per_ep_dp_rank": evaluator.total_batches,
            "elapsed_seconds": time.monotonic() - started,
        }
        if get_rank() == 0:
            log.info("Finished %s: %s", task, metrics)
        del evaluator
        gc_cuda()
    return results


def main() -> None:
    args = _parse_args()
    if args.ep_degree <= 0:
        raise ValueError("--ep-degree must be positive")
    if args.rank_batch_size <= 0 or args.rank_batch_size % args.max_sequence_length != 0:
        raise ValueError("--rank-batch-size must be a positive multiple of --max-sequence-length")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size % args.ep_degree != 0:
        raise ValueError(
            f"WORLD_SIZE ({world_size}) must be divisible by --ep-degree ({args.ep_degree})"
        )

    os.environ.setdefault("OLMO_USE_OWN_SYMM_MEM", "1")
    os.environ.setdefault("OLMO_EP_MP_HIGH_PRIORITY_GROUP", "1")
    os.environ.setdefault("OLMO_OWN_SYMM_PREWARM", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    prepare_training_environment()

    try:
        from olmo_eval import HFTokenizer

        checkpoint = Path(args.checkpoint).resolve()
        config_path = _config_path(checkpoint, args.config)
        with config_path.open() as f:
            raw_config = json.load(f)

        model, module_config, checkpoint_kind = _build_model_and_module_config(
            raw_config,
            ep_degree=args.ep_degree,
            max_sequence_length=args.max_sequence_length,
            rank_batch_size=args.rank_batch_size,
        )
        train_module = module_config.build(model, eval_only=True)
        state_dir = _checkpoint_state_dir(checkpoint)
        log.info("Loading %s checkpoint from %s", checkpoint_kind, state_dir)
        train_module.load_state_dict_direct(
            state_dir,
            process_group=dist.group.WORLD,
            thread_count=args.checkpoint_load_threads,
            load_optim_state=False,
        )

        tokenizer = HFTokenizer(
            args.tokenizer,
            pad_token_id=args.pad_token_id,
            eos_token_id=args.eos_token_id,
        )
        tasks: List[str] = args.tasks if args.tasks else list(TASK_GROUPS[args.task_group])
        results = _evaluate_tasks(
            train_module,
            tasks,
            tokenizer,
            max_batches=args.max_batches,
        )

        output_label = args.task_group if not args.tasks else "custom"
        output = (
            Path(args.output)
            if args.output
            else _default_output(checkpoint, output_label, args.max_batches is not None)
        )
        payload = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint),
            "checkpoint_state_dir": str(state_dir),
            "checkpoint_kind": checkpoint_kind,
            "config": str(config_path),
            "git": _git_revision(),
            "protocol": {
                "harness": "ai2-olmo-eval",
                "task_group": None if args.tasks else args.task_group,
                "tasks": tasks,
                "partial": args.max_batches is not None,
                "max_batches_per_task": args.max_batches,
                "max_sequence_length": args.max_sequence_length,
                "rank_batch_size_tokens": args.rank_batch_size,
                "world_size": world_size,
                "ep_degree": args.ep_degree,
                "ep_dp_degree": world_size // args.ep_degree,
                "attention_backend": "flex",
            },
            "results": results,
        }
        if get_rank() == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(output.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            temporary.replace(output)
            log.info("Wrote results to %s", output)
        dist.barrier()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
