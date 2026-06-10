"""Molmo2 image-QA benchmark evaluation using OLMo-core.

Runs the 11 Molmo2 image-QA benchmarks (ChartQA, VQA v2, DocVQA, InfoVQA,
TextVQA, RealWorldQA, MMMU, MathVista, CountBench QA, PixMo Count, AI2D)
with the OLMo-core ``MultimodalLM`` and scores them with the
matching tasks from olmo-eval-internal (prompt construction, metrics, and
data loading all live there — this script only does inference glue).

Usage (GPU smoke test — 8 examples per task into a fresh output dir):

    python image_qa_eval.py --model allenai/Molmo2-4B \\
        --tasks chart_qa,ai2d,pixmo_count,mmmu --limit 8 --output /tmp/iqa_smoke

Full run of all 11 benchmarks (slow — no KV cache; shard across GPUs):

    torchrun --nproc-per-node=8 image_qa_eval.py --output /path/to/run

Environment variables:
    MOLMO_DATA_DIR      Data root (default /weka/oe-training-default/mm-olmo).
    HF_DATASETS_CACHE   HF datasets cache for MMMU / MathVista / RealWorldQA
                        (point at an offline cache + HF_DATASETS_OFFLINE=1).
    OPENAI_API_KEY      Only for the ``math_vista:gpt`` variant.

Cache semantics:
    Model outputs are cached per task under ``--predictions-cache`` (default
    ``{output}/cache/``).  Outputs found there from *previous runs of this
    script* are reused and only missing examples are inferenced; pass
    ``--recompute`` to force fresh inference.  The pre-existing mm_olmo
    outputs (released ``predictions-ck2000-*`` dumps, shared ``gpt4-cache``)
    are never read as a cache source and never written to.

Notes:
    * ``--max-crops`` defaults to 24, matching the original mm_olmo benchmark
      eval (``eval_molmo2.py``); the released HF processor default is 8.
    * Prompts use the native Molmo2 layout (image tokens before
      ``<|im_start|>user``), verified against the released prediction dumps.
    * Use Molmo2-4B or Molmo2-8B (not Molmo2-O-7B).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

if "LOCAL_RANK" not in os.environ:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from molmo2_infer import (  # noqa: E402
    DEFAULT_MODEL_ID,
    build_input_ids,
    greedy_decode,
    hf_generate,
    load_hf_pipeline,
    load_model,
    load_prediction_cache,
    preprocess_image_multicrop,
    save_prediction_cache,
)

DEFAULT_TASKS = (
    "chart_qa",
    "vqa2",
    "doc_qa",
    "info_qa",
    "text_vqa",
    "real_world_qa",
    "mmmu",
    "math_vista",
    "countbench_qa",
    "pixmo_count",
    "ai2d",
)
DEFAULT_MAX_CROPS = 24  # matches the original mm_olmo benchmark eval


def _instance_key(index: int, instance) -> str:
    """Stable cache key for an instance (index in deterministic task order + id)."""
    example_id = instance.metadata.get("example_id", "")
    return f"{index}:{example_id}"


def _run_task(
    spec: str,
    args,
    model_state: dict,
    device: torch.device,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
) -> None:
    from olmo_eval.common.types import LMOutput, Response
    from olmo_eval.evals.tasks.common.image_qa_base import load_instance_image
    from olmo_eval.evals.tasks.common.registry import get_task

    from olmo_core.distributed.utils import barrier

    task_file = spec.replace(":", "_")
    cache_path = str(Path(args.predictions_cache) / f"{task_file}.json")
    rank_cache_path = f"{cache_path}.rank{rank}" if world_size > 1 else cache_path

    overrides = {"limit": args.limit} if args.limit is not None else None
    task = get_task(spec, overrides)
    instances = list(task.instances)
    max_new_tokens = args.max_new_tokens or task.config.sampling_params.max_tokens
    logger.info("[%s] %d instances, max_new_tokens=%d", spec, len(instances), max_new_tokens)

    # ── Prediction cache (this run's own cache dir; resume unless --recompute) ─
    cached: dict[str, str] = {}
    if not args.recompute:
        cached = load_prediction_cache(cache_path)
        if world_size > 1:
            cached.update(load_prediction_cache(rank_cache_path))
        if cached:
            logger.info("[%s] %d cached predictions found", spec, len(cached))

    keyed = [(_instance_key(i, inst), inst) for i, inst in enumerate(instances)]
    needs_inference = [(k, inst) for k, inst in keyed if k not in cached]
    if args.cache_only and needs_inference:
        sys.exit(f"[{spec}] {len(needs_inference)} cache misses with --cache-only set")
    shard = needs_inference[rank::world_size]

    # ── Inference ─────────────────────────────────────────────────────────────
    predictions: dict[str, str] = dict(cached)
    if shard:
        if model_state.get("model") is None:
            if args.backend == "hf":
                model_state["model"], model_state["processor"] = load_hf_pipeline(
                    args.model, device, dtype
                )
            else:
                model_state["model"], model_state["tokenizer"] = load_model(
                    args.model, device, dtype, hf_config_id=args.hf_config
                )
        logger.info("[%s] rank %d/%d: running %d examples", spec, rank, world_size, len(shard))
        for i, (key, instance) in enumerate(shard):
            try:
                pil_img = load_instance_image(instance)
                if args.backend == "hf":
                    prediction = hf_generate(
                        model_state["model"],
                        model_state["processor"],
                        instance.question,
                        pil_img,
                        device,
                        max_new_tokens=max_new_tokens,
                        max_crops=args.max_crops,
                    )
                else:
                    model, tokenizer = model_state["model"], model_state["tokenizer"]
                    if pil_img is not None:
                        images_t, pooling_idx_t, image_grid = preprocess_image_multicrop(
                            pil_img, dtype, device, max_crops=args.max_crops
                        )
                    else:
                        images_t = pooling_idx_t = image_grid = None
                    input_ids = build_input_ids(tokenizer, instance.question, image_grid, device)
                    prediction = greedy_decode(
                        model,
                        input_ids,
                        images_t,
                        pooling_idx_t,
                        tokenizer,
                        max_new_tokens=max_new_tokens,
                    )
            except Exception as exc:
                logger.warning("[%s] inference failed for %s: %s", spec, key, exc)
                prediction = ""
            predictions[key] = prediction
            if (i + 1) % 25 == 0 or (i + 1) == len(shard):
                logger.info("[%s] [%d/%d] %s → %r", spec, i + 1, len(shard), key, prediction[:60])
            save_prediction_cache(rank_cache_path, predictions)

    barrier()
    if rank != 0:
        return

    # ── Rank 0: merge shards, score, write metrics ────────────────────────────
    if world_size > 1:
        merged = load_prediction_cache(cache_path)
        for r in range(world_size):
            merged.update(load_prediction_cache(f"{cache_path}.rank{r}"))
        predictions = merged
        save_prediction_cache(cache_path, predictions)

    missing = [k for k, _ in keyed if k not in predictions]
    if missing:
        logger.warning("[%s] %d instances have no prediction (scored as empty)", spec, len(missing))

    import asyncio

    responses = [
        Response(
            instance=instance,
            request=task.format_request(instance),
            outputs=[LMOutput(text=predictions.get(key, ""))],
        )
        for key, instance in keyed
    ]
    if any(getattr(s, "requires_async", False) for s in task._get_scorers().values()):
        from olmo_eval.common.execution import ScoringContext

        responses = asyncio.run(task.score_responses(responses, ScoringContext()))
    else:
        responses = asyncio.run(task.score_responses(responses))
    nested = task.compute_metrics(responses)
    flat = {name: next(iter(by_scorer.values())) for name, by_scorer in nested.items()}

    print(f"\n=== {spec} ({len(responses)} examples) ===")
    for name, value in flat.items():
        print(f"  {name:28s}: {value:.4f}")

    task_out_dir = Path(args.output) / task_file
    task_out_dir.mkdir(parents=True, exist_ok=True)
    with open(task_out_dir / "metrics.json", "w") as f:
        json.dump({"metrics": flat, "nested": nested, "n_examples": len(responses)}, f, indent=2)
    rows = [
        {
            "key": key,
            "example_id": instance.metadata.get("example_id"),
            "question": instance.question,
            "prediction": predictions.get(key, ""),
        }
        for key, instance in keyed
    ]
    with open(task_out_dir / "predictions.json", "w") as f:
        json.dump(rows, f, indent=2)
    logger.info("[%s] wrote %s", spec, task_out_dir / "metrics.json")

    # DocVQA / InfographicVQA test splits have no public answers: also write the
    # RRC evaluation-server submission file ([{"questionId": ..., "answer": ...}]).
    # The metrics.json above is meaningless for these (scored vs. placeholder "").
    from olmo_eval.common.types import Split

    if task.config.name in ("doc_qa", "info_qa") and task.config.split == Split.TEST:
        submission = [
            {
                "questionId": instance.metadata["example_id"],
                "answer": predictions.get(key, ""),
            }
            for key, instance in keyed
        ]
        with open(task_out_dir / "submission.json", "w") as f:
            json.dump(submission, f)
        logger.info(
            "[%s] wrote %s (upload to the RRC evaluation server)",
            spec,
            task_out_dir / "submission.json",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Molmo2 image-QA benchmark eval (OLMo-core)")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_ID, help="HF model ID or local OLMo-core checkpoint path"
    )
    parser.add_argument(
        "--hf-config",
        default=DEFAULT_MODEL_ID,
        metavar="MODEL_ID",
        help="HF config ID for native .distcp checkpoints (no weights download)",
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated task specs (e.g. chart_qa,ai2d:test,math_vista:gpt)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max examples per task")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override per-task max new tokens (default: task-specific)",
    )
    parser.add_argument(
        "--max-crops",
        type=int,
        default=DEFAULT_MAX_CROPS,
        help="Max image crops (24 = original mm_olmo eval; HF default is 8)",
    )
    parser.add_argument(
        "--backend",
        choices=["hf", "olmo_core"],
        default="hf",
        help="Inference backend. 'hf' = the released HF Molmo2 pipeline (KV-cached, "
        "implements bidirectional image attention; reproduces the original eval "
        "exactly). 'olmo_core' = MultimodalLM (supports native .distcp "
        "checkpoints, but currently lacks bidirectional image attention — outputs "
        "can drift from the original eval)",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp32"],
        default="bf16",
        help="Model dtype (the original eval ran amp_bf16; fp32 for maximum fidelity)",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Run output dir: per-task metrics.json + predictions.json",
    )
    parser.add_argument(
        "--predictions-cache",
        default=None,
        metavar="DIR",
        help="Model-output cache dir (default: {output}/cache). Reused across "
        "this script's own runs; never points at pre-existing mm_olmo outputs.",
    )
    parser.add_argument(
        "--recompute", action="store_true", help="Ignore cached predictions and rerun inference"
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Fail instead of running inference on cache misses",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU required.")

    if "LOCAL_RANK" in os.environ:
        from olmo_core.distributed.utils import init_distributed
        from olmo_core.utils import prepare_cli_environment

        init_distributed()
        prepare_cli_environment()

    from olmo_core.distributed.utils import get_local_rank, get_rank, get_world_size

    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{get_local_rank()}")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    try:
        import olmo_eval  # noqa: F401
    except ImportError:
        sys.exit(
            "olmo-eval-internal not installed. Run: pip install -e /path/to/olmo-eval-internal"
        )

    if args.predictions_cache is None:
        args.predictions_cache = str(Path(args.output) / "cache")
    if rank == 0:
        Path(args.predictions_cache).mkdir(parents=True, exist_ok=True)
        Path(args.output).mkdir(parents=True, exist_ok=True)

    model_state: dict = {"model": None, "tokenizer": None}
    for spec in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        _run_task(spec, args, model_state, device, dtype, rank, world_size)


if __name__ == "__main__":
    main()
