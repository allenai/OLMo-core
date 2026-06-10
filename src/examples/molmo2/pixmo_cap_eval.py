"""Pixmo-cap dense-caption evaluation using OLMo-core Molmo2.

Loads allenai/Molmo2-4B via the OLMo-core MultimodalTransformer arch,
captions a sample of pixmo-cap images with greedy decoding, then scores
them with the ``dense_caption`` task from olmo-eval-internal.

Usage (smoke test — 12 images):
    python pixmo_cap_eval.py --limit 12 --output predictions.json

Full run (all 2730 images):
    python pixmo_cap_eval.py

Environment variables:
    OPENAI_API_KEY          Required for GPT-judge calls on cache misses.
    DENSE_CAPTION_EVAL_DIR  Override default eval data root (optional).
    MOLMO_DATA_DIR          Override default image data root (optional).
    HF_HOME                 Override HuggingFace cache root (optional).

Multi-GPU usage:
    torchrun --nproc-per-node=N pixmo_cap_eval.py --predictions-cache /path/preds.json ...
    Each GPU processes a shard of the dataset. ``--predictions-cache`` is required.
    GPT scoring and metric aggregation run on rank 0 after all ranks finish.

Notes / known risks:
    * Shared inference helpers (preprocessing, model loading, decoding) live in
      ``molmo2_infer.py``; no mm_olmo dependency.
    * This script keeps its original prompt layout (image block inside the user
      turn); the image-QA eval uses the native layout — see ``molmo2_infer``.
    * Use Molmo2-4B or Molmo2-8B (not Molmo2-O-7B), as the O-7B variant uses
      per-layer YaRN attention scaling not yet supported by MultimodalTransformer.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from molmo2_infer import DEFAULT_MODEL_ID as _DEFAULT_MODEL_ID
from molmo2_infer import build_input_ids_placeholder_style as _build_input_ids
from molmo2_infer import greedy_decode, load_model
from molmo2_infer import load_prediction_cache as _load_prediction_cache
from molmo2_infer import preprocess_image_multicrop
from PIL import Image

if "LOCAL_RANK" not in os.environ:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_HF_CONFIG_ID = _DEFAULT_MODEL_ID
_MAX_NEW_TOKENS = 448


def _preprocess_image_multicrop(pil_img: Image.Image, dtype: torch.dtype, device: torch.device):
    """Multi-crop preprocessing with this script's original settings (8 crops)."""
    return preprocess_image_multicrop(pil_img, dtype, device, max_crops=8)


def _save_prediction_cache(path: str, url_to_caption: dict[str, str]) -> None:
    """Save url→caption mapping as a JSON list (same format as predictions.json)."""
    preds = [{"image_url": url, "prediction": cap} for url, cap in url_to_caption.items()]
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(preds, f, indent=2)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pixmo-cap dense-caption eval with Molmo2")
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL_ID,
        help="HF model ID or local OLMo-core checkpoint path "
        "(step dir or run root containing step*/ subdirs)",
    )
    parser.add_argument(
        "--hf-config",
        default=_DEFAULT_HF_CONFIG_ID,
        metavar="MODEL_ID",
        help="HF model ID used to fetch the architecture config when --model is a "
        "local OLMo-core .distcp checkpoint. Config JSON only — no weights "
        "are downloaded. Ignored when --model is an HF model ID. "
        "(default: %(default)s)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Max number of images to evaluate")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=_MAX_NEW_TOKENS,
        help="Max new tokens per caption (use 64–128 for fast smoke tests)",
    )

    # ── Output / cache paths ────────────────────────────────────────────────
    parser.add_argument(
        "--predictions-cache",
        default=None,
        metavar="PATH",
        help="JSON file for Molmo2 caption cache ({image_url: caption}). "
        "Captions already present are reused; new ones are appended. "
        "Use --recompute to ignore and overwrite.",
    )
    parser.add_argument(
        "--gpt-cache-dir",
        default=None,
        metavar="DIR",
        help="Directory for GPT response cache. Defaults to a sibling 'gpt-cache/' "
        "next to --predictions-cache, or /tmp/dense_caption_gpt_cache.",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Write final scored predictions JSON to this path.",
    )

    # ── Cache control ───────────────────────────────────────────────────────
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore ALL existing caches (Molmo2 predictions and GPT responses) "
        "and rerun from scratch. New results still overwrite the cache files.",
    )
    parser.add_argument("--cache-only", action="store_true", help="Fail on any cache miss")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA GPU required. Run on a machine with at least one A100/H100.")

    # ── Distributed setup (torchrun sets LOCAL_RANK; absent in single-GPU runs) ─
    if "LOCAL_RANK" in os.environ:
        from olmo_core.distributed.utils import init_distributed
        from olmo_core.utils import prepare_cli_environment

        init_distributed()  # sets torch.cuda device per-rank, inits NCCL
        prepare_cli_environment()  # rich logging, rank-0-only INFO filter

    from olmo_core.distributed.utils import barrier, get_local_rank, get_rank, get_world_size

    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{get_local_rank()}")
    dtype = torch.bfloat16

    # ── Resolve cache paths ──────────────────────────────────────────────────
    pred_cache_path = args.predictions_cache
    if world_size > 1 and not pred_cache_path:
        sys.exit("--predictions-cache is required when running with multiple GPUs")

    if args.gpt_cache_dir:
        gpt_cache_dir = args.gpt_cache_dir
    elif pred_cache_path:
        gpt_cache_dir = str(Path(pred_cache_path).parent / "gpt-cache")
    else:
        gpt_cache_dir = "/tmp/dense_caption_gpt_cache"
    if rank == 0:
        os.makedirs(gpt_cache_dir, exist_ok=True)
    logger.info("Molmo2 prediction cache: %s", pred_cache_path or "(none, no caching)")
    logger.info("GPT response cache dir : %s", gpt_cache_dir)

    # Each rank writes to its own shard file; rank 0 merges at the end.
    rank_cache_path = (
        (pred_cache_path + f".rank{rank}")
        if (world_size > 1 and pred_cache_path)
        else pred_cache_path
    )

    # ── Load task (data) ─────────────────────────────────────────────────────
    try:
        from olmo_eval.common.execution import ScoringContext
        from olmo_eval.common.scorers.dense_caption_judge import DenseCaptionJudgeScorer
        from olmo_eval.common.types import LMOutput, Response
        from olmo_eval.evals.tasks.common.registry import _configs, _tasks
        from olmo_eval.evals.tasks.dense_caption import (
            DenseCaptionAvgMetric,
            DenseCaptionConsistencyMetric,
            DenseCaptionNumStatementsMetric,
            DenseCaptionRecallAt10Metric,
            DenseCaptionRecallMetric,
        )
    except ImportError:
        sys.exit(
            "olmo-eval-internal not installed. " "Run: pip install -e /path/to/olmo-eval-internal"
        )

    from dataclasses import replace as dc_replace

    cfg = dc_replace(_configs["dense_caption"], limit=args.limit)
    task = _tasks["dense_caption"](cfg)
    instances = list(task.instances)
    logger.info("Total instances: %d", len(instances))

    # ── Load Molmo2 prediction cache ──────────────────────────────────────────
    cached_captions: dict[str, str] = {}
    if pred_cache_path and not args.recompute:
        cached_captions = _load_prediction_cache(pred_cache_path)
        if world_size > 1 and rank_cache_path:
            # Also recover captions from an interrupted previous run on this rank.
            cached_captions.update(_load_prediction_cache(rank_cache_path))
        logger.info("Loaded %d cached captions", len(cached_captions))

    # ── Determine which instances need inference, then shard by rank ─────────
    needs_inference = [inst for inst in instances if inst.metadata["url"] not in cached_captions]
    needs_inference = needs_inference[rank::world_size]

    model = tokenizer = None
    if needs_inference:
        logger.info(
            "Rank %d/%d: %d images need fresh inference.", rank, world_size, len(needs_inference)
        )
        model, tokenizer = load_model(args.model, device, dtype, hf_config_id=args.hf_config)
    else:
        logger.info("Rank %d/%d: all captions cached — skipping model load.", rank, world_size)

    # ── Caption each image ────────────────────────────────────────────────────
    url_to_caption: dict[str, str] = dict(cached_captions)
    prompt = "Describe this image."

    for i, instance in enumerate(needs_inference):
        url = instance.metadata["url"]
        image_path = instance.metadata.get("image_path", "")
        if not image_path or not Path(image_path).exists():
            logger.warning("[%d/%d] Image not found: %s", i + 1, len(needs_inference), image_path)
            caption = ""
        else:
            try:
                pil_img = Image.open(image_path)
                images_t, pooling_idx_t, image_grid = _preprocess_image_multicrop(
                    pil_img, dtype, device
                )
                input_ids = _build_input_ids(tokenizer, prompt, image_grid, device)
                caption = greedy_decode(
                    model,
                    input_ids,
                    images_t,
                    pooling_idx_t,
                    tokenizer,
                    max_new_tokens=args.max_new_tokens,
                )
            except Exception as exc:
                logger.warning(
                    "[%d/%d] Inference failed for %s: %s",
                    i + 1,
                    len(needs_inference),
                    image_path,
                    exc,
                )
                caption = ""

        url_to_caption[url] = caption
        if (i + 1) % 10 == 0 or (i + 1) == len(needs_inference):
            logger.info(
                "[%d/%d] %s → %s…",
                i + 1,
                len(needs_inference),
                Path(image_path).name if image_path else "?",
                caption[:60],
            )
        # Checkpoint to this rank's shard file after every image.
        if rank_cache_path:
            _save_prediction_cache(rank_cache_path, url_to_caption)

    # ── Barrier: wait for all ranks, then rank 0 merges shards ─────────────────
    barrier()

    if rank != 0:
        return  # non-zero ranks are done

    if world_size > 1:
        merged: dict[str, str] = _load_prediction_cache(pred_cache_path) if pred_cache_path else {}
        for r in range(world_size):
            merged.update(_load_prediction_cache(pred_cache_path + f".rank{r}"))
        url_to_caption = merged
        if pred_cache_path:
            _save_prediction_cache(pred_cache_path, url_to_caption)
        logger.info(
            "Merged %d total captions from %d rank shards.", len(url_to_caption), world_size
        )

    # ── Build Response objects ────────────────────────────────────────────────
    import asyncio

    responses: list[Response] = []
    for instance in instances:
        caption = url_to_caption.get(instance.metadata["url"], "")
        req = task.format_request(instance)
        output = LMOutput(text=caption)
        output.extracted_answer = caption
        output.metadata = {}
        responses.append(Response(instance=instance, request=req, outputs=[output], scores={}))

    # ── Score with GPT judge ──────────────────────────────────────────────────
    judge = DenseCaptionJudgeScorer(
        cache_dir=gpt_cache_dir,
        cache_only=args.cache_only,
        recompute=args.recompute,
    )
    context = ScoringContext()
    logger.info("Running GPT judge on %d responses …", len(responses))

    async def _score_all() -> None:
        sem = asyncio.Semaphore(8)

        async def _score_one(resp: Response) -> None:
            async with sem:
                await judge.ascore_with_context(resp.instance, resp.outputs[0], context)

        await asyncio.gather(*[_score_one(r) for r in responses])

    asyncio.run(_score_all())

    # ── Aggregate and print metrics ──────────────────────────────────────────
    metrics = {
        "recall": DenseCaptionRecallMetric().compute(responses),
        "consistency": DenseCaptionConsistencyMetric().compute(responses),
        "recall_at_10": DenseCaptionRecallAt10Metric().compute(responses),
        "num_statements": DenseCaptionNumStatementsMetric().compute(responses),
        "avg": DenseCaptionAvgMetric().compute(responses),
    }
    n_valid_recall = sum(
        1
        for r in responses
        for o in r.outputs
        if o.metadata and o.metadata.get("dense_caption_result", {}).get("recall_valid")
    )
    n_valid_cons = sum(
        1
        for r in responses
        for o in r.outputs
        if o.metadata and o.metadata.get("dense_caption_result", {}).get("consistency_valid")
    )
    print("\n=== Dense-Caption Eval Results ===")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"  {'valid_recall':20s}: {n_valid_recall}/{len(responses)}")
    print(f"  {'valid_consistency':20s}: {n_valid_cons}/{len(responses)}")

    # ── Save final output ────────────────────────────────────────────────────
    if args.output:
        out_data = [
            {
                "image_url": r.instance.metadata["url"],
                "prediction": r.outputs[0].text,
                "dense_caption_result": r.outputs[0].metadata.get("dense_caption_result", {}),
                "metrics": metrics,
            }
            for r in responses
        ]
        with open(args.output, "w") as f:
            json.dump(out_data, f, indent=2)
        logger.info("Output written to %s", args.output)


if __name__ == "__main__":
    main()
