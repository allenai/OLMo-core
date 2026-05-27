"""
Image-side caption F1 ("cap F1") eval for Molmo2-style VLMs.

Reproduces the Molmo v1 paper's image cap F1 protocol on the
``dense-caption-eval`` set (2,730 images, multi-transcript references,
GPT-judged statement recall + consistency, harmonic-mean F1).

Task definitions and scoring live in ``olmo-eval-internal``
(:class:`olmo_eval.evals.tasks.pixmo_cap_eval.PixmoCapEvalTask`).  OLMo-core
contributes the multimodal preprocessor
(:class:`~olmo_core.data.multimodal.MultimodalPreprocessor`) and greedy
generator (:class:`~olmo_core.eval.MultimodalGenerator`).

Two ways to run:

1. **Multi-GPU on Beaker** — via the sibling ``launch_cap_f1_eval.py``
   launcher. The eval driver below uses ``torchrun``-style env vars
   (``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``) to data-parallel-split the
   examples across GPUs. Rank 0 gathers predictions and drives the OpenAI
   judge; all other ranks exit after generation.

2. **Single GPU** for a smoke test::

    python src/scripts/eval/molmo2_cap_f1_eval.py \\
        --model allenai/Molmo2-4B --limit 16 \\
        --out runs/molmo2-4b-cap-f1.json

Outputs a JSON file with aggregate {cap_f1, recall, consistency} and the
per-example breakdown (prediction text + gold statements + judge verdicts).

Dependencies
------------
``OPENAI_API_KEY`` must be set for judging.  Pass ``--predictions-only`` to
skip the judge and just dump the raw generations.
"""

import argparse
import json
import logging
import os
import time
from typing import List, Optional

import torch

from olmo_core.data.multimodal import (
    MultimodalPreprocessor,
    MultimodalPreprocessorConfig,
    MultimodalTokenizerConfig,
)
from olmo_core.data.multimodal.image_preprocessor import ImagePreprocessorConfig
from olmo_core.data.multimodal.multicrop import MultiCropPreprocessorConfig
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.eval.multimodal_generator import MultimodalGenerator
from olmo_core.nn.vision import MultimodalTransformer
from olmo_core.nn.vision.molmo2_loader import (
    ensure_default_rope_registered,
    molmo2_config_from_hf_config,
    molmo2_hf_state_dict_to_multimodal_transformer,
)

# Benchmark task + scoring from olmo-eval-internal.
from olmo_eval.evals.tasks.pixmo_cap_eval import PixmoCapEvalTask, PixmoCapEvalTaskConfig
from olmo_eval.evals.tasks.common.cap_f1_judge import DEFAULT_JUDGE_MODEL

log = logging.getLogger("molmo2_cap_f1_eval")


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def _dist_info() -> tuple[int, int, int]:
    """Return ``(rank, world_size, local_rank)`` from torchrun env vars,
    defaulting to single-process (0, 1, 0) when unset."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def _init_distributed_if_needed(world_size: int) -> None:
    if world_size > 1:
        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")


def _gather_objects(obj_list: list, world_size: int) -> Optional[list]:
    """All-gather a list of Python objects from every rank onto rank 0.
    Returns the concatenated list on rank 0, ``None`` on other ranks."""
    if world_size == 1:
        return obj_list
    import torch.distributed as dist

    rank = dist.get_rank()
    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(obj_list, gathered, dst=0)
    if rank != 0:
        return None
    flat: list = []
    for sub in gathered:  # type: ignore[union-attr]
        if sub:
            flat.extend(sub)
    return flat


# ---------------------------------------------------------------------------
# Model loading + preprocessor
# ---------------------------------------------------------------------------


def _load_molmo2(model_id: str, device: torch.device, dtype: torch.dtype):
    from transformers import AutoModelForImageTextToText

    ensure_default_rope_registered()
    log.info("loading HF Molmo2 from %s", model_id)
    t0 = time.perf_counter()
    hf = AutoModelForImageTextToText.from_pretrained(
        model_id, trust_remote_code=True, local_files_only=True
    )
    mm_cfg = molmo2_config_from_hf_config(hf.config)
    converted = molmo2_hf_state_dict_to_multimodal_transformer(hf.state_dict(), mm_cfg)
    del hf

    ours = MultimodalTransformer(mm_cfg, init_device="meta")
    ours.to_empty(device=torch.device("cpu"))
    ours.load_state_dict(converted, strict=False)
    del converted
    ours = ours.to(device=device, dtype=dtype).eval()
    log.info("loaded in %.1fs → %s (%s)", time.perf_counter() - t0, device, dtype)
    return ours, mm_cfg


def _build_preprocessor(mm_cfg, model_id: str) -> tuple[MultimodalPreprocessor, object]:
    from transformers import AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
    base_vocab_size = mm_cfg.image_patch_token_id - 2  # <im_patch> at idx 2
    tok_cfg = TokenizerConfig(
        vocab_size=base_vocab_size,
        eos_token_id=hf_tok.eos_token_id,
        pad_token_id=hf_tok.pad_token_id or hf_tok.eos_token_id,
        identifier=model_id,
    )
    mm_tok_cfg = MultimodalTokenizerConfig(base=tok_cfg)

    pre_cfg = MultimodalPreprocessorConfig(
        tokenizer=mm_tok_cfg,
        max_sequence_length=4096,
        add_eos=False,
        prompt_template=("<|im_start|>user\n{image}\n{prompt}<|im_end|>\n<|im_start|>assistant\n"),
        response_template="{response}",
        multicrop=MultiCropPreprocessorConfig(
            base_image_input_size=mm_cfg.vision.image_default_input_size,
            image_preprocessor=ImagePreprocessorConfig(patch_size=mm_cfg.vision.image_patch_size),
        ),
    )
    return MultimodalPreprocessor(pre_cfg, hf_tok), hf_tok


def _encode_prompt(pre: MultimodalPreprocessor, prompt: str, image, image_dtype: torch.dtype):
    out = pre(prompt=prompt, response="", image=image)
    mask = out["loss_masks"] == 0
    input_ids = torch.from_numpy(out["input_tokens"][mask]).unsqueeze(0).long()
    images = torch.from_numpy(out["images"]).unsqueeze(0).to(image_dtype)
    pooled = torch.from_numpy(out["pooled_patches_idx"]).unsqueeze(0).long()
    return input_ids, images, pooled


# ---------------------------------------------------------------------------
# Eval loop (single rank)
# ---------------------------------------------------------------------------


def _generate_predictions(
    *,
    task: PixmoCapEvalTask,
    model,
    preprocessor,
    hf_tok,
    max_new_tokens: int,
    dtype: torch.dtype,
    rank: int,
    world_size: int,
) -> tuple[list[str], list]:
    """Generate captions on this rank's shard of the task's instances.

    Returns ``(predictions, instances)`` for the rank's subset only.
    """
    generator = MultimodalGenerator(model)
    predictions: list[str] = []
    rank_instances = []

    for i, instance in enumerate(task.instances):
        if i % world_size != rank:
            continue
        image = instance.images[0]  # PIL image
        prompt = instance.question
        input_ids, images_t, pooled = _encode_prompt(preprocessor, prompt, image, dtype)
        t0 = time.perf_counter()
        out = generator.generate(
            input_ids=input_ids,
            images=images_t,
            pooled_patches_idx=pooled,
            max_new_tokens=max_new_tokens,
            eos_token_id=hf_tok.eos_token_id,
        )
        gen_s = time.perf_counter() - t0
        pred_text = hf_tok.decode(out.token_ids, skip_special_tokens=True).strip()
        predictions.append(pred_text)
        rank_instances.append(instance)

        n = len(predictions)
        if n % (16 * max(1, world_size // world_size)) == 0:
            log.info(
                "[rank %d] %d generated; last took %.2fs (%d tokens)",
                rank, n, gen_s, len(out.token_ids),
            )

    return predictions, rank_instances


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", type=str, default="allenai/Molmo2-4B")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--limit", type=int, default=0, help="0 = full split (2730)")
    p.add_argument("--prompt", type=str, default="Describe this image.")
    p.add_argument("--max-new-tokens", type=int, default=448)
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("float32", "bfloat16", "float16"),
    )
    p.add_argument("--out", type=str, required=True, help="Output JSON path.")
    p.add_argument(
        "--predictions-only",
        action="store_true",
        help="Skip the LLM judge; just dump generations.",
    )
    p.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--judge-threads", type=int, default=16)
    p.add_argument("--judge-cache-dir", type=str, default=None)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )
    args = parse_args(argv)

    rank, world_size, local_rank = _dist_info()
    _init_distributed_if_needed(world_size)
    if world_size > 1:
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        args.dtype
    ]

    # ---- model + preprocessor ----
    model, mm_cfg = _load_molmo2(args.model, device=device, dtype=dtype)
    preprocessor, hf_tok = _build_preprocessor(mm_cfg, args.model)

    # ---- task (data + scoring) ----
    task = PixmoCapEvalTask(
        PixmoCapEvalTaskConfig(
            name="pixmo_cap_eval",
            data_root=args.data_root,
            limit=args.limit if args.limit > 0 else None,
            prompt=args.prompt,
            judge_model=args.judge_model,
            judge_cache_dir=args.judge_cache_dir,
            judge_threads=args.judge_threads,
        )
    )
    if rank == 0:
        log.info("DLC-Bench task ready (data_root=%s)", args.data_root)

    # ---- generation ----
    t0 = time.perf_counter()
    predictions, instances = _generate_predictions(
        task=task,
        model=model,
        preprocessor=preprocessor,
        hf_tok=hf_tok,
        max_new_tokens=args.max_new_tokens,
        dtype=dtype,
        rank=rank,
        world_size=world_size,
    )
    log.info(
        "[rank %d] generation done in %.1fs (%d items)", rank, time.perf_counter() - t0, len(predictions)
    )

    # ---- gather to rank 0 ----
    gathered_preds = _gather_objects(predictions, world_size)
    gathered_insts = _gather_objects(instances, world_size)
    if rank != 0:
        return 0

    assert gathered_preds is not None and gathered_insts is not None
    log.info("gathered %d predictions on rank 0", len(gathered_preds))

    if args.predictions_only:
        per_example = [
            {
                "image_id": inst.metadata.get("image_id", str(i)),
                "prediction": pred,
                "gold_caption": inst.gold_answer or "",
                "gold_statements": inst.metadata.get("atomic_statements", []),
            }
            for i, (pred, inst) in enumerate(zip(gathered_preds, gathered_insts))
        ]
        summary = {
            "model": args.model,
            "n_evaluated": len(gathered_preds),
            "prompt": args.prompt,
            "judge_model": None,
            "cap_f1": None,
            "recall": None,
            "consistency": None,
        }
    else:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            log.error(
                "OPENAI_API_KEY not set — required for judging. Use --predictions-only to skip."
            )
            return 2

        log.info(
            "running judge via PixmoCapEvalTask.score_all (n=%d, threads=%d, model=%s)",
            len(gathered_preds),
            args.judge_threads,
            args.judge_model,
        )
        t_judge = time.perf_counter()
        agg = task.score_all(gathered_preds, gathered_insts)
        log.info("judging done in %.1fs", time.perf_counter() - t_judge)

        summary = {
            "model": args.model,
            "n_evaluated": len(gathered_preds),
            "prompt": args.prompt,
            "judge_model": args.judge_model,
            **agg,
        }
        per_example = [
            {
                "image_id": inst.metadata.get("image_id", str(i)),
                "prediction": pred,
                "gold_caption": inst.gold_answer or "",
                "gold_statements": inst.metadata.get("atomic_statements", []),
            }
            for i, (pred, inst) in enumerate(zip(gathered_preds, gathered_insts))
        ]

    # ---- write ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "predictions": per_example}, f, indent=2)
    log.info("wrote %s", args.out)

    print("\n=== DLC-Bench cap F1 ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
