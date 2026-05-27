"""
PixMo-Cap caption-quality benchmark for Molmo2-style VLMs.

Standalone single-GPU eval script. The model under test is loaded from a
HuggingFace Molmo2 checkpoint (the only checkpoint format we can construct
a model from end-to-end today) — converted to our
:class:`MultimodalTransformer` via :mod:`olmo_core.nn.vision.molmo2_loader`.
This is the Stage-1 evaluation Molmo2 uses to track caption pretraining;
Token-F1 is the headline number.

Example:

.. code-block:: bash

    python src/scripts/eval/molmo2_caption_eval.py \\
        --model allenai/Molmo2-4B \\
        --split validation \\
        --limit 64 \\
        --max-new-tokens 256 \\
        --out runs/molmo2-4b-cap.json

Output is a JSON file with the aggregate metrics and a per-example breakdown
(prediction text, references, individual scores). The exit code is 0 on
success; nonzero on hard errors (model load, no data).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import torch
from PIL import Image

from olmo_core.data.multimodal import (
    MultimodalPreprocessor,
    MultimodalPreprocessorConfig,
    MultimodalTokenizerConfig,
)
from olmo_core.data.multimodal.image_preprocessor import ImagePreprocessorConfig
from olmo_core.data.multimodal.multicrop import MultiCropPreprocessorConfig
from olmo_core.data.multimodal.pixmo_cap import (
    DEFAULT_MOLMO_DATA_DIR,
    PixmoCapDataset,
    PixmoCapDatasetConfig,
)
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.eval.multimodal_generator import MultimodalGenerator

# Scoring / judge from olmo-eval-internal.
from olmo_eval.evals.tasks.common.cap_f1_judge import DEFAULT_JUDGE_MODEL, CapF1Judge
from olmo_core.nn.vision import MultimodalTransformer
from olmo_core.nn.vision.molmo2_loader import (
    ensure_default_rope_registered,
    molmo2_config_from_hf_config,
    molmo2_hf_state_dict_to_multimodal_transformer,
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_molmo2_from_hf(model_id: str, device: torch.device, dtype: torch.dtype):
    """Load a HF Molmo2 checkpoint, convert to ours, move to ``device``.

    Returns ``(model, multimodal_config, hf_config)``.
    """
    from transformers import AutoModelForImageTextToText

    # transformers ≥ 5 dropped ROPE_INIT_FUNCTIONS["default"]; Molmo2's
    # bundled modeling code still asks for it by name. Reinstall a
    # mathematically-identical entry before instantiation.
    ensure_default_rope_registered()

    print(f"[load] HF model: {model_id}", flush=True)
    t0 = time.perf_counter()
    hf = AutoModelForImageTextToText.from_pretrained(
        model_id, trust_remote_code=True
    )
    mm_cfg = molmo2_config_from_hf_config(hf.config)
    converted = molmo2_hf_state_dict_to_multimodal_transformer(hf.state_dict(), mm_cfg)
    # Free the HF model — we have the weights as a plain dict now.
    del hf.model  # type: ignore[attr-defined]
    hf_config = hf.config
    del hf

    ours = MultimodalTransformer(mm_cfg, init_device="meta")
    ours.to_empty(device=torch.device("cpu"))
    missing, unexpected = ours.load_state_dict(converted, strict=False)
    if unexpected:
        print(f"[load] WARNING: {len(unexpected)} unexpected keys: {unexpected[:3]}…", flush=True)
    if missing:
        # Some fused / norm keys are intentionally not in the HF dump
        # (e.g. position-bias buffers). Warn but don't abort.
        print(f"[load] {len(missing)} missing keys (using random init): {missing[:3]}…", flush=True)
    del converted

    ours = ours.to(device=device, dtype=dtype).eval()
    print(
        f"[load] done in {time.perf_counter() - t0:.1f}s — moved to {device}, dtype={dtype}",
        flush=True,
    )
    return ours, mm_cfg, hf_config


# ---------------------------------------------------------------------------
# Prompt-only token encoding
# ---------------------------------------------------------------------------


def _encode_prompt_only(
    pre: MultimodalPreprocessor,
    prompt: str,
    image: Image.Image,
    image_dtype: torch.dtype,
):
    """Run the multimodal preprocessor with an empty response, then return
    only the prompt-side portion for generation.

    Returns ``(input_ids[1, prompt_len], images[1, n_crops, n_patches, dim], pooled_patches_idx[1, n_pooled, pool])``.
    """
    out = pre(prompt=prompt, response="", image=image)
    # ``loss_masks == 0`` marks the prompt-side region (including image tokens).
    # The trailing response (just `add_eos`'s EOS if enabled) lives where
    # loss_masks == 1; drop it.
    prompt_mask = out["loss_masks"] == 0
    prompt_ids = out["input_tokens"][prompt_mask]
    input_ids = torch.from_numpy(prompt_ids).unsqueeze(0).long()
    images = torch.from_numpy(out["images"]).unsqueeze(0).to(image_dtype)
    pooled = torch.from_numpy(out["pooled_patches_idx"]).unsqueeze(0).long()
    return input_ids, images, pooled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model",
        type=str,
        default="allenai/Molmo2-4B",
        help="HuggingFace model id (must be present in the local HF cache).",
    )
    p.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=("train", "validation"),
        help="PixMo-Cap split.",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override the on-disk PixMo-Cap dataset directory. Defaults to "
        f"$MOLMO_DATA_DIR/torch_datasets/pixmo_datasets/cap (root: {DEFAULT_MOLMO_DATA_DIR}).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=64,
        help="Max examples to evaluate. PixMo-Cap validation is 2048 examples; "
        "use --limit 0 for the full split.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Prompt string sent before the image-conditioned response.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Generation cap. Molmo2's PixMo-Cap captions are typically 100-300 tokens.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 = greedy. Any positive value enables sampling.",
    )
    p.add_argument("--top-p", type=float, default=1.0, help="Nucleus-sampling cutoff.")
    p.add_argument("--device", type=str, default="cuda", help="Torch device (cuda/cuda:N/cpu).")
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("float32", "bfloat16", "float16"),
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help="OpenAI model used for recall/consistency judging.",
    )
    p.add_argument(
        "--judge-cache-dir",
        type=str,
        default=None,
        help="Disk cache dir for judge API calls (makes re-runs free).",
    )
    p.add_argument(
        "--judge-threads",
        type=int,
        default=16,
        help="Parallel threads for judge API calls.",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path. If omitted, no file is written (metrics still printed).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available — aborting.", file=sys.stderr)
        return 2
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
        args.dtype
    ]

    # --- model ---------------------------------------------------------------
    model, mm_cfg, _hf_cfg = _load_molmo2_from_hf(args.model, device=device, dtype=dtype)
    generator = MultimodalGenerator(model)

    # --- tokenizer + preprocessor -------------------------------------------
    # Load Molmo2's bundled HF tokenizer directly — it already registers the
    # six image special tokens at the same offsets our `MultimodalTokenizerConfig`
    # convention expects (base.vocab_size + i), so the resulting `image_patch_id`
    # lines up with the model's `image_patch_token_id`.
    from transformers import AutoTokenizer

    hf_tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    # Some variants (e.g. Molmo2-O-7B) use <|endoftext|> as the registered
    # eos_token but <|im_end|> as the turn separator.  Add the turn separator
    # as an extra stop token so generation doesn't run past the reply boundary.
    _im_end_id = hf_tok.convert_tokens_to_ids("<|im_end|>")
    _extra_stop: tuple = () if _im_end_id == hf_tok.eos_token_id else (_im_end_id,)
    base_vocab_size = mm_cfg.image_patch_token_id - 2  # <im_patch> at idx 2
    base_cfg = TokenizerConfig(
        vocab_size=base_vocab_size,
        eos_token_id=hf_tok.eos_token_id,
        pad_token_id=hf_tok.pad_token_id or hf_tok.eos_token_id,
        identifier=args.model,
    )
    mm_tok_cfg = MultimodalTokenizerConfig(base=base_cfg)
    # Match the model's vision config — Molmo2 uses 378×378 / 14-patch
    # (27×27 = 729 patches per crop); the preprocessor default is 336×336.
    img_size = mm_cfg.vision.image_default_input_size
    patch_size = mm_cfg.vision.image_patch_size
    # Molmo2's tokenizer uses Qwen3's <|im_start|>/<|im_end|> chat template;
    # an un-templated prompt produces near-empty completions because the model
    # has only ever seen the templated form during training.
    pre_cfg = MultimodalPreprocessorConfig(
        tokenizer=mm_tok_cfg,
        max_sequence_length=4096,  # leave headroom for crops + prompt
        add_eos=False,  # eval: do not append EOS into the prompt
        prompt_template=("<|im_start|>user\n{image}\n{prompt}<|im_end|>\n<|im_start|>assistant\n"),
        response_template="{response}",
        multicrop=MultiCropPreprocessorConfig(
            base_image_input_size=img_size,
            image_preprocessor=ImagePreprocessorConfig(patch_size=patch_size),
        ),
    )
    preproc = MultimodalPreprocessor(pre_cfg, hf_tok)

    # --- data ---------------------------------------------------------------
    ds_cfg = PixmoCapDatasetConfig(
        data_dir=args.data_dir,
        split=args.split,
        prompt=args.prompt,
        limit=args.limit if args.limit > 0 else None,
        shuffle=False,
    )
    print(f"[data] PixMo-Cap dir: {ds_cfg.resolve_data_dir()}", flush=True)
    ds = PixmoCapDataset(ds_cfg)

    # --- run ----------------------------------------------------------------
    items: List[dict] = []
    t_gen_total = 0.0
    n = 0
    for prompt_text, gold_caption, image in ds:
        n += 1
        input_ids, images_t, pooled = _encode_prompt_only(
            preproc, prompt_text, image, image_dtype=dtype
        )

        t0 = time.perf_counter()
        out = generator.generate(
            input_ids=input_ids,
            images=images_t,
            pooled_patches_idx=pooled,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=hf_tok.eos_token_id,
            stop_token_ids=_extra_stop,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        t_gen_total += time.perf_counter() - t0

        pred_text = hf_tok.decode(out.token_ids, skip_special_tokens=True).strip()
        items.append(
            {
                "image_id": str(n - 1),
                "prediction": pred_text,
                "gold_statements": [],  # extracted on-the-fly by the judge
                "gold_caption": gold_caption,
                "finished_reason": out.finished_reason,
                "n_generated_tokens": len(out.token_ids),
                "gen_seconds": time.perf_counter() - t0,
            }
        )

        if n % 8 == 0:
            print(
                f"[gen] {n}/{args.limit or '?'} examples — avg {t_gen_total / n:.2f}s/ex",
                flush=True,
            )

    if n == 0:
        print("No examples evaluated (data iterator was empty).", file=sys.stderr)
        return 3

    # --- GPT judge ----------------------------------------------------------
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        print("[judge] OPENAI_API_KEY not set — skipping judge, writing generations only.", flush=True)
        recall_avg = consistency_avg = f1_avg = None
    else:
        judge = CapF1Judge(
            api_key=openai_key,
            model=args.judge_model,
            cache_dir=args.judge_cache_dir,
        )
        # Extract gold statements from each gold caption first (recall side).
        print(f"[judge] extracting gold statements for {n} examples…", flush=True)
        gold_stmts: dict = {}
        with ThreadPoolExecutor(max_workers=args.judge_threads) as pool:
            futs = {pool.submit(judge.extract_statements, it["gold_caption"]): i for i, it in enumerate(items)}
            for fut in as_completed(futs):
                gold_stmts[futs[fut]] = fut.result()
        for i, it in enumerate(items):
            it["gold_statements"] = gold_stmts[i]

        print(f"[judge] scoring {n} examples (threads={args.judge_threads}, model={args.judge_model})…", flush=True)
        t_judge = time.perf_counter()
        scores = judge.score_batch(items, n_threads=args.judge_threads)
        print(f"[judge] done in {time.perf_counter() - t_judge:.1f}s", flush=True)

        recall_avg = sum(s.recall for s in scores) / len(scores)
        consistency_avg = sum(s.consistency for s in scores) / len(scores)
        denom = recall_avg + consistency_avg
        f1_avg = 2 * recall_avg * consistency_avg / denom if denom > 0 else 0.0

        for i, s in enumerate(scores):
            items[i]["recall"] = s.recall
            items[i]["consistency"] = s.consistency
            items[i]["f1"] = s.f1
            items[i]["n_gold_statements"] = s.n_gold_statements
            items[i]["n_pred_statements"] = s.n_pred_statements

    # --- metrics summary ----------------------------------------------------
    metrics = {
        "model": args.model,
        "split": args.split,
        "n_evaluated": n,
        "recall": recall_avg,
        "consistency": consistency_avg,
        "f1": f1_avg,
        "avg_seconds_per_example": t_gen_total / n,
        "max_new_tokens": args.max_new_tokens,
        "judge_model": args.judge_model if openai_key else None,
    }

    print("\n=== PixMo-Cap metrics ===")
    print(f"  model:           {args.model}")
    print(f"  split:           {args.split}")
    print(f"  n_evaluated:     {n}")
    if recall_avg is not None:
        print(f"  recall:          {recall_avg:.4f}")
        print(f"  consistency:     {consistency_avg:.4f}")
        print(f"  F1:              {f1_avg:.4f}   (primary)")
    print(f"  s/example (gen): {t_gen_total / n:.2f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"metrics": metrics, "predictions": items}, f, indent=2)
        print(f"\n[out] wrote {args.out}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
