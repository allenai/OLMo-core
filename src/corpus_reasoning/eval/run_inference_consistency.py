"""
Run the inference-consistency check against a **real trained checkpoint** and a real eval set.

For every example this scores the gold continuation twice -- once with a single teacher-forced
forward pass (the number a loss or perplexity script reports) and once through the actual generation
path with the gold tokens forced in (the distributions the model produces while decoding) -- and
reports whether the two agree.

The two are supposed to be the same conditional distribution computed two ways. Where they are not,
one of the numbers a run is being judged on describes a model that does not exist: either the loss
curve is measuring a function the eval harness never serves, or the eval harness is serving a
function that was never trained. Both have happened in this repo, and both looked like modelling
results at the time.

Usage::

    python -m corpus_reasoning.eval.run_inference_consistency \\
        --model-path /weka/.../step10000 \\
        --data data/contradiction_eval_pubmed_both_n100_k3.jsonl \\
        --task contradiction \\
        --tokenizer Qwen/Qwen3.5-0.8B \\
        --out consistency.json

The variant (dense / document-chunked / summary-token / landmark) is detected from the built model,
so the same command works for every arm of a sweep.
"""

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["detect_variant", "VariantInfo", "main"]


@dataclass
class VariantInfo:
    """
    What the built model turned out to be, and what that implies for how to score it.

    :param name: Detected variant name.
    :param is_landmark: Whether generation rewrites the prompt with landmark tokens.
    :param block_multiple: Block alignment the eager reference forward requires, if any.
    :param mem_freq: Landmark ``mem_freq``, if applicable.
    :param num_landmarks: Landmarks per block, if applicable.
    :param expect: ``"identical"`` if the two paths should agree, ``"gap"`` if they diverge by
        design.
    :param notes: Human-readable detail about the detection.
    """

    name: str
    is_landmark: bool = False
    block_multiple: Optional[int] = None
    mem_freq: Optional[int] = None
    num_landmarks: Optional[int] = None
    expect: str = "identical"
    notes: List[str] = field(default_factory=list)


def detect_variant(gm, *, topk_disabled: bool = False) -> VariantInfo:
    """
    Work out which attention variant a built generation module is.

    Detection reads the model rather than trusting a flag, because the thing most worth catching is a
    checkpoint that is not the variant its run name claims.

    :param gm: The built :class:`~olmo_core.generate.generation_module.TransformerGenerationModule`.
    :param topk_disabled: Whether hard top-k landmark retrieval has been turned off for this run,
        which changes what agreement a landmark model owes.

    :returns: The detected variant info.
    """
    from olmo_core.nn.attention import DocumentChunkedAttention
    from olmo_core.nn.attention.summary_token import SummaryTokenAttention

    landmark_layers = gm._landmark_attention_layers()
    attn_types = {type(b.attention).__name__ for b in gm.model.blocks.values()}
    notes = [f"attention layer types: {sorted(attn_types)}"]

    if landmark_layers:
        mem_freq = int(getattr(landmark_layers[0], "mem_freq"))
        num_landmarks = int(getattr(landmark_layers[0], "num_landmarks", 1))
        name = type(landmark_layers[0]).__name__
        notes.append(f"{len(landmark_layers)} landmark layers, mem_freq={mem_freq}")
        # Hard top-k retrieval is the dominant cause of the landmark decode gap, and it runs on
        # decode steps only. With it off the decode gates densely over all past blocks, exactly as
        # the batched forward does, and the two paths should agree -- so the same run means something
        # different depending on this flag. The residual cause, block drift once the continuation
        # passes the prompt's final block, is bounded by --max-gold-tokens.
        notes.append(
            "hard top-k retrieval DISABLED -- decode and forward should agree"
            if topk_disabled
            else "hard top-k retrieval ACTIVE (the default eval config) -- a gap is expected"
        )
        return VariantInfo(
            name=name,
            is_landmark=True,
            block_multiple=mem_freq + num_landmarks,
            mem_freq=mem_freq,
            num_landmarks=num_landmarks,
            expect="identical" if topk_disabled else "gap",
            notes=notes,
        )

    if any(isinstance(b.attention, SummaryTokenAttention) for b in gm.model.blocks.values()):
        mode = getattr(gm.model, "_summary_eval_mask_mode", "?")
        notes.append(f"summary serving mask mode = {mode!r}")
        return VariantInfo(name="summary_token", expect="identical", notes=notes)

    if any(isinstance(b.attention, DocumentChunkedAttention) for b in gm.model.blocks.values()):
        return VariantInfo(name="document_chunked", expect="identical", notes=notes)

    return VariantInfo(name="dense", expect="identical", notes=notes)


def _truncate_gold(gold: List[int], stop_ids: set, max_tokens: int) -> List[int]:
    """
    Cut the gold continuation before the first stop token and to at most ``max_tokens``.

    A gold continuation containing EOS would end the decode loop early, leaving the trace with fewer
    steps than gold tokens; the harness raises on that rather than silently comparing a prefix, so
    trim here instead.

    :param gold: The gold token ids.
    :param stop_ids: Ids that would terminate decoding.
    :param max_tokens: Cap on length.

    :returns: The trimmed gold continuation.
    """
    out: List[int] = []
    for t in gold[:max_tokens]:
        if t in stop_ids:
            break
        out.append(int(t))
    return out


def _summarize(key: str, values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    vs = sorted(values)
    return {
        f"{key}_mean": statistics.fmean(vs),
        f"{key}_median": vs[len(vs) // 2],
        f"{key}_p95": vs[min(len(vs) - 1, int(0.95 * len(vs)))],
        f"{key}_max": vs[-1],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True, help="step dir: config.json + model_and_optim/")
    ap.add_argument("--data", required=True, help="unified-format eval JSONL")
    ap.add_argument("--task", default="contradiction", help="task name for prompt construction")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    ap.add_argument("--out", required=True, help="where to write the JSON report")
    ap.add_argument(
        "--family",
        default=None,
        help="reserved-id set for landmark / marker tokens (see RESERVED_IDS in "
        "olmo_core.data.document_chunk_landmark); required for landmark checkpoints",
    )
    ap.add_argument("--eval-size", type=int, default=500, help="number of examples to score")
    ap.add_argument("--max-length", type=int, default=65536)
    ap.add_argument(
        "--max-gold-tokens",
        type=int,
        default=32,
        help="cap on gold continuation length; every one of these is a decode step, so this is the "
        "main cost knob",
    )
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument(
        "--skip-too-long",
        action="store_true",
        help="skip over-budget prompts instead of failing. Off by default: an over-budget prompt is "
        "a config error, and silently dropping it changes which examples the report describes.",
    )
    ap.add_argument(
        "--no-topk",
        action="store_true",
        help="landmark models only: disable hard top-k decode retrieval "
        "(landmark_top_k_fraction defaults to 0.1, so it is ON otherwise). Top-k is applied on "
        "decode steps but never during the batched prefill, and is the dominant cause of the "
        "landmark decode/forward gap; with it off the two paths should agree, which turns this run "
        "into a real correctness check on the landmark KV-cache decode.",
    )
    ap.add_argument("--limit-print", type=int, default=5, help="per-example lines to print")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    from corpus_reasoning.eval.evaluate import load_unified_examples
    from corpus_reasoning.eval.inference_consistency import (
        compare_paths,
        forced_generate_batch,
        reference_step_logits,
    )
    from olmo_core.config import DType
    from olmo_core.generate.generation_module.config import GenerationConfig
    from olmo_core.generate.generation_module.transformer import (
        TransformerGenerationModuleConfig,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id != eos_id else eos_id

    reserved = None
    if args.family:
        from olmo_core.data.document_chunk_landmark import RESERVED_IDS

        if args.family not in RESERVED_IDS:
            sys.exit(
                f"[consistency] unknown --family {args.family!r}; known: {sorted(RESERVED_IDS)}"
            )
        reserved = RESERVED_IDS[args.family]

    gen_cfg_kwargs: Dict[str, Any] = dict(
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        max_length=args.max_length,
        use_cache=True,
        do_sample=False,
    )
    if reserved is not None:
        gen_cfg_kwargs["landmark_mem_id"] = reserved.landmark
        gen_cfg_kwargs["landmark_pad_id"] = reserved.pad
    if args.no_topk:
        # Both are needed: clearing only the block count still defers to the fraction, which
        # defaults to 0.1, so top-k would silently stay on.
        gen_cfg_kwargs["landmark_top_k_blocks"] = None
        gen_cfg_kwargs["landmark_top_k_fraction"] = None

    t0 = time.time()
    gm = TransformerGenerationModuleConfig(
        GenerationConfig(**gen_cfg_kwargs),
        float8_config=None,
        dtype=DType(args.dtype),
        compile_model=False,
    ).build(checkpoint_dir=args.model_path, device=device)
    print(
        f"[consistency] built model from {args.model_path} in {time.time() - t0:.1f}s", flush=True
    )

    variant = detect_variant(gm, topk_disabled=args.no_topk)
    print(f"[consistency] detected variant: {variant.name} (expect={variant.expect})")
    for n in variant.notes:
        print(f"[consistency]   {n}")
    if variant.is_landmark and reserved is None:
        sys.exit(
            "[consistency] this is a landmark checkpoint but --family was not given, so the "
            "landmark token id is unknown and generation cannot build the prompt it was trained on."
        )

    examples = load_unified_examples(args.data, args.eval_size, task=args.task, use_alpaca=True)
    print(f"[consistency] loaded {len(examples)} examples from {args.data}")

    stop_ids = {i for i in (eos_id, tok.pad_token_id) if i is not None}
    cap = args.max_length - args.max_gold_tokens

    reports = []
    per_example: List[Dict[str, Any]] = []
    n_skipped_long = 0
    n_skipped_empty = 0

    for i, ex in enumerate(examples):
        prompt_ids = tok(ex["prompt"], add_special_tokens=False)["input_ids"]
        gold_ids = tok(ex["expected_output"], add_special_tokens=False)["input_ids"]
        gold = _truncate_gold(gold_ids, stop_ids, args.max_gold_tokens)

        if not gold:
            n_skipped_empty += 1
            continue
        if len(prompt_ids) > cap:
            if not args.skip_too_long:
                sys.exit(
                    f"[consistency] example {i} builds a {len(prompt_ids)}-token prompt, past the "
                    f"{cap}-token cap (--max-length {args.max_length} minus --max-gold-tokens "
                    f"{args.max_gold_tokens}). Re-run with a larger --max-length, or pass "
                    f"--skip-too-long to exclude these and have the count reported."
                )
            n_skipped_long += 1
            continue

        prompt_t = torch.tensor([prompt_ids], device=device)

        model_space = None
        if variant.is_landmark:
            from olmo_core.generate.generation_module.transformer.generation_module import (
                _build_landmark_prompt,
            )

            assert reserved is not None
            model_space = _build_landmark_prompt(
                prompt_t,
                variant.mem_freq,
                reserved.landmark,
                mode=gm._generation_config.landmark_decode_mode,
                pad_id=reserved.pad,
                num_landmarks=variant.num_landmarks or 1,
            )

        try:
            trace = forced_generate_batch(
                gm, prompt_t, gold, model_space_prompt=model_space, log_timing=False
            )
            ref = reference_step_logits(
                gm,
                trace,
                pad_to_multiple=variant.block_multiple,
                pad_id=reserved.pad if reserved is not None else (pad_id or 0),
            )
            rep = compare_paths(variant.name, trace, ref)
        except Exception as e:  # noqa: BLE001 - one bad example should not lose the whole sweep
            print(f"[consistency] example {i} failed: {type(e).__name__}: {e}", flush=True)
            continue

        reports.append(rep)
        per_example.append(
            {
                "index": i,
                "n_steps": rep.n_steps,
                "prompt_tokens": len(prompt_ids),
                "ce_forward": rep.ce_forward,
                "ce_generate": rep.ce_generate,
                "ce_delta": rep.ce_delta,
                "mean_kl": rep.mean_kl,
                "max_kl": rep.max_kl,
                "top1_agreement": rep.top1_agreement,
                "first_divergent_step": rep.first_divergent_step,
            }
        )
        if len(reports) <= args.limit_print:
            print(rep.summary(), flush=True)

    if not reports:
        sys.exit("[consistency] no examples scored successfully; nothing to report.")

    eval_size = len(reports)
    agg: Dict[str, Any] = {
        "variant": variant.name,
        "expect": variant.expect,
        "model_path": args.model_path,
        "data": args.data,
        "task": args.task,
        "tokenizer": args.tokenizer,
        "dtype": args.dtype,
        "eval_size": eval_size,
        "n_skipped_too_long": n_skipped_long,
        "n_skipped_empty_gold": n_skipped_empty,
        "max_gold_tokens": args.max_gold_tokens,
        "landmark_topk_disabled": bool(args.no_topk),
        "variant_detection": asdict(variant),
    }
    agg.update(_summarize("ce_forward", [r.ce_forward for r in reports]))
    agg.update(_summarize("ce_generate", [r.ce_generate for r in reports]))
    agg.update(_summarize("abs_ce_delta", [abs(r.ce_delta) for r in reports]))
    agg.update(_summarize("mean_kl", [r.mean_kl for r in reports]))
    agg.update(_summarize("max_kl", [r.max_kl for r in reports]))
    agg.update(_summarize("top1_agreement", [r.top1_agreement for r in reports]))
    agg["fraction_examples_fully_agreeing"] = sum(
        1 for r in reports if r.top1_agreement == 1.0
    ) / len(reports)
    agg["per_example"] = per_example

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(agg, f, indent=2)

    print("\n" + "=" * 72)
    print(f"INFERENCE CONSISTENCY -- {variant.name}  (expect={variant.expect})")
    print(f"  eval_size={eval_size}", end="")
    if eval_size < 500:
        # Repo rule: a sub-500 eval must be flagged inline, next to the number, never presented bare.
        print("   [!] fewer than 500 examples -- treat differences at this size as provisional")
    else:
        print()
    print(f"  CE forward  mean={agg['ce_forward_mean']:.6f}")
    print(f"  CE generate mean={agg['ce_generate_mean']:.6f}")
    print(f"  |CE delta|  mean={agg['abs_ce_delta_mean']:.6f}  max={agg['abs_ce_delta_max']:.6f}")
    print(
        f"  KL(fwd||gen) mean={agg['mean_kl_mean']:.3e}  worst-example max={agg['max_kl_max']:.3e}"
    )
    print(
        f"  top-1 agreement mean={agg['top1_agreement_mean']:.4f}   "
        f"examples fully agreeing: {agg['fraction_examples_fully_agreeing']:.3f}"
    )
    if variant.expect == "identical" and agg["max_kl_max"] > 1e-2:
        print(
            "\n  [!] This variant's two paths are supposed to compute the same function, but they "
            "do not.\n      The cross-entropy reported for this checkpoint and the distributions it "
            "generates from\n      are describing different models. Investigate before trusting "
            "either number."
        )
    print("=" * 72)
    print(f"[consistency] wrote {args.out}")


if __name__ == "__main__":
    main()
