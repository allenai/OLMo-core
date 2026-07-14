"""Per-example exact-text-match sweep for rightpad batched chunked eval.

Picks N prompts from the eval set, generates with single-example
(evaluate.generate_hf) as reference, then runs rightpad batched at each
batch size in --batch-sizes. Reports exact-text-match count and wall
time per batch size.

This is the focused diagnostic for the past session's open question:
math+SDPA+rightpad was bit-exact at bs=2 but F1 dropped 0.06 at bs=8.
Either bs=2 was a lucky pair (and even bs=2 will show drift on a
diverse sample) or drift compounds with batch size.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch

from corpus_reasoning.lib.chunked_attention import AttentionPattern, wrap_documents
from corpus_reasoning.lib.data_format import build_prompt
from corpus_reasoning.eval.batched_chunked_rightpad import generate_hf_batched_rightpad
from corpus_reasoning.eval.batched_chunked_serial_prefill import (
    generate_hf_batched_serial_prefill,
)
from corpus_reasoning.eval.evaluate import generate_hf


def _load_prompts(path, task, query_position, n_examples, seed=42):
    with open(path) as f:
        examples = [json.loads(line) for line in f]
    random.seed(seed)
    if n_examples < len(examples):
        examples = random.sample(examples, n_examples)
    prompts = []
    for ex in examples:
        p, _ = build_prompt(ex, task=task, query_position=query_position, use_alpaca=True)
        prompts.append(wrap_documents(p))
    return prompts


def _run_single(model, tokenizer, prompts, doc_start_id, doc_end_id,
                pattern, max_new_tokens):
    stop_ids = {tokenizer.eos_token_id}
    out = []
    t0 = time.time()
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        text = generate_hf(
            model, tokenizer, ids, doc_start_id, doc_end_id,
            max_new_tokens=max_new_tokens, stop_token_ids=stop_ids,
            backend="chunked-sdpa", attention_pattern=pattern,
        )
        out.append(text)
    return out, time.time() - t0


def _run_variant(variant, model, tokenizer, prompts, doc_start_id, doc_end_id,
                  pattern, max_new_tokens, batch_size):
    if variant == "rightpad":
        gen_fn = generate_hf_batched_rightpad
    elif variant == "serial":
        gen_fn = generate_hf_batched_serial_prefill
    else:
        raise SystemExit(f"unknown variant {variant!r}")
    t0 = time.time()
    out = gen_fn(
        model, tokenizer, prompts,
        doc_start_id=doc_start_id, doc_end_id=doc_end_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        attention_pattern=pattern,
        batch_size=batch_size,
    )
    return out, time.time() - t0


def _compare(ref, cand, label, tokenizer):
    exact = 0
    for i, (r, c) in enumerate(zip(ref, cand)):
        if r == c:
            exact += 1
            continue
        # Find first token-level divergence so we can see whether outputs
        # disagree at the very first token (full-prompt drift) or only
        # late (likely bf16 tie-breaking).
        r_tok = tokenizer(r, add_special_tokens=False).input_ids
        c_tok = tokenizer(c, add_special_tokens=False).input_ids
        common = 0
        for a, b in zip(r_tok, c_tok):
            if a != b:
                break
            common += 1
        print(f"  [{i}] {label}: DIFFER  common_prefix_tok={common}/"
              f"{max(len(r_tok), len(c_tok))}")
    return exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--task", default="outlier")
    ap.add_argument("--query-position", default="both")
    ap.add_argument("--n-examples", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=500)
    ap.add_argument("--attention-pattern", default="chunked")
    ap.add_argument("--batch-sizes", default="1,2,4,8",
                    help="Comma-separated list of batch sizes to test.")
    ap.add_argument("--variant", default="rightpad",
                    choices=("rightpad", "serial"),
                    help="Which batched-decode variant to test.")
    args = ap.parse_args()

    # Load model via the standard evaluate path so adapter handling, tokenizer
    # resize, and chunked-sdpa attn_impl all match production.
    import types
    from corpus_reasoning.eval.evaluate import load_hf_model
    model_args = types.SimpleNamespace(
        base_model=args.base_model, lora_path=args.lora, backend="chunked-sdpa",
        language_model_only=False, attn_impl_override=None,
    )
    print(f"Loading model: base={args.base_model}  lora={args.lora}")
    model, tokenizer, doc_start_id, doc_end_id = load_hf_model(model_args)
    pattern = AttentionPattern(name=args.attention_pattern)

    prompts = _load_prompts(args.eval_data, args.task, args.query_position,
                            args.n_examples)
    lens = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts]
    print(f"Loaded {len(prompts)} prompts. Length range: "
          f"{min(lens)}..{max(lens)} (mean {sum(lens)//len(lens)})")

    # Reference: single-example.
    print(f"\n=== single-example reference ===")
    ref, ref_t = _run_single(model, tokenizer, prompts, doc_start_id, doc_end_id,
                              pattern, args.max_new_tokens)
    print(f"  wall: {ref_t:.1f}s  ({ref_t/len(prompts):.2f}s/ex)")

    summary = []
    for bs in [int(b) for b in args.batch_sizes.split(",")]:
        print(f"\n=== {args.variant} bs={bs} ===")
        try:
            out, t = _run_variant(
                args.variant, model, tokenizer, prompts, doc_start_id,
                doc_end_id, pattern, args.max_new_tokens, bs,
            )
            print(f"  wall: {t:.1f}s  ({t/len(prompts):.2f}s/ex)  "
                  f"speedup={ref_t/t:.2f}x")
            exact = _compare(ref, out, f"bs={bs}", tokenizer)
            print(f"  exact match vs single: {exact}/{len(prompts)}")
            summary.append((bs, t, exact))
        except Exception as e:
            import traceback
            traceback.print_exc()
            summary.append((bs, float("nan"), -1))

    print("\n=== summary ===")
    print(f"  reference:    {ref_t:6.1f}s")
    for bs, t, exact in summary:
        sp = ref_t / t if t and t == t else float("nan")
        print(f"  {args.variant} bs={bs}: {t:6.1f}s  speedup={sp:5.2f}x  "
              f"exact={exact}/{len(prompts)}")


if __name__ == "__main__":
    main()
