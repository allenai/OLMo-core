"""Does the batched chunked eval produce the same task metric as single?

Single-example already proved F1 = 0.849 EM = 0.512 on n=12 chunked cot_enum
(outputs/eval_results/contradiction_n12_chunked_cot_enum.json). Per-token
comparison showed ~25% exact text match, which could be either (a) benign
bf16 tie-breaking on late tokens or (b) a real correctness bug.

This test answers that by running batched generation on the same 488
prompts and computing contradiction F1 on the parsed outputs. If F1
drops noticeably, we have a real bug to chase; if it matches, we ship.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from corpus_reasoning.lib.chunked_attention import AttentionPattern
from corpus_reasoning.eval.evaluate import parse_pairs, pair_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--attention-pattern", default="chunked")
    ap.add_argument("--max-test-samples", type=int, default=488)
    ap.add_argument("--variant", default="leftpad", choices=("leftpad", "rightpad"))
    ap.add_argument("--attn-impl", default="sdpa", choices=("sdpa", "eager"))
    args = ap.parse_args()

    from corpus_reasoning.eval.test_batched_chunked_generate import load_model
    model, tokenizer, doc_start_id, doc_end_id = load_model(
        args.base_model, args.lora, attn_impl=args.attn_impl,
    )

    # Load prompts and ground-truth gold pairs (mirrors evaluate.py sampling).
    import random
    with open(args.eval_data) as f:
        examples = [json.loads(line) for line in f]
    random.seed(42)
    if args.max_test_samples < len(examples):
        examples = random.sample(examples, args.max_test_samples)

    from corpus_reasoning.lib.data_format import build_prompt
    from corpus_reasoning.lib.chunked_attention import wrap_documents
    prompts, golds = [], []
    for ex in examples:
        p, _ = build_prompt(ex, task="contradiction", query_position="both", use_alpaca=True)
        prompts.append(wrap_documents(p))
        golds.append([sorted([int(a), int(b)]) for a, b in ex["gold_doc_indices"]])
    print(f"Loaded {len(prompts)} prompts")

    pattern = AttentionPattern(name=args.attention_pattern)
    if args.variant == "leftpad":
        from corpus_reasoning.eval.batched_chunked_generate import (
            generate_hf_batched as gen_fn,
        )
    elif args.variant == "rightpad":
        from corpus_reasoning.eval.batched_chunked_rightpad import (
            generate_hf_batched_rightpad as gen_fn,
        )
    else:
        raise SystemExit(f"unknown variant {args.variant}")

    t0 = time.time()
    responses = gen_fn(
        model, tokenizer, prompts,
        doc_start_id=doc_start_id, doc_end_id=doc_end_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=args.max_new_tokens,
        attention_pattern=pattern,
        batch_size=args.batch_size,
    )
    wall = time.time() - t0
    print(f"Batched generation ({args.variant}): {wall:.1f}s "
          f"({wall/len(prompts):.3f}s/ex, bs={args.batch_size})")

    # Score
    tp_total = fp_total = fn_total = 0
    em_count = parse_failures = 0
    for resp, gold in zip(responses, golds):
        pred = parse_pairs(resp)
        if pred is None:
            parse_failures += 1
            pred = []
        m = pair_metrics(pred, gold)
        pred_set = {tuple(p) for p in pred}
        gold_set = {tuple(p) for p in gold}
        tp_total += len(pred_set & gold_set)
        fp_total += len(pred_set - gold_set)
        fn_total += len(gold_set - pred_set)
        if pred_set == gold_set:
            em_count += 1
    P = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    R = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    EM = em_count / len(golds)
    parse = 1.0 - parse_failures / len(golds)
    print()
    print(f"=== batched baseline metrics (eval_size={len(golds)}) ===")
    print(f"  P={P:.3f}  R={R:.3f}  F1={F1:.3f}  EM={EM:.3f}  parse={parse:.3f}")
    print(f"  (reference single-example: F1=0.849  EM=0.512  parse=1.000)")


if __name__ == "__main__":
    main()
