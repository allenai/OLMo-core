"""Is the batched pipeline itself correct at B=1 (no padding)?

Compares batched vs single-example output token-by-token on several
prompts. If all match at bs=1 with no padding, the bug is left-padding
specific (attention sinks, RoPE, etc.). If they don't match, we have a
bug in the batched pipeline unrelated to padding.
"""

from __future__ import annotations

import argparse
import json
import random
import time

import torch

from corpus_reasoning.lib.chunked_attention import AttentionPattern
from corpus_reasoning.eval.test_batched_chunked_generate import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--n-examples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    args = ap.parse_args()

    model, tokenizer, doc_start_id, doc_end_id = load_model(
        args.base_model, args.lora, attn_impl="sdpa",
    )
    pattern = AttentionPattern(name="chunked")

    from corpus_reasoning.lib.data_format import build_prompt
    from corpus_reasoning.lib.chunked_attention import wrap_documents

    with open(args.eval_data) as f:
        examples = [json.loads(line) for line in f]
    random.seed(42)
    examples = random.sample(examples, args.n_examples)

    prompts = []
    for ex in examples:
        p, _ = build_prompt(ex, task="contradiction", query_position="both", use_alpaca=True)
        prompts.append(wrap_documents(p))

    from corpus_reasoning.eval.evaluate import generate_hf
    from corpus_reasoning.eval.batched_chunked_generate import generate_hf_batched

    stop_ids = {tokenizer.eos_token_id}

    print(f"\n=== single-example ===")
    t0 = time.time()
    single_out = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        single_out.append(generate_hf(
            model, tokenizer, ids, doc_start_id, doc_end_id,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=stop_ids, backend="chunked-sdpa",
            attention_pattern=pattern,
        ))
    print(f"  wall: {time.time()-t0:.1f}s")

    print(f"\n=== batched bs=1 (no padding) ===")
    t0 = time.time()
    bs1_out = generate_hf_batched(
        model, tokenizer, prompts,
        doc_start_id=doc_start_id, doc_end_id=doc_end_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=args.max_new_tokens,
        attention_pattern=pattern, batch_size=1,
    )
    print(f"  wall: {time.time()-t0:.1f}s")

    print(f"\n=== agreement (bs=1) ===")
    exact = 0
    for i, (s, b) in enumerate(zip(single_out, bs1_out)):
        if s == b:
            exact += 1
        else:
            s_tok = tokenizer(s, add_special_tokens=False).input_ids
            b_tok = tokenizer(b, add_special_tokens=False).input_ids
            common = 0
            for a, c in zip(s_tok, b_tok):
                if a != c:
                    break
                common += 1
            print(f"  [{i}] DIFFER  common_prefix_tok={common}/{max(len(s_tok),len(b_tok))}")
            print(f"    single: {s[:120]!r}")
            print(f"    batch : {b[:120]!r}")
    print(f"\n  exact: {exact}/{len(prompts)}")


if __name__ == "__main__":
    main()
