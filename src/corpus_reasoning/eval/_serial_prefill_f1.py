"""Full-eval F1 verification: single-example vs serial-prefill+batched-decode.

Runs both paths on the same N prompts from an outlier eval file, scores
both via evaluate._eval_outlier, and reports side-by-side F1/EM/parse.

Pass criteria for shipping:
  - F1 within ~0.01 of single-example
  - parse_rate identical (both should reach a parseable Outliers: line)
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from corpus_reasoning.lib.chunked_attention import AttentionPattern, wrap_documents
from corpus_reasoning.lib.data_format import build_prompt
from corpus_reasoning.eval.batched_chunked_serial_prefill import (
    generate_hf_batched_serial_prefill,
)
from corpus_reasoning.eval.evaluate import generate_hf, _eval_outlier


def _load_examples(path, n_max, seed=42):
    with open(path) as f:
        examples = [json.loads(line) for line in f]
    import random
    random.seed(seed)
    if n_max < len(examples):
        examples = random.sample(examples, n_max)
    return examples


def _build(examples, task, query_position):
    entries, prompts = [], []
    for ex in examples:
        p, _ = build_prompt(ex, task=task, query_position=query_position,
                            use_alpaca=True)
        prompts.append(wrap_documents(p))
        entries.append({"ex": ex})
    return entries, prompts


def _gen_single(model, tokenizer, prompts, doc_start_id, doc_end_id, pattern,
                max_new_tokens):
    stop_ids = {tokenizer.eos_token_id}
    out = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        ids = tokenizer(p, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        text = generate_hf(
            model, tokenizer, ids, doc_start_id, doc_end_id,
            max_new_tokens=max_new_tokens, stop_token_ids=stop_ids,
            backend="chunked-sdpa", attention_pattern=pattern,
        )
        out.append(text)
        if (i + 1) % 50 == 0:
            print(f"  single: {i+1}/{len(prompts)}  elapsed={time.time()-t0:.0f}s",
                  flush=True)
    return out, time.time() - t0


def _gen_serial(model, tokenizer, prompts, doc_start_id, doc_end_id, pattern,
                max_new_tokens, batch_size):
    t0 = time.time()
    out = generate_hf_batched_serial_prefill(
        model, tokenizer, prompts,
        doc_start_id=doc_start_id, doc_end_id=doc_end_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        attention_pattern=pattern,
        batch_size=batch_size,
    )
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--task", default="outlier")
    ap.add_argument("--query-position", default="both")
    ap.add_argument("--n-examples", type=int, default=400)
    ap.add_argument("--max-new-tokens", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    import types
    from corpus_reasoning.eval.evaluate import load_hf_model
    margs = types.SimpleNamespace(
        base_model=args.base_model, lora_path=args.lora, backend="chunked-sdpa",
        language_model_only=False, attn_impl_override=None,
    )
    print(f"Loading model: base={args.base_model}  lora={args.lora}", flush=True)
    model, tokenizer, doc_start_id, doc_end_id = load_hf_model(margs)
    pattern = AttentionPattern(name="chunked")

    examples = _load_examples(args.eval_data, args.n_examples)
    entries, prompts = _build(examples, args.task, args.query_position)
    lens = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts]
    print(f"Loaded {len(prompts)} prompts. Length range: "
          f"{min(lens)}..{max(lens)} (mean {sum(lens)//len(lens)})", flush=True)

    print(f"\n=== single-example reference ===", flush=True)
    single_resp, single_t = _gen_single(
        model, tokenizer, prompts, doc_start_id, doc_end_id, pattern,
        args.max_new_tokens,
    )
    print(f"  total wall: {single_t:.0f}s  ({single_t/len(prompts):.2f}s/ex)", flush=True)

    print(f"\n=== serial-prefill + batched-decode bs={args.batch_size} ===", flush=True)
    serial_resp, serial_t = _gen_serial(
        model, tokenizer, prompts, doc_start_id, doc_end_id, pattern,
        args.max_new_tokens, args.batch_size,
    )
    print(f"  total wall: {serial_t:.0f}s  ({serial_t/len(prompts):.2f}s/ex)  "
          f"speedup={single_t/serial_t:.2f}x", flush=True)

    # Score both.
    print(f"\n=== scoring ===", flush=True)
    single_metrics, _ = _eval_outlier(entries, single_resp)
    serial_metrics, _ = _eval_outlier(entries, serial_resp)
    print(f"  single:  P={single_metrics['precision']:.4f}  "
          f"R={single_metrics['recall']:.4f}  "
          f"F1={single_metrics['f1']:.4f}  "
          f"EM={single_metrics['exact_match']:.4f}  "
          f"parse={single_metrics['parse_rate']:.4f}")
    print(f"  serial:  P={serial_metrics['precision']:.4f}  "
          f"R={serial_metrics['recall']:.4f}  "
          f"F1={serial_metrics['f1']:.4f}  "
          f"EM={serial_metrics['exact_match']:.4f}  "
          f"parse={serial_metrics['parse_rate']:.4f}")
    print(f"  delta:   P={serial_metrics['precision']-single_metrics['precision']:+.4f}  "
          f"R={serial_metrics['recall']-single_metrics['recall']:+.4f}  "
          f"F1={serial_metrics['f1']-single_metrics['f1']:+.4f}  "
          f"EM={serial_metrics['exact_match']-single_metrics['exact_match']:+.4f}  "
          f"parse={serial_metrics['parse_rate']-single_metrics['parse_rate']:+.4f}")

    # Per-example string identity
    exact = sum(1 for s, r in zip(single_resp, serial_resp) if s == r)
    print(f"\n  exact text match: {exact}/{len(prompts)} "
          f"({100*exact/len(prompts):.1f}%)")

    if args.output_json:
        out = {
            "single_metrics": single_metrics,
            "serial_metrics": serial_metrics,
            "single_wall_s": single_t, "serial_wall_s": serial_t,
            "speedup": single_t / serial_t,
            "exact_text_match": exact,
            "n_examples": len(prompts),
        }
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
