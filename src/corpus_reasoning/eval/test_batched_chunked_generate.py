"""Correctness + benchmark harness for the batched chunked-eval variants.

Exercises each registered variant against the existing single-example
path in `evaluate.py::generate_hf` and reports (a) per-example text
agreement and (b) wall-clock speedup. The single-example baseline is
run first, then each variant runs against the same prompts.

Variants registered below:

  * baseline    — scripts/eval/batched_chunked_generate.py
  * prune       — scripts/eval/batched_chunked_prune.py
  * compile     — scripts/eval/batched_chunked_compile.py  (torch.compile)
  * flex        — scripts/eval/batched_chunked_flex.py     (requires
                    model loaded with attn_implementation=flex_attention)

Select with `--variants baseline,prune,compile` (comma-separated). Flex
is excluded by default because it needs a different model load; pass
`--load-flex` to reload the model with flex_attention before running it.

Example:
    conda activate corpus-reasoning-train
    python scripts/eval/test_batched_chunked_generate.py \
        --lora outputs/checkpoints/contradiction-pubmed-n12-k3-qwen08-lora-chunked-cot-enum \
        --base-model Qwen/Qwen3.5-0.8B-Base \
        --eval-data data/contradiction_eval_pubmed_both_n12_k3.jsonl \
        --n-examples 8 --batch-size 4 --max-new-tokens 400 \
        --variants baseline,prune,compile
"""

from __future__ import annotations

import argparse
import json
import random
import time
from typing import Callable

import torch

from corpus_reasoning.lib.chunked_attention import AttentionPattern


# ---------------------------------------------------------------------------
# Model + prompt loading
# ---------------------------------------------------------------------------

def load_model(base_model: str, lora_path: str, attn_impl: str = "sdpa"):
    """Reuse evaluate.load_hf_model so tokenizer resize + adapter-key
    normalization stays in one place."""
    import types

    from corpus_reasoning.eval.evaluate import load_hf_model

    backend = "chunked-flex" if attn_impl == "flex_attention" else "chunked-sdpa"
    args = types.SimpleNamespace(
        base_model=base_model,
        lora_path=lora_path,
        backend=backend,
        language_model_only=False,
        # load_hf_model picks attn_impl from backend unless this override is set.
        attn_impl_override=attn_impl if attn_impl == "eager" else None,
    )
    return load_hf_model(args)


def load_prompts(eval_data: str, task: str, n_examples: int, seed: int = 42):
    from corpus_reasoning.lib.data_format import build_prompt
    from corpus_reasoning.lib.chunked_attention import wrap_documents

    with open(eval_data) as f:
        examples = [json.loads(line) for line in f]
    random.seed(seed)
    if n_examples < len(examples):
        examples = random.sample(examples, n_examples)

    prompts = []
    for ex in examples:
        p, _ = build_prompt(ex, task=task, query_position="both", use_alpaca=True)
        p = wrap_documents(p)
        prompts.append(p)
    return prompts


# ---------------------------------------------------------------------------
# Variant callables
# ---------------------------------------------------------------------------

def run_single(model, tokenizer, prompts, doc_start_id, doc_end_id,
               max_new_tokens, pattern, **_):
    from corpus_reasoning.eval.evaluate import generate_hf
    stop_ids = {tokenizer.eos_token_id}
    results = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        text = generate_hf(
            model, tokenizer, ids, doc_start_id, doc_end_id,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_ids,
            backend="chunked-sdpa",
            attention_pattern=pattern,
        )
        results.append(text)
    return results


def run_baseline(model, tokenizer, prompts, doc_start_id, doc_end_id,
                 max_new_tokens, pattern, batch_size, **_):
    from corpus_reasoning.eval.batched_chunked_generate import generate_hf_batched
    return generate_hf_batched(
        model, tokenizer, prompts,
        doc_start_id=doc_start_id, doc_end_id=doc_end_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        attention_pattern=pattern,
        batch_size=batch_size,
    )


def run_prune(model, tokenizer, prompts, doc_start_id, doc_end_id,
              max_new_tokens, pattern, batch_size, **_):
    from corpus_reasoning.eval.batched_chunked_prune import generate_hf_batched_prune
    return generate_hf_batched_prune(
        model, tokenizer, prompts,
        doc_start_id=doc_start_id, doc_end_id=doc_end_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        attention_pattern=pattern,
        batch_size=batch_size,
    )


def run_compile(model, tokenizer, prompts, doc_start_id, doc_end_id,
                max_new_tokens, pattern, batch_size, **_):
    from corpus_reasoning.eval.batched_chunked_compile import generate_hf_batched_compiled
    return generate_hf_batched_compiled(
        model, tokenizer, prompts,
        doc_start_id=doc_start_id, doc_end_id=doc_end_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        attention_pattern=pattern,
        batch_size=batch_size,
    )


def run_flex(model, tokenizer, prompts, doc_start_id, doc_end_id,
             max_new_tokens, pattern, batch_size, **_):
    from corpus_reasoning.eval.batched_chunked_flex import generate_hf_batched_flex
    return generate_hf_batched_flex(
        model, tokenizer, prompts,
        doc_start_id=doc_start_id, doc_end_id=doc_end_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        attention_pattern=pattern,
        batch_size=batch_size,
    )


VARIANTS: dict[str, Callable] = {
    "baseline": run_baseline,
    "prune": run_prune,
    "compile": run_compile,
    "flex": run_flex,
}


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare(tokenizer, reference: list[str], candidate: list[str], label: str):
    exact = 0
    for i, (r, c) in enumerate(zip(reference, candidate)):
        if r == c:
            exact += 1
            continue
        r_tok = tokenizer(r, add_special_tokens=False).input_ids
        c_tok = tokenizer(c, add_special_tokens=False).input_ids
        common = 0
        for a, b in zip(r_tok, c_tok):
            if a != b:
                break
            common += 1
        print(f"  [{i}] {label}: DIFFER  common_prefix_tok={common}/{max(len(r_tok), len(c_tok))}")
        print(f"       ref:  {r[:140]!r}")
        print(f"       cand: {c[:140]!r}")
    return exact


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--task", default="contradiction")
    ap.add_argument("--n-examples", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--attention-pattern", default="chunked")
    ap.add_argument(
        "--variants", default="baseline",
        help="Comma-separated subset of " + ",".join(VARIANTS),
    )
    ap.add_argument(
        "--attn-impl", default="sdpa", choices=("sdpa", "flex_attention"),
        help="Model attention implementation at load time. Use flex_attention "
             "when including the 'flex' variant.",
    )
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"Unknown variant '{v}'. Options: {list(VARIANTS)}")

    pattern = AttentionPattern(name=args.attention_pattern)
    print(f"Loading model: base={args.base_model}  lora={args.lora}  attn={args.attn_impl}")
    model, tokenizer, doc_start_id, doc_end_id = load_model(
        args.base_model, args.lora, attn_impl=args.attn_impl,
    )

    prompts = load_prompts(args.eval_data, args.task, args.n_examples)
    print(f"Loaded {len(prompts)} prompts. Lengths:",
          [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts])

    kwargs = dict(
        doc_start_id=doc_start_id, doc_end_id=doc_end_id,
        max_new_tokens=args.max_new_tokens, pattern=pattern,
        batch_size=args.batch_size,
    )

    # Baseline (single-example, existing path).
    print(f"\n=== single (baseline) ===")
    t0 = time.time()
    single_out = run_single(model, tokenizer, prompts, **kwargs)
    single_t = time.time() - t0
    print(f"  wall: {single_t:.2f}s  ({single_t/len(prompts):.2f}s/ex)")

    summary = []
    for v in variants:
        print(f"\n=== {v} (bs={args.batch_size}) ===")
        try:
            if v == "compile" and len(prompts) >= args.batch_size:
                t_warm = time.time()
                _ = VARIANTS[v](model, tokenizer, prompts[: args.batch_size], **kwargs)
                print(f"  (compile warmup: {time.time()-t_warm:.1f}s)")

            t0 = time.time()
            out = VARIANTS[v](model, tokenizer, prompts, **kwargs)
            elapsed = time.time() - t0
            print(f"  wall: {elapsed:.2f}s  ({elapsed/len(prompts):.2f}s/ex)")
            print(f"  speedup vs single: {single_t/elapsed:.2f}x")
            exact = compare(tokenizer, single_out, out, v)
            print(f"  exact matches vs single: {exact}/{len(prompts)}")
            summary.append((v, elapsed, exact))
        except Exception as e:
            import traceback
            print(f"  FAILED ({type(e).__name__}): {e}")
            traceback.print_exc()
            summary.append((v, float("nan"), -1))

    print("\n=== summary ===")
    print(f"  single: {single_t:.2f}s")
    for v, t, e in summary:
        print(f"  {v:10s}: {t:>6.2f}s  speedup={single_t/t:4.2f}x  exact={e}/{len(prompts)}")


if __name__ == "__main__":
    main()
