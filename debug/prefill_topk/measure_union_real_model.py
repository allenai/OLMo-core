"""
The number that decides whether the EXACT ("union") sparse prefill is fast.

``union`` mode keeps per-token top-k exactly, by having each 64-query block iterate the union of its
rows' selections. That saves work only to the extent that neighbouring queries agree. Random q/k
gives a worst case; this measures a real trained landmark model on a real long-context prompt.

    python debug/prefill_topk/measure_union_real_model.py \
        --model-path /data/prasann/olmo_ckpts/q06b-comp-contra-n20-sft-local/step750 \
        --data /scratch/users/prasann/corpus-reasoning/data/contradiction_eval_pubmed_both_ctx32k.jsonl \
        --tokenizer Qwen/Qwen3-0.6B --top-k 8 --max-length 16384

Reports, per layer and overall: mean union size vs k, and what fraction of the past blocks the union
covers (1.0 = no saving; k/past = perfect agreement).
"""

import argparse
import json
import types

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--top-k-fraction", type=float, default=None)
    ap.add_argument("--max-length", type=int, default=16384)
    ap.add_argument("--n-examples", type=int, default=2)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from olmo_core.config import DType
    from olmo_core.generate.generation_module.config import GenerationConfig
    from olmo_core.generate.generation_module.transformer import TransformerGenerationModuleConfig
    from olmo_core.nn.attention.landmark_fast import FastLandmarkAttention
    from olmo_core.nn.attention.landmark_prefill_sparse import selection_stats

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    device = torch.device("cuda:0")
    gen_cfg = GenerationConfig(
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
        max_length=args.max_length + 512, use_cache=True,
    )
    gm = TransformerGenerationModuleConfig(
        gen_cfg, float8_config=None, dtype=DType("bfloat16"), compile_model=False
    ).build(checkpoint_dir=args.model_path, device=device)

    layers = [m for m in gm.model.modules() if isinstance(m, FastLandmarkAttention)]
    print(f"[stats] {len(layers)} landmark layers, top_k={args.top_k}", flush=True)
    collected = []

    def _prefill_stats(self, q, k, v):
        Lb = self.block_size
        T = q.shape[2]
        pad = (-T) % Lb
        if pad:
            q = torch.nn.functional.pad(q, (0, 0, 0, pad))
            k = torch.nn.functional.pad(k, (0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, 0, 0, pad))
        n_blocks = q.shape[2] // Lb
        tk = args.top_k
        if args.top_k_fraction is not None:
            tk = max(1, int(args.top_k_fraction * n_blocks + 0.999))
        if n_blocks > tk + 1:
            st = selection_stats(
                q, k, block_size=Lb, softmax_scale=self.softmax_scale, top_k=tk
            )
            st["layer"] = self._stat_layer
            st["n_blocks"] = n_blocks
            collected.append(st)
        return self._prefill_orig_stats(q[:, :, : q.shape[2]], k, v)[:, :, :T]

    for i, attn in enumerate(layers):
        attn._stat_layer = i
        attn._prefill_orig_stats = attn._prefill
        attn._prefill = types.MethodType(_prefill_stats, attn)

    # Build prompts exactly as the eval does (raw JSONL fields are documents/claims, not a prompt).
    from ctc_eval.eval.evaluate import load_unified_examples

    rows = load_unified_examples(
        args.data, args.n_examples, task="contradiction", query_position="both"
    )
    for i, row in enumerate(rows):
        text = tok.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        ids = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length,
                  add_special_tokens=False)["input_ids"].to(device)
        print(f"[stats] example {i}: {ids.shape[1]} tokens", flush=True)
        with torch.no_grad():
            gm.generate_batch(input_ids=ids, max_new_tokens=1, log_timing=False)

    if not collected:
        print("no stats collected (context too short for the chosen top_k)")
        return
    import statistics

    mean_u = statistics.mean(s["union_mean"] for s in collected)
    mean_ratio = statistics.mean(s["union_over_k"] for s in collected)
    mean_frac = statistics.mean(s["frac_of_past_blocks"] for s in collected)
    print(f"\n=== union statistics (top_k={args.top_k}, n_blocks={collected[0]['n_blocks']}) ===")
    print(f"  mean union size      : {mean_u:.1f}  ({mean_ratio:.1f}x top_k)")
    print(f"  covers               : {mean_frac*100:.1f}% of each query's past blocks")
    print(f"  => exact 'union' mode speedup ceiling vs dense landmark: {1/max(mean_frac,1e-6):.1f}x")
    print("\n  per-layer union/k:")
    for s in collected[: len(layers)]:
        print(f"    layer {s['layer']:>2}: {s['union_mean']:>7.1f} ({s['union_over_k']:>5.1f}x k), "
              f"covers {s['frac_of_past_blocks']*100:>5.1f}%")


if __name__ == "__main__":
    main()
