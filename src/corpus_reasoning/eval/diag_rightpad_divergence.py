"""Pinpoint where right-pad batched diverges from single-example.

Runs three configurations on a controlled set of prompts and compares
them layer-by-layer and token-by-token:

  (a) single-example  — the reference.
  (b) bs=2, SAME-length prompts (no padding anywhere)
  (c) bs=2, DIFFERENT-length prompts  — only this one has padding.

After prefill:
  * compare first-token logits for the shorter prompt across (a), (b), (c).
  * compare layer-wise hidden-state L2 error for each layer's output at
    the last-real-token position.

This tells us whether (b) matches — i.e., batching alone is fine — and
whether (c)'s divergence is localized to specific layers (suggesting a
specific kernel interaction) or smeared across all layers (suggesting a
systemic numerics issue).
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Optional

import torch

from corpus_reasoning.lib.chunked_attention import AttentionPattern, wrap_documents
from corpus_reasoning.lib.data_format import build_prompt
from corpus_reasoning.eval.chunked_batch_helpers import build_prefill_mask
from corpus_reasoning.eval.batched_chunked_rightpad import (
    _build_chunk_ids_right_padded, _pad_batch_right,
)
from corpus_reasoning.eval.test_batched_chunked_generate import load_model


def _load_prompt_pair(eval_data, n_pair_examples, seed=42):
    """Sample two prompts of different lengths."""
    with open(eval_data) as f:
        examples = [json.loads(line) for line in f]
    random.seed(seed)
    random.shuffle(examples)
    chosen = []
    for ex in examples:
        p, _ = build_prompt(ex, task="contradiction", query_position="both", use_alpaca=True)
        chosen.append((wrap_documents(p), ex))
        if len(chosen) >= n_pair_examples:
            break
    return chosen


@torch.no_grad()
def prefill_and_capture_hidden(model, tokenizer, prompt_ids, pattern,
                               doc_start_id, doc_end_id, pad_token_id,
                               companion_prompt_ids=None):
    """Run prefill, return (last_real_logits, hidden_states_per_layer).

    If companion_prompt_ids is provided, builds a 2-item batch using
    right-padding, then extracts the target prompt's outputs.
    """
    device = next(model.parameters()).device

    if companion_prompt_ids is None:
        # Single (no batching).
        input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        lens = [len(prompt_ids)]
        B, S = 1, len(prompt_ids)
        target_slot = 0
    else:
        # Batch of 2 right-padded.
        input_ids, lens = _pad_batch_right(
            [prompt_ids, companion_prompt_ids], pad_token_id, device,
        )
        B, S = input_ids.shape
        target_slot = 0  # target is first item in batch

    chunk_ids, n_docs_list = _build_chunk_ids_right_padded(
        input_ids, lens, doc_start_id, doc_end_id,
    )
    if B == 1:
        # Single-slot: just build the unbatched mask so attention matches
        # evaluate.generate_hf exactly (no pad-diagonal fix at all).
        from corpus_reasoning.lib.chunked_attention import build_dense_bool_mask
        bool_mask = build_dense_bool_mask(pattern, chunk_ids)
        dtype = torch.bfloat16
        min_val = torch.finfo(dtype).min
        mask = torch.where(
            bool_mask,
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), min_val, dtype=dtype, device=device),
        ).unsqueeze(1)  # (B, 1, S, S)
    else:
        mask = build_prefill_mask(chunk_ids, input_ids, doc_end_id, n_docs_list, pattern)

    outputs = model(
        input_ids=input_ids,
        attention_mask=mask,
        use_cache=True,
        output_hidden_states=True,
    )
    # Logits at target's last real token.
    L_target = lens[target_slot]
    last_logit = outputs.logits[target_slot, L_target - 1, :]  # (V,)
    # Hidden states at target's last real token per layer.
    per_layer = [h[target_slot, L_target - 1, :].clone() for h in outputs.hidden_states]
    return last_logit, per_layer, outputs.past_key_values


def compare_layers(ref_layers, cand_layers, label):
    print(f"  {label}  layer-by-layer L2 error at last-real-token:")
    for i, (r, c) in enumerate(zip(ref_layers, cand_layers)):
        err = (r.float() - c.float()).norm().item()
        rel = err / (r.float().norm().item() + 1e-9)
        flag = "  <— diverged" if rel > 1e-3 else ""
        print(f"    layer {i:2d}: L2={err:8.4f}  rel={rel:.2e}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--attn-impl", default="sdpa", choices=("sdpa", "eager"))
    ap.add_argument(
        "--sdp-backend", default="default",
        choices=("default", "math", "mem_efficient", "flash", "cudnn"),
        help="For attn_impl=sdpa: force a specific SDPA kernel to isolate "
             "numerical drift between backends. The math backend runs "
             "softmax in fp32 and should be bit-exact; mem-efficient/flash "
             "use block-tiled online softmax that drifts with pad columns.",
    )
    args = ap.parse_args()

    print(f"attn_impl = {args.attn_impl}  sdp_backend = {args.sdp_backend}")
    model, tokenizer, doc_start_id, doc_end_id = load_model(
        args.base_model, args.lora, attn_impl=args.attn_impl,
    )

    if args.attn_impl == "sdpa" and args.sdp_backend != "default":
        # Force a specific SDPA kernel. torch.nn.attention.sdpa_kernel accepts
        # a list of allowed backends; by using a singleton we exclude others.
        from torch.nn.attention import sdpa_kernel, SDPBackend
        backend_map = {
            "math": SDPBackend.MATH,
            "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
            "flash": SDPBackend.FLASH_ATTENTION,
            "cudnn": SDPBackend.CUDNN_ATTENTION,
        }
        sdp_cm = sdpa_kernel([backend_map[args.sdp_backend]])
    else:
        import contextlib
        sdp_cm = contextlib.nullcontext()
    pattern = AttentionPattern(name="chunked")
    prompt_pairs = _load_prompt_pair(args.eval_data, n_pair_examples=8)

    # Find one short prompt (target) and one longer prompt (companion).
    prompt_ids_all = [
        tokenizer(p, add_special_tokens=False)["input_ids"] for p, _ in prompt_pairs
    ]
    idx_sorted = sorted(range(len(prompt_ids_all)), key=lambda i: len(prompt_ids_all[i]))
    short_idx, long_idx = idx_sorted[0], idx_sorted[-1]
    target_ids = prompt_ids_all[short_idx]
    longer_ids = prompt_ids_all[long_idx]
    same_ids = target_ids  # copy of same length

    print(f"Target (short) len={len(target_ids)}")
    print(f"Longer companion len={len(longer_ids)}")
    print(f"\nConfigurations:")
    print(f"  (a) single-example  (no batch, no pad)")
    print(f"  (b) bs=2 same-len   (no pad anywhere)")
    print(f"  (c) bs=2 + padding  (target padded up to longer prompt)")

    with sdp_cm:
        print(f"\n=== (a) single-example baseline ===")
        logit_a, layers_a, _ = prefill_and_capture_hidden(
            model, tokenizer, target_ids, pattern,
            doc_start_id, doc_end_id, tokenizer.pad_token_id,
            companion_prompt_ids=None,
        )
        print(f"  last-real-token argmax: {logit_a.argmax().item()}  "
              f"top-1 token: {tokenizer.decode([int(logit_a.argmax())])!r}")

        print(f"\n=== (b) bs=2 with identical-length prompts ===")
        logit_b, layers_b, _ = prefill_and_capture_hidden(
            model, tokenizer, target_ids, pattern,
            doc_start_id, doc_end_id, tokenizer.pad_token_id,
            companion_prompt_ids=same_ids,
        )
        same_argmax_b = logit_a.argmax().item() == logit_b.argmax().item()
        max_layer_err_b = max(
            (r.float() - c.float()).norm().item() for r, c in zip(layers_a, layers_b)
        )
        print(f"  last-real-token argmax match vs single: {same_argmax_b}")
        print(f"  max layer L2 error vs single: {max_layer_err_b:.4f}")
        compare_layers(layers_a, layers_b, "(b) vs (a)")

        print(f"\n=== (c) bs=2 with different-length prompts (right-pad) ===")
        logit_c, layers_c, _ = prefill_and_capture_hidden(
            model, tokenizer, target_ids, pattern,
            doc_start_id, doc_end_id, tokenizer.pad_token_id,
            companion_prompt_ids=longer_ids,
        )
        same_argmax_c = logit_a.argmax().item() == logit_c.argmax().item()
        max_layer_err_c = max(
            (r.float() - c.float()).norm().item() for r, c in zip(layers_a, layers_c)
        )
        print(f"  last-real-token argmax match vs single: {same_argmax_c}")
        print(f"  max layer L2 error vs single: {max_layer_err_c:.4f}")
        compare_layers(layers_a, layers_c, "(c) vs (a)")


if __name__ == "__main__":
    main()
