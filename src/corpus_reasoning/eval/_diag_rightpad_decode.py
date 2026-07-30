"""Per-layer hidden-state diff at PREFILL+1-DECODE-STEP.

Extends scripts/eval/diag_rightpad_divergence.py to capture hidden
states one decode step into generation, not just at end-of-prefill.

Goal: if right-padding pollutes the GDN linear-attention state with pad
tokens, the pollution stays hidden during prefill (the diag's
last-real-token slot is at L_b-1, BEFORE any pads in right-padding) but
shows up during decode (the new token consumes a polluted state).

If divergence appears at a GDN layer (Qwen3.5-0.8B layers 0-4 — first
softmax layer is layer 5) when comparing (c) right-padded vs (a) single,
GDN state pollution is confirmed. If divergence is still only in
softmax layers (5+), the issue is pure SDPA kernel drift and we need a
different fix.

Three configs (same as diag_rightpad_divergence.py):
  (a) single-example  — reference
  (b) bs=2 same-length — no pad anywhere; tests batching alone
  (c) bs=2 different-length — target right-padded to companion length
"""
from __future__ import annotations

import argparse
import json
import random

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from corpus_reasoning.lib.chunked_attention import (
    AttentionPattern,
    FREE_CHUNK_ID,
    PAD_CHUNK_ID,
    build_dense_bool_mask,
    find_chunk_spans,
    wrap_documents,
)
from corpus_reasoning.lib.data_format import build_prompt
from corpus_reasoning.eval.chunked_batch_helpers import build_prefill_mask
from corpus_reasoning.eval.batched_chunked_rightpad import (
    _build_chunk_ids_right_padded, _pad_batch_right,
)


def _load_model(base_model, lora_path):
    import types
    from corpus_reasoning.eval.evaluate import load_hf_model
    args = types.SimpleNamespace(
        base_model=base_model, lora_path=lora_path, backend="chunked-sdpa",
        language_model_only=False, attn_impl_override=None,
    )
    return load_hf_model(args)


def _load_prompt_pair(eval_data, n_examples=10, seed=42, task="outlier",
                      query_position="both"):
    with open(eval_data) as f:
        examples = [json.loads(line) for line in f]
    random.seed(seed)
    random.shuffle(examples)
    out = []
    for ex in examples[:n_examples]:
        p, _ = build_prompt(ex, task=task, query_position=query_position,
                            use_alpaca=True)
        out.append(wrap_documents(p))
    return out


@torch.no_grad()
def _prefill_then_one_decode(model, tokenizer, target_ids, pattern,
                             doc_start_id, doc_end_id, pad_token_id,
                             companion_ids=None, force_math=True):
    """Run prefill on (target [+ companion right-padded]) under MATH SDPA,
    sample the first decode token, run ONE decode step, and return:
      - prefill_last_real_logit at slot L_target-1
      - decode_step1_logit
      - per-layer hidden states at the decode step (target slot 0)
      - (B, n_layers) extra: also the prefill-time hidden state at L_target-1
    """
    device = next(model.parameters()).device

    if companion_ids is None:
        input_ids = torch.tensor(target_ids, dtype=torch.long,
                                  device=device).unsqueeze(0)
        lens = [len(target_ids)]
        B, S = 1, len(target_ids)
        target_slot = 0
    else:
        input_ids, lens = _pad_batch_right(
            [target_ids, companion_ids], pad_token_id, device,
        )
        B, S = input_ids.shape
        target_slot = 0

    chunk_ids, n_docs_list = _build_chunk_ids_right_padded(
        input_ids, lens, doc_start_id, doc_end_id,
    )
    if B == 1:
        bool_mask = build_dense_bool_mask(pattern, chunk_ids)
        dtype = torch.bfloat16
        min_val = torch.finfo(dtype).min
        prefill_mask = torch.where(
            bool_mask, torch.zeros((), dtype=dtype, device=device),
            torch.full((), min_val, dtype=dtype, device=device),
        ).unsqueeze(1)
    else:
        prefill_mask = build_prefill_mask(
            chunk_ids, input_ids, doc_end_id, n_docs_list, pattern,
        )

    cm = sdpa_kernel([SDPBackend.MATH]) if force_math else _nullcontext()
    with cm:
        # PREFILL
        outputs = model(
            input_ids=input_ids, attention_mask=prefill_mask,
            use_cache=True, output_hidden_states=True,
        )
        L_target = lens[target_slot]
        prefill_last_real_logit = outputs.logits[target_slot, L_target - 1, :].clone()
        prefill_layers = [h[target_slot, L_target - 1, :].clone()
                          for h in outputs.hidden_states]
        past_kv = outputs.past_key_values

        # Sample first decode token from prefill last-real-token logit
        first_token = prefill_last_real_logit.argmax().unsqueeze(0)  # (1,)
        if B == 1:
            decode_input = first_token.unsqueeze(0)  # (1, 1)
            decode_pos = torch.tensor([[L_target]], device=device)
            # Build a 2D attention mask over all past + new tokens.
            decode_mask = torch.ones((1, S + 1), dtype=torch.long, device=device)
        else:
            # In bs=2: sample the target's first decode token. Companion's
            # decode token doesn't matter for the diagnostic; use 0.
            decode_input = torch.zeros((B, 1), dtype=torch.long, device=device)
            decode_input[target_slot, 0] = first_token[0]
            # Position_ids per example: target uses L_target, companion uses L_b
            lens_t = torch.tensor(lens, dtype=torch.long, device=device)
            decode_pos = lens_t.unsqueeze(-1)  # (B, 1)
            decode_mask = torch.zeros((B, S + 1), dtype=torch.long, device=device)
            for b, L in enumerate(lens):
                decode_mask[b, :L] = 1
            decode_mask[:, S:] = 1

        # ONE decode step
        outputs2 = model(
            input_ids=decode_input,
            attention_mask=decode_mask,
            position_ids=decode_pos,
            past_key_values=past_kv,
            use_cache=True,
            output_hidden_states=True,
        )
        decode_logit = outputs2.logits[target_slot, 0, :].clone()
        decode_layers = [h[target_slot, 0, :].clone()
                         for h in outputs2.hidden_states]

    return {
        "prefill_logit": prefill_last_real_logit,
        "prefill_layers": prefill_layers,
        "decode_logit": decode_logit,
        "decode_layers": decode_layers,
        "first_token": int(first_token.item()),
    }


def _nullcontext():
    import contextlib
    return contextlib.nullcontext()


def _compare_layers(ref_layers, cand_layers, label):
    print(f"  {label}  layer-by-layer L2 error:")
    for i, (r, c) in enumerate(zip(ref_layers, cand_layers)):
        err = (r.float() - c.float()).norm().item()
        rel = err / (r.float().norm().item() + 1e-9)
        flag = "  <— diverged" if rel > 1e-3 else ""
        print(f"    layer {i:2d}: L2={err:9.4f}  rel={rel:.2e}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-0.8B-Base")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--eval-data", required=True)
    ap.add_argument("--task", default="outlier")
    ap.add_argument("--query-position", default="both")
    ap.add_argument("--no-math", action="store_true",
                    help="Run without forcing MATH SDPA backend (default: forced).")
    args = ap.parse_args()

    print(f"Loading model: base={args.base_model}  lora={args.lora}")
    model, tokenizer, doc_start_id, doc_end_id = _load_model(
        args.base_model, args.lora,
    )
    pattern = AttentionPattern(name="chunked")

    prompts = _load_prompt_pair(args.eval_data, n_examples=10,
                                 task=args.task,
                                 query_position=args.query_position)
    enc = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
    idx_sorted = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    short_idx, long_idx = idx_sorted[0], idx_sorted[-1]
    target_ids = enc[short_idx]
    longer_ids = enc[long_idx]
    same_ids = list(target_ids)

    print(f"Target (short) len={len(target_ids)}")
    print(f"Longer companion len={len(longer_ids)}  "
          f"(pad amount = {len(longer_ids)-len(target_ids)})")
    force_math = not args.no_math
    print(f"force_math = {force_math}")

    print(f"\n=== (a) single-example baseline ===")
    a = _prefill_then_one_decode(
        model, tokenizer, target_ids, pattern,
        doc_start_id, doc_end_id, tokenizer.pad_token_id,
        companion_ids=None, force_math=force_math,
    )
    print(f"  prefill argmax: {a['prefill_logit'].argmax().item()}  "
          f"first_decode_token: {a['first_token']}  "
          f"decode_argmax: {a['decode_logit'].argmax().item()}")

    print(f"\n=== (b) bs=2 same-length ===")
    b = _prefill_then_one_decode(
        model, tokenizer, target_ids, pattern,
        doc_start_id, doc_end_id, tokenizer.pad_token_id,
        companion_ids=same_ids, force_math=force_math,
    )
    print(f"  prefill argmax match: {a['prefill_logit'].argmax().item() == b['prefill_logit'].argmax().item()}  "
          f"decode_argmax match: {a['decode_logit'].argmax().item() == b['decode_logit'].argmax().item()}")
    print(f"  --- PREFILL hidden-state diffs (last real token) ---")
    _compare_layers(a["prefill_layers"], b["prefill_layers"], "(b) vs (a) prefill")
    print(f"  --- DECODE-STEP-1 hidden-state diffs (new token) ---")
    _compare_layers(a["decode_layers"], b["decode_layers"], "(b) vs (a) decode")

    print(f"\n=== (c) bs=2 different-length (right-padded) ===")
    c = _prefill_then_one_decode(
        model, tokenizer, target_ids, pattern,
        doc_start_id, doc_end_id, tokenizer.pad_token_id,
        companion_ids=longer_ids, force_math=force_math,
    )
    print(f"  prefill argmax match: {a['prefill_logit'].argmax().item() == c['prefill_logit'].argmax().item()}  "
          f"decode_argmax match: {a['decode_logit'].argmax().item() == c['decode_logit'].argmax().item()}")
    print(f"  --- PREFILL hidden-state diffs (last real token) ---")
    _compare_layers(a["prefill_layers"], c["prefill_layers"], "(c) vs (a) prefill")
    print(f"  --- DECODE-STEP-1 hidden-state diffs (new token) ---")
    _compare_layers(a["decode_layers"], c["decode_layers"], "(c) vs (a) decode")

    # Pinpoint the FIRST diverging layer for (c) decode.
    print(f"\n=== first-diverging-layer summary ===")
    for label, cand in [("(b) prefill", b["prefill_layers"]),
                          ("(b) decode", b["decode_layers"]),
                          ("(c) prefill", c["prefill_layers"]),
                          ("(c) decode", c["decode_layers"])]:
        ref = a["prefill_layers"] if "prefill" in label else a["decode_layers"]
        first_div = None
        for i, (r, ca) in enumerate(zip(ref, cand)):
            rel = (r.float() - ca.float()).norm().item() / (r.float().norm().item() + 1e-9)
            if rel > 1e-3:
                first_div = (i, rel)
                break
        if first_div is None:
            print(f"  {label}: no divergence (all rel <= 1e-3)")
        else:
            print(f"  {label}: first divergence at layer {first_div[0]} "
                  f"(rel={first_div[1]:.2e})")


if __name__ == "__main__":
    main()
