"""Batched chunked-family generator using SERIAL prefill + BATCHED decode.

Designed for Qwen3.5's hybrid architecture (6 softmax + 18 GDN linear-
attention layers in the 0.8B). Right-padded prefill pollutes the GDN
recurrent state with pad tokens — the recurrence has no mask, so every
input token steps the state. After right-padded prefill, the cached GDN
state for an example of real length L_b reflects L_b real + (S - L_b)
pad steps, not L_b real steps. That polluted state then feeds into
decode and breaks outputs (rel error ~0.5 starting at the first GDN
decode layer; see scripts/eval/_diag_rightpad_decode.py).

Strategy:
  1. Run prefill ONCE PER EXAMPLE on a single-example forward — no
     padding, so each example's softmax KV cache and GDN state are
     produced cleanly.
  2. Stack the per-example caches into a B-batched cache:
       - softmax (DynamicLayer): pad each example's K/V to max_L, stack.
       - GDN (LinearAttentionLayer): concat fixed-size states along bs dim.
  3. Run the decode loop batched. Per-example position_ids advance from
     L_b (so the first generated token has RoPE L_b for example b, even
     though its cache slot is max_L). 2D attention_mask zeroes slots
     [L_b, max_L) so the new Q never attends through pad-padded K columns
     of the stacked KV cache for shorter examples.

Trade-offs vs the right-padded variant:
  - Prefill is serial, losing batched-prefill GPU utilization. Chunked
    attention's prefill is O(N²/k) so each prompt is fast, but for
    short-context tasks this approach gives less of a win. For long-
    context eval (decode-dominated), it's strictly better.
  - Decode is correctly batched and reaches up to ~B× decode-step speed.
"""
from __future__ import annotations

import copy
from typing import List, Optional, Sequence

import torch
from tqdm.auto import tqdm

from ctc_eval.eval.chunked_batch_helpers import (
    encode_and_sort, resolve_stop_set, make_stop_tester,
    truncate_and_decode_batch,
)
from ctc_eval.lib.chunked_attention import (
    AttentionPattern, FREE_CHUNK_ID,
    build_dense_bool_mask, find_chunk_spans,
    build_random_doc_edges, build_random_token_keep,
)


def _build_chunk_ids_one(input_ids: torch.Tensor, doc_start_id: int,
                          doc_end_id: int) -> torch.Tensor:
    """Build chunk_ids for a single (1, L) prompt. Real tokens get
    FREE_CHUNK_ID by default; tokens inside a doc-span get the doc index."""
    L = input_ids.size(1)
    device = input_ids.device
    chunk_ids = torch.full((1, L), FREE_CHUNK_ID, dtype=torch.int32,
                            device=device)
    spans = find_chunk_spans(input_ids[0], doc_start_id, doc_end_id)
    for idx, (s, e) in enumerate(spans):
        chunk_ids[0, s:e] = idx
    return chunk_ids


@torch.no_grad()
def _prefill_one(model, prompt_ids: List[int], doc_start_id: int,
                  doc_end_id: int, pattern: AttentionPattern):
    """Single-example prefill. Returns (last_real_logit, past_kv, L)."""
    device = next(model.parameters()).device
    input_ids = torch.tensor(prompt_ids, dtype=torch.long,
                              device=device).unsqueeze(0)
    L = input_ids.size(1)
    chunk_ids = _build_chunk_ids_one(input_ids, doc_start_id, doc_end_id)
    kwargs = {}
    if pattern.needs_anchor_tensor():
        kwargs["is_anchor"] = (input_ids[0] == doc_end_id).unsqueeze(0)
    if pattern.needs_random_edges():
        n_docs = int((chunk_ids[0] >= 0).any() and chunk_ids[0].max().item() + 1 or 0)
        adj = build_random_doc_edges(
            num_docs=n_docs, num_edges=pattern.num_random_doc_edges,
            seed=pattern.random_seed, max_docs=max(n_docs, 1),
        ).to(device)
        kwargs["doc_random"] = adj.unsqueeze(0)
    if pattern.needs_random_token_mask():
        rk = build_random_token_keep(
            seq_len=L, keep_prob=pattern.keep_prob, seed=pattern.random_seed,
        ).to(device)
        kwargs["random_keep"] = rk.unsqueeze(0)
    bool_mask = build_dense_bool_mask(pattern, chunk_ids, **kwargs)  # (1, L, L)
    dtype = torch.bfloat16
    min_val = torch.finfo(dtype).min
    mask = torch.where(
        bool_mask, torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_val, dtype=dtype, device=device),
    ).unsqueeze(1)  # (1, 1, L, L)
    outputs = model(input_ids=input_ids, attention_mask=mask, use_cache=True)
    return outputs.logits[0, -1, :].clone(), outputs.past_key_values, L


def _stack_olmohybrid_caches(caches: list, lens: List[int], stacked, B: int, max_L: int):
    """OlmoHybridDynamicCache uses parallel lists indexed by global layer_idx
    (no per-layer wrapper objects). layer_types[i] dispatches softmax vs GDN."""
    for layer_idx, ltype in enumerate(stacked.layer_types):
        if ltype == "full_attention":
            k0 = caches[0].key_cache[layer_idx]
            if k0 is None or (hasattr(k0, "numel") and k0.numel() == 0):
                continue
            num_heads = k0.shape[1]
            head_dim = k0.shape[3]
            stacked_k = torch.zeros(
                (B, num_heads, max_L, head_dim), dtype=k0.dtype, device=k0.device,
            )
            stacked_v = torch.zeros_like(stacked_k)
            for b, (c, L) in enumerate(zip(caches, lens)):
                stacked_k[b, :, :L, :] = c.key_cache[layer_idx][0]
                stacked_v[b, :, :L, :] = c.value_cache[layer_idx][0]
            stacked.key_cache[layer_idx] = stacked_k
            stacked.value_cache[layer_idx] = stacked_v
        elif ltype == "linear_attention":
            stacked.conv_states_q[layer_idx] = torch.cat(
                [c.conv_states_q[layer_idx] for c in caches], dim=0)
            stacked.conv_states_k[layer_idx] = torch.cat(
                [c.conv_states_k[layer_idx] for c in caches], dim=0)
            stacked.conv_states_v[layer_idx] = torch.cat(
                [c.conv_states_v[layer_idx] for c in caches], dim=0)
            stacked.recurrent_states[layer_idx] = torch.cat(
                [c.recurrent_states[layer_idx] for c in caches], dim=0)
        else:
            raise RuntimeError(f"OlmoHybrid: unknown layer_type {ltype!r} "
                               f"at layer {layer_idx}")
    return stacked


def _stack_caches(caches: list, lens: List[int]):
    """Stack B per-example DynamicCache instances (each B=1) into a single
    B-batch DynamicCache. Pads softmax K/V to max_L; concats fixed-size GDN
    states. Mutates a deepcopy of caches[0] in place and returns it."""
    B = len(caches)
    max_L = max(lens)

    stacked = copy.deepcopy(caches[0])
    if type(stacked).__name__ == "OlmoHybridDynamicCache":
        return _stack_olmohybrid_caches(caches, lens, stacked, B, max_L)

    for layer_idx in range(len(stacked.layers)):
        layer_caches = [c.layers[layer_idx] for c in caches]
        cls_name = type(layer_caches[0]).__name__

        if cls_name in ("DynamicLayer", "DynamicSlidingWindowLayer"):
            # DynamicSlidingWindowLayer (Olmo-3 sliding layers, etc.) is a
            # subclass of DynamicLayer that truncates self.keys/values to
            # the last sliding_window-1 tokens on each update. When the
            # whole prefill fits in the window (L_b < sliding_window),
            # the layer still holds the full prefill and we can stack
            # identically to DynamicLayer; we just need to also align
            # cumulative_length with the stacked layout (= max_L) so the
            # layer's get_mask_sizes returns kv_length consistent with
            # the decode-time attention mask the caller builds.
            is_sliding = cls_name == "DynamicSlidingWindowLayer"
            if is_sliding:
                sw = layer_caches[0].sliding_window
                for lc, L in zip(layer_caches, lens):
                    if lc.keys.shape[-2] != L:
                        raise NotImplementedError(
                            f"Sliding-window cache truncation in "
                            f"_stack_caches not yet supported "
                            f"(layer {layer_idx}: prefill_len={L}, "
                            f"sliding_window={sw}, cache_len="
                            f"{lc.keys.shape[-2]}). Extend this branch "
                            f"to right-align truncated caches before "
                            f"running eval with prefill > sliding_window."
                        )
            num_heads = layer_caches[0].keys.shape[1]
            head_dim = layer_caches[0].keys.shape[3]
            dtype = layer_caches[0].keys.dtype
            device = layer_caches[0].keys.device
            stacked_k = torch.zeros(
                (B, num_heads, max_L, head_dim), dtype=dtype, device=device,
            )
            stacked_v = torch.zeros_like(stacked_k)
            for b, (lc, L) in enumerate(zip(layer_caches, lens)):
                stacked_k[b, :, :L, :] = lc.keys[0]
                stacked_v[b, :, :L, :] = lc.values[0]
            stacked.layers[layer_idx].keys = stacked_k
            stacked.layers[layer_idx].values = stacked_v
            if is_sliding:
                stacked.layers[layer_idx].cumulative_length = max_L

        elif cls_name == "LinearAttentionLayer":
            # GDN states are fixed-size — just concat along the batch dim.
            conv = torch.cat([lc.conv_states for lc in layer_caches], dim=0)
            rec = torch.cat([lc.recurrent_states for lc in layer_caches], dim=0)
            stacked.layers[layer_idx].conv_states = conv
            stacked.layers[layer_idx].recurrent_states = rec
            stacked.layers[layer_idx].max_batch_size = B

        else:
            raise RuntimeError(f"Unknown per-layer cache class: {cls_name}")

    return stacked


@torch.no_grad()
def generate_hf_batched_serial_prefill(
    model,
    tokenizer,
    prompts: List[str],
    doc_start_id: int,
    doc_end_id: int,
    pad_token_id: int,
    max_new_tokens: int = 200,
    stop_token_ids: Optional[Sequence[int]] = None,
    attention_pattern: Optional[AttentionPattern] = None,
    batch_size: int = 4,
) -> List[str]:
    pattern = attention_pattern or AttentionPattern(name="chunked")
    stop_set = resolve_stop_set(stop_token_ids, tokenizer)

    enc, order = encode_and_sort(tokenizer, prompts)
    outputs: List[Optional[str]] = [None] * len(enc)

    pbar = tqdm(range(0, len(order), batch_size), desc="  chunked-eval",
                unit="batch", total=(len(order) + batch_size - 1) // batch_size)
    for start in pbar:
        idx_batch = order[start:start + batch_size]
        texts = _generate_one_batch(
            model, tokenizer,
            [enc[i] for i in idx_batch],
            doc_start_id, doc_end_id, pad_token_id,
            max_new_tokens, stop_set, pattern,
        )
        for i, t in zip(idx_batch, texts):
            outputs[i] = t

    return outputs  # type: ignore[return-value]


def _generate_one_batch(model, tokenizer, batch_ids, doc_start_id, doc_end_id,
                         pad_token_id, max_new_tokens, stop_set, pattern):
    device = next(model.parameters()).device
    B = len(batch_ids)

    # Serial prefill — clean per-example KV / GDN cache, no padding.
    per_example_caches = []
    first_tokens = []
    lens: List[int] = []
    for prompt_ids in batch_ids:
        last_logit, pkv, L = _prefill_one(
            model, prompt_ids, doc_start_id, doc_end_id, pattern,
        )
        first_tokens.append(last_logit.argmax())  # scalar
        per_example_caches.append(pkv)
        lens.append(L)

    # Stack per-example caches into a B-batch cache.
    past_kv = _stack_caches(per_example_caches, lens)
    max_L = max(lens)

    next_tokens = torch.stack(first_tokens, dim=0).unsqueeze(-1)  # (B, 1)
    lens_t = torch.tensor(lens, dtype=torch.long, device=device)
    next_pos = lens_t.clone().unsqueeze(-1)  # (B, 1)

    # Decode-time 2D attention mask. Per example: real KV slots [0, L_b) = 1;
    # pad slots [L_b, max_L) = 0 (zero KV from stacking, masked out);
    # generated slots [max_L, max_L+step) = 1 (always attended).
    decode_mask_full = torch.zeros(
        (B, max_L + max_new_tokens), dtype=torch.long, device=device,
    )
    for b, L in enumerate(lens):
        decode_mask_full[b, :L] = 1
    decode_mask_full[:, max_L:] = 1

    is_stop = make_stop_tester(stop_set, B, device, next_tokens.dtype)
    gen_ids = torch.empty((B, max_new_tokens), dtype=torch.long, device=device)
    gen_ids[:, 0] = next_tokens[:, 0]
    finished = is_stop(next_tokens[:, 0])

    produced = 1
    for step in range(1, max_new_tokens):
        if finished.all():
            break
        mask_len = max_L + step
        outputs = model(
            input_ids=next_tokens,
            attention_mask=decode_mask_full[:, :mask_len],
            position_ids=next_pos,
            past_key_values=past_kv,
            use_cache=True,
        )
        past_kv = outputs.past_key_values
        next_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen_ids[:, step] = next_tokens[:, 0]
        produced = step + 1
        finished |= is_stop(next_tokens[:, 0])
        next_pos = next_pos + 1

    gen_ids = gen_ids[:, :produced]
    return truncate_and_decode_batch(gen_ids, stop_set, tokenizer)
