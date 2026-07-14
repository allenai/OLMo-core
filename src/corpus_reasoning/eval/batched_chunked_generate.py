"""Batched HF greedy generation for chunked-family attention models.

Baseline batched variant. Prefill uses per-example 4D chunked masks;
decode uses the HF forward pass with past_key_values (DynamicCache).

Shared setup (padding, chunk-ids, prefill mask, position_ids) lives in
`chunked_batch_helpers.py` so this module only owns the decode loop.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from corpus_reasoning.eval.chunked_batch_helpers import (
    build_chunk_ids_padded,
    build_decode_mask_full,
    build_prefill_mask,
    build_prefill_position_ids,
    encode_and_sort,
    make_stop_tester,
    pad_batch_left,
    resolve_stop_set,
    truncate_and_decode_batch,
)
from corpus_reasoning.lib.chunked_attention import AttentionPattern


@torch.no_grad()
def generate_hf_batched(
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
    """Greedy batched generation. Returns decoded text per prompt, in order."""
    pattern = attention_pattern or AttentionPattern(name="chunked")
    stop_set = resolve_stop_set(stop_token_ids, tokenizer)

    enc, order = encode_and_sort(tokenizer, prompts)
    outputs: List[Optional[str]] = [None] * len(enc)

    for start in range(0, len(order), batch_size):
        idx_batch = order[start : start + batch_size]
        texts = _generate_single_batch(
            model, tokenizer,
            [enc[i] for i in idx_batch],
            doc_start_id, doc_end_id, pad_token_id,
            max_new_tokens, stop_set, pattern,
        )
        for i, t in zip(idx_batch, texts):
            outputs[i] = t

    return outputs  # type: ignore[return-value]


def _generate_single_batch(
    model, tokenizer, batch_ids, doc_start_id, doc_end_id, pad_token_id,
    max_new_tokens, stop_set, pattern,
):
    device = next(model.parameters()).device
    input_ids, lens = pad_batch_left(batch_ids, pad_token_id, device)
    B, S = input_ids.shape

    chunk_ids, n_docs_list = build_chunk_ids_padded(
        input_ids, lens, doc_start_id, doc_end_id,
    )
    prefill_mask = build_prefill_mask(
        chunk_ids, input_ids, doc_end_id, n_docs_list, pattern,
    )
    position_ids = build_prefill_position_ids(lens, S, device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=prefill_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    past_kv = outputs.past_key_values
    next_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)

    decode_mask_full = build_decode_mask_full(lens, S, max_new_tokens, device)
    is_stop = make_stop_tester(stop_set, B, device, next_tokens.dtype)

    gen_ids = torch.empty((B, max_new_tokens), dtype=torch.long, device=device)
    gen_ids[:, 0] = next_tokens[:, 0]
    finished = is_stop(next_tokens[:, 0])

    # Per-example next-position counter (RoPE). Prefill consumed positions
    # [0, L_b) so the first generated token is at position L_b.
    next_pos = torch.tensor(lens, dtype=torch.long, device=device).unsqueeze(-1)

    produced = 1
    for step in range(1, max_new_tokens):
        if finished.all():
            break
        mask_len = S + step
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
