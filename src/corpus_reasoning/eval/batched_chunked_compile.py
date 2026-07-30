"""Batched chunked-family generator with torch.compile on the decode step.

Each model call in the decode loop has fixed input shape (B, 1) for the
new query and grows KV cache length by 1 per step. Inductor can compile
this to fused kernels if we pin the batch size and tell it the cache
length is dynamic.

Prefill is left uncompiled — its shape varies per batch and the 4D
chunked mask makes compilation brittle. The decode loop dominates
wall-time anyway.

Validated against the non-compiled baseline by the test harness.
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


# Cache of compiled forwards keyed by (model id, batch_size). We compile
# once per batch size; subsequent calls reuse the compiled artifact.
_COMPILED_CACHE: dict[tuple[int, int], "object"] = {}


def _get_compiled_forward(model, batch_size: int):
    key = (id(model), batch_size)
    if key not in _COMPILED_CACHE:
        # `dynamic=True` lets inductor keep the cache length as a symbolic
        # dim so we don't recompile every decode step.
        _COMPILED_CACHE[key] = torch.compile(model, mode="default", dynamic=True)
    return _COMPILED_CACHE[key]


@torch.no_grad()
def generate_hf_batched_compiled(
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

    for start in range(0, len(order), batch_size):
        idx_batch = order[start : start + batch_size]
        texts = _generate_single_batch_compiled(
            model, tokenizer,
            [enc[i] for i in idx_batch],
            doc_start_id, doc_end_id, pad_token_id,
            max_new_tokens, stop_set, pattern,
            actual_bs=len(idx_batch),
        )
        for i, t in zip(idx_batch, texts):
            outputs[i] = t
    return outputs  # type: ignore[return-value]


def _generate_single_batch_compiled(
    model, tokenizer, batch_ids, doc_start_id, doc_end_id, pad_token_id,
    max_new_tokens, stop_set, pattern, actual_bs,
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

    # Prefill uses the eager model (4D mask is awkward to compile).
    outputs = model(
        input_ids=input_ids,
        attention_mask=prefill_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    past_kv = outputs.past_key_values
    next_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    decode_mask_full = build_decode_mask_full(lens, S, max_new_tokens, device)
    is_stop = make_stop_tester(stop_set, B, device, next_tokens.dtype)

    gen_ids = torch.empty((B, max_new_tokens), dtype=torch.long, device=device)
    gen_ids[:, 0] = next_tokens[:, 0]
    finished = is_stop(next_tokens[:, 0])

    next_pos = torch.tensor(lens, dtype=torch.long, device=device).unsqueeze(-1)
    compiled_forward = _get_compiled_forward(model, actual_bs)

    produced = 1
    for step in range(1, max_new_tokens):
        if finished.all():
            break
        mask_len = S + step
        outputs = compiled_forward(
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
