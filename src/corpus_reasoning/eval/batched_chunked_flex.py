"""Batched chunked-family generator using FlexAttention for prefill.

`torch.nn.attention.flex_attention.create_block_mask` builds a block-sparse
BlockMask from a mask-mod closure. For a chunked pattern, most of the
dense mask is zero — FlexAttention skips those blocks entirely, which is
much faster than SDPA on a dense bf16 mask at long sequence lengths.

Decode still uses eager model.forward with a 2D attention_mask; each
decode step has only one query token so the block-sparse win doesn't
apply there.

Requires: model loaded with `attn_implementation="flex_attention"`.
Otherwise the BlockMask is silently ignored and results diverge.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from corpus_reasoning.eval.chunked_batch_helpers import (
    build_chunk_ids_padded,
    build_decode_mask_full,
    build_prefill_position_ids,
    encode_and_sort,
    make_stop_tester,
    pad_batch_left,
    resolve_stop_set,
    truncate_and_decode_batch,
)
from corpus_reasoning.lib.chunked_attention import (
    AttentionPattern,
    build_flex_mask_mod,
    build_random_doc_edges,
)


def _build_batched_block_mask(
    chunk_ids: torch.Tensor,        # (B, S)
    input_ids: torch.Tensor,        # (B, S)
    doc_end_id: Optional[int],
    n_docs_list: Sequence[int],
    pattern: AttentionPattern,
):
    from torch.nn.attention.flex_attention import create_block_mask

    B, S = chunk_ids.shape
    device = chunk_ids.device
    kwargs = {}
    if pattern.needs_anchor_tensor():
        if doc_end_id is None:
            raise ValueError("Pattern needs anchors but doc_end_id is None")
        kwargs["is_anchor"] = (input_ids == doc_end_id)
    if pattern.needs_random_edges():
        max_docs = max(max(n_docs_list), 1)
        doc_random = torch.zeros((B, max_docs, max_docs), dtype=torch.bool, device=device)
        for b, nd in enumerate(n_docs_list):
            doc_random[b] = build_random_doc_edges(
                num_docs=nd,
                num_edges=pattern.num_random_doc_edges,
                seed=pattern.random_seed + b,
                max_docs=max_docs,
            ).to(device)
        kwargs["doc_random"] = doc_random

    mask_mod = build_flex_mask_mod(pattern, chunk_ids, **kwargs)
    return create_block_mask(mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S, device=device)


@torch.no_grad()
def generate_hf_batched_flex(
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
        texts = _generate_single_batch_flex(
            model, tokenizer,
            [enc[i] for i in idx_batch],
            doc_start_id, doc_end_id, pad_token_id,
            max_new_tokens, stop_set, pattern,
        )
        for i, t in zip(idx_batch, texts):
            outputs[i] = t
    return outputs  # type: ignore[return-value]


def _generate_single_batch_flex(
    model, tokenizer, batch_ids, doc_start_id, doc_end_id, pad_token_id,
    max_new_tokens, stop_set, pattern,
):
    device = next(model.parameters()).device
    input_ids, lens = pad_batch_left(batch_ids, pad_token_id, device)
    B, S = input_ids.shape

    chunk_ids, n_docs_list = build_chunk_ids_padded(
        input_ids, lens, doc_start_id, doc_end_id,
    )
    block_mask = _build_batched_block_mask(
        chunk_ids, input_ids, doc_end_id, n_docs_list, pattern,
    )
    position_ids = build_prefill_position_ids(lens, S, device)

    # Prefill via FlexAttention. The model's internal attention impl must
    # be "flex_attention" for the BlockMask to be honored.
    outputs = model(
        input_ids=input_ids,
        attention_mask=block_mask,
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
