"""Batched chunked-family generator using RIGHT-padding.

The left-padded variant (batched_chunked_generate.py) correctly handles
RoPE and the chunked mask, but its first absolute position is always a
pad token. Modern LLMs learn attention-sink heads that funnel stray
attention to the first absolute position. With left-padding those sinks
collapse onto masked-out pads, forcing attention to redistribute across
real tokens — which perturbs outputs enough to drop F1 by ~0.15 on this
task.

Right-padding keeps the first absolute position as the first real token
for every example, matching single-example behaviour exactly. The
tradeoff is decode bookkeeping: after prefill we pick the last-real-token
logits per example (index L_b - 1, not S - 1), and decode uses an
explicit RoPE position of L_b for each example's first generated token
so the model sees the same relative positions it was trained with.

Numerical-correctness note: PyTorch SDPA's default backend with a 4D
float mask dispatches to mem_efficient attention, whose block-tiled
online softmax produces ~rel 2e-3 drift per layer relative to the
single-example path when K has extra pad columns. This compounds over
24 layers × 400 decode steps into ~6 F1 points. The MATH backend runs
full fp32 softmax without tiling, and sum + 0 = sum exactly in fp32 —
so pad-column contributions vanish cleanly and outputs match single
bit-for-bit. We wrap prefill + decode in `sdpa_kernel([MATH])` to get
that guarantee.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from corpus_reasoning.eval.chunked_batch_helpers import (
    build_decode_mask_full,
    build_prefill_mask,
    encode_and_sort,
    make_stop_tester,
    resolve_stop_set,
    truncate_and_decode_batch,
)
from corpus_reasoning.lib.chunked_attention import (
    AttentionPattern,
    FREE_CHUNK_ID,
    PAD_CHUNK_ID,
    find_chunk_spans,
)


def _pad_batch_right(batch_ids: list[list[int]], pad_token_id: int, device):
    """Right-pad token-id lists to the max length. Returns (input_ids, lens)."""
    B = len(batch_ids)
    lens = [len(ids) for ids in batch_ids]
    S = max(lens)
    input_ids = torch.full((B, S), pad_token_id, dtype=torch.long, device=device)
    for b, ids in enumerate(batch_ids):
        input_ids[b, : lens[b]] = torch.tensor(ids, dtype=torch.long, device=device)
    return input_ids, lens


def _build_chunk_ids_right_padded(
    input_ids_padded: torch.Tensor,  # (B, S)
    real_lens: Sequence[int],
    doc_start_id: int,
    doc_end_id: int,
):
    """(B, S) chunk_ids with real tokens at [0, L_b) and PAD_CHUNK_ID beyond."""
    B, S = input_ids_padded.shape
    device = input_ids_padded.device
    chunk_ids = torch.full((B, S), PAD_CHUNK_ID, dtype=torch.int32, device=device)
    n_docs_list: list[int] = []
    for b in range(B):
        L = real_lens[b]
        chunk_ids[b, :L] = FREE_CHUNK_ID
        spans = find_chunk_spans(input_ids_padded[b, :L], doc_start_id, doc_end_id)
        for idx, (s, e) in enumerate(spans):
            chunk_ids[b, s:e] = idx
        n_docs_list.append(len(spans))
    return chunk_ids, n_docs_list


@torch.no_grad()
def generate_hf_batched_rightpad(
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
    input_ids, lens = _pad_batch_right(batch_ids, pad_token_id, device)
    B, S = input_ids.shape

    chunk_ids, n_docs_list = _build_chunk_ids_right_padded(
        input_ids, lens, doc_start_id, doc_end_id,
    )
    prefill_mask = build_prefill_mask(
        chunk_ids, input_ids, doc_end_id, n_docs_list, pattern,
    )

    # Force the math SDPA backend so pad-column softmax contributions vanish
    # exactly (sum + 0 = sum in fp32). See module docstring.
    with sdpa_kernel([SDPBackend.MATH]):
        # With right-padding, standard [0, 1, ..., S-1] positions are exactly
        # what HF would have picked by default — and what single-example used.
        # No override needed; real tokens get RoPE 0..L_b-1, pads get RoPE
        # L_b..S-1 (never attended to, so never materialize as actual values).
        outputs = model(input_ids=input_ids, attention_mask=prefill_mask, use_cache=True)
        past_kv = outputs.past_key_values

        # Logits at the last REAL token (L_b - 1) per example.
        lens_t = torch.tensor(lens, dtype=torch.long, device=device)
        last_logits = outputs.logits[torch.arange(B, device=device), lens_t - 1, :]  # (B, V)
        next_tokens = last_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

        # Decode-time padding-aware mask. Real positions [0, L_b) are keepable;
        # pad positions [L_b, S) are masked throughout decode. Newly generated
        # tokens are never padding.
        decode_mask_full = torch.zeros((B, S + max_new_tokens), dtype=torch.long, device=device)
        for b, L in enumerate(lens):
            decode_mask_full[b, :L] = 1
        decode_mask_full[:, S:] = 1

        is_stop = make_stop_tester(stop_set, B, device, next_tokens.dtype)
        gen_ids = torch.empty((B, max_new_tokens), dtype=torch.long, device=device)
        gen_ids[:, 0] = next_tokens[:, 0]
        finished = is_stop(next_tokens[:, 0])

        # Each example's "next RoPE position" is L_b (continuing from last
        # real token at L_b - 1). Increment by 1 each decode step. Pads
        # between L_b and S are never attended to, so they're outside the
        # relative-position view the new token sees.
        next_pos = lens_t.clone().unsqueeze(-1)  # (B, 1)

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
