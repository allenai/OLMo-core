"""Batched chunked-family generator that drops finished examples mid-decode.

Over a long-CoT workload, output lengths vary. In the baseline batched
generator every slot keeps getting fed until the slowest finishes, so
early finishers burn GPU cycles producing garbage. This variant calls
`past_kv.batch_select_indices(...)` between decode steps to shrink the
batch as examples hit EOS, leaving only live examples in the forward
pass.

Correctness must match the baseline — validated by the test harness.
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
)
from corpus_reasoning.lib.chunked_attention import AttentionPattern


@torch.no_grad()
def generate_hf_batched_prune(
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
        texts = _generate_single_batch_pruned(
            model, tokenizer,
            [enc[i] for i in idx_batch],
            doc_start_id, doc_end_id, pad_token_id,
            max_new_tokens, stop_set, pattern,
        )
        for i, t in zip(idx_batch, texts):
            outputs[i] = t
    return outputs  # type: ignore[return-value]


def _generate_single_batch_pruned(
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
    next_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    decode_mask_full = build_decode_mask_full(lens, S, max_new_tokens, device)

    # Per-batch-slot -> original index into batch_ids.
    slot_to_orig = list(range(B))
    per_example_tokens: list[list[int]] = [[] for _ in range(B)]
    # Seed per_example_tokens with the prefill-output token (one sync total).
    first_tokens = next_tokens[:, 0].tolist()
    for b, tok in enumerate(first_tokens):
        per_example_tokens[b].append(tok)

    stop_tensor = (torch.tensor(list(stop_set), device=device, dtype=next_tokens.dtype)
                   if stop_set else None)

    def _is_stop_col(tok_col: torch.Tensor) -> torch.Tensor:
        if stop_tensor is None:
            return torch.zeros(tok_col.size(0), dtype=torch.bool, device=device)
        return (tok_col.unsqueeze(-1) == stop_tensor).any(dim=-1)

    # Prune any example that finished after the prefill token itself.
    just_finished = _is_stop_col(next_tokens[:, 0])
    keep_mask = (~just_finished).tolist()
    if not all(keep_mask):
        keep_idx = torch.tensor(
            [i for i, k in enumerate(keep_mask) if k],
            device=device, dtype=torch.long,
        )
        past_kv.batch_select_indices(keep_idx)
        decode_mask_full = decode_mask_full[keep_idx]
        next_tokens = next_tokens[keep_idx]
        slot_to_orig = [slot_to_orig[i] for i, k in enumerate(keep_mask) if k]
        lens = [lens[i] for i, k in enumerate(keep_mask) if k]

    next_pos = torch.tensor(lens, dtype=torch.long, device=device).unsqueeze(-1)

    for step in range(1, max_new_tokens):
        if next_tokens.size(0) == 0:
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
        new_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # One sync per step: pull the whole column to CPU as a list.
        new_tok_list = new_tokens[:, 0].tolist()
        for slot, orig in enumerate(slot_to_orig):
            per_example_tokens[orig].append(new_tok_list[slot])

        # Decide which slots survive to the next iteration.
        just_finished = [t in stop_set for t in new_tok_list]
        next_pos = next_pos + 1
        if any(just_finished):
            keep = [i for i, done in enumerate(just_finished) if not done]
            if len(keep) < len(just_finished):
                if not keep:
                    break
                keep_idx = torch.tensor(keep, device=device, dtype=torch.long)
                past_kv.batch_select_indices(keep_idx)
                decode_mask_full = decode_mask_full[keep_idx]
                new_tokens = new_tokens[keep_idx]
                next_pos = next_pos[keep_idx]
                slot_to_orig = [slot_to_orig[i] for i in keep]
        next_tokens = new_tokens

    # Decode, trimming at first stop token (baseline-compatible).
    texts = []
    for toks in per_example_tokens:
        cut = len(toks)
        for i, t in enumerate(toks):
            if t in stop_set:
                cut = i
                break
        texts.append(tokenizer.decode(toks[:cut], skip_special_tokens=True))
    return texts
