"""Shared helpers for batched chunked-family HF eval.

Each variant (baseline batched, compile+StaticCache, batch-prune, flex
prefill) reuses the same left-pad / chunk-id / 4D prefill mask logic;
they differ only in how the decode loop is driven. Put those shared
steps here so each variant is a thin wrapper on top.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from ctc_eval.lib.chunked_attention import (
    AttentionPattern,
    FREE_CHUNK_ID,
    PAD_CHUNK_ID,
    build_dense_bool_mask,
    build_random_doc_edges,
    find_chunk_spans,
)


# ---------------------------------------------------------------------------
# Tokenization + padding
# ---------------------------------------------------------------------------

def encode_and_sort(tokenizer, prompts: List[str]) -> tuple[list[list[int]], list[int]]:
    """Tokenize prompts; return (encoded_ids, sorted_order) where `sorted_order`
    lists original indices in ascending prompt-length order.
    """
    enc = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    return enc, order


def pad_batch_left(batch_ids: list[list[int]], pad_token_id: int, device) -> tuple[torch.Tensor, list[int]]:
    """Left-pad a batch of token-id lists to the max length. Returns (B, S)."""
    B = len(batch_ids)
    lens = [len(ids) for ids in batch_ids]
    S = max(lens)
    input_ids = torch.full((B, S), pad_token_id, dtype=torch.long, device=device)
    for b, ids in enumerate(batch_ids):
        input_ids[b, S - lens[b] :] = torch.tensor(ids, dtype=torch.long, device=device)
    return input_ids, lens


# ---------------------------------------------------------------------------
# Per-example chunk ids + prefill mask
# ---------------------------------------------------------------------------

def build_chunk_ids_padded(
    input_ids_padded: torch.Tensor,  # (B, S)
    real_lens: Sequence[int],
    doc_start_id: int,
    doc_end_id: int,
) -> tuple[torch.Tensor, list[int]]:
    """Return (B, S) chunk_ids plus per-example n_docs for a left-padded batch."""
    B, S = input_ids_padded.shape
    device = input_ids_padded.device
    chunk_ids = torch.full((B, S), PAD_CHUNK_ID, dtype=torch.int32, device=device)
    n_docs_list: list[int] = []
    for b in range(B):
        L = real_lens[b]
        real = input_ids_padded[b, S - L:]
        chunk_ids[b, S - L:] = FREE_CHUNK_ID
        spans = find_chunk_spans(real, doc_start_id, doc_end_id)
        for idx, (s, e) in enumerate(spans):
            chunk_ids[b, S - L + s : S - L + e] = idx
        n_docs_list.append(len(spans))
    return chunk_ids, n_docs_list


def build_prefill_mask(
    chunk_ids: torch.Tensor,  # (B, S)
    input_ids_padded: torch.Tensor,  # (B, S)
    doc_end_id: Optional[int],
    n_docs_list: Sequence[int],
    pattern: AttentionPattern,
) -> torch.Tensor:
    """Build (B, 1, S, S) bf16 additive mask from the pattern spec."""
    B, S = chunk_ids.shape
    device = chunk_ids.device
    kwargs = {}
    if pattern.needs_anchor_tensor():
        if doc_end_id is None:
            raise ValueError("Pattern needs anchors but doc_end_id is None")
        kwargs["is_anchor"] = (input_ids_padded == doc_end_id)
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

    bool_mask = build_dense_bool_mask(pattern, chunk_ids, **kwargs)  # (B, S, S)

    # Training paths only ever right-padded, so the chunked mask_mod never
    # saw an all-masked row. With left-padding for batched eval, a pad
    # token's query row is fully masked (can't attend to anything), which
    # makes softmax(-inf, ..., -inf) = NaN. Those NaNs stay at pad
    # positions but can still perturb the model in subtle ways (LayerNorm,
    # rotary, etc.). Fix: let each pad token attend to itself on the
    # diagonal. This keeps pad rows valid without letting pads exchange
    # anything with real tokens (real rows still mask out pad columns).
    S = bool_mask.size(-1)
    diag = torch.eye(S, dtype=torch.bool, device=device).unsqueeze(0)
    bool_mask = bool_mask | diag

    dtype = torch.bfloat16
    min_val = torch.finfo(dtype).min
    mask = torch.where(
        bool_mask,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), min_val, dtype=dtype, device=device),
    )
    return mask.unsqueeze(1)  # (B, 1, S, S)


def build_prefill_position_ids(lens: Sequence[int], S: int, device) -> torch.Tensor:
    """(B, S) position_ids: real tokens count 0..L_b-1; padded dummy positions = 0."""
    B = len(lens)
    position_ids = torch.zeros((B, S), dtype=torch.long, device=device)
    for b, L in enumerate(lens):
        position_ids[b, S - L :] = torch.arange(L, device=device)
    return position_ids


def build_decode_mask_full(lens: Sequence[int], S: int, max_new_tokens: int, device) -> torch.Tensor:
    """(B, S + max_new_tokens) 2D attention mask that keeps padded KV positions
    masked throughout the whole decode loop. Generated positions are always 1.
    """
    B = len(lens)
    total = S + max_new_tokens
    mask = torch.zeros((B, total), dtype=torch.long, device=device)
    for b, L in enumerate(lens):
        mask[b, S - L :] = 1
    mask[:, S:] = 1
    return mask


# ---------------------------------------------------------------------------
# Stop / decoding
# ---------------------------------------------------------------------------

def resolve_stop_set(stop_token_ids: Optional[Sequence[int]], tokenizer) -> set[int]:
    stop_set = set(stop_token_ids or [])
    if tokenizer.eos_token_id is not None:
        stop_set.add(tokenizer.eos_token_id)
    return stop_set


def make_stop_tester(stop_set: set[int], B: int, device, dtype):
    """Return a fast closure that tests (B,) token column against stop_set."""
    if not stop_set:
        empty = torch.zeros(B, dtype=torch.bool, device=device)
        return lambda tok_col: empty
    stop_tensor = torch.tensor(list(stop_set), device=device, dtype=dtype)

    def _is_stop(tok_col):
        return (tok_col.unsqueeze(-1) == stop_tensor).any(dim=-1)
    return _is_stop


def truncate_and_decode_batch(
    gen_ids: torch.Tensor, stop_set: set[int], tokenizer,
) -> list[str]:
    """Per-example: truncate at first stop-token (exclusive), decode to text."""
    B = gen_ids.size(0)
    texts = []
    for b in range(B):
        ids = gen_ids[b].tolist()
        cut = len(ids)
        for i, t in enumerate(ids):
            if t in stop_set:
                cut = i
                break
        texts.append(tokenizer.decode(ids[:cut], skip_special_tokens=True))
    return texts
