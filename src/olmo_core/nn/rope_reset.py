"""
Document-level RoPE position resets ("rope-reset").

Within one example, positions count up from 0 through the preamble (chat header, instruction, the
leading copy of the query). Every document marker ``<|box_start|>`` then restarts the count at
``P``, the position of the example's first ``<|box_start|>``, so every document occupies the
same position range ``P, P+1, ...``. The trailing segment (the text after the example's last
``<|box_end|>``: the second copy of the query, the assistant header and the answer) restarts at
``P`` too, as if it were one more document. The largest position a model ever sees is therefore
``P + max(longest document, trailing segment)``, independent of how many documents the context
holds.

An example ends at EOS (packed SFT windows) or at a non-valid (left-padding) token. An example
without any ``<|box_start|>`` keeps plain ``0, 1, 2, ...`` positions.

Positions depend only on the prefix up to each token -- the anchors are the first
``<|box_start|>`` and the last ``<|box_end|>``, and an answer never contains a marker -- so an
eval prompt gets exactly the positions its training sequence gave the same tokens. Decoding
continues from the last prompt position (``+1`` per generated token).
"""

from typing import Optional

import torch

__all__ = ["build_rope_reset_position_ids"]


def build_rope_reset_position_ids(
    input_ids: torch.Tensor,
    *,
    doc_start_id: int,
    doc_end_id: int,
    eos_id: int,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build per-token RoPE positions that reset at every document and at the trailing query.

    :param input_ids: Token ids of shape ``(batch_size, seq_len)``.
    :param doc_start_id: The document-start marker id (``<|box_start|>``).
    :param doc_end_id: The document-end marker id (``<|box_end|>``).
    :param eos_id: The EOS id. The token after an EOS starts a new example.
    :param valid_mask: Optional boolean ``(batch_size, seq_len)`` mask, ``False`` on padding. Each
        padding token is its own one-token example (position 0), so a left-padded row's prompt
        starts at position 0.

    :returns: A ``torch.long`` tensor of shape ``(batch_size, seq_len)``.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be (batch_size, seq_len), got {tuple(input_ids.shape)}")
    B, T = input_ids.shape
    device = input_ids.device
    if T == 0:
        return torch.zeros_like(input_ids, dtype=torch.long)

    t = torch.arange(T, device=device).expand(B, T)
    valid = (
        torch.ones_like(input_ids, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.to(device=device, dtype=torch.bool)
    )
    is_start = input_ids.eq(doc_start_id) & valid
    is_end = input_ids.eq(doc_end_id) & valid

    # Example boundaries: row start, after an EOS, at and after every padding token.
    seg_start = torch.zeros_like(valid)
    seg_start[:, 0] = True
    seg_start[:, 1:] |= input_ids[:, :-1].eq(eos_id) | ~valid[:, :-1]
    seg_start |= ~valid
    # Global segment ids (row starts are always segment starts, so segments never span rows).
    seg = torch.cumsum(seg_start.reshape(-1).long(), dim=0) - 1
    n_seg = B * T  # an upper bound; avoids a host sync on the real count
    flat_t = t.reshape(-1)

    big = torch.full((n_seg,), T, device=device, dtype=torch.long)
    seg_begin = big.scatter_reduce(0, seg, flat_t, reduce="amin")
    first_start = big.scatter_reduce(
        0, seg, torch.where(is_start.reshape(-1), flat_t, T), reduce="amin"
    )
    last_end = torch.full((n_seg,), -1, device=device, dtype=torch.long).scatter_reduce(
        0, seg, torch.where(is_end.reshape(-1), flat_t, -1), reduce="amax"
    )
    has_docs = first_start < T
    reset_pos = (first_start - seg_begin).clamp(min=0)  # P

    # Anchors: segment start (value 0), each doc start (value P), and the first token after the
    # last doc end, if it lies inside the same segment (value P).
    tok_seg = seg.view(B, T)
    tok_has_docs = has_docs[tok_seg]
    trailing = tok_has_docs & (t == (last_end[tok_seg] + 1)) & (last_end[tok_seg] >= 0)
    trailing &= ~seg_start  # the trailing anchor must not begin a new example
    reset_anchor = (is_start | trailing) & tok_has_docs
    is_anchor = seg_start | reset_anchor
    anchor_val = torch.where(reset_anchor & ~seg_start, reset_pos[tok_seg], 0)

    anchor_idx = torch.where(is_anchor, t, -1).cummax(dim=1).values  # t=0 is always an anchor
    base = anchor_val.gather(1, anchor_idx)
    return (base + (t - anchor_idx)).long()
