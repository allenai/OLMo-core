"""
Token exit ("layer skipping for dropped tokens") on top of the positional-block input drop.

The input-drop arms of ``olmo3_cptdrop`` (``--variant softtoken --st-block-size 512
--st-gold-blind --st-drop-slots``; I-rand / I-fl) remove the tokens their keep rule drops BEFORE
block 0. Token exit runs the SAME keep rule but lets the dropped tokens through the first
``exit_layer`` blocks on the full row, and only then compacts them out:

* blocks ``[0, L)`` run on the full row (plain causal attention, contiguous positions);
* at the input of block ``L`` the hidden states are gathered to the kept tokens, which keep their
  ORIGINAL positions (``position_ids``), so RoPE matches and the sliding-window layers mask by
  original position (:mod:`olmo_core.nn.attention.position_window` triggers on the gapped
  ``position_ids``);
* blocks ``[L, n)`` and the LM head run on the kept tokens only; the labels are the compacted
  labels of the input-drop path, so a dropped token's own next-token label gets no loss.

``L = 0`` is exactly the input-drop path (the gather picks the kept tokens' embeddings). Kept
tokens can absorb what the dropped ones said in the first ``L`` blocks, which a CPT objective can
teach. Ported from the dropstruct worktree (``prasann/dropstruct``) without its pseudo-document
markers, learned router and straight-through coefficient, and without the GDN pieces: the keep
selection comes from :meth:`Transformer._compact_pooled_soft_tokens` (positional blocks).

FLOPs: blocks ``[0, L)`` are charged at the full length, the rest and the LM head at the
compacted length (:func:`token_exit_example_flops`, read by
:class:`~olmo_core.train.callbacks.flop_meter.FlopMeterCallback`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

__all__ = [
    "TokenExitHolder",
    "gather_kept",
    "prefix_flops_per_token",
    "token_exit_example_flops",
]


class TokenExitHolder:
    """Configuration and running statistics of token exit (one per model)."""

    def __init__(self, *, exit_layer: int, log_every: int = 50):
        self.exit_layer = int(exit_layer)
        self.log_every = int(log_every)
        self.calls = 0
        self.cum_in = 0
        self.cum_kept = 0
        self.last: Dict[str, float] = {}

    def record(self, n_in: int, n_kept: int) -> None:
        self.calls += 1
        self.cum_in += int(n_in)
        self.cum_kept += int(n_kept)
        self.last = {"keep_frac": n_kept / max(1, n_in)}
        c = self.calls
        if c <= 5 or c == 10 or (self.log_every and c % self.log_every == 0):
            print(
                f"[token-exit] call#{c} L={self.exit_layer} tokens_in={n_in} kept={n_kept} "
                f"({n_kept / max(1, n_in):.4f}) cum_keep={self.cum_kept / max(1, self.cum_in):.4f}",
                flush=True,
            )


def gather_kept(h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``h[b, idx[b, j]]`` for a ``(B, T, D)`` hidden state and a ``(B, T')`` column index."""
    return torch.gather(h, 1, idx.to(h.device)[..., None].expand(-1, -1, h.shape[-1]))


def prefix_flops_per_token(model: Any, seq_len: int, exit_layer: int) -> int:
    """FLOPs per token of blocks ``[0, exit_layer)`` at ``seq_len`` (the model's own formula)."""
    blocks = list(model.blocks.values())
    return int(sum(b.num_flops_per_token(int(seq_len)) for b in blocks[: int(exit_layer)]))


def token_exit_example_flops(
    model: Any,
    full_len: int,
    kept_len: int,
    exit_layer: int,
    cache: Optional[Dict[Any, int]] = None,
) -> float:
    """
    Training FLOPs of one example under token exit: ``full_len`` tokens through blocks
    ``[0, L)`` and ``kept_len`` tokens through blocks ``[L, n)`` plus the LM head, each block at
    its own row length (attention term included), i.e.

    ``full_len * prefix(full_len) + kept_len * (fpt(kept_len) - prefix(kept_len))``

    with ``fpt`` = :meth:`Transformer.num_flops_per_token` and ``prefix`` =
    :func:`prefix_flops_per_token`. ``exit_layer = 0`` reduces to the input-drop charge
    ``kept_len * fpt(kept_len)``.
    """
    c = cache if cache is not None else {}

    def fpt(n: int) -> int:
        key = ("fpt", n)
        if key not in c:
            c[key] = int(model.num_flops_per_token(int(n)))
        return c[key]

    def pre(n: int) -> int:
        key = ("pre", n, int(exit_layer))
        if key not in c:
            c[key] = prefix_flops_per_token(model, n, exit_layer)
        return c[key]

    full_len, kept_len = int(full_len), int(kept_len)
    return float(full_len) * pre(full_len) + float(kept_len) * (fpt(kept_len) - pre(kept_len))
