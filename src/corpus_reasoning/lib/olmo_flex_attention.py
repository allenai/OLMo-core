"""Runtime FlexAttention masking for olmo-core models (chunked / modified SWA).

This is layered *on top of* an already-built olmo-core ``Transformer`` with **zero
edits to the olmo-core library**. At install time we:

  1. register a ``forward_pre_hook`` on the ``Transformer`` that, each step, reads the
     batch's ``input_ids``, derives a per-token *role* tensor ``(B, S)`` (which token is
     a document / the instruction-prefix sink / a free query-answer token / padding),
     builds a FlexAttention :class:`~torch.nn.attention.flex_attention.BlockMask` for the
     chosen pattern, and stashes it on a shared holder; and
  2. monkeypatch each :class:`olmo_core.nn.attention.Attention` *instance*'s bound
     ``sdpa`` method to run :func:`torch.nn.attention.flex_attention.flex_attention`
     with that BlockMask instead of the default backend.

Everything else in the olmo-core attention forward (QKV projection, QK-norm, RoPE,
gating, output projection) is untouched. We never set ``cu_doc_lens``, so RoPE stays
*absolute* — token-window offsets and causal positions remain meaningful.

Patterns
--------
``chunked``       Port of the repo's block-diagonal rule (scripts/lib/chunked_attention):
                  ``causal & (same_doc | q_free | kv_free)``. Documents are isolated; the
                  instruction prefix and the query/answer are "free" (globally visible).

``modified_swa``  Token-level sliding window of width ``W`` with two global exceptions:
                  every position may always attend to the **instruction/prompt prefix**
                  (the "sink", everything before the first document) and the
                  **query/answer** tokens attend to everything prior. Concretely::

                      causal & not_pad & ( kv_is_sink
                                           | (0 <= q-kv <= W)     # token window
                                           | q_is_free            # query/answer see all
                                           | kv_is_free )

Scope
-----
Single-rank, full-sequence (CP=1). Ring context parallelism is **not** supported:
FlexAttention has no ring kernel. Ulysses CP (head-sharding) is reachable later (each
rank holds the full sequence for its heads), but is not wired here. See
``results/olmo_flex_attention.md``.
"""

from __future__ import annotations

import random
import types
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# NOTE: olmo_core is imported lazily inside install_flex_attention so that the pure-torch
# mask helpers (build_roles / mask_mods / dense refs) can be imported and unit-tested
# without the full olmo-core environment.

from corpus_reasoning.lib.chunked_attention import (
    FREE_CHUNK_ID,
    PAD_CHUNK_ID,
    AttentionPattern,
    build_flex_mask_mod,
)

# Roles reuse chunked_attention's conventions (doc idx >= 0, FREE = -1, PAD = -2) and
# add a SINK role for the instruction/prompt prefix used by ``modified_swa``.
SINK_CHUNK_ID = -3

# Patterns this installer can build. "chunked" (and the other context-context variants
# in chunked_attention) are delegated to build_flex_mask_mod; "modified_swa" is local.
OLMO_FLEX_PATTERNS = ("chunked", "modified_swa")


# ---------------------------------------------------------------------------
# Role construction
# ---------------------------------------------------------------------------

def build_roles(
    input_ids: torch.Tensor,
    doc_start_id: int,
    doc_end_id: int,
    eos_id: int,
    mode: str = "chunked",
) -> torch.Tensor:
    """Per-token role tensor ``(B, S)`` int32 derived from ``input_ids``.

    Values: ``>= 0`` document index; ``-1`` FREE (query / answer / instruction);
    ``-2`` PAD; ``-3`` SINK (instruction prefix before the first document — only
    emitted when ``mode == "modified_swa"``).

    Padding is identified as everything strictly **after the first** ``eos_id`` in
    each row. olmo-core's ``NumpyPaddedFSLDataset`` writes one EOS-terminated example
    per instance and right-pads to ``sequence_length``; the real example always ends
    in exactly one in-stream EOS, so "after first EOS" is the pad region regardless of
    whether ``pad_token_id == eos_token_id`` (it does for Qwen3-Base).
    """
    ids2d = input_ids.tolist()
    out: List[List[int]] = []
    for ids in ids2d:
        S = len(ids)
        row = [FREE_CHUNK_ID] * S

        # Pad = positions after the first eos (the real document terminator).
        try:
            pad_from = ids.index(eos_id) + 1
        except ValueError:
            pad_from = S

        cur: Optional[int] = None  # open document index, if inside a <doc_start>..<doc_end>
        idx = -1                   # last assigned document index
        first_doc_start: Optional[int] = None
        for i, t in enumerate(ids):
            if i >= pad_from:
                row[i] = PAD_CHUNK_ID
                continue
            if t == doc_start_id:
                idx += 1
                cur = idx
                if first_doc_start is None:
                    first_doc_start = i
                row[i] = idx
            elif t == doc_end_id and cur is not None:
                row[i] = cur  # the end marker belongs to its document
                cur = None
            elif cur is not None:
                row[i] = cur
            # else: leave as FREE

        if mode == "modified_swa" and first_doc_start is not None:
            # Everything free *before* the first document is the prefix sink.
            for i in range(min(first_doc_start, pad_from)):
                if row[i] == FREE_CHUNK_ID:
                    row[i] = SINK_CHUNK_ID

        out.append(row)
    return torch.tensor(out, dtype=torch.int32)


# ---------------------------------------------------------------------------
# mask_mod factories
# ---------------------------------------------------------------------------

def modified_swa_mask_mod(roles: torch.Tensor, window: int):
    """FlexAttention ``mask_mod`` for the ``modified_swa`` pattern.

    ``roles`` is the ``(B, S)`` tensor from :func:`build_roles` (with SINK marked).
    A diagonal term is always allowed so padding rows are never fully masked (which
    would NaN the softmax); those outputs are dropped by the loss mask anyway.
    """
    w = int(window)

    def mask_mod(b, h, q_idx, kv_idx):
        qc = roles[b, q_idx]
        kc = roles[b, kv_idx]
        causal = q_idx >= kv_idx
        q_not_pad = qc != PAD_CHUNK_ID
        kv_not_pad = kc != PAD_CHUNK_ID
        kv_sink = kc == SINK_CHUNK_ID
        q_free = qc == FREE_CHUNK_ID
        kv_free = kc == FREE_CHUNK_ID
        diff = q_idx - kv_idx
        within = (diff >= 0) & (diff <= w)
        diag = q_idx == kv_idx  # NaN guard for fully-masked (pad) rows
        return diag | (
            causal & q_not_pad & kv_not_pad & (kv_sink | within | q_free | kv_free)
        )

    return mask_mod


def make_mask_mod(pattern: str, roles: torch.Tensor, window: int):
    """Dispatch to the right ``mask_mod`` for ``pattern``.

    ``modified_swa`` is handled locally; every other name is delegated to
    chunked_attention.build_flex_mask_mod (roles double as its ``chunk_ids``; SINK is
    only present for modified_swa so other patterns see standard FREE/PAD/doc ids).
    """
    if pattern == "modified_swa":
        return modified_swa_mask_mod(roles, window)
    ap = AttentionPattern(name=pattern, token_window_w=window)
    return build_flex_mask_mod(ap, roles)


def modified_swa_dense_bool(roles: torch.Tensor, window: int) -> torch.Tensor:
    """Dense ``(B, S, S)`` boolean reference for ``modified_swa`` (True = attend).

    Mirrors :func:`modified_swa_mask_mod` exactly; used by the unit tests and as a
    readable reference. Not used on the training hot path.
    """
    if roles.dim() == 1:
        roles = roles.unsqueeze(0)
    B, S = roles.shape
    dev = roles.device
    w = int(window)

    q = torch.arange(S, device=dev)
    kv = torch.arange(S, device=dev)
    causal = q.unsqueeze(1) >= kv.unsqueeze(0)  # (S, S)
    diff = q.unsqueeze(1) - kv.unsqueeze(0)
    within = (diff >= 0) & (diff <= w)
    diag = torch.eye(S, dtype=torch.bool, device=dev)

    qc = roles.unsqueeze(2)  # (B, S, 1)
    kc = roles.unsqueeze(1)  # (B, 1, S)
    q_not_pad = qc != PAD_CHUNK_ID
    kv_not_pad = kc != PAD_CHUNK_ID
    kv_sink = kc == SINK_CHUNK_ID
    q_free = qc == FREE_CHUNK_ID
    kv_free = kc == FREE_CHUNK_ID

    allowed = causal.unsqueeze(0) & q_not_pad & kv_not_pad & (
        kv_sink | within.unsqueeze(0) | q_free | kv_free
    )
    return allowed | diag.unsqueeze(0)


# ---------------------------------------------------------------------------
# Runtime installer
# ---------------------------------------------------------------------------

@dataclass
class FlexMaskHolder:
    """Shared, per-step holder for the current BlockMask (set by the pre-hook,
    read by every patched ``sdpa``)."""
    block_mask: object = None
    n_patched: int = 0
    forward_idx: int = 0  # microbatch-forward counter (drives the mix curriculum)


def install_flex_attention(
    model: torch.nn.Module,
    *,
    pattern: str,
    doc_start_id: int,
    doc_end_id: int,
    eos_id: int,
    window: int = 512,
    compile_flex: bool = False,
    mix_start_p: float = 0.0,
    mix_end_p: float = 0.0,
    mix_total_forwards: int = 0,
    mix_seed: int = 0,
) -> FlexMaskHolder:
    """Install FlexAttention chunked / modified-SWA masking on ``model`` in place.

    :param model: a built olmo-core ``Transformer`` (call after ``train_module.build``).
    :param pattern: one of :data:`OLMO_FLEX_PATTERNS` (or any chunked_attention pattern).
    :param doc_start_id, doc_end_id: marker token ids that wrap documents in the stream.
    :param eos_id: real tokenizer eos id (used to locate the pad region).
    :param window: token-window width ``W`` for ``modified_swa`` (ignored by ``chunked``).
    :param compile_flex: torch.compile the BlockMask build + flex kernel. Leave ``False``
        for the first correctness pass; flip on once verified.
    :param mix_start_p, mix_end_p, mix_total_forwards, mix_seed: **mix curriculum**. When
        ``mix_total_forwards > 0``, each microbatch-forward independently collapses every
        example to plain causal (all roles -> FREE, padding kept) with probability ``p``
        that anneals **linearly** from ``mix_start_p`` to ``mix_end_p`` over
        ``mix_total_forwards`` forwards. This is the runtime counterpart of the static
        tokenize-time ``standard_mix_prob`` (a markerless example is all-FREE -> causal),
        but here the fraction changes over training: e.g. start 0.8 (mostly standard
        attention, easy) and end 0.0 (fully chunked). The data must be tokenized WITH
        markers and WITHOUT static mixing (``standard_mix_prob=0``). The per-(forward,
        example) coin is seeded by ``mix_seed`` so a resumed/rerun job is deterministic.
    :returns: the :class:`FlexMaskHolder` (also records how many modules were patched).
    """
    from olmo_core.nn.attention import Attention

    if pattern not in OLMO_FLEX_PATTERNS:
        # Allow the other chunked_attention variants too, but be explicit.
        from corpus_reasoning.lib.chunked_attention import ATTENTION_PATTERNS
        if pattern not in ATTENTION_PATTERNS:
            raise ValueError(f"unknown attn pattern {pattern!r}")

    holder = FlexMaskHolder()
    _create_block_mask = torch.compile(create_block_mask) if compile_flex else create_block_mask
    _flex = torch.compile(flex_attention) if compile_flex else flex_attention

    def pre_hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None:
            return
        roles_cpu = build_roles(input_ids, doc_start_id, doc_end_id, eos_id, mode=pattern)
        device = getattr(module, "device", None) or input_ids.device
        roles = roles_cpu.to(device)
        B, S = roles.shape
        # Mix curriculum: anneal the per-example "treat as plain causal" probability from
        # mix_start_p -> mix_end_p over mix_total_forwards forwards. Collapsing an example's
        # roles to FREE (keeping PAD) makes the chunked/SWA mask_mod degenerate to causal,
        # identical to a markerless (static-mixed) example -- just scheduled over training.
        if mix_total_forwards > 0:
            prog = min(1.0, holder.forward_idx / mix_total_forwards)
            p = mix_start_p + (mix_end_p - mix_start_p) * prog
            holder.forward_idx += 1
            if p > 0.0:
                for b in range(B):
                    # str seed (tuples are not a supported Random seed type); deterministic
                    # across runs/platforms so a resumed/rerun job makes the same choices.
                    if random.Random(f"{mix_seed}:{holder.forward_idx}:{b}").random() < p:
                        rb = roles[b]
                        roles[b] = torch.where(
                            rb == PAD_CHUNK_ID, rb, torch.full_like(rb, FREE_CHUNK_ID))
                if holder.forward_idx % 500 == 1:
                    print(f"[curriculum] forward={holder.forward_idx} "
                          f"prog={prog:.2f} p_standard={p:.3f}", flush=True)
        mask_mod = make_mask_mod(pattern, roles, window)
        holder.block_mask = _create_block_mask(
            mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S, device=device
        )

    model.register_forward_pre_hook(pre_hook, with_kwargs=True)

    def flex_sdpa(self, q, k, v, **_ignored):
        # q: (B, S, n_heads, head_dim); k, v: (B, S, n_kv_heads, head_dim).
        # FlexAttention wants (B, H, S, D).
        qb = q.transpose(1, 2)
        kb = k.transpose(1, 2)
        vb = v.transpose(1, 2)
        out = _flex(
            qb, kb, vb,
            block_mask=holder.block_mask,
            enable_gqa=(self.n_kv_heads != self.n_heads),
            scale=getattr(self.backend, "scale", None),
        )
        return out.transpose(1, 2)  # back to (B, S, n_heads, head_dim)

    n = 0
    for m in model.modules():
        if isinstance(m, Attention):
            m.sdpa = types.MethodType(flex_sdpa, m)
            n += 1
    holder.n_patched = n
    return holder
