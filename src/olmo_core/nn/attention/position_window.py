"""
Sliding-window attention by ORIGINAL token position, for compacted sequences.

Input-level token dropping physically removes tokens from a row and keeps the survivors' original
RoPE positions in an explicit ``position_ids`` tensor
(:meth:`Transformer._compact_pooled_soft_tokens`, :func:`olmo_core.nn.block_skip._compact_kwargs`).
Every attention backend applies a sliding window in *index* space (flash:
``window_size=(W - 1, 0)``), so after dropping a layer with window ``W`` reaches
back ``W`` *kept* tokens -- about ``W / keep`` original tokens (4,096 kept = ~16k original at 25%
keep on OLMo 3's 24 SWA layers), which pretraining never did and full-context eval never does.

This module masks by position instead: key ``k`` is visible to query ``q`` iff
``k_idx <= q_idx`` (causal), ``0 <= pos_q - pos_k < W`` (window), and both sit in the same
document when ``cu_doc_lens`` packs several into a row. With contiguous positions that is exactly
the index window, so :func:`needs_position_window` sends those rows (dense training, keep-all
compaction, decode) to the unchanged backend path; only genuinely compacted rows pay for the
FlexAttention kernel here.

The :class:`~torch.nn.attention.flex_attention.BlockMask` is built at block resolution from each
tile's min/max position and document (no token-level ``T x T`` mask is materialised, so 32k rows
are cheap), and both it and the contiguity verdict are cached on the ``position_ids`` tensor: the
model threads the same tensor to every block, so one forward builds one mask for all its SWA
layers, and the activation-checkpoint recompute reuses it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention

    _flex_attention = torch.compile(flex_attention, dynamic=True)
    _HAS_FLEX = True
except Exception:  # pragma: no cover - flex unavailable on old torch
    _HAS_FLEX = False

__all__ = [
    "needs_position_window",
    "position_window_attention",
    "position_window_allowed",
    "apply_position_window_to_bias",
]

#: FlexAttention tile size (matches the flash tile on H100/H200).
FLEX_BLOCK_SIZE = 128

#: Rows at most this long fall back to a dense boolean mask when flex is unavailable (CPU, old
#: torch). Beyond it the ``B x T x T`` mask is too big to be a sensible fallback.
DENSE_FALLBACK_MAX_LEN = 8192

_CONTIG_ATTR = "_olmo_pos_window_contiguous"
_MASK_ATTR = "_olmo_pos_window_block_masks"


def _doc_ids(cu_doc_lens: Optional[torch.Tensor], B: int, T: int, device) -> Optional[torch.Tensor]:
    """``(B, T)`` document index per token from cumulative lengths over the flattened batch."""
    if cu_doc_lens is None:
        return None
    cu = cu_doc_lens.to(device=device)
    flat = torch.arange(B * T, device=device, dtype=cu.dtype)
    return torch.searchsorted(cu[1:].contiguous(), flat, right=True).view(B, T)


@torch._dynamo.disable
def needs_position_window(
    position_ids: Optional[torch.Tensor], cu_doc_lens: Optional[torch.Tensor] = None
) -> bool:
    """
    ``True`` iff ``position_ids`` is non-contiguous somewhere a sliding window could see it, i.e.
    some token's position is not its predecessor's ``+ 1`` within the same document. Contiguous
    positions (any per-row or per-document offset) make the index window and the position window
    identical, so the caller keeps the normal backend path.

    One host sync per ``(position_ids, cu_doc_lens)`` pair; the verdict is cached on the tensor.
    """
    if position_ids is None or position_ids.shape[-1] < 2:
        return False
    cached = getattr(position_ids, _CONTIG_ATTR, None)
    if cached is not None and cached[0] is cu_doc_lens:
        return cached[1]
    pos = position_ids if position_ids.dim() == 2 else position_ids.view(1, -1)
    B, T = pos.shape
    step_ok = (pos[:, 1:] - pos[:, :-1]) == 1
    doc = _doc_ids(cu_doc_lens, B, T, pos.device)
    if doc is not None:
        step_ok = step_ok | (doc[:, 1:] != doc[:, :-1])
    verdict = not bool(step_ok.all())
    try:
        setattr(position_ids, _CONTIG_ATTR, (cu_doc_lens, verdict))
    except Exception:  # pragma: no cover - e.g. a tensor subclass refusing attributes
        pass
    return verdict


def position_window_allowed(
    position_ids: torch.Tensor, window: int, cu_doc_lens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    The reference ``(B, T, T)`` boolean visibility mask (query, key): causal by index, window by
    position, same document. Dense -- for tests, short rows and the additive-bias path.
    """
    pos = position_ids if position_ids.dim() == 2 else position_ids.view(1, -1)
    B, T = pos.shape
    idx = torch.arange(T, device=pos.device)
    d = pos[:, :, None].long() - pos[:, None, :].long()  # (B, q, k)
    allowed = (idx[None, None, :] <= idx[None, :, None]) & (d >= 0) & (d < window)
    doc = _doc_ids(cu_doc_lens, B, T, pos.device)
    if doc is not None:
        allowed = allowed & (doc[:, :, None] == doc[:, None, :])
    return allowed


def apply_position_window_to_bias(
    attn_bias: torch.Tensor, position_ids: Optional[torch.Tensor], window: int
) -> torch.Tensor:
    """
    Fold the position window into a ``(B, 1, T, T)`` additive bias (the masked-SDPA path used by
    soft-token aux matching / slot biases / ``soft_keep``), which otherwise ignores the layer's
    window entirely. ``position_ids=None`` means contiguous positions. The diagonal is always
    visible (distance 0), so no row is fully masked by this.
    """
    B, _, T, _ = attn_bias.shape
    if position_ids is None:
        pos = torch.arange(T, device=attn_bias.device)[None].expand(B, T)
    else:
        pos = position_ids if position_ids.dim() == 2 else position_ids.view(1, -1)
        pos = pos.to(attn_bias.device)
    d = pos[:, :, None].long() - pos[:, None, :].long()
    in_window = ((d >= 0) & (d < window))[:, None]
    floor = torch.finfo(attn_bias.dtype).min if attn_bias.is_floating_point() else -1e9
    return torch.where(in_window, attn_bias, torch.full_like(attn_bias, floor))


def _tile_minmax(x: torch.Tensor, n_blocks: int, bs: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-tile min/max of a ``(B, T)`` tensor, padding the last tile with its edge values."""
    B, T = x.shape
    pad = n_blocks * bs - T
    if pad:
        x_lo = torch.cat([x, x[:, -1:].expand(B, pad)], dim=1)
    else:
        x_lo = x
    x_t = x_lo.view(B, n_blocks, bs)
    return x_t.amin(-1), x_t.amax(-1)


@torch._dynamo.disable
def _build_block_mask(
    position_ids: torch.Tensor, window: int, cu_doc_lens: Optional[torch.Tensor]
) -> "BlockMask":
    """
    The FlexAttention mask for :func:`position_window_attention`, at BLOCK resolution.

    A (query tile ``i``, key tile ``j``) pair is listed when it may hold a visible entry: ``j <= i``
    (index causality), the tile's position-distance interval
    ``[min pos_q - max pos_k, max pos_q - min pos_k]`` meets ``[0, W - 1]``, and the two tiles'
    document ranges intersect. It is FULL (run without ``mask_mod``) only when every entry is
    visible: ``j < i``, the distance interval lies inside ``[0, W - 1]``, and one document spans
    both tiles. Min/max bounds make the listed set an over-approximation (costs compute, never
    correctness) and the full set an under-approximation (the only direction that is safe), with
    no assumption that positions are monotone.
    """
    cache: Dict[Any, Any] = getattr(position_ids, _MASK_ATTR, None) or {}
    key = (int(window), id(cu_doc_lens))
    hit = cache.get(key)
    if hit is not None and hit[0] is cu_doc_lens:
        return hit[1]

    bs = FLEX_BLOCK_SIZE
    pos = position_ids if position_ids.dim() == 2 else position_ids.view(1, -1)
    pos = pos.to(torch.int64)
    B, T = pos.shape
    device = pos.device
    nb = -(-T // bs)
    p_min, p_max = _tile_minmax(pos, nb, bs)  # (B, nb)
    blk = torch.arange(nb, device=device)
    causal_any = blk[None, :] <= blk[:, None]  # (q, k)
    causal_all = blk[None, :] < blk[:, None]
    d_lo = p_min[:, :, None] - p_max[:, None, :]  # (B, q, k) smallest distance in the tile
    d_hi = p_max[:, :, None] - p_min[:, None, :]  # largest distance in the tile
    any_ok = causal_any[None] & (d_hi >= 0) & (d_lo <= window - 1)
    all_ok = causal_all[None] & (d_lo >= 0) & (d_hi <= window - 1)

    doc = _doc_ids(cu_doc_lens, B, T, device)
    if doc is not None:
        d_min, d_max = _tile_minmax(doc, nb, bs)
        any_ok = (
            any_ok
            & (d_min[:, None, :] <= d_max[:, :, None])
            & (d_max[:, None, :] >= d_min[:, :, None])
        )
        one_doc = d_min == d_max
        all_ok = (
            all_ok
            & one_doc[:, :, None]
            & one_doc[:, None, :]
            & (d_min[:, :, None] == d_min[:, None, :])
        )
    all_ok = all_ok & any_ok
    partial = any_ok & ~all_ok

    # A private copy: ``pos`` may BE ``position_ids`` (already int64), and the cached mask's
    # closure must not hold the tensor it is cached on (a reference cycle that defers freeing).
    pos_c = pos.contiguous().clone()
    doc_c = doc.contiguous() if doc is not None else None
    W = int(window)

    def mask_mod(b, h, q_idx, kv_idx):
        dist = pos_c[b, q_idx] - pos_c[b, kv_idx]
        ok = (kv_idx <= q_idx) & (dist >= 0) & (dist < W)
        if doc_c is not None:
            ok = ok & (doc_c[b, q_idx] == doc_c[b, kv_idx])
        return ok

    kv_num = partial.sum(-1, dtype=torch.int32)[:, None]  # (B, 1, nb)
    kv_idx = torch.argsort((~partial).to(torch.int8), dim=-1, stable=True).to(torch.int32)[:, None]
    full_num = all_ok.sum(-1, dtype=torch.int32)[:, None]
    full_idx = torch.argsort((~all_ok).to(torch.int8), dim=-1, stable=True).to(torch.int32)[:, None]
    block_mask = BlockMask.from_kv_blocks(
        kv_num,
        kv_idx,
        full_num,
        full_idx,
        BLOCK_SIZE=bs,
        mask_mod=mask_mod,
        seq_lengths=(T, T),
    )
    cache[key] = (cu_doc_lens, block_mask)
    try:
        setattr(position_ids, _MASK_ATTR, cache)
    except Exception:  # pragma: no cover
        pass
    return block_mask


def position_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position_ids: torch.Tensor,
    window: int,
    *,
    cu_doc_lens: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Causal sliding-window attention with the window measured in ORIGINAL positions.

    :param q: ``(B, T, H, D)`` queries (already rotated).
    :param k: ``(B, T, H_kv, D)`` keys (already rotated).
    :param v: ``(B, T, H_kv, D)`` values.
    :param position_ids: ``(B, T)`` original token positions.
    :param window: Window size ``W``: key visible iff ``0 <= pos_q - pos_k < W``.
    :param cu_doc_lens: Optional cumulative document lengths over the flattened ``B * T`` tokens
        (intra-document masking, as in the backends).
    :param scale: Softmax scale; ``None`` means ``D ** -0.5``.
    :returns: ``(B, T, H, D)``.
    """
    B, T, Hq, D = q.shape
    Hk = k.shape[2]
    if scale is None:
        scale = D**-0.5
    if _HAS_FLEX and q.is_cuda:
        block_mask = _build_block_mask(position_ids, window, cu_doc_lens)
        out = _flex_attention(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            block_mask=block_mask,
            scale=scale,
            enable_gqa=(Hq != Hk),
        )
        return out.transpose(1, 2).contiguous()

    if T > DENSE_FALLBACK_MAX_LEN:
        raise RuntimeError(
            f"position-window attention needs FlexAttention on CUDA for rows longer than "
            f"{DENSE_FALLBACK_MAX_LEN} tokens (got T={T}, device={q.device})"
        )
    from .backend import _repeat_kv

    allowed = position_window_allowed(position_ids.to(q.device), window, cu_doc_lens)
    n_rep = Hq // Hk
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        _repeat_kv(k, n_rep).transpose(1, 2),
        _repeat_kv(v, n_rep).transpose(1, 2),
        attn_mask=allowed[:, None],
        scale=scale,
    )
    return out.transpose(1, 2).contiguous()
