"""
Landmark top-k prefill that actually **skips** work, instead of masking it.

:mod:`landmark_prefill_topk` answers the accuracy question but is the *masked* form: it computes the
scores for every past block and then floors the non-selected ones out of the gate softmax. Cost is
therefore independent of ``k`` and stays O(T^2) -- measured at 2.7 s/prompt at 32k against 1.5 s for
the ordinary landmark prefill kernel and 0.9 s for dense flash attention, i.e. top-k made prefill
*slower*.

This module iterates only over a per-query-block **candidate list** of key blocks, so the work really
does scale with the number of selected blocks. Two ways to build that list, sharing one kernel:

``mode="union"`` (default, **exact**)
    Take the per-token top-k exactly as :mod:`landmark_prefill_topk` defines it, then let each query
    block iterate the *union* of its 64 rows' selections, re-applying the per-row cutoff inside the
    kernel via the same midpoint-threshold trick. Output is bit-comparable to the masked path -- the
    accuracy numbers already measured for this checkpoint carry over unchanged -- and the saving is
    whatever the union leaves out. How well that pays depends entirely on how much 64 neighbouring
    queries agree, which is an empirical question: :func:`selection_stats` measures it.

``mode="qblock"`` (**approximate**)
    Pool the 64 queries' landmark scores (mean) and give the whole query block one top-k set, so the
    candidate list is exactly ``k`` long. Always fast, but it changes which blocks a token sees and so
    needs its own accuracy run.

Note the published alternatives resolve this differently and neither is a drop-in here: NSA shares
the selection across the query heads of a **GQA group** (its kernel puts the query loop in the grid,
which wants a bigger group than Qwen3's ``n_rep = 4``), and MoBA keeps per-token selection but
inverts the loop -- sort queries by assigned block, varlen-attend per block, recombine with online
softmax. ``union`` is the cheapest thing that preserves our exact semantics; ``qblock`` is the
cheapest thing overall. See ``debug/sparse_landmark_training/DESIGN.md``.
"""

import math
from typing import Optional, Tuple

import torch

from .landmark_prefill_topk import landmark_topk_thresholds

__all__ = [
    "landmark_topk_prefill_sparse",
    "selection_stats",
    "build_candidates",
    "enable_prefill_sparse",
    "disable_prefill_sparse",
]

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore


if triton is not None:

    @triton.jit
    def _fwd_kernel_prefill_sparse(
        Q,
        K,
        V,
        sm_scale,
        Out,
        sqz,
        sqh,
        sqm,
        sqd,
        skz,
        skh,
        skn,
        skd,
        svz,
        svh,
        svn,
        svd,
        soz,
        soh,
        som,
        sod,
        Thresh,  # fp32 (Z*H, N_CTX): per-query cutoff, -inf to keep every candidate
        Cand,  # int32 (Z*H, N_QBLOCKS, MAXSEL): candidate key-block ids, ascending
        NSel,  # int32 (Z*H, N_QBLOCKS): valid entries in Cand for this query block
        Z,
        H,
        N_CTX,
        N_QBLOCKS,
        MAXSEL,
        BLOCK: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        COMPRESSIVE: tl.constexpr,
    ):
        # One program per (query block, batch*head). The key-block loop walks the candidate list
        # instead of every past block, with a DATA-DEPENDENT bound -- that is where the saving is.
        # Candidates are strictly-past blocks, so no causal mask is needed inside the loop (only the
        # diagonal block below needs one).
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)

        BLOCK_M: tl.constexpr = BLOCK
        BLOCK_N: tl.constexpr = BLOCK

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m_real = offs_m + tl.where(tl.arange(0, BLOCK_M) == BLOCK_M - 1, -1, 0)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        offs_q = off_hz * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd
        q_vals = tl.load(Q + offs_q, mask=offs_m[:, None] < N_CTX, other=0)
        thresh = tl.load(Thresh + off_hz * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)

        m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        nsel = tl.load(NSel + off_hz * N_QBLOCKS + start_m)
        cand_base = Cand + off_hz * N_QBLOCKS * MAXSEL + start_m * MAXSEL

        for i in range(0, nsel):
            blk = tl.load(cand_base + i)
            n_off = blk * BLOCK_N + offs_n
            k_vals = tl.load(
                K + off_hz * skh + n_off[None, :] * skn + offs_d[:, None] * skd,
                mask=n_off[None, :] < N_CTX,
                other=0,
            )
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q_vals.dtype)
            qk += tl.dot(q_vals, k_vals, allow_tf32=False)
            qk *= sm_scale

            landmark_qk = tl.max(
                tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, qk, float("-inf")), 1
            )
            # Per-row cutoff: a candidate that this particular row did not select is floored out.
            # Finite floor (not -inf) for the same reason as the masked kernel -- see
            # landmark_prefill_topk._fwd_kernel_prefill_topk.
            landmark_qk = tl.where(landmark_qk >= thresh, landmark_qk, -1e30)

            if COMPRESSIVE:
                within_qk = qk
            else:
                within_qk = tl.where(
                    tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, float("-inf"), qk
                )
            within_m = tl.max(within_qk, 1)
            within_p = tl.exp(within_qk - within_m[:, None])
            within_denom = tl.sum(within_p, 1)

            m_curr = tl.maximum(landmark_qk, m_prev)
            l_prev *= tl.exp(m_prev - m_curr)
            landmark_p = tl.exp(landmark_qk - m_curr)
            l_curr = landmark_p + l_prev
            l_rcp = 1.0 / l_curr
            landmark_p *= l_rcp

            acc *= (l_prev * l_rcp)[:, None]
            v_vals = tl.load(
                V + off_hz * svh + n_off[:, None] * svn + offs_d[None, :] * svd,
                mask=n_off[:, None] < N_CTX,
                other=0,
            )
            acc += tl.dot(
                (landmark_p[:, None] * within_p / within_denom[:, None]).to(Q.dtype.element_ty),
                v_vals,
                allow_tf32=False,
            )
            l_prev = l_curr
            m_prev = m_curr

        # Diagonal (local) block: plain causal softmax, never gated.
        n_off = start_m * BLOCK_N + offs_n
        k_vals = tl.load(
            K + off_hz * skh + n_off[None, :] * skn + offs_d[:, None] * skd,
            mask=n_off[None, :] < N_CTX,
            other=0,
        )
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q_vals.dtype)
        qk += tl.dot(q_vals, k_vals, allow_tf32=False)
        qk *= sm_scale
        qk = tl.where(offs_m_real[:, None] >= n_off[None, :], qk, float("-inf"))

        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        l_prev *= tl.exp(m_prev - m_curr)
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        l_rcp = 1.0 / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        v_vals = tl.load(
            V + off_hz * svh + n_off[:, None] * svn + offs_d[None, :] * svd,
            mask=n_off[:, None] < N_CTX,
            other=0,
        )
        acc += tl.dot(p.to(Q.dtype.element_ty), v_vals, allow_tf32=False)

        offs_o = off_hz * soh + offs_m[:, None] * som + offs_d[None, :] * sod
        tl.store(Out + offs_o, acc, mask=offs_m[:, None] < N_CTX)


def _pack(sel_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Boolean ``(BH, n_qblocks, n_blocks)`` selection -> padded ascending id list + counts.

    A stable descending argsort of the mask brings the selected ids to the front in ascending order,
    which keeps the kernel's key reads roughly sequential.
    """
    counts = sel_mask.sum(-1).to(torch.int32)  # (BH, n_qblocks)
    maxsel = max(1, int(counts.max().item()))
    order = torch.argsort(sel_mask.to(torch.int8), dim=-1, descending=True, stable=True)
    return order[..., :maxsel].contiguous().to(torch.int32), counts, maxsel


@torch.no_grad()
def build_candidates(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    block_size: int,
    softmax_scale: float,
    top_k: Optional[int],
    mode: str = "union",
    query_tile: int = 2048,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """
    Per-query-block candidate key blocks, plus the per-row cutoff the kernel re-applies.

    :returns: ``(cand, counts, maxsel, thresh)`` -- ``cand`` int32 ``(BH, n_qblocks, maxsel)``,
        ``counts`` int32 ``(BH, n_qblocks)``, ``thresh`` fp32 ``(BH, T)``.
    """
    B, H, T, _ = q.shape
    Lb = block_size
    n_blocks = T // Lb
    device = q.device
    BH = B * H

    if top_k is None or top_k >= n_blocks:
        # Everything past is a candidate; the kernel still skips the strictly-future blocks.
        blk = torch.arange(n_blocks, device=device)
        sel = blk[None, :] < blk[:, None]  # (n_qblocks, n_blocks)
        sel = sel[None].expand(BH, n_blocks, n_blocks)
        cand, counts, maxsel = _pack(sel)
        thresh = torch.full((BH, T), -float("inf"), device=device, dtype=torch.float32)
        return cand, counts, maxsel, thresh

    lm_k = k[:, :, Lb - 1 :: Lb, :].float()
    blk_idx = torch.arange(n_blocks, device=device)
    sel_mask = torch.zeros(BH, n_blocks, n_blocks, dtype=torch.bool, device=device)
    thresh = torch.empty(B, H, T, device=device, dtype=torch.float32)

    for m0 in range(0, T, query_tile):
        m1 = min(m0 + query_tile, T)
        pos = torch.arange(m0, m1, device=device)
        q_block = pos // Lb
        s = torch.matmul(q[:, :, m0:m1].float(), lm_k.transpose(-1, -2)) * softmax_scale
        s = s.masked_fill(~(blk_idx[None, :] < q_block[:, None]), -float("inf"))

        if mode == "union":
            top = s.topk(top_k + 1, dim=-1)
            thresh[:, :, m0:m1] = 0.5 * (top.values[..., -2] + top.values[..., -1])
            keep = torch.zeros_like(s, dtype=torch.bool).scatter_(
                -1, top.indices[..., :top_k], True
            )
            keep &= blk_idx[None, :] < q_block[:, None]
            # union over the rows of each query block
            u = keep.view(B, H, (m1 - m0) // Lb, Lb, n_blocks).any(dim=3)
            sel_mask.view(B, H, n_blocks, n_blocks)[:, :, m0 // Lb : m1 // Lb] = u
        elif mode == "qblock":
            # one pooled selection for the whole query block
            pooled = s.view(B, H, (m1 - m0) // Lb, Lb, n_blocks).mean(dim=3)
            pooled = pooled.masked_fill(torch.isnan(pooled), -float("inf"))
            kk = min(top_k, n_blocks)
            idx = pooled.topk(kk, dim=-1).indices
            u = torch.zeros_like(pooled, dtype=torch.bool).scatter_(-1, idx, True)
            qb = torch.arange(m0 // Lb, m1 // Lb, device=device)
            u &= blk_idx[None, None, None, :] < qb[None, None, :, None]
            sel_mask.view(B, H, n_blocks, n_blocks)[:, :, m0 // Lb : m1 // Lb] = u
            thresh[:, :, m0:m1] = -float("inf")
        else:
            raise ValueError(f"mode must be 'union' or 'qblock' (got {mode!r})")

    cand, counts, maxsel = _pack(sel_mask)
    return cand, counts, maxsel, thresh.reshape(BH, T)


@torch.no_grad()
def selection_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    block_size: int,
    softmax_scale: float,
    top_k: int,
) -> dict:
    """
    How much do 64 neighbouring queries agree on their top-k blocks?

    This is the number that decides whether the exact ``union`` mode is fast: a union of size ``k``
    means perfect agreement and the full saving; a union of ``64 * k`` means none.

    :returns: dict with mean/median/max union size, the ratio to ``top_k``, and the fraction of past
        blocks the union covers (1.0 = no saving at all).
    """
    B, H, T, _ = q.shape
    Lb = block_size
    n_blocks = T // Lb
    _, counts, _, _ = build_candidates(
        q, k, block_size=Lb, softmax_scale=softmax_scale, top_k=top_k, mode="union"
    )
    qb = torch.arange(n_blocks, device=q.device).view(1, n_blocks).expand(B * H, n_blocks)
    past = qb.clamp(min=1).float()
    c = counts.float()
    live = qb > 0
    return {
        "top_k": top_k,
        "union_mean": c[live].mean().item(),
        "union_median": c[live].median().item(),
        "union_max": c[live].max().item(),
        "union_over_k": c[live].mean().item() / top_k,
        "frac_of_past_blocks": (c[live] / past[live]).mean().item(),
    }


@torch.no_grad()
def landmark_topk_prefill_sparse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int,
    softmax_scale: float,
    top_k: Optional[int],
    compressive: bool,
    mode: str = "union",
) -> torch.Tensor:
    """
    Top-k landmark prefill that iterates only the selected key blocks.

    :param mode: ``"union"`` keeps the exact per-token selection (each query block iterates the union
        of its rows' picks and the per-row cutoff is re-applied in-kernel); ``"qblock"`` gives the
        whole query block one pooled top-k, which is faster but changes the attention.

    :returns: ``(B, H, T, D)``.
    """
    if triton is None:
        raise RuntimeError("the sparse prefill path requires triton + CUDA")
    B, H, T, D = q.shape
    Lb = block_size
    if T % Lb != 0:
        raise ValueError(f"sequence length {T} is not a multiple of block_size {Lb}")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    cand, counts, maxsel, thresh = build_candidates(
        q, k, block_size=Lb, softmax_scale=softmax_scale, top_k=top_k, mode=mode
    )
    o = torch.empty_like(q)
    n_qblocks = T // Lb
    grid = (n_qblocks, B * H, 1)
    _fwd_kernel_prefill_sparse[grid](
        q,
        k,
        v,
        softmax_scale,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        thresh.contiguous(),
        cand,
        counts.contiguous(),
        B,
        H,
        T,
        n_qblocks,
        maxsel,
        BLOCK=Lb,
        BLOCK_DMODEL=D,
        COMPRESSIVE=compressive,
        num_warps=8 if D > 128 else 4,
        num_stages=2 if D > 128 else 3,
    )
    return o


def enable_prefill_sparse(
    model: torch.nn.Module,
    *,
    top_k: Optional[int] = None,
    top_k_fraction: Optional[float] = None,
    mode: str = "qblock",
) -> int:
    """
    Route every landmark layer's prefill through :func:`landmark_topk_prefill_sparse`.

    ``mode="union"`` preserves the exact per-token selection of
    :mod:`landmark_prefill_topk`; ``mode="qblock"`` pools the selection over each 64-query block,
    which is much faster but is a different attention and needs its own accuracy measurement.

    :returns: Number of layers patched.
    """
    import types as _types

    from .landmark_compressive import FastCompressiveLandmarkAttention
    from .landmark_fast import FastLandmarkAttention

    if mode not in ("union", "qblock"):
        raise ValueError(f"mode must be 'union' or 'qblock' (got {mode!r})")
    layers = [m for m in model.modules() if isinstance(m, FastLandmarkAttention)]
    if not layers:
        raise RuntimeError("no landmark attention layers found")
    for attn in layers:
        attn._prefill_sparse_cfg = {
            "top_k": top_k,
            "top_k_fraction": top_k_fraction,
            "compressive": isinstance(attn, FastCompressiveLandmarkAttention),
            "mode": mode,
        }
        if not hasattr(attn, "_prefill_orig_sparse"):
            attn._prefill_orig_sparse = attn._prefill
        attn._prefill = _types.MethodType(sparse_prefill, attn)
    return len(layers)


def disable_prefill_sparse(model: torch.nn.Module) -> int:
    """Undo :func:`enable_prefill_sparse`."""
    from .landmark_fast import FastLandmarkAttention

    n = 0
    for attn in model.modules():
        if isinstance(attn, FastLandmarkAttention) and hasattr(attn, "_prefill_orig_sparse"):
            attn._prefill = attn._prefill_orig_sparse
            del attn._prefill_orig_sparse
            n += 1
    return n


def sparse_prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Instance-bound ``_prefill`` replacement; config comes from ``self._prefill_sparse_cfg``."""
    cfg = self._prefill_sparse_cfg
    Lb = self.block_size
    T = q.shape[2]
    pad = (-T) % Lb
    if pad:
        q = torch.nn.functional.pad(q, (0, 0, 0, pad))
        k = torch.nn.functional.pad(k, (0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad))
    n_blocks = q.shape[2] // Lb
    top_k = cfg["top_k"]
    if top_k is None and cfg["top_k_fraction"] is not None:
        top_k = max(1, math.ceil(cfg["top_k_fraction"] * n_blocks))
    att = landmark_topk_prefill_sparse(
        q,
        k,
        v,
        block_size=Lb,
        softmax_scale=self.softmax_scale,
        top_k=top_k,
        compressive=cfg["compressive"],
        mode=cfg["mode"],
    )
    return att[:, :, :T]
