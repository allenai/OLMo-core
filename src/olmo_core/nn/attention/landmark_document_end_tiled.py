"""Query-tiled exact document-end compressive attention, with native GQA.

All document softmaxes in a query tile are segmented reductions; there are no
Python loops over tokens, heads, or documents. Backward recomputes one tile at a
time and saves only Q/K/V and O(T) role/group metadata. This is a portable tensor
backend (CPU/CUDA), not a fused Triton kernel. Arithmetic remains quadratic.
"""

from typing import Optional

import torch
from torch.autograd.function import once_differentiable

from .summary_mask import ROLE_DOC_ID, ROLE_EXAMPLE_ID, ROLE_KIND, TokenKind

# Bound EACH score-shaped tensor to this many elements, except when a single
# query row is already larger. Several such tensors coexist during backward.
_DEFAULT_SCORE_BUDGET = 1 << 20


def _qk(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """(B,Hq,Q,D) @ shared (B,Hkv,K,D), without expanding K/V heads."""
    B, H, Q, D = q.shape
    Hkv, K = k.shape[1:3]
    return (q.reshape(B, Hkv, H // Hkv, Q, D) @ k[:, :, None].transpose(-1, -2)).reshape(B, H, Q, K)


def _pv(p: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, H, Q, K = p.shape
    Hkv, D = v.shape[1], v.shape[-1]
    return (p.reshape(B, Hkv, H // Hkv, Q, K) @ v[:, :, None]).reshape(B, H, Q, D)


def _kv_grad(p: torch.Tensor, x: torch.Tensor, n_kv_heads: int) -> torch.Tensor:
    """Contract all query heads in a KV group in one matmul (no per-head dK/dV)."""
    B, H, Q, K = p.shape
    R, D = H // n_kv_heads, x.shape[-1]
    left = p.reshape(B, n_kv_heads, R, Q, K).permute(0, 1, 4, 2, 3)
    return left.reshape(B, n_kv_heads, K, R * Q) @ x.reshape(B, n_kv_heads, R * Q, D)


def _tile_probs(q, k, roles, groups, n_groups, start, scale):
    """Two softmax levels and their product for a contiguous absolute query range."""
    Q, K = q.shape[2], k.shape[2]
    scores = _qk(q, k) * scale
    doc = roles[:, ROLE_DOC_ID, :K]
    kind = roles[:, ROLE_KIND, :K]
    example = roles[:, ROLE_EXAMPLE_ID, :K]
    qdoc = roles[:, ROLE_DOC_ID, start : start + Q, None]
    qkind = roles[:, ROLE_KIND, start : start + Q, None]
    qexample = roles[:, ROLE_EXAMPLE_ID, start : start + Q, None]
    key_pos = torch.arange(K, device=q.device)
    query_pos = torch.arange(start, start + Q, device=q.device)
    visible = (
        (key_pos[None, :] <= query_pos[:, None])
        & (qexample == example[:, None])
        & (qkind != int(TokenKind.PAD))
        & (kind[:, None] != int(TokenKind.PAD))
    )[:, None]
    is_lm = (kind == int(TokenKind.SUMMARY))[:, None, None]
    local = (
        visible
        & ~is_lm
        & (
            (kind[:, None, None] == int(TokenKind.INSTRUCTION))
            | ((doc[:, None, None] == qdoc[:, None]) & (doc[:, None, None] >= 0))
        )
    )
    past = visible & (doc[:, None, None] < qdoc[:, None]) & (doc[:, None, None] >= 0)
    past = past & ((kind[:, None, None] == int(TokenKind.DOC_CONTENT)) | is_lm)
    gate_mask = local | (past & is_lm)
    gate_scores = scores.masked_fill(~gate_mask, -torch.inf)
    gate_max = gate_scores.amax(-1, keepdim=True)
    # PAD rows have no keys. Replace their maximum before exponentiation so
    # neither forward nor backward has a NaN from (-inf)-(-inf).
    gate_max = torch.where(torch.isfinite(gate_max), gate_max, 0)
    gate_exp = (gate_scores - gate_max).exp()
    gate = gate_exp / gate_exp.sum(-1, keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)

    index = groups[:, None, None, :K].expand_as(scores)
    shape = (*scores.shape[:-1], n_groups)
    within_scores = scores.masked_fill(~past, -torch.inf)
    group_max = scores.new_full(shape, -torch.inf)
    group_max.scatter_reduce_(-1, index, within_scores, reduce="amax", include_self=True)
    # Only past positions participate; all-masked groups must yield exactly 0.
    within_exp = torch.where(past, scores - group_max.gather(-1, index), -torch.inf).exp()
    denom = scores.new_zeros(shape).scatter_add_(-1, index, within_exp)
    within = within_exp / denom.gather(-1, index).clamp_min(torch.finfo(q.dtype).tiny)
    group_gate = scores.new_zeros(shape).scatter_add_(-1, index, gate * is_lm)
    probs = torch.where(local, gate, within * group_gate.gather(-1, index))
    return probs, within, gate, local, past & is_lm, index


class _DocumentEndTiled(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, roles, scale, query_start, tile_size):
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        lm = roles[:, ROLE_KIND] == int(TokenKind.SUMMARY)
        # The closing landmark belongs to its preceding group. Group indices do
        # not reset at packed-example boundaries; visibility still tests example ID.
        groups = lm.long().cumsum(-1) - lm.long()
        n_groups = int(groups.max().item()) + 1
        ka, va = k.to(dtype), v.to(dtype)
        out = torch.empty_like(q)
        for lo in range(0, q.shape[2], tile_size):
            hi = min(lo + tile_size, q.shape[2])
            end = query_start + hi  # no tile ever reads future K/V
            p = _tile_probs(
                q[:, :, lo:hi].to(dtype),
                ka[:, :, :end],
                roles,
                groups,
                n_groups,
                query_start + lo,
                scale,
            )[0]
            out[:, :, lo:hi] = _pv(p, va[:, :, :end]).to(q.dtype)
            del p
        ctx.save_for_backward(q, k, v, roles, groups)
        ctx.scale, ctx.query_start = scale, query_start
        ctx.tile_size, ctx.n_groups = tile_size, n_groups
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        with torch.autocast(device_type=grad_out.device.type, enabled=False):
            return _DocumentEndTiled._backward(ctx, grad_out)

    @staticmethod
    def _backward(ctx, grad_out):
        q, k, v, roles, groups = ctx.saved_tensors
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        ka, va = k.to(dtype), v.to(dtype)
        dq = torch.empty_like(q)
        dk, dv = torch.zeros_like(ka), torch.zeros_like(va)
        for lo in range(0, q.shape[2], ctx.tile_size):
            hi = min(lo + ctx.tile_size, q.shape[2])
            end, start = ctx.query_start + hi, ctx.query_start + lo
            qt = q[:, :, lo:hi].to(dtype)
            do = grad_out[:, :, lo:hi].to(dtype)
            p, within, gate, local, lm, index = _tile_probs(
                qt,
                ka[:, :, :end],
                roles,
                groups,
                ctx.n_groups,
                start,
                ctx.scale,
            )
            # u_i = dO . V_i; delta = dO . O. Compute from the unrounded
            # probabilities so low-precision output rounding does not affect ds.
            u = _qk(do, va[:, :, :end])
            pu = p * u
            delta = pu.sum(-1, keepdim=True)
            z = p.new_zeros((*p.shape[:-1], ctx.n_groups))
            z.scatter_add_(-1, index, torch.where(local, 0, pu))
            z = z.gather(-1, index)  # z_d = G_d * sum_i f_i u_i
            # Past content: G_d f_i (u_i - E_f[u]). A landmark also
            # contributes the gate derivative G_d (E_f[u] - delta).
            ds = torch.where(local, p * (u - delta), pu - within * z)
            ds = ds + torch.where(lm, z - gate * delta, 0)
            dq[:, :, lo:hi] = (_pv(ds, ka[:, :, :end]) * ctx.scale).to(q.dtype)
            dk[:, :, :end] += _kv_grad(ds, qt, k.shape[1]) * ctx.scale
            dv[:, :, :end] += _kv_grad(p, do, v.shape[1])
            del p, within, gate, local, lm, index, u, pu, delta, z, ds
        return dq, dk.to(k.dtype), dv.to(v.dtype), None, None, None, None


def tiled_document_end_compressive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    roles: torch.Tensor,
    softmax_scale: Optional[float] = None,
    *,
    query_start: int = 0,
    query_tile_size: int = 64,
    score_budget: int = _DEFAULT_SCORE_BUDGET,
) -> torch.Tensor:
    """Exact attention with bounded query tiles and recomputed first-order backward.

    Q: (B,Hq,Q,D); K/V: (B,Hkv,K,D); roles: (B,4,K). Each query is the
    token at ``query_start + its row`` in roles. Supports packed examples,
    padding, arbitrary complete document lengths, and cached trailing queries.
    ``score_budget`` bounds the elements in one score-shaped workspace (not
    total bytes); at least one query row is processed. Inputs must follow the
    validated one-landmark-per-document layout. Higher derivatives are unsupported.
    """
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("Expected rank-4 Q/K/V with matching K/V shapes")
    B, H, Q, D = q.shape
    if min(B, H, Q, D, k.shape[1], k.shape[2]) < 1:
        raise ValueError("Empty attention dimensions are unsupported")
    if k.shape[0] != B or k.shape[-1] != D or H % k.shape[1]:
        raise ValueError("Invalid GQA head or batch dimensions")
    if roles.shape != (B, 4, k.shape[2]):
        raise ValueError("Expected roles with shape (B, 4, K)")
    if query_start < 0 or query_start + Q > k.shape[2]:
        raise ValueError("Query range is outside the key sequence")
    if query_tile_size < 1 or score_budget < 1:
        raise ValueError("Query tile size and score budget must be positive")
    if q.device != k.device or q.device != v.device or q.device != roles.device:
        raise ValueError("Q/K/V and roles must be on the same device")
    if (
        q.dtype != k.dtype
        or q.dtype != v.dtype
        or q.dtype
        not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        )
    ):
        raise ValueError("Q/K/V must share a supported floating-point dtype")
    scale = D**-0.5 if softmax_scale is None else softmax_scale
    tile = min(query_tile_size, max(1, score_budget // (B * H * k.shape[2])))
    # Disable ambient autocast: softmax statistics and gradient accumulation use
    # FP32 (FP64 for numerical tests), including all grouped matrix products.
    with torch.autocast(device_type=q.device.type, enabled=False):
        return _DocumentEndTiled.apply(q, k, v, roles, scale, query_start, tile)
