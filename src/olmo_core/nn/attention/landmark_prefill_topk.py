"""
Hard top-k landmark retrieval applied to the **whole prefill**, not just the decode steps.

Motivation
----------
Landmark / compressive-landmark eval today is top-k only at *decode*: each generated query (plus the
final prompt token, which the generation loop decodes rather than prefills) scores the cached
landmark keys, keeps the top ``k`` blocks and zeroes the rest
(:meth:`~olmo_core.nn.attention.landmark_fast.FastLandmarkAttention.set_landmark_eval_decode`). The
**prefill** is untouched: every prompt token still soft-gates over *all* past blocks, i.e. the prompt
representations the decode reads from were built with dense attention. So the reported numbers are
not what a genuinely sparse (O(k * block) per query) landmark model would produce.

This module implements the missing half: an eager, forward-only landmark attention where **every
query position** -- prompt tokens included -- keeps only its top-k past landmark blocks. Pointing an
eval at it answers "how much accuracy do we lose when top-k is applied everywhere, not just at
decode?".

It is deliberately standalone: nothing here modifies the training kernels or the existing decode
path. :func:`enable_prefill_topk` monkeypatches ``_prefill`` on the landmark layers of an
already-built model at eval time, and :func:`disable_prefill_topk` restores it.

Semantics
---------
For a query at absolute position ``p`` in block ``qb = p // block_size``:

* **past blocks** are blocks ``b < qb``; the block's gate score is its landmark key's score;
* the ``top_k`` highest-scoring past blocks are *selected*, all other past blocks get exactly zero
  weight (matching :meth:`FastLandmarkAttention._decode_apply_topk_landmark_retrieval`); a query with
  ``<= top_k`` past blocks keeps all of them;
* the **gate softmax** runs over the selected blocks' landmark keys plus the query's own (local)
  block, exactly as in training/decode;
* each selected block's gate weight is distributed over the block's tokens by a within-block
  softmax: content-only for the plain landmark variant, content **+ landmark** for the compressive
  variant (the landmark's value is the block's compressed summary);
* ``nonselected_mass`` (compressive only, ``alpha``) optionally reserves a fixed fraction of the mass
  for the *non-selected* blocks' landmark tokens, split by a softmax over their landmark scores --
  the prefill counterpart of
  :meth:`~olmo_core.nn.attention.landmark_compressive.FastCompressiveLandmarkAttention._compressive_decode_probs`.

With ``top_k=None`` and ``nonselected_mass=0`` this reduces to the exact dense grouped-softmax that
the fused Triton kernels compute, which is how the implementation is validated (see
``debug/prefill_topk/test_prefill_topk.py``).

Cost
----
This is an *accuracy* probe, not a speed one: it materializes ``(B, H, tile, T)`` score tiles, so it
is O(T^2) like dense attention (the kernel it replaces is too). ``query_tile`` bounds the peak
memory.
"""

import math
import types
from typing import Any, List, Optional

import torch

__all__ = [
    "landmark_topk_prefill_attention",
    "enable_prefill_topk",
    "disable_prefill_topk",
]


@torch.no_grad()
def landmark_topk_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int,
    softmax_scale: float,
    top_k: Optional[int] = None,
    compressive: bool = False,
    nonselected_mass: float = 0.0,
    query_tile: int = 128,
) -> torch.Tensor:
    """
    Landmark self-attention over a full prompt with per-query hard top-k block retrieval.

    :param q: Queries ``(B, H, T, D)``. ``T`` must be a multiple of ``block_size``.
    :param k: Keys ``(B, H, T, D)`` (GQA heads already expanded); no history KV.
    :param v: Values ``(B, H, T, D)``.
    :param block_size: Landmark block size (``mem_freq + 1``); the landmark token of a block is its
        last position.
    :param softmax_scale: Score scale (usually ``head_dim ** -0.5``).
    :param top_k: Number of past blocks each query may attend, or ``None`` for dense soft gating
        over all past blocks (the training/prefill default).
    :param compressive: Include each block's landmark token in the within-block softmax (the
        compressive variant), instead of gating content only.
    :param nonselected_mass: ``alpha`` in ``[0, 1)``: mass reserved for the non-selected blocks'
        landmark tokens. Only meaningful with ``top_k`` set and ``compressive=True``.
    :param query_tile: Number of query positions processed at once (peak memory knob).

    :returns: Attention output ``(B, H, T, D)``.
    """
    B, H, T, _ = q.shape
    if k.shape[2] != T or v.shape[2] != T:
        raise ValueError(
            f"prefill top-k expects self-attention with no history KV (got q T={T}, k T={k.shape[2]})"
        )
    Lb = block_size
    if T % Lb != 0:
        raise ValueError(f"sequence length {T} is not a multiple of block_size {Lb}")
    if not (0.0 <= nonselected_mass < 1.0):
        raise ValueError(f"nonselected_mass must be in [0, 1) (got {nonselected_mass})")
    n_blocks = T // Lb
    device = q.device

    key_pos = torch.arange(T, device=device)
    key_block = key_pos // Lb
    is_lm_key = (key_pos % Lb) == (Lb - 1)
    blk_idx = torch.arange(n_blocks, device=device)

    kf = k.float()
    out = torch.empty_like(q)

    for m0 in range(0, T, query_tile):
        m1 = min(m0 + query_tile, T)
        M = m1 - m0
        pos = torch.arange(m0, m1, device=device)
        q_block = pos // Lb
        # A landmark-position query does not attend to itself (the training kernel decrements the
        # causal bound on a block's last row), so its bound is p - 1.
        bound = torch.where((pos % Lb) == (Lb - 1), pos - 1, pos)

        causal = key_pos[None, :] <= bound[:, None]  # (M, T)
        past = key_block[None, :] < q_block[:, None]  # (M, T)
        local = (key_block[None, :] == q_block[:, None]) & causal  # (M, T)

        # (B, H, M, T) scores in fp32 (the kernel accumulates its dots in fp32 too).
        scores = torch.matmul(q[:, :, m0:m1].float(), kf.transpose(-1, -2)) * softmax_scale
        neg = torch.finfo(scores.dtype).min

        scores_blk = scores.view(B, H, M, n_blocks, Lb)
        lm_scores = scores_blk[..., Lb - 1]  # (B, H, M, n_blocks) block gate scores
        valid_blk = blk_idx[None, :] < q_block[:, None]  # (M, n_blocks) strictly-past blocks

        if top_k is not None and top_k < n_blocks:
            masked_lm = lm_scores.masked_fill(~valid_blk, neg)
            idx = masked_lm.topk(top_k, dim=-1).indices
            keep = torch.zeros_like(masked_lm, dtype=torch.bool).scatter_(-1, idx, True)
            # Rows with fewer than top_k past blocks: the padding picks belong to invalid blocks and
            # are dropped here, which is exactly "keep every past block".
            keep = keep & valid_blk
        else:
            keep = valid_blk.expand(B, H, M, n_blocks)

        keep_key = keep.repeat_interleave(Lb, dim=-1)  # (B, H, M, T)
        gate_set = (keep_key & is_lm_key[None, None, None, :] & past[None, None]) | local[
            None, None
        ]
        gate_w = torch.softmax(scores.masked_fill(~gate_set, neg), dim=-1)  # (B, H, M, T)

        # Local (own-block) keys keep their gate weight directly.
        probs = torch.where(local[None, None], gate_w, gate_w.new_zeros(()))

        # Past blocks: the block's gate weight spreads over its tokens by the within-block softmax.
        within_logits = scores_blk
        if not compressive:
            # Plain landmark: the landmark token itself contributes no value.
            within_logits = within_logits.clone()
            within_logits[..., Lb - 1] = neg
        within = torch.softmax(within_logits, dim=-1)  # (B, H, M, n_blocks, Lb)
        block_gate = gate_w.view(B, H, M, n_blocks, Lb)[..., Lb - 1]  # 0 for non-selected blocks
        contrib = (block_gate * keep).unsqueeze(-1) * within
        probs = probs + contrib.reshape(B, H, M, T) * past[None, None]

        if nonselected_mass > 0.0 and top_k is not None:
            nonsel = valid_blk & ~keep  # (B, H, M, n_blocks)
            has_ns = nonsel.any(-1)  # (B, H, M)
            ns_key = (
                nonsel.repeat_interleave(Lb, dim=-1)
                & is_lm_key[None, None, None, :]
                & past[None, None]
            )
            ns_w = torch.softmax(scores.masked_fill(~ns_key, neg), dim=-1)
            alpha = (has_ns.to(probs.dtype) * nonselected_mass).unsqueeze(-1)
            probs = probs * (1.0 - alpha) + alpha * ns_w

        out[:, :, m0:m1] = torch.matmul(probs.to(v.dtype), v)

    return out


try:  # optional: the fused fast path needs triton + CUDA
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore


if triton is not None:

    @triton.jit
    def _fwd_kernel_prefill_topk(
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
        Thresh,  # fp32 (Z*H, N_CTX_Q): per-query k-th largest landmark score, or -inf to keep all
        Z,
        H,
        N_CTX,
        BLOCK: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        COMPRESSIVE: tl.constexpr,
    ):
        # Forward-only landmark attention with per-query hard top-k block retrieval. Structurally the
        # same as landmark_kernel._fwd_kernel / landmark_compressive._fwd_kernel_compressive with
        # N_PREFIX_Q=0 and no doc/chunk masking, plus one extra line: a past block whose landmark
        # score falls below this query's threshold is floored to -1e30, so its gate weight underflows
        # to 0 and it contributes nothing. The floor is finite (not -inf) for the same reason as
        # DOC_MASK's: if EVERY block so far is dropped the running max stays at the floor and the
        # partial accumulator is garbage, but the diagonal block that always follows has finite
        # scores, so the rescale exp(-1e30 - finite) = 0 wipes it. -inf would produce inf - inf = nan.
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)

        BLOCK_M: tl.constexpr = BLOCK
        BLOCK_N: tl.constexpr = BLOCK

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # Last row of a block is its landmark token, which does not attend to itself.
        offs_m_real = offs_m + tl.where(tl.arange(0, BLOCK_M) == BLOCK_M - 1, -1, 0)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        offs_q = off_hz * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd
        offs_k = off_hz * skh + offs_n[None, :] * skn + offs_d[:, None] * skd
        offs_v = off_hz * svh + offs_n[:, None] * svn + offs_d[None, :] * svd

        thresh = tl.load(Thresh + off_hz * N_CTX + offs_m, mask=offs_m < N_CTX, other=0.0)

        m_prev = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        q_vals = tl.load(Q + offs_q, mask=offs_m[:, None] < N_CTX, other=0)

        for _ in range(0, start_m):
            k_vals = tl.load(K + offs_k, mask=offs_n[None, :] < N_CTX, other=0)
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q_vals.dtype)
            qk += tl.dot(q_vals, k_vals, allow_tf32=False)
            qk *= sm_scale
            qk = tl.where(offs_m_real[:, None] >= offs_n[None, :], qk, float("-inf"))

            landmark_qk = tl.max(
                tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, qk, float("-inf")), 1
            )
            # ---- the whole point: drop blocks outside this query's top-k ----
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
            v_vals = tl.load(V + offs_v, mask=offs_n[:, None] < N_CTX, other=0)
            acc += tl.dot(
                (landmark_p[:, None] * within_p / within_denom[:, None]).to(Q.dtype.element_ty),
                v_vals,
                allow_tf32=False,
            )

            l_prev = l_curr
            m_prev = m_curr

            offs_n += BLOCK_N
            offs_k += BLOCK_N * skn
            offs_v += BLOCK_N * svn

        # Diagonal (local) block: plain causal softmax, never gated by top-k.
        k_vals = tl.load(K + offs_k, mask=offs_n[None, :] < N_CTX, other=0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=q_vals.dtype)
        qk += tl.dot(q_vals, k_vals, allow_tf32=False)
        qk *= sm_scale
        qk = tl.where(offs_m_real[:, None] >= offs_n[None, :], qk, float("-inf"))

        m_curr = tl.maximum(tl.max(qk, 1), m_prev)
        l_prev *= tl.exp(m_prev - m_curr)
        p = tl.exp(qk - m_curr[:, None])
        l_curr = tl.sum(p, 1) + l_prev
        l_rcp = 1.0 / l_curr
        p *= l_rcp[:, None]
        acc *= (l_prev * l_rcp)[:, None]
        v_vals = tl.load(V + offs_v, mask=offs_n[:, None] < N_CTX, other=0)
        acc += tl.dot(p.to(Q.dtype.element_ty), v_vals, allow_tf32=False)

        offs_o = off_hz * soh + offs_m[:, None] * som + offs_d[None, :] * sod
        tl.store(Out + offs_o, acc, mask=offs_m[:, None] < N_CTX)


@torch.no_grad()
def landmark_topk_thresholds(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    block_size: int,
    softmax_scale: float,
    top_k: Optional[int],
    query_tile: int = 4096,
) -> torch.Tensor:
    """
    Per-query cutoff for hard top-k landmark retrieval, for use as ``landmark_score >= thresh``.

    The cutoff is the **midpoint between the k-th and (k+1)-th largest** score over the query's
    strictly-past landmark keys, not the k-th value itself. The k-th value would be fragile: this
    pass computes scores with a torch fp32 matmul while the kernel recomputes them with a bf16
    ``tl.dot``, so at ``top_k=1`` the arg-max block's own score can land a rounding step *below* its
    threshold and get dropped -- selecting nothing at all. A midpoint tolerates any noise smaller
    than half the gap between the last kept and first dropped block.

    ``-inf`` (keep every past block) is returned when the query has ``<= top_k`` past blocks, which
    falls out for free: the (k+1)-th value is then ``-inf`` and so is the midpoint.

    Costs ``T x n_blocks`` instead of ``T x T``.

    :returns: fp32 ``(B * H, T)`` thresholds, laid out for the fused kernel.
    """
    B, H, T, _ = q.shape
    Lb = block_size
    n_blocks = T // Lb
    device = q.device
    if top_k is None or top_k >= n_blocks:
        return torch.full((B * H, T), -float("inf"), device=device, dtype=torch.float32)

    lm_k = k[:, :, Lb - 1 :: Lb, :].float()  # (B, H, n_blocks, D)
    blk_idx = torch.arange(n_blocks, device=device)
    thresh = torch.empty(B, H, T, device=device, dtype=torch.float32)

    for m0 in range(0, T, query_tile):
        m1 = min(m0 + query_tile, T)
        pos = torch.arange(m0, m1, device=device)
        q_block = pos // Lb
        s = torch.matmul(q[:, :, m0:m1].float(), lm_k.transpose(-1, -2)) * softmax_scale
        valid = blk_idx[None, :] < q_block[:, None]  # (M, n_blocks) strictly-past blocks
        s = s.masked_fill(~valid, -float("inf"))
        top = s.topk(top_k + 1, dim=-1).values  # (B, H, M, top_k + 1)
        thresh[:, :, m0:m1] = 0.5 * (top[..., -2] + top[..., -1])
    return thresh.reshape(B * H, T)


@torch.no_grad()
def landmark_topk_prefill_attention_fast(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int,
    softmax_scale: float,
    top_k: Optional[int],
    compressive: bool,
) -> torch.Tensor:
    """
    Fused (Triton) equivalent of :func:`landmark_topk_prefill_attention` with
    ``nonselected_mass=0``. Two passes: landmark-only scores -> per-query threshold, then the
    landmark forward kernel with blocks below the threshold floored out.

    Roughly the speed of the ordinary fused prefill kernel, vs ~55x slower for the eager path at 32k.
    """
    if triton is None:
        raise RuntimeError("the fused prefill top-k path requires triton + CUDA")
    B, H, T, D = q.shape
    Lb = block_size
    if T % Lb != 0:
        raise ValueError(f"sequence length {T} is not a multiple of block_size {Lb}")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    thresh = landmark_topk_thresholds(
        q, k, block_size=Lb, softmax_scale=softmax_scale, top_k=top_k
    ).contiguous()
    o = torch.empty_like(q)
    grid = (T // Lb, B * H, 1)
    num_warps = 8 if D > 128 else 4
    num_stages = 2 if D > 128 else 3
    _fwd_kernel_prefill_topk[grid](
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
        thresh,
        B,
        H,
        T,
        BLOCK=Lb,
        BLOCK_DMODEL=D,
        COMPRESSIVE=compressive,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o


def _prefill_topk(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Replacement for ``FastLandmarkAttention._prefill`` that applies per-query top-k retrieval."""
    cfg = self._prefill_topk_cfg
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
    # The fused path is ~55x faster at 32k but only implements the hard drop (alpha = 0).
    use_fast = cfg["backend"] == "fused" or (
        cfg["backend"] == "auto"
        and triton is not None
        and q.is_cuda
        and cfg["nonselected_mass"] == 0.0
    )
    if use_fast:
        att = landmark_topk_prefill_attention_fast(
            q,
            k,
            v,
            block_size=Lb,
            softmax_scale=self.softmax_scale,
            top_k=top_k,
            compressive=cfg["compressive"],
        )
    else:
        att = landmark_topk_prefill_attention(
            q,
            k,
            v,
            block_size=Lb,
            softmax_scale=self.softmax_scale,
            top_k=top_k,
            compressive=cfg["compressive"],
            nonselected_mass=cfg["nonselected_mass"],
            query_tile=cfg["query_tile"],
        )
    return att[:, :, :T]


def _landmark_layers(model: torch.nn.Module) -> List[Any]:
    """All fast-landmark attention layers of ``model`` (``FastCompressiveLandmarkAttention``
    subclasses ``FastLandmarkAttention``, so one isinstance check covers both)."""
    from .landmark_fast import FastLandmarkAttention

    return [m for m in model.modules() if isinstance(m, FastLandmarkAttention)]


def enable_prefill_topk(
    model: torch.nn.Module,
    *,
    top_k: Optional[int] = None,
    top_k_fraction: Optional[float] = None,
    nonselected_mass: Optional[float] = None,
    query_tile: int = 128,
    backend: str = "auto",
) -> int:
    """
    Route every landmark layer's prefill through the top-k attention above, so *all* prompt
    positions -- not just the decoded ones -- see only their top-k landmark blocks.

    Inference only; the training path is untouched. Call :func:`disable_prefill_topk` to restore.

    :param model: A built (eval-mode) transformer with fast-landmark attention layers.
    :param top_k: Fixed number of past blocks per query. Takes precedence over ``top_k_fraction``.
    :param top_k_fraction: ``k = ceil(fraction * num_prompt_blocks)``, the same rule the decode uses.
    :param nonselected_mass: ``alpha`` for compressive layers; ``None`` reuses each layer's own
        :attr:`nonselected_landmark_mass`. Note ``alpha > 0`` forces the slow eager backend.
    :param query_tile: Query positions per tile (eager backend peak-memory knob).
    :param backend: ``"auto"`` (fused when it applies -- CUDA, triton, ``alpha == 0``),
        ``"fused"`` (force the Triton path; ignores ``alpha``), or ``"eager"``.

    :returns: Number of layers patched.

    :raises RuntimeError: If the model has no fast-landmark attention layers.
    """
    from .landmark_compressive import FastCompressiveLandmarkAttention

    layers = _landmark_layers(model)
    if not layers:
        raise RuntimeError(
            "no FastLandmarkAttention / FastCompressiveLandmarkAttention layers found; "
            "prefill top-k only applies to landmark models"
        )
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1 or None (got {top_k})")
    if backend not in ("auto", "fused", "eager"):
        raise ValueError(f"backend must be auto|fused|eager (got {backend!r})")
    for attn in layers:
        compressive = isinstance(attn, FastCompressiveLandmarkAttention)
        alpha = 0.0
        if compressive:
            alpha = (
                float(attn.nonselected_landmark_mass)
                if nonselected_mass is None
                else float(nonselected_mass)
            )
        attn._prefill_topk_cfg = {
            "top_k": top_k,
            "top_k_fraction": top_k_fraction,
            "compressive": compressive,
            "nonselected_mass": alpha,
            "query_tile": query_tile,
            "backend": backend,
        }
        if not hasattr(attn, "_prefill_orig"):
            attn._prefill_orig = attn._prefill
        attn._prefill = types.MethodType(_prefill_topk, attn)
    return len(layers)


def disable_prefill_topk(model: torch.nn.Module) -> int:
    """Undo :func:`enable_prefill_topk`. Returns the number of layers restored."""
    n = 0
    for attn in _landmark_layers(model):
        if hasattr(attn, "_prefill_orig"):
            attn._prefill = attn._prefill_orig
            del attn._prefill_orig
            n += 1
    return n
