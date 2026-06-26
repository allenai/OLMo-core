"""
``FastLandmarkAttention`` -- a standalone landmark-attention sequence mixer with a much faster
backward, independent of (and not modifying) the original :mod:`landmark_kernel` / ``LandmarkAttention``.

The forward is identical to the original landmark kernel (it reuses the original forward Triton
kernel via ``FusedLandmarkAttention``); only the **backward** is replaced. The original
``_bwd_kernel`` launches one program per ``(batch, head)`` (~16 programs) and loops over key blocks
serially, so the backward is badly under-parallelized and dominates the step (~26x FlashAttention at
4k). Here the backward is split FlashAttention-2-style into two atomic-free kernels:

  * :func:`_bwd_kv_kernel` -- one program per ``(key-block, head)``, computes ``dk``/``dv``.
  * :func:`_bwd_q_kernel`  -- one program per ``(query-block, head)``, computes ``dq`` (causal half).

Gradients follow the original's accumulation order (``dk``/``dv`` per key block; ``dq`` ascending
with no atomics) and match it up to last-ulp reassociation noise (the two backwards use different
``num_warps``, which retiles the ``tl.dot`` reductions); the forward output is bit-identical.
fwd+bwd is ~17-20x faster at 4k-8k (~1.4x FlashAttention). Tuned launch config (H200, head_dim
128): ``num_warps=4, num_stages=2``. Head dims up to 256 (e.g. Qwen3.5) are supported:
``head_dim > 128`` switches to ``num_warps=8`` (and fewer stages) to fit register/shared-memory
budgets, leaving the ``<= 128`` launch configs untouched. Note that at head_dim 256 with the usual
``block_size=64``, the *fp32* backward exceeds H100 shared memory (the dot/trans operand tiles
alone need ~278KB); bf16/fp16 -- what training uses -- fit fine.

``FastLandmarkAttention`` is a drop-in :class:`Attention` variant (a *new* sequence mixer, selected
via ``AttentionType.fast_landmark``); it supports Ulysses context parallelism exactly as the
original landmark attention does. The fused Triton kernel is required (CUDA + triton).
"""

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F

from olmo_core.distributed.parallel.context_parallel import (
    all_to_all_cp2hp,
    all_to_all_single_cp2hp,
    all_to_all_single_hp2cp,
)
from olmo_core.distributed.utils import get_rank
from olmo_core.exceptions import OLMoConfigurationError

from . import Attention  # base mixer (defined before the end-of-module import in __init__)
from .kv_cache import KVCacheManager
from .landmark import (
    build_block_doc_id,
    build_local_packed_position_ids,
    landmark_grouped_softmax,
    repeat_kv,
)
from .landmark_kernel import (
    FusedLandmarkAttention,
    _bwd_preprocess,
    _fwd_kernel,
    has_landmark_kernel,
)
from .ring import RingContextParallelStyle, UlyssesContextParallelStyle

try:
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:
    triton = None  # type: ignore
    tl = None  # type: ignore


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


if triton is not None:

    @triton.jit
    def _bwd_kv_kernel(
        Q,
        K,
        V,
        sm_scale,
        Out,
        DO,
        DQ,
        DK,
        DV,
        L,
        M,
        D,
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
        DocId,  # int32 (Z, N_BLOCKS) per-block document id, or dummy when DOC_MASK is False
        Z,
        H,
        N_CTX_Q,
        N_CTX_KV,
        N_BLOCKS,
        BLOCK: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        N_PREFIX_Q: tl.constexpr,
        DOC_MASK: tl.constexpr,
    ):
        # dk/dv only, one program per (key-block, head); atomic-free. dk/dv accumulation order is
        # the same as the original kernel -> bit-identical.
        off_hz = tl.program_id(0)
        off_z = off_hz // H
        off_h = off_hz % H

        BLOCK_M: tl.constexpr = BLOCK
        BLOCK_N: tl.constexpr = BLOCK

        Q += off_z * sqz + off_h * sqh
        K += off_z * skz + off_h * skh
        V += off_z * svz + off_h * svh
        DO += off_z * sqz + off_h * sqh
        DK += off_z * skz + off_h * skh
        DV += off_z * svz + off_h * svh

        offs_d = tl.arange(0, BLOCK_DMODEL)
        D_ptrs = D + off_hz * N_CTX_Q
        m_ptrs = M + off_hz * N_CTX_Q

        start_n = tl.program_id(1) * BLOCK_N
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K + (offs_n[:, None] * skn + offs_d[None, :] * skd)
        v_ptrs = V + (offs_n[:, None] * svn + offs_d[None, :] * svd)

        # Document id of this key block (for intra-document / packing masking). Only the landmark-
        # grouping loop over strictly-future query blocks needs the cross-document gate; the
        # diagonal (own-block) contribution below is always same-document.
        if DOC_MASK:
            k_doc = tl.load(DocId + off_z * N_BLOCKS + (start_n // BLOCK_N))

        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)

        if start_n < N_PREFIX_Q * BLOCK_M:
            start_q_index = 0
        elif N_CTX_Q <= start_n - N_PREFIX_Q * BLOCK_M:
            start_q_index = start_n - N_PREFIX_Q * BLOCK_M
        else:
            first_start_m = start_n - N_PREFIX_Q * BLOCK_M
            first_start_m = tl.multiple_of(first_start_m, BLOCK_M)
            offs_m = first_start_m + tl.arange(0, BLOCK_M)
            offs_m_real = offs_m + N_PREFIX_Q * BLOCK_M
            offs_m_real += tl.where(tl.arange(0, BLOCK_M) == BLOCK_M - 1, -1, 0)

            q_ptrs = Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            do_ptrs = DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)

            q = tl.load(q_ptrs)
            qk = tl.dot(q, tl.trans(k), allow_tf32=False)
            qk = tl.where(offs_m_real[:, None] >= (offs_n[None, :]), qk, float("-inf"))

            m = tl.load(m_ptrs + offs_m)
            last_p = tl.exp(qk * sm_scale - m[:, None])

            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(last_p.to(Q.dtype.element_ty)), do, allow_tf32=False)

            Di = tl.load(D_ptrs + offs_m)
            last_dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            last_dp += tl.dot(do, tl.trans(v), allow_tf32=False)
            ds = last_p * last_dp * sm_scale

            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q, allow_tf32=False)
            start_q_index = first_start_m + BLOCK_M

        # Only strictly-future query blocks contribute; runtime *upper* bound avoids keep==0 waste
        # (a runtime *lower* bound would trip the TritonGPUCoalesce bug). Skipped blocks added 0.
        for i in range(0, N_CTX_Q - start_q_index, BLOCK_M):
            start_m = start_q_index + i
            start_m = tl.multiple_of(start_m, BLOCK_M)
            offs_m = start_m + tl.arange(0, BLOCK_M)

            # Cross-document query blocks received zero weight on this key block in the forward, so
            # they get zero gradient here. ``doc_keep`` is 1.0 for same-document, 0.0 otherwise.
            doc_keep = 1.0
            if DOC_MASK:
                q_doc = tl.load(DocId + off_z * N_BLOCKS + (start_m // BLOCK_M))
                doc_keep = (q_doc == k_doc).to(tl.float32)

            q_ptrs = Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)
            do_ptrs = DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd)

            q = tl.load(q_ptrs)
            qk = tl.dot(q, tl.trans(k), allow_tf32=False)
            qk *= sm_scale

            landmark_qk = tl.max(
                tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, qk, float("-inf")), 1
            )
            normal_qk = tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, float("-inf"), qk)

            m = tl.load(m_ptrs + offs_m)
            p = tl.exp(landmark_qk - m)

            do = tl.load(do_ptrs)

            normal_m = tl.max(normal_qk, 1)
            normal_p = tl.exp(normal_qk - normal_m[:, None])
            normal_p_normalized = normal_p / tl.sum(normal_p, 1)[:, None]
            normal_kv = tl.dot(normal_p_normalized.to(Q.dtype.element_ty), v, allow_tf32=False)

            normal_D = tl.sum(do * normal_kv, 1)

            dv += tl.dot(
                tl.trans((doc_keep * p[:, None] * normal_p_normalized).to(Q.dtype.element_ty)),
                do,
                allow_tf32=False,
            )

            Di = tl.load(D_ptrs + offs_m)
            dp = tl.zeros([BLOCK_M], dtype=tl.float32) - Di
            dp += normal_D
            landmark_ds = p * dp
            normal_dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - normal_D[:, None]
            normal_dp += tl.dot(do, tl.trans(v), allow_tf32=False)
            normal_ds = p[:, None] * normal_p_normalized * normal_dp
            ds = tl.where(
                tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, landmark_ds[:, None], normal_ds
            )
            ds *= sm_scale * doc_keep
            dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q, allow_tf32=False)

        dv_ptrs = DV + (offs_n[:, None] * svn + offs_d[None, :] * svd)
        dk_ptrs = DK + (offs_n[:, None] * skn + offs_d[None, :] * skd)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)

    @triton.jit
    def _bwd_q_kernel(
        Q,
        K,
        V,
        sm_scale,
        Out,
        DO,
        DQ,
        DK,
        DV,
        L,
        M,
        D,
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
        DocId,  # int32 (Z, N_BLOCKS) per-block document id, or dummy when DOC_MASK is False
        Z,
        H,
        N_CTX_Q,
        N_CTX_KV,
        N_BLOCKS,
        BLOCK: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        N_PREFIX_Q: tl.constexpr,
        DOC_MASK: tl.constexpr,
    ):
        # dq only, one program per (query-block, head). Causal-only key-block loop via a runtime
        # *upper* bound, atomic-free. dq accumulates ascending -> bit-identical to the original.
        # Only implemented for N_PREFIX_Q == 0 (the caller guards this).
        off_hz = tl.program_id(0)
        off_z = off_hz // H
        off_h = off_hz % H

        BLOCK_M: tl.constexpr = BLOCK
        BLOCK_N: tl.constexpr = BLOCK

        Q += off_z * sqz + off_h * sqh
        K += off_z * skz + off_h * skh
        V += off_z * svz + off_h * svh
        DO += off_z * sqz + off_h * sqh
        DQ += off_z * sqz + off_h * sqh

        offs_d = tl.arange(0, BLOCK_DMODEL)
        D_ptrs = D + off_hz * N_CTX_Q
        m_ptrs = M + off_hz * N_CTX_Q

        start_m = tl.program_id(1) * BLOCK_M
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + tl.arange(0, BLOCK_M)

        q = tl.load(Q + (offs_m[:, None] * sqm + offs_d[None, :] * sqd))
        do = tl.load(DO + (offs_m[:, None] * sqm + offs_d[None, :] * sqd))
        m = tl.load(m_ptrs + offs_m)
        Di = tl.load(D_ptrs + offs_m)

        if DOC_MASK:
            q_doc = tl.load(DocId + off_z * N_BLOCKS + (start_m // BLOCK_M))

        dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        for start_n in range(0, start_m, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(K + (offs_n[:, None] * skn + offs_d[None, :] * skd))
            v = tl.load(V + (offs_n[:, None] * svn + offs_d[None, :] * svd))

            # Cross-document prior key blocks contributed nothing in the forward -> zero gradient.
            doc_keep = 1.0
            if DOC_MASK:
                k_doc = tl.load(DocId + off_z * N_BLOCKS + (start_n // BLOCK_N))
                doc_keep = (q_doc == k_doc).to(tl.float32)

            qk = tl.dot(q, tl.trans(k), allow_tf32=False)
            qk *= sm_scale
            landmark_qk = tl.max(
                tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, qk, float("-inf")), 1
            )
            normal_qk = tl.where(tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, float("-inf"), qk)
            p = tl.exp(landmark_qk - m)
            normal_m = tl.max(normal_qk, 1)
            normal_p = tl.exp(normal_qk - normal_m[:, None])
            normal_p_normalized = normal_p / tl.sum(normal_p, 1)[:, None]
            normal_kv = tl.dot(normal_p_normalized.to(Q.dtype.element_ty), v, allow_tf32=False)
            normal_D = tl.sum(do * normal_kv, 1)
            dp = tl.zeros([BLOCK_M], dtype=tl.float32) - Di
            dp += normal_D
            landmark_ds = p * dp
            normal_dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - normal_D[:, None]
            normal_dp += tl.dot(do, tl.trans(v), allow_tf32=False)
            normal_ds = p[:, None] * normal_p_normalized * normal_dp
            ds = tl.where(
                tl.arange(0, BLOCK_N)[None, :] == BLOCK_N - 1, landmark_ds[:, None], normal_ds
            )
            ds *= sm_scale * doc_keep
            dq += tl.dot(ds.to(Q.dtype.element_ty), k, allow_tf32=False)

        # diagonal key block (start_n == start_m): within-block causal attention
        offs_n = start_m + tl.arange(0, BLOCK_N)
        k = tl.load(K + (offs_n[:, None] * skn + offs_d[None, :] * skd))
        v = tl.load(V + (offs_n[:, None] * svn + offs_d[None, :] * svd))
        offs_m_real = offs_m + tl.where(tl.arange(0, BLOCK_M) == BLOCK_M - 1, -1, 0)
        qk = tl.dot(q, tl.trans(k), allow_tf32=False)
        qk = tl.where(offs_m_real[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        last_p = tl.exp(qk * sm_scale - m[:, None])
        last_dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
        last_dp += tl.dot(do, tl.trans(v), allow_tf32=False)
        ds = last_p * last_dp * sm_scale
        dq += tl.dot(ds.to(Q.dtype.element_ty), k, allow_tf32=False)

        tl.store(DQ + (offs_m[:, None] * sqm + offs_d[None, :] * sqd), dq)


class _FusedLandmarkAttentionFast(FusedLandmarkAttention):
    """
    Same numerics as the original ``FusedLandmarkAttention``, but: the forward relaunches the
    original ``_fwd_kernel`` with a tuned launch config (num_warps=4, num_stages=3 -> ~1.5x), and
    the backward is the FA2-style two-kernel split (~17-20x). Leaves ``landmark_kernel`` untouched.
    """

    @staticmethod
    def forward(ctx, q, k, v, n_prefix_q, sm_scale, block_size, doc_id=None, chunk_ids=None):
        if triton is None:
            raise RuntimeError("Landmark attention requires 'triton' (and a CUDA device).")
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        batch, nheads, seqlen_q, d = q.shape
        assert d <= 256 and q.dtype == k.dtype == v.dtype and q.is_cuda

        BLOCK = block_size
        n_blocks = k.shape[2] // BLOCK
        doc_mask = doc_id is not None
        if doc_mask:
            assert doc_id.shape == (batch, n_blocks), (doc_id.shape, (batch, n_blocks))
            doc_id = doc_id.to(device=q.device, dtype=torch.int32).contiguous()
        doc_id_arg = doc_id if doc_mask else torch.empty(1, dtype=torch.int32, device=q.device)
        # Per-token document-chunked masking (mutually exclusive with the per-block doc_id packing).
        chunk_mask = chunk_ids is not None
        if chunk_mask:
            assert not doc_mask, "chunk_ids and doc_id (packing) are mutually exclusive"
            assert chunk_ids.shape == (batch, k.shape[2]), (chunk_ids.shape, (batch, k.shape[2]))
            chunk_ids = chunk_ids.to(device=q.device, dtype=torch.int32).contiguous()
        chunk_ids_arg = (
            chunk_ids if chunk_mask else torch.empty(1, dtype=torch.int32, device=q.device)
        )

        o = torch.empty_like(q)
        grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        # Tuned forward launch (H200, head_dim 128): num_warps=4, num_stages=3 (~1.5x over the
        # original 8/2). Env-overridable. Identical numerics to the original forward.
        # head_dim > 128 (e.g. Qwen3.5's 256) gets its own defaults: more warps to spread the
        # (BLOCK, 256) fp32 accumulator across registers, fewer stages so the pipelined K/V tiles
        # fit in shared memory. The <= 128 defaults are untouched.
        if d > 128:
            num_warps = _env_int("LM_FAST_FWD_WARPS", 8)
            num_stages = _env_int("LM_FAST_FWD_STAGES", 2)
        else:
            num_warps = _env_int("LM_FAST_FWD_WARPS", 4)
            num_stages = _env_int("LM_FAST_FWD_STAGES", 3)
        _fwd_kernel[grid](
            q,
            k,
            v,
            sm_scale,
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
            L,
            m,
            doc_id_arg,
            chunk_ids_arg,
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k.shape[2],
            n_blocks,
            BLOCK=BLOCK,
            BLOCK_DMODEL=d,
            N_PREFIX_Q=n_prefix_q,
            DOC_MASK=doc_mask,
            CHUNK_MASK=chunk_mask,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        ctx.save_for_backward(q, k, v, o, L, m)
        ctx.doc_id = doc_id  # None when not packing
        ctx.chunk_ids = chunk_ids  # None unless per-token chunked masking
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = d
        ctx.N_PREFIX_Q = n_prefix_q
        ctx.BLOCK = BLOCK
        return o

    @staticmethod
    def backward(ctx, do):
        # The backward kernels are not yet chunk-mask-aware; refuse rather than return wrong grads.
        if getattr(ctx, "chunk_ids", None) is not None:
            raise NotImplementedError(
                "DocumentLandmark fused-kernel backward (per-token chunk mask) is not implemented "
                "yet; use the eager path for training."
            )
        # Fast path only supports no history KV; defer to the original backward otherwise.
        if ctx.N_PREFIX_Q != 0:
            return FusedLandmarkAttention.backward(ctx, do) + (None,)

        BLOCK = ctx.BLOCK
        q, k, v, o, lse, m = ctx.saved_tensors
        doc_id = ctx.doc_id
        doc_mask = doc_id is not None
        n_blocks = k.shape[2] // BLOCK
        doc_id_arg = doc_id if doc_mask else torch.empty(1, dtype=torch.int32, device=q.device)
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        do_scaled = torch.empty_like(do)
        delta = torch.empty_like(lse)
        _bwd_preprocess[(ctx.grid[0], ctx.grid[1])](
            o,
            o.stride(0),
            o.stride(1),
            o.stride(2),
            o.stride(3),
            do,
            lse,
            lse.stride(0),
            lse.stride(1),
            do_scaled,
            delta,
            q.shape[2],
            BLOCK_M=BLOCK,
            D_HEAD=ctx.BLOCK_DMODEL,
        )
        args = (
            q,
            k,
            v,
            ctx.sm_scale,
            o,
            do_scaled,
            dq,
            dk,
            dv,
            lse,
            m,
            delta,
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
            doc_id_arg,
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k.shape[2],
            n_blocks,
        )
        const = dict(
            BLOCK=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            N_PREFIX_Q=ctx.N_PREFIX_Q,
            DOC_MASK=doc_mask,
        )
        # head_dim > 128 needs 8 warps: the dk/dv (and dq) fp32 accumulators are (BLOCK, 256), and
        # at 4 warps they alone exceed the per-thread register budget. It also needs num_stages=1:
        # at 2 stages the pipelined (BLOCK, 256) Q/DO tiles of _bwd_kv_kernel overflow H100 shared
        # memory (279KB required vs 232KB available at BLOCK=64, fp32). <= 128 defaults untouched.
        if ctx.BLOCK_DMODEL > 128:
            warps = _env_int("LM_FAST_WARPS", 8)
            stages = _env_int("LM_FAST_STAGES", 1)
        else:
            warps = _env_int("LM_FAST_WARPS", 4)
            stages = _env_int("LM_FAST_STAGES", 2)
        n_kv_blocks = triton.cdiv(k.shape[2], BLOCK)
        _bwd_kv_kernel[(ctx.grid[1], n_kv_blocks)](
            *args, **const, num_warps=warps, num_stages=stages
        )
        _bwd_q_kernel[(ctx.grid[1], ctx.grid[0])](
            *args, **const, num_warps=warps, num_stages=stages
        )
        return dq, dk, dv, None, None, None, None, None


def fused_landmark_attention_fast(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_mem: torch.Tensor,
    sm_scale: float = None,  # type: ignore[assignment]
    block_size: int = 64,
    doc_id: Optional[torch.Tensor] = None,
    chunk_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Drop-in for ``landmark_kernel.fused_landmark_attention`` with the fast (FA2-style) backward.

    ``doc_id`` is an optional int32 ``(batch, seq_len_k // block_size)`` per-block document id for
    sequence packing (see :func:`~olmo_core.nn.attention.landmark.build_block_doc_id`); when given,
    cross-document key blocks are masked out.
    """
    expected_is_mem = torch.arange(0, is_mem.shape[-1], device=is_mem.device) % block_size == (
        block_size - 1
    )
    assert (is_mem == expected_is_mem).all()
    n_history_kv = k.shape[-2] - q.shape[-2]
    assert n_history_kv % block_size == 0
    n_history_blocks = n_history_kv // block_size
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))
    return _FusedLandmarkAttentionFast.apply(
        q, k, v, n_history_blocks, sm_scale, block_size, doc_id, chunk_ids
    )


class FastLandmarkAttention(Attention):
    """
    Landmark attention with the fast FA2-style backward -- a standalone :class:`Attention` variant
    (``AttentionType.fast_landmark``). Numerically identical to the original ``LandmarkAttention``
    (fused-kernel path), just much faster to train. Requires the fused Triton kernel (mem_freq >= 15).
    Supports Ulysses context parallelism, and the optional output gate inherited from
    :class:`Attention` (``att * sigmoid(w_g(x))``), so it drops into gated models like Qwen3.5.
    """

    def __init__(self, *, mem_freq: int, softmax_scale: Optional[float] = None, **kwargs):
        if kwargs.get("window_size") is not None:
            raise OLMoConfigurationError(
                "FastLandmarkAttention does not support sliding window attention"
            )
        super().__init__(softmax_scale=softmax_scale, **kwargs)
        if mem_freq is None or mem_freq < 15:
            raise OLMoConfigurationError(
                f"FastLandmarkAttention requires mem_freq >= 15 (got {mem_freq}); the fused kernel "
                f"tiles by block_size = mem_freq + 1 and tl.dot needs tile dims >= 16."
            )
        self.mem_freq = mem_freq
        self.block_size = mem_freq + 1
        self.softmax_scale = softmax_scale if softmax_scale is not None else self.head_dim**-0.5
        self._cp_pg: Optional[torch.distributed.ProcessGroup] = None
        self._cp_world_size: int = 1
        # Eval-decode state (set by the generation module for landmark HELMET/RULER-style eval). When
        # ``_eval_prompt_len`` is not None, the decode step treats all post-prompt positions as one
        # growing local block instead of continuing the fixed per-block structure. See
        # :meth:`set_landmark_eval_decode`.
        self._eval_prompt_len: Optional[int] = None
        self._eval_decode_mode: str = "extend_last_block"
        self._eval_top_k: Optional[int] = None

    def set_landmark_eval_decode(
        self, prompt_len: int, mode: str = "extend_last_block", top_k: Optional[int] = None
    ) -> None:
        """Enable "one long local block" decoding (see :class:`GenerationConfig.landmark_decode_mode`).

        :param prompt_len: Length of the (landmark-inserted) prompt. Generated tokens occupy absolute
            positions ``>= prompt_len`` and are never treated as landmarks.
        :param mode: ``"extend_last_block"`` or ``"generation_only"``.
        :param top_k: If set, decode uses hard top-k landmark block retrieval as in the landmark
            attention paper's inference procedure (Mohtashami & Jaggi 2023, section 3.2): each head
            scores the query against the cached landmark keys, keeps the ``top_k`` highest-scoring
            blocks, and gives every other past block exactly zero attention weight (the grouped
            softmax renormalizes over the local block plus the retrieved blocks' landmarks). ``None``
            keeps the dense soft gating over all past blocks. Prefill is unaffected either way.
        """
        if mode not in ("extend_last_block", "generation_only"):
            raise OLMoConfigurationError(
                f"Unknown landmark decode mode {mode!r} "
                "(expected 'extend_last_block' or 'generation_only')."
            )
        if top_k is not None and top_k < 1:
            raise OLMoConfigurationError(f"top_k must be >= 1 or None (got {top_k})")
        self._eval_prompt_len = prompt_len
        self._eval_decode_mode = mode
        self._eval_top_k = top_k

    def clear_landmark_eval_decode(self) -> None:
        """Disable "one long local block" decoding, restoring the default per-block decode."""
        self._eval_prompt_len = None
        self._eval_top_k = None

    def init_kv_cache_manager(self, batch_size: int, max_seq_len: int):
        # Fast landmark attention implements its own cached prefill/decode in ``_forward_generate``
        # (it does not route through the flash backend), so skip the backend KV-cache assertion.
        self.kv_cache_manager = KVCacheManager(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            device=self.w_k.weight.device,
            dtype=self.w_k.weight.dtype,  # eager decode matmuls q against the cache directly
        )

    def apply_cp(
        self,
        cp_mesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        # Ulysses only (gathers the full sequence per rank); same constraints as LandmarkAttention.
        if ring is not None:
            raise OLMoConfigurationError(
                "FastLandmarkAttention only supports Ulysses context parallelism, not ring/zigzag CP."
            )
        if uly is None:
            raise ValueError("One of 'ring' or 'uly' must be specified")
        cp_size = cp_mesh.size()
        if self.n_heads % cp_size != 0 or self.n_kv_heads % cp_size != 0:
            raise OLMoConfigurationError(
                f"Ulysses CP degree ({cp_size}) must divide n_heads ({self.n_heads}) and "
                f"n_kv_heads ({self.n_kv_heads})"
            )
        self._cp_pg = cp_mesh.get_group()
        self._cp_world_size = cp_size

    @property
    def cp_enabled(self) -> bool:
        return self._cp_pg is not None

    @torch.compiler.disable
    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if any(
            v is not None
            for v in (
                cu_doc_lens_q,
                cu_doc_lens_k,
                max_doc_len_q,
                max_doc_len_k,
                local_k_slice,
            )
        ):
            raise NotImplementedError(
                "FastLandmarkAttention supports symmetric intra-document masking via 'cu_doc_lens' "
                "only; the cross-attention variants are not supported"
            )
        # Generation path: incremental decode / prefill with a KV cache.
        if self.kv_cache_manager is not None:
            if self.cp_enabled:
                raise NotImplementedError(
                    "Context parallelism is not supported with landmark generation"
                )
            return self._forward_generate(x, pos_sin, pos_cos, freqs_cis, cache_leftpad)
        if cache_leftpad is not None:
            raise NotImplementedError(
                "cache_leftpad is only supported together with a KV cache manager"
            )

        B, T_local, _ = x.shape
        # Per-document RoPE for sequence packing. Without CP the local shard *is* the full sequence,
        # so the standard ``cu_doc_lens`` RoPE path (positions reset to 0 per document) applies. Under
        # Ulysses CP the shard is a contiguous slice of the full sequence while ``cu_doc_lens`` still
        # describes the full sequence, and RoPE runs on the slice *before* the all-to-all gather; so
        # we pass explicit per-document positions for this rank's slice -- correct even for documents
        # that straddle a rank boundary (their positions stay continuous across the boundary).
        rope_cu_doc_lens, position_ids = cu_doc_lens, None
        if cu_doc_lens is not None and self.cp_enabled:
            assert self._cp_pg is not None
            position_ids = build_local_packed_position_ids(
                cu_doc_lens, B, T_local, get_rank(self._cp_pg), self._cp_world_size
            )
            rope_cu_doc_lens = None
        q, k, v = self._prepare_qkv(
            x,
            pos_sin=pos_sin,
            pos_cos=pos_cos,
            freqs_cis=freqs_cis,
            cu_doc_lens=rope_cu_doc_lens,
            position_ids=position_ids,
        )
        if self.cp_enabled:
            assert self._cp_pg is not None
            q = all_to_all_single_cp2hp(q, self._cp_pg)
            k, v = all_to_all_cp2hp([k, v], self._cp_pg)

        T = q.shape[1]
        if T % self.block_size != 0:
            raise OLMoConfigurationError(
                f"Sequence length ({T}) must be a multiple of the landmark block size "
                f"(mem_freq + 1 = {self.block_size})."
            )

        n_rep = q.shape[2] // k.shape[2]
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), n_rep)
        v = repeat_kv(v.transpose(1, 2), n_rep)

        # Per-block document ids for sequence packing (None for the single-document path).
        doc_id = (
            build_block_doc_id(cu_doc_lens, B, T, self.block_size)
            if cu_doc_lens is not None
            else None
        )
        att = self._attn_core(q, k, v, doc_id=doc_id)

        att = att.transpose(1, 2)
        if self.cp_enabled:
            assert self._cp_pg is not None
            att = all_to_all_single_hp2cp(att.contiguous(), self._cp_pg)
        att = att.contiguous().view(B, T_local, -1)
        att = self._apply_gate(att, x)
        return self.w_out(att)

    def _attn_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        doc_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Original (hierarchical) landmark self-attention on ``(B, H, T, D)``, ``T`` a multiple of
        block_size. Requires the fused Triton kernel (CUDA). ``doc_id`` enables packing masking."""
        if not has_landmark_kernel():
            raise RuntimeError(
                "FastLandmarkAttention requires the fused Triton kernel (install 'triton', run on CUDA)."
            )
        T = q.shape[2]
        is_mem = (torch.arange(T, device=q.device) % self.block_size) == (self.block_size - 1)
        return fused_landmark_attention_fast(
            q, k, v, is_mem, sm_scale=self.softmax_scale, block_size=self.block_size, doc_id=doc_id
        )

    def _forward_generate(
        self,
        x: torch.Tensor,
        pos_sin: Optional[torch.Tensor],
        pos_cos: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
        cache_leftpad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Generation with a KV cache: single-shot prefill (T>1) or incremental decode (T==1).

        Blocks follow absolute position, so generation must be left-pad free (real tokens start at
        absolute position 0) -- i.e. ``batch_size == 1`` for these evals.
        """
        kvm = self.kv_cache_manager
        assert kvm is not None
        if cache_leftpad is not None and bool(cache_leftpad.ne(0).any()):
            raise NotImplementedError(
                "Landmark generation requires batch_size=1 / no left-padding "
                "(blocks are tied to absolute position)."
            )

        B, T, _ = x.shape
        start_pos = int(kvm.current_position())
        q, k, v = self._prepare_qkv(
            x, pos_sin=pos_sin, pos_cos=pos_cos, freqs_cis=freqs_cis, cu_doc_lens=None
        )

        kvm.k_cache[:, start_pos : start_pos + T].copy_(k)
        kvm.v_cache[:, start_pos : start_pos + T].copy_(v)
        kvm.update_seqlen(T)
        total = start_pos + T

        n_rep = q.shape[2] // k.shape[2]
        qh = q.transpose(1, 2)  # (B, H, T, D)

        if T == 1:
            kh = repeat_kv(kvm.k_cache[:, :total].transpose(1, 2), n_rep)
            vh = repeat_kv(kvm.v_cache[:, :total].transpose(1, 2), n_rep)
            att = self._decode_one(qh, kh, vh, start_pos)
        else:
            if start_pos != 0:
                raise NotImplementedError(
                    "Landmark multi-token forward with a non-empty cache is not supported "
                    "(only single-shot prefill from position 0)."
                )
            kh = repeat_kv(k.transpose(1, 2), n_rep)
            vh = repeat_kv(v.transpose(1, 2), n_rep)
            att = self._prefill(qh, kh, vh)

        att = att.transpose(1, 2).contiguous().view(B, T, -1)
        att = self._apply_gate(att, x)
        return self.w_out(att)

    def _prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Prefill over an arbitrary-length prompt: right-pad to a multiple of block_size, run the
        fused kernel, slice off the padded tail (padding is future-only, never attended causally).
        """
        T = q.shape[2]
        pad = (-T) % self.block_size
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        att = self._attn_core(q, k, v)
        return att[:, :, :T]

    def _apply_topk_landmark_retrieval(
        self, scores: torch.Tensor, is_mem: torch.Tensor
    ) -> torch.Tensor:
        """Hard top-k landmark block retrieval (the paper's inference procedure, section 3.2).

        Masks the scores of all but the ``top_k`` highest-scoring landmark keys (independently per
        batch/head) to ``-inf``. A masked landmark gets zero probability in the grouped softmax's
        top-level group, so its block's content is gated to exactly zero weight and the remaining
        mass renormalizes over the local section plus the retrieved blocks -- matching the paper's
        retrieve-then-``GroupedSoftmax`` order. Content scores are left untouched: a retrieved
        block's within-block softmax is unchanged.

        :param scores: Attention logits of shape ``(B, H, 1, total)``.
        :param is_mem: Boolean landmark-key mask of shape ``(1, 1, 1, total)``.
        """
        top_k = self._eval_top_k
        if top_k is None:
            return scores
        lm_idx = is_mem.view(-1).nonzero(as_tuple=True)[0]  # (n_lm,)
        if lm_idx.numel() <= top_k:
            return scores
        lm_scores = scores[..., lm_idx]  # (B, H, 1, n_lm)
        keep = torch.zeros_like(lm_scores, dtype=torch.bool)
        keep.scatter_(-1, lm_scores.topk(top_k, dim=-1).indices, True)
        scores = scores.clone()
        scores[..., lm_idx] = lm_scores.masked_fill(~keep, float("-inf"))
        return scores

    def _decode_one(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qpos: int
    ) -> torch.Tensor:
        """Single-query decode using the eager grouped-softmax reference (numerically matched to the
        training kernel). Query at absolute position ``qpos`` attends to cached keys ``0..total-1``
        (all <= qpos, so causal masking is implicit).

        A landmark-position query (the inserted memory token, ``qpos % block_size == block_size-1``)
        does not attend to itself: the training kernel decrements the causal bound on the last row of
        a block, so the landmark token sees only its block's content tokens ``[block_start, qpos-1]``
        plus past blocks via landmark gating. Drop the self key to match.

        In eval mode only *generated* queries (``qpos >= prompt_len``) use the "one long local block"
        decode (:meth:`_decode_one_eval`). Prompt-position queries (``qpos < prompt_len``) keep the
        per-block decode below so they reproduce prefill -- this is the path taken by the final prompt
        token, which the generation loop decodes first so hard top-k retrieval also gates the first
        generated token (prefill itself never applies top-k).
        """
        Lb = self.block_size
        if self._eval_prompt_len is not None and qpos >= self._eval_prompt_len:
            return self._decode_one_eval(q, k, v, qpos)
        if qpos % Lb == Lb - 1:
            k = k[:, :, :qpos]  # keys 0..qpos-1 (drop the landmark's own position)
            v = v[:, :, :qpos]
        total = k.shape[2]
        j = torch.arange(total, device=q.device)
        is_mem = ((j % Lb) == (Lb - 1)).view(1, 1, 1, total)
        last_section = ((j // Lb) == (qpos // Lb)).view(1, 1, 1, total)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.softmax_scale  # (B, H, 1, total)
        scores = self._apply_topk_landmark_retrieval(scores, is_mem)
        Bsz, Hn = scores.shape[0], scores.shape[1]
        probs = landmark_grouped_softmax(
            scores,
            dim=-1,
            is_mem=is_mem.expand(Bsz, Hn, 1, total),
            last_section_mask=last_section.expand(Bsz, Hn, 1, total),
        )
        return torch.matmul(probs.to(v.dtype), v)

    def _decode_one_eval(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qpos: int
    ) -> torch.Tensor:
        """Decode a generated token as part of "one long local block" (landmark eval mode).

        Generated tokens (absolute position ``>= prompt_len``) are never landmarks. They attend
        directly to every key in the growing local block ``[section_start, qpos]`` and reach earlier
        prompt blocks only through those blocks' landmark tokens. ``section_start`` is the start of
        the prompt's final block (``extend_last_block``) or the end of the prompt
        (``generation_only``). See :meth:`set_landmark_eval_decode`.
        """
        Lb = self.block_size
        P = self._eval_prompt_len
        assert P is not None
        section_start = (P // Lb) * Lb if self._eval_decode_mode == "extend_last_block" else P

        total = k.shape[2]  # = qpos + 1 (generated query attends to keys 0..qpos)
        j = torch.arange(total, device=q.device)
        # Only the prompt's landmarks (below the local block) gate access to past blocks; generated
        # positions are never landmarks.
        is_mem = (((j % Lb) == (Lb - 1)) & (j < section_start)).view(1, 1, 1, total)
        last_section = (j >= section_start).view(1, 1, 1, total)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.softmax_scale  # (B, H, 1, total)
        scores = self._apply_topk_landmark_retrieval(scores, is_mem)
        Bsz, Hn = scores.shape[0], scores.shape[1]
        probs = landmark_grouped_softmax(
            scores,
            dim=-1,
            is_mem=is_mem.expand(Bsz, Hn, 1, total),
            last_section_mask=last_section.expand(Bsz, Hn, 1, total),
        )
        return torch.matmul(probs.to(v.dtype), v)
