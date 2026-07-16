"""
``SharedVectorLandmarkAttention`` -- a landmark-attention variant that appends a small, learned,
*block-positional* vector to every value before the attention aggregation.

Motivation
----------
Ordinary landmark attention (:class:`~olmo_core.nn.attention.landmark_fast.FastLandmarkAttention`)
gates each past block by the attention weight assigned to that block's landmark ("memory") token, but
the landmark token's own value never enters the output. This variant keeps the *attention weights*
exactly as in non-compressive fast landmark attention (same scores, same landmark-gated grouped
softmax), and only changes the **values** that those weights aggregate: each value gets a length
``vec_dim`` (default 32) tail concatenated onto it, so the per-head attention output grows from
``head_dim`` to ``head_dim + vec_dim``. The output projection is correspondingly enlarged, but stored
as *two* matrices -- the inherited ``w_out`` (over the ``head_dim`` part) plus a new ``w_out_vec``
(over the ``vec_dim`` tail), summed. Keeping ``w_out`` at its base shape lets a dense base checkpoint
warm-start into it unchanged, with only ``w_out_vec`` / ``weight_landmark`` / ``base`` newly created.

The tail is *block-specific* (identical for every key within a block):

* For a key in a **past** block ``B`` (a fully-completed block that gates via its landmark), the tail
  is ``e_B = weight_landmark_h @ v_landmark_B`` -- the block's landmark **value** vector mapped
  through a per-head, learnable ``weight_landmark`` matrix of shape ``(head_dim, vec_dim)``. This is
  a learned code for "how far back / which block this came from".
* For a key in the query's **own current block** (the local/"last" section, where the block's
  landmark has not yet come into play as a gate), the tail is ``base_h`` -- a single per-head,
  learnable ``vec_dim`` vector (think of it as a bias).

Because the tail is constant within a block, the ``vec_dim`` part of a query's output collapses to a
gate-weighted mixture of per-block codes::

    output_tail(query i) = sum_{past blocks B} gate_i(B) * e_B  +  local_mass_i * base_h

where ``gate_i(B)`` is the total attention mass query ``i`` places on block ``B`` (its landmark gate
weight) and ``local_mass_i`` is the mass on its own block. So the model reads out a soft,
gate-weighted summary of *which* blocks it attended to, in a learned ``vec_dim``-dim space.

Implementation
--------------
Two paths, selected by ``use_kernel``:

* **Eager** (``use_kernel=False``): a dense, fully-autograd reference. The ``head_dim`` output is the
  standard (non-compressive) landmark grouped-softmax output; the ``vec_dim`` tail is computed from
  the same block/gate structure. ``O(T^2)`` memory -- for testing and small models.
* **Kernel** (``use_kernel=True``, the default): the ``head_dim`` output uses the existing fused
  FA2-style landmark kernel (:func:`~olmo_core.nn.attention.landmark_fast.fused_landmark_attention_fast`),
  and the ``vec_dim`` tail is computed by :meth:`_shared_vector_tail` from the landmark-gate softmax
  over ``(past landmarks + local block)`` -- an ``O(T * n_blocks)`` side computation that never
  materializes the dense ``(T, T)`` matrix. This is the long-context training path (CUDA + triton).

Both paths call the same :meth:`_shared_vector_tail`, so the tail is validated once (against a dense
brute-force reference) and shared; only the ``head_dim`` output differs between them.

.. note::
    Generation / KV-caching, the output gate, and sliding windows are not supported by this variant.
"""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from olmo_core.distributed.parallel.context_parallel import (
    all_to_all_cp2hp,
    all_to_all_single_cp2hp,
    all_to_all_single_hp2cp,
)
from olmo_core.distributed.utils import get_rank
from olmo_core.exceptions import OLMoConfigurationError

from .landmark import (
    build_block_doc_id,
    build_landmark_masks,
    build_local_packed_position_ids,
    landmark_grouped_softmax,
    repeat_kv,
)
from .landmark_fast import FastLandmarkAttention, fused_landmark_attention_fast
from .landmark_kernel import has_landmark_kernel

__all__ = ["SharedVectorLandmarkAttention"]


class SharedVectorLandmarkAttention(FastLandmarkAttention):
    """
    Landmark attention that appends a learned, per-block positional vector to every value
    (``AttentionType.shared_vector_landmark``). See the module docstring for the math.

    :param mem_freq: Regular tokens between landmarks; the block size is ``mem_freq + 1``.
    :param vec_dim: Length of the per-block vector appended to each value (default 32). This enlarges
        the output projection: the inherited ``w_out`` still maps the ``head_dim`` attention output,
        and a *separate* ``w_out_vec`` maps the ``vec_dim`` tail, with the two summed. Keeping ``w_out``
        at its base shape lets a dense base checkpoint load into it unchanged (only ``w_out_vec``,
        ``weight_landmark`` and ``base`` are new).
    :param use_kernel: Use the fused landmark kernel for the ``head_dim`` output (default ``True``).
        Set ``False`` for the eager reference path (CPU-friendly, ``O(T^2)`` memory).

    See :class:`~olmo_core.nn.attention.landmark_fast.FastLandmarkAttention` for the remaining
    parameters.
    """

    #: Number of query positions processed per chunk in :meth:`_shared_vector_tail`, bounding the
    #: fp32 score/softmax working set to ``O(chunk * n_blocks)`` instead of ``O(seq_len * n_blocks)``.
    #: Rounded down to a whole number of blocks; a value >= ``seq_len`` reproduces the dense path.
    _tail_query_chunk: int = 4096

    def __init__(
        self,
        *,
        mem_freq: int,
        vec_dim: int = 32,
        use_kernel: bool = True,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ):
        if kwargs.get("gate") is not None:
            raise OLMoConfigurationError(
                "SharedVectorLandmarkAttention does not support the output gate (the augmented "
                "head_dim + vec_dim output would need a matching gate projection)."
            )
        super().__init__(mem_freq=mem_freq, softmax_scale=softmax_scale, **kwargs)
        if vec_dim < 1:
            raise OLMoConfigurationError(f"vec_dim must be >= 1 (got {vec_dim})")
        self.vec_dim = vec_dim
        self.use_kernel = use_kernel

        dtype = self.w_out.weight.dtype
        device = self.w_out.weight.device
        bias = self.w_out.bias is not None

        # Enlarge the output projection by a *separate* branch for the vec_dim tail (summed with the
        # inherited head_dim ``w_out``). The inherited ``w_out`` keeps its base shape so a dense base
        # checkpoint loads into it unchanged; ``w_out_vec`` is a new, zero-initialized branch.
        self.w_out_vec = nn.Linear(
            self.n_heads * vec_dim, self.d_model, bias=bias, dtype=dtype, device=device
        )
        # Per-head map from a block's landmark VALUE vector -> its vec_dim positional code.
        self.weight_landmark = nn.Parameter(
            torch.empty(self.n_heads, self.head_dim, vec_dim, dtype=dtype, device=device)
        )
        # Per-head "base" (bias) tail used for the query's own (current) block.
        self.base = nn.Parameter(torch.empty(self.n_heads, vec_dim, dtype=dtype, device=device))
        self.reset_shared_vector_parameters()

    def reset_shared_vector_parameters(self) -> None:
        """Initialize the new shared-vector parameters.

        ``weight_landmark`` gets a small normal init; ``base`` and the ``w_out_vec`` weight are
        zeroed so that at initialization the vec_dim tail contributes *nothing* to the output. This
        means a model warm-started from a dense base checkpoint reproduces the plain (non-compressive)
        landmark model exactly at step 0, and learns the positional read-out from there.
        """
        with torch.no_grad():
            self.weight_landmark.normal_(mean=0.0, std=self.head_dim**-0.5)
            self.base.zero_()
            self.w_out_vec.weight.zero_()
            if self.w_out_vec.bias is not None:
                self.w_out_vec.bias.zero_()

    def init_weights(self, **kwargs) -> None:
        # Initialize the inherited projections (w_q/w_k/w_v/w_out), then our own new params.
        super().init_weights(**kwargs)
        self.reset_shared_vector_parameters()

    # ------------------------------------------------------------------ forward

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
        """Landmark attention with the appended per-block positional vector.

        Mirrors :meth:`FastLandmarkAttention.forward` (RoPE/QK-norm via ``_prepare_qkv``, optional
        Ulysses CP, ``cu_doc_lens`` sequence packing), but the ``head_dim`` output and the
        ``vec_dim`` tail are projected by ``w_out`` and ``w_out_vec`` respectively and summed.
        KV-cached generation (prefill + decode) is supported via :meth:`_forward_generate`.
        """
        if any(
            v is not None
            for v in (
                cu_doc_lens_q,
                cu_doc_lens_k,
                max_doc_len,
                max_doc_len_q,
                max_doc_len_k,
                local_k_slice,
            )
        ):
            raise NotImplementedError(
                "SharedVectorLandmarkAttention supports symmetric intra-document masking via "
                "'cu_doc_lens' only; the cross-attention variants are not supported"
            )
        # Generation path: cached prefill / incremental decode (produces main + tail, both projected).
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
        # Per-document RoPE for sequence packing (see FastLandmarkAttention.forward for the Ulysses
        # CP subtlety: positions are computed on this rank's local shard, before the all-to-all).
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

        # Per-block document ids for sequence packing (None for the single-document path). Built
        # after the CP all-to-all so ``T`` is the full (un-sharded) sequence length, matching
        # ``cu_doc_lens``'s convention.
        doc_id = (
            build_block_doc_id(cu_doc_lens, B, T, self.block_size)
            if cu_doc_lens is not None
            else None
        )

        # shape: (B, H, T, head_dim) and (B, H, T, vec_dim)
        main = self._attn_core(q, k, v, doc_id=doc_id, cu_doc_lens=cu_doc_lens)
        tail = self._shared_vector_tail(q, k, v, doc_id=doc_id).to(main.dtype)

        # Concatenate along the head-dim so a single Ulysses all-to-all scatters the sequence back and
        # gathers the heads, then split for the two output projections.
        combined = torch.cat([main, tail], dim=-1).transpose(1, 2)  # (B, T, H, head_dim + vec_dim)
        if self.cp_enabled:
            assert self._cp_pg is not None
            combined = all_to_all_single_hp2cp(combined.contiguous(), self._cp_pg)
        combined = combined.contiguous().view(
            B, T_local, self.n_heads, self.head_dim + self.vec_dim
        )

        main_flat = combined[..., : self.head_dim].reshape(B, T_local, -1)
        tail_flat = combined[..., self.head_dim :].reshape(B, T_local, -1)
        return self.w_out(main_flat) + self.w_out_vec(tail_flat)

    def _attn_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        doc_id: Optional[torch.Tensor] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The plain (non-compressive) landmark ``head_dim`` output ``(B, H, T, head_dim)``. Overrides
        the (kernel-only) base ``_attn_core`` to add the eager path; also used by ``_prefill``.

        :param doc_id: Per-block document ids (see :func:`build_block_doc_id`), used by the fused
            kernel path.
        :param cu_doc_lens: Cumulative document lengths, used by the eager (dense) path.
        """
        if self.use_kernel:
            if not has_landmark_kernel():
                raise RuntimeError(
                    "SharedVectorLandmarkAttention(use_kernel=True) requires the fused Triton "
                    "kernel (install 'triton', run on CUDA). Use use_kernel=False for the eager path."
                )
            T = q.shape[2]
            is_mem = (torch.arange(T, device=q.device) % self.block_size) == (self.block_size - 1)
            return fused_landmark_attention_fast(
                q,
                k,
                v,
                is_mem,
                sm_scale=self.softmax_scale,
                block_size=self.block_size,
                doc_id=doc_id,
            )
        return self._main_dense(q, k, v, cu_doc_lens=cu_doc_lens)

    def _main_dense(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Eager (dense) non-compressive landmark output ``(B, H, T, head_dim)`` -- identical to
        ``LandmarkAttention._eager_forward``, optionally with ``cu_doc_lens`` block-diagonal
        document masking for sequence packing."""
        B, H, T, _ = q.shape
        attn_mask, is_mem, last_section_mask = build_landmark_masks(
            T, self.block_size, q.device, q.dtype, cu_doc_lens=cu_doc_lens, batch_size=B
        )

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.softmax_scale
        attn = attn + attn_mask
        attn = torch.maximum(
            attn, torch.tensor(torch.finfo(attn.dtype).min, device=attn.device, dtype=attn.dtype)
        )
        probs = landmark_grouped_softmax(
            attn,
            dim=-1,
            is_mem=is_mem.expand(B, H, T, T),
            last_section_mask=last_section_mask.expand(B, 1, T, T),
        ).to(q.dtype)
        return torch.matmul(probs, v)

    def _shared_vector_tail(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        doc_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the ``vec_dim`` output tail ``(B, H, T, vec_dim)`` for every query.

        The tail is ``sum_{past B} gate_i(B) e_B + local_mass_i base``. We obtain the block gate
        masses ``gate_i(B)`` and ``local_mass_i`` directly from the *landmark-gate* softmax -- a
        single softmax, per query, over ``[scores to past-block landmarks] ++ [scores to the query's
        own-block content]`` -- so we never build the dense ``(T, T)`` attention matrix. This matches
        the top-level (cross-block) softmax of the landmark grouped softmax by construction.

        :param doc_id: Optional per-block document ids ``(B, nb)`` (see :func:`build_block_doc_id`)
            for sequence packing. A past block only gates a query if it lies strictly before the
            query's block *and* shares its document id; a block is wholly inside one document by
            construction (boundaries are required to be block-aligned), so the own-block ("local")
            gate group needs no such check.
        """
        B, H, T, D = q.shape
        Lb = self.block_size
        nb = T // Lb
        scale = self.softmax_scale
        device = q.device
        neg_inf = torch.finfo(torch.float32).min

        # Under Ulysses CP the tail runs head-parallel: after the cp2hp all-to-all this rank holds
        # only H = n_heads / cp_degree heads -- the contiguous global slice [rank*H : (rank+1)*H]
        # (see all_to_all_single_cp2hp). The per-head parameters are replicated at full n_heads, so
        # select this rank's slice to match ``v``/``q``. Without CP, H == n_heads and this is a no-op.
        if self.cp_enabled:
            assert self._cp_pg is not None
            h0 = dist.get_rank(self._cp_pg) * H
            weight_landmark = self.weight_landmark[h0 : h0 + H]
            base = self.base[h0 : h0 + H]
        else:
            weight_landmark = self.weight_landmark
            base = self.base

        # Per-block landmark value/key vectors and the block codes e_B -- computed once (all cheap,
        # O(nb) not O(T)) and shared across query chunks.
        mem_pos = torch.arange(Lb - 1, T, Lb, device=device)  # (nb,)
        v_lm = v[:, :, mem_pos, :]  # (B, H, nb, D)
        k_lm = k[:, :, mem_pos, :]  # (B, H, nb, D)
        # e_B = v_landmark_B @ weight_landmark_h  -> (B, H, nb, vec_dim)
        e = torch.einsum("bhnd,hde->bhne", v_lm.float(), weight_landmark.float())
        base_t = base.float().view(1, H, 1, self.vec_dim)

        a_idx = torch.arange(Lb, device=device).view(Lb, 1)
        c_idx = torch.arange(Lb, device=device).view(1, Lb)
        local_valid = (c_idx <= a_idx) & (
            c_idx != (Lb - 1)
        )  # causal within block, drop own landmark
        block_idx = torch.arange(nb, device=device).view(1, nb)  # (1, nb)

        def _tail_chunk(
            q_c: torch.Tensor,
            k_c: torch.Tensor,
            k_lm: torch.Tensor,
            e: torch.Tensor,
            base_t: torch.Tensor,
            nb0: int,
            doc_id: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Tail ``(B, H, C, vec)`` for a contiguous, block-aligned span of ``C = mc*Lb`` queries
            starting at block ``nb0``. Each query's tail is independent, so this equals the matching
            slice of the dense computation -- we only chunk to bound the ``(C, nb)`` fp32 working set.
            """
            Bc, Hc, C, _ = q_c.shape
            mc = C // Lb

            # gate group 1: scores to each past block's landmark. (B, H, C, nb)
            sl = torch.einsum("bhid,bhnd->bhin", q_c.float(), k_lm.float()) * scale
            block_of_query = (nb0 + torch.arange(C, device=device) // Lb).view(C, 1)  # (C, 1)
            past_valid = block_idx < block_of_query  # (C, nb): block strictly before query's block
            if doc_id is not None:
                # doc id of each query's own block, broadcast to its Lb tokens: (B, C).
                query_doc = doc_id[:, nb0 : nb0 + mc].repeat_interleave(Lb, dim=1)
                same_doc = query_doc.unsqueeze(-1) == doc_id.unsqueeze(1)  # (B, C, nb)
                past_valid = past_valid.view(1, C, nb) & same_doc  # (B, C, nb)
                sl = sl.masked_fill(~past_valid.unsqueeze(1), neg_inf)
            else:
                sl = sl.masked_fill(~past_valid.view(1, 1, C, nb), neg_inf)

            # gate group 2: scores to the query's own-block content (causal, excludes the landmark).
            qb = q_c.float().reshape(Bc, Hc, mc, Lb, D)
            kb = k_c.float().reshape(Bc, Hc, mc, Lb, D)
            local = (
                torch.einsum("bhnad,bhncd->bhnac", qb, kb) * scale
            )  # (B, H, mc, Lb, Lb): a=query, c=key
            local = local.masked_fill(~local_valid.view(1, 1, 1, Lb, Lb), neg_inf)
            local = local.reshape(Bc, Hc, C, Lb)

            # Single softmax over [past landmarks (nb) ++ own-block content (Lb)] per query.
            logits = torch.cat([sl, local], dim=-1)  # (B, H, C, nb + Lb)
            logits = logits - logits.amax(dim=-1, keepdim=True)
            w = torch.softmax(logits, dim=-1)
            gate = w[..., :nb]  # (B, H, C, nb): mass on each past block
            local_mass = w[..., nb:].sum(dim=-1)  # (B, H, C)

            tail = torch.einsum("bhtn,bhne->bhte", gate, e)  # sum_B gate_B e_B
            return tail + local_mass.unsqueeze(-1) * base_t

        # Chunk over queries so the dense ``(B, H, T, nb)`` fp32 score/softmax tensors are never
        # materialized (peak drops from O(T*nb) to O(chunk*nb)). Chunks are block-aligned so each
        # query's own block lies wholly within its chunk. Under autograd, recompute each chunk in
        # backward (checkpoint) so the intermediates aren't retained across the whole sequence.
        chunk_blocks = max(1, self._tail_query_chunk // Lb)
        use_ckpt = torch.is_grad_enabled() and (
            q.requires_grad or k.requires_grad or v.requires_grad
        )
        parts = []
        for nb0 in range(0, nb, chunk_blocks):
            nb1 = min(nb0 + chunk_blocks, nb)
            t0, t1 = nb0 * Lb, nb1 * Lb
            q_c, k_c = q[:, :, t0:t1], k[:, :, t0:t1]
            if use_ckpt:
                part = checkpoint(
                    _tail_chunk, q_c, k_c, k_lm, e, base_t, nb0, doc_id, use_reentrant=False
                )
            else:
                part = _tail_chunk(q_c, k_c, k_lm, e, base_t, nb0, doc_id)
            parts.append(part)
        return torch.cat(parts, dim=2)

    # ------------------------------------------------------------------ generation

    def _forward_generate(
        self,
        x: torch.Tensor,
        pos_sin: Optional[torch.Tensor],
        pos_cos: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
        cache_leftpad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """KV-cached generation: single-shot prefill (T>1) or incremental decode (T==1).

        Mirrors :meth:`FastLandmarkAttention._forward_generate` but additionally computes the
        ``vec_dim`` tail and projects the two branches (``w_out(main) + w_out_vec(tail)``). As in the
        base, blocks follow absolute position, so generation must be left-pad free (``batch_size==1``).
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
            probs, v_used, section_start = self._decode_probs(qh, kh, vh, start_pos)
            main = torch.matmul(probs.to(v_used.dtype), v_used)  # (B, H, 1, head_dim)
            tail = self._decode_tail(probs, v_used, section_start)  # (B, H, 1, vec_dim)
        else:
            if start_pos != 0:
                raise NotImplementedError(
                    "Landmark multi-token forward with a non-empty cache is not supported "
                    "(only single-shot prefill from position 0)."
                )
            kh = repeat_kv(k.transpose(1, 2), n_rep)
            vh = repeat_kv(v.transpose(1, 2), n_rep)
            main = self._prefill(qh, kh, vh)  # (B, H, T, head_dim)
            # _shared_vector_tail computes nb = T // block_size and silently drops any partial
            # trailing block, so an arbitrary-length prompt must be right-padded to a block-aligned
            # length first (mirroring FastLandmarkAttention._prefill) and sliced back to T after.
            pad = (-T) % self.block_size
            if pad:
                qp = F.pad(qh, (0, 0, 0, pad))
                kp = F.pad(kh, (0, 0, 0, pad))
                vp = F.pad(vh, (0, 0, 0, pad))
            else:
                qp, kp, vp = qh, kh, vh
            tail = self._shared_vector_tail(qp, kp, vp)[:, :, :T]  # (B, H, T, vec_dim)

        main_flat = main.transpose(1, 2).contiguous().view(B, T, -1)
        tail_flat = tail.to(main.dtype).transpose(1, 2).contiguous().view(B, T, -1)
        return self.w_out(main_flat) + self.w_out_vec(tail_flat)

    def _decode_tail(
        self, probs: torch.Tensor, v_used: torch.Tensor, section_start: int
    ) -> torch.Tensor:
        """The ``vec_dim`` tail for a single decode query, from its grouped-softmax probabilities.

        ``tail = sum_{past B} mass(B) e_B + local_mass base``, where ``mass(B)`` is the total decode
        probability on past block ``B`` (its content, summed; the landmark itself carries 0 in the
        non-compressive softmax) and ``e_B = weight_landmark @ v_landmark_B``. This is the single-query
        analogue of :meth:`_shared_vector_tail`, reusing the decode probabilities so it tracks the
        per-block / eval "one long local block" / top-k decode structure exactly.

        :param probs: Decode probabilities ``(B, H, 1, total)`` from :meth:`_decode_probs`.
        :param v_used: The (possibly self-sliced) cached values ``(B, H, total, D)``.
        :param section_start: Start of the local section; the past region ``[0, section_start)``
            partitions into whole landmark blocks.
        """
        B, H = probs.shape[0], probs.shape[1]
        Lb = self.block_size
        S = section_start
        vec = self.vec_dim
        wl = self.weight_landmark.float()
        if S > 0:
            nb = S // Lb
            lm_pos = torch.arange(Lb - 1, S, Lb, device=probs.device)
            v_lm = v_used[:, :, lm_pos, :].float()  # (B, H, nb, D)
            e = torch.einsum("bhnd,hde->bhne", v_lm, wl)  # (B, H, nb, vec)
            block_mass = probs[..., :S].reshape(B, H, 1, nb, Lb).sum(-1).float()  # (B, H, 1, nb)
            tail = torch.einsum("bhqn,bhne->bhqe", block_mass, e)  # (B, H, 1, vec)
        else:
            tail = torch.zeros(B, H, 1, vec, device=probs.device, dtype=torch.float32)
        local_mass = probs[..., S:].sum(-1).float()  # (B, H, 1)
        tail = tail + local_mass.unsqueeze(-1) * self.base.float().view(1, H, 1, vec)
        return tail

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"mem_freq={self.mem_freq}, vec_dim={self.vec_dim}, use_kernel={self.use_kernel}"
