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
import torch.nn as nn

from olmo_core.distributed.parallel.context_parallel import (
    all_to_all_cp2hp,
    all_to_all_single_cp2hp,
    all_to_all_single_hp2cp,
)
from olmo_core.exceptions import OLMoConfigurationError

from .landmark import build_landmark_masks, landmark_grouped_softmax, repeat_kv
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
        Ulysses CP), but the ``head_dim`` output and the ``vec_dim`` tail are projected by ``w_out``
        and ``w_out_vec`` respectively and summed. Generation / KV-caching is not supported.
        """
        if any(
            v is not None
            for v in (
                cu_doc_lens,
                cu_doc_lens_q,
                cu_doc_lens_k,
                max_doc_len,
                max_doc_len_q,
                max_doc_len_k,
                local_k_slice,
            )
        ):
            raise NotImplementedError(
                "Intra-document masking (cu_doc_lens) is not supported with landmark attention"
            )
        if cache_leftpad is not None or self.kv_cache_manager is not None:
            raise NotImplementedError(
                "Generation / KV-caching is not supported with SharedVectorLandmarkAttention"
            )

        B, T_local, _ = x.shape
        q, k, v = self._prepare_qkv(
            x, pos_sin=pos_sin, pos_cos=pos_cos, freqs_cis=freqs_cis, cu_doc_lens=None
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

        # shape: (B, H, T, head_dim) and (B, H, T, vec_dim)
        main = self._main(q, k, v)
        tail = self._shared_vector_tail(q, k, v).to(main.dtype)

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

    def _main(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """The plain (non-compressive) landmark ``head_dim`` output ``(B, H, T, head_dim)``."""
        if self.use_kernel:
            if not has_landmark_kernel():
                raise RuntimeError(
                    "SharedVectorLandmarkAttention(use_kernel=True) requires the fused Triton "
                    "kernel (install 'triton', run on CUDA). Use use_kernel=False for the eager path."
                )
            T = q.shape[2]
            is_mem = (torch.arange(T, device=q.device) % self.block_size) == (self.block_size - 1)
            return fused_landmark_attention_fast(
                q, k, v, is_mem, sm_scale=self.softmax_scale, block_size=self.block_size
            )
        return self._main_dense(q, k, v)

    def _main_dense(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Eager (dense) non-compressive landmark output ``(B, H, T, head_dim)`` -- identical to
        ``LandmarkAttention._eager_forward``."""
        B, H, T, _ = q.shape
        attn_mask, is_mem, last_section_mask = build_landmark_masks(
            T, self.block_size, q.device, q.dtype
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
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Compute the ``vec_dim`` output tail ``(B, H, T, vec_dim)`` for every query.

        The tail is ``sum_{past B} gate_i(B) e_B + local_mass_i base``. We obtain the block gate
        masses ``gate_i(B)`` and ``local_mass_i`` directly from the *landmark-gate* softmax -- a
        single softmax, per query, over ``[scores to past-block landmarks] ++ [scores to the query's
        own-block content]`` -- so we never build the dense ``(T, T)`` attention matrix. This matches
        the top-level (cross-block) softmax of the landmark grouped softmax by construction.
        """
        B, H, T, D = q.shape
        Lb = self.block_size
        nb = T // Lb
        scale = self.softmax_scale
        device = q.device

        # Per-block landmark value vectors and their vec_dim codes e_B.
        mem_pos = torch.arange(Lb - 1, T, Lb, device=device)  # (nb,)
        v_lm = v[:, :, mem_pos, :]  # (B, H, nb, D)
        # e_B = v_landmark_B @ weight_landmark_h  -> (B, H, nb, vec_dim)
        e = torch.einsum("bhnd,hde->bhne", v_lm.float(), self.weight_landmark.float())

        # --- gate group 1: scores to each past block's landmark. (B, H, T, nb) ---
        k_lm = k[:, :, mem_pos, :]  # (B, H, nb, D)
        sl = torch.einsum("bhid,bhnd->bhin", q.float(), k_lm.float()) * scale
        block_of_query = (torch.arange(T, device=device) // Lb).view(T, 1)  # (T, 1)
        block_idx = torch.arange(nb, device=device).view(1, nb)  # (1, nb)
        past_valid = block_idx < block_of_query  # (T, nb): block is strictly before query's block
        neg_inf = torch.finfo(torch.float32).min
        sl = sl.masked_fill(~past_valid.view(1, 1, T, nb), neg_inf)

        # --- gate group 2: scores to the query's own-block content (causal, excludes the landmark).
        qb = q.view(B, H, nb, Lb, D).float()
        kb = k.view(B, H, nb, Lb, D).float()
        local = (
            torch.einsum("bhnad,bhncd->bhnac", qb, kb) * scale
        )  # (B, H, nb, Lb, Lb): a=query, c=key
        a_idx = torch.arange(Lb, device=device).view(Lb, 1)
        c_idx = torch.arange(Lb, device=device).view(1, Lb)
        local_valid = (c_idx <= a_idx) & (
            c_idx != (Lb - 1)
        )  # causal within block, drop own landmark
        local = local.masked_fill(~local_valid.view(1, 1, 1, Lb, Lb), neg_inf)
        local = local.reshape(B, H, T, Lb)  # row i=(n,a) -> its own block's content scores

        # Single softmax over [past landmarks (nb) ++ own-block content (Lb)] per query.
        logits = torch.cat([sl, local], dim=-1)  # (B, H, T, nb + Lb)
        logits = logits - logits.amax(dim=-1, keepdim=True)
        w = torch.softmax(logits, dim=-1)
        gate = w[..., :nb]  # (B, H, T, nb): mass on each past block
        local_mass = w[..., nb:].sum(dim=-1)  # (B, H, T)

        tail = torch.einsum("bhtn,bhne->bhte", gate, e)  # sum_B gate_B e_B
        tail = tail + local_mass.unsqueeze(-1) * self.base.float().view(1, H, 1, self.vec_dim)
        return tail

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"mem_freq={self.mem_freq}, vec_dim={self.vec_dim}, use_kernel={self.use_kernel}"
