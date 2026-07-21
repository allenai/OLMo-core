"""
``CompressiveGQAGroupedAttention`` -- a *training-time* compressive landmark variant where each past
block's cross-block **gate** (the rescaling ``G_b`` that decides how much attention mass a block
receives) is computed from the **mean of the KV group's landmark scores** for that block, instead of
each query head's own landmark score. The within-block softmax (which token *inside* the block) and
the local/diagonal section stay per-head. It is the differentiable, trainable analogue of the
inference-only ``group_landmark_selection`` (which only re-ranked a hard top-k); here the soft gate
itself is group-shared and learned.

**Key simplification.** The block gate logit for head ``h`` is the landmark dot product
``s_b^h = q_h . k_b``. Under GQA the landmark key ``k_b`` is *shared* across a group's ``n_rep`` query
heads, so ``mean_h(q_h . k_b) = (mean_h q_h) . k_b = q̄_g . k_b``: the mean-over-heads block score is
exactly the group-mean query dotted with the (shared) landmark key. So the whole variant reduces to
feeding a second, group-averaged query ``q̄`` into the **gate** path only, keeping the per-head query
``q`` for content + local. Building ``q̄`` with plain autograd-tracked ops lets autograd distribute the
gate gradient back across the group (each head gets ``1/n_rep`` of the block's gate gradient) -- no
hand-written cross-head reduction.

**Gate-only.** Only the block rescaling is grouped. Note the block SCORE is grouped but the final gate
WEIGHT is still normalized per-head (the softmax denominator includes each head's own local scores), so
two heads in a group share the same block *score* / gate *ratio*, not the same gate *weight*.

**MHA (``n_rep == 1``) is a no-op** -- ``q̄ == q`` -- reducing bit-identically to
:class:`FastCompressiveLandmarkAttention`.
"""

from typing import Optional

import torch

from olmo_core.exceptions import OLMoConfigurationError

from .landmark import build_landmark_masks, compressive_landmark_grouped_softmax
from .landmark_compressive import _NO_OVERRIDE, FastCompressiveLandmarkAttention
from .landmark_kernel import has_landmark_kernel


class CompressiveGQAGroupedAttention(FastCompressiveLandmarkAttention):
    """
    Compressive landmark attention with GQA group-mean block gating
    (``AttentionType.compressive_gqa_grouped``). See the module docstring for the math.

    :param use_kernel: Use the fused Triton kernel (GPU) for the grouped forward/backward. Defaults to
        ``True``; falls back to the eager grouped-softmax path when the kernel is unavailable (CPU / no
        triton). The eager path is autograd-differentiable and is the reference the kernel is validated
        against, but it materializes the dense ``(T, T)`` attention matrix.
    :param decode_gate_mode: How the cross-block gate is computed at *decode* time (train/prefill always
        use the group-mean gate). Two eval variants for a grouped-trained model:

        * ``"grouped"`` (default, **Version A**): the decode gate logit at each past landmark is the
          group-mean ``q̄·k_landmark``, exactly matching the group-mean gating the model was trained
          with. With top-k retrieval the ranking is then naturally group-shared, so
          ``group_landmark_selection`` is redundant (leave it ``None``).
        * ``"selection_only"`` (**Version B**): fall back to the inherited *per-head* decode gate; use
          ``group_landmark_selection="mean"`` to make only the top-k block *selection* group-shared
          (the gate *weights* stay per-head). This does NOT match training's soft gate -- it is the
          weaker, inference-only approximation, provided for A/B comparison.

    See :class:`FastCompressiveLandmarkAttention` for ``mem_freq`` / ``nonselected_landmark_mass`` /
    ``group_landmark_selection`` and the remaining parameters.
    """

    def __init__(
        self,
        *,
        mem_freq: int,
        nonselected_landmark_mass: float = 0.1,
        softmax_scale: Optional[float] = None,
        group_landmark_selection: Optional[str] = None,
        use_kernel: bool = True,
        decode_gate_mode: str = "grouped",
        **kwargs,
    ):
        super().__init__(
            mem_freq=mem_freq,
            nonselected_landmark_mass=nonselected_landmark_mass,
            softmax_scale=softmax_scale,
            group_landmark_selection=group_landmark_selection,
            **kwargs,
        )
        self.use_kernel = use_kernel
        if decode_gate_mode not in ("grouped", "selection_only"):
            raise OLMoConfigurationError(
                "decode_gate_mode must be 'grouped' or 'selection_only' "
                f"(got {decode_gate_mode!r})"
            )
        self.decode_gate_mode = decode_gate_mode

    def _group_mean_query(self, q: torch.Tensor) -> torch.Tensor:
        """Group-mean query ``q̄`` for the gate path. ``q``: ``(B, H_local, T, D)`` (post-``repeat_kv``,
        so heads ``[g*n_rep, (g+1)*n_rep)`` share KV group ``g``). Returns the per-group mean, expanded
        back to ``H_local`` -- so every head in a group carries its group's averaged query. ``n_rep==1``
        returns ``q`` unchanged (the MHA no-op).

        Uses the *local* head count ``H_local = q.shape[1]`` (not ``self.n_heads``): under Ulysses CP
        each rank holds only ``n_heads / cp`` query heads (and ``n_kv_heads / cp`` KV heads), so the
        number of KV groups on this rank is ``H_local // n_rep``, not ``self.n_kv_heads``. The ratio
        ``n_rep`` is CP-invariant (CP divides ``n_heads`` and ``n_kv_heads`` equally), and the head
        scatter keeps whole groups contiguous on a rank, so this reshape is correct with or without CP.
        """
        n_rep = self.n_heads // self.n_kv_heads
        if n_rep == 1:
            return q
        B, H, T, D = q.shape
        n_groups = H // n_rep  # local KV-group count (== n_kv_heads without CP)
        grouped = q.view(B, n_groups, n_rep, T, D)
        gmean = grouped.mean(dim=2, keepdim=True)
        return gmean.expand(B, n_groups, n_rep, T, D).reshape(B, H, T, D)

    def _decode_gate_scores(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Group-mean gate logits for decode (Version A), matching the group-mean gate used at
        train/prefill. Returns ``q̄ @ kᵀ * scale`` when :attr:`decode_gate_mode` == ``"grouped"``;
        ``None`` (per-head gate, the inherited behavior) for ``"selection_only"`` (Version B). When
        ``n_rep == 1`` (MHA) ``q̄ == q`` so the returned scores equal ``scores`` -- an exact no-op."""
        if self.decode_gate_mode != "grouped":
            return None
        q_gate = self._group_mean_query(q)
        return torch.matmul(q_gate, k.transpose(-1, -2)) * self.softmax_scale

    def set_landmark_eval_decode(
        self,
        prompt_len: int,
        mode: str = "extend_last_block",
        top_k: Optional[int] = None,
        nonselected_landmark_mass: Optional[float] = None,
        group_landmark_selection: Optional[str] = _NO_OVERRIDE,  # type: ignore[assignment]
        decode_gate_mode: Optional[str] = None,
    ) -> None:
        """Enable eval decode (see :class:`FastCompressiveLandmarkAttention`). Adds
        ``decode_gate_mode`` (``"grouped"``/``"selection_only"``) to switch the decode gate between the
        two eval variants; ``None`` leaves :attr:`decode_gate_mode` unchanged."""
        super().set_landmark_eval_decode(
            prompt_len,
            mode,
            top_k=top_k,
            nonselected_landmark_mass=nonselected_landmark_mass,
            group_landmark_selection=group_landmark_selection,
        )
        if decode_gate_mode is not None:
            if decode_gate_mode not in ("grouped", "selection_only"):
                raise OLMoConfigurationError(
                    "decode_gate_mode must be 'grouped' or 'selection_only' "
                    f"(got {decode_gate_mode!r})"
                )
            self.decode_gate_mode = decode_gate_mode

    def _attn_core(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        doc_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_gate = self._group_mean_query(q)
        if self.use_kernel and has_landmark_kernel():
            return self._kernel_forward(q, q_gate, k, v, doc_id)
        if doc_id is not None:
            raise NotImplementedError(
                "The eager CompressiveGQAGroupedAttention path does not support sequence packing "
                "(doc_id); run on a CUDA device with the fused kernel (use_kernel=True)."
            )
        return self._eager_forward(q, q_gate, k, v)

    def _kernel_forward(
        self,
        q: torch.Tensor,
        q_gate: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        doc_id: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Filled in by the fused grouped kernel (see landmark_compressive_gqa kernel work). Until then
        # callers on GPU can force the eager path with use_kernel=False.
        from .landmark_compressive_gqa_kernel import fused_compressive_gqa_grouped_attention

        T = q.shape[2]
        is_mem = (torch.arange(T, device=q.device) % self.block_size) == (self.block_size - 1)
        return fused_compressive_gqa_grouped_attention(
            q,
            q_gate,
            k,
            v,
            is_mem,
            sm_scale=self.softmax_scale,
            block_size=self.block_size,
            doc_id=doc_id,
        )

    def _eager_forward(
        self,
        q: torch.Tensor,
        q_gate: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Eager (dense) grouped-gate compressive attention on ``(B, n_heads, T, D)``. The gate softmax
        reads group-mean landmark logits (``q_gate @ kᵀ``); the within-block and local softmaxes read
        the per-head logits (``q @ kᵀ``). Autograd-differentiable; the fused-kernel parity reference.
        """
        B, H, T, _ = q.shape
        attn_mask, is_mem, last_section_mask = build_landmark_masks(
            T, self.block_size, q.device, q.dtype
        )
        scale = self.softmax_scale
        x = torch.matmul(q, k.transpose(-1, -2)) * scale + attn_mask
        x = torch.maximum(x, torch.tensor(torch.finfo(x.dtype).min, device=x.device, dtype=x.dtype))
        # Gate logits: per-head everywhere except landmark COLUMNS, which use the group-mean query.
        # (By linearity q̄·k_landmark == mean_h(q_h·k_landmark); the additive mask is shared.)
        x_gate_full = torch.matmul(q_gate, k.transpose(-1, -2)) * scale + attn_mask
        is_mem_col = (
            (torch.arange(T, device=q.device) % self.block_size) == (self.block_size - 1)
        ).view(1, 1, 1, T)
        gate_logits = torch.where(is_mem_col, x_gate_full, x)

        probs = compressive_landmark_grouped_softmax(
            x,
            dim=-1,
            is_mem=is_mem.expand(B, H, T, T),
            last_section_mask=last_section_mask.expand(B, 1, T, T),
            gate_logits=gate_logits,
        ).to(q.dtype)
        return torch.matmul(probs, v)
