"""
``DocumentLandmarkAttention`` -- OLMo-core landmark attention (grouped two-level softmax) combined
with the **chunked-document attention logic** ported from corpus-reasoning.

The landmark *mechanism* is unchanged from :class:`LandmarkAttention`: within whatever set of keys a
query is allowed to see, attention is the grouped two-level softmax (a query attends to its own
block fully and to earlier blocks gated by their landmark tokens). What changes is **which keys a
query may see**, which is now governed by a :class:`~olmo_core.nn.attention.chunked_mask.AttentionPattern`
over per-token ``chunk_id`` roles (``>= 0`` context-chunk index; ``-1`` FREE = query/answer/
instruction; ``-2`` PAD; ``-3`` SINK), exactly as in corpus-reasoning ``chunked_attention``:

    allowed = causal & not_pad & (context_ok | q_free | kv_free)

The default pattern is ``"chunked"``: **context chunks are mutually isolated**, while **FREE tokens
(the trailing query/answer, plus any instruction) bridge across chunks** -- and because the bridge
runs through the landmark grouped softmax, the query/answer retrieves each earlier chunk's content
gated by that chunk's landmarks. Other patterns (``doc_window``, ``last_token_anchor``,
``token_window``, ``random_token``, ``standard``) select different context-context connectivity and
are the pluggable cross-document policy.

The chunk roles reach the attention as a ``chunk_ids`` tensor threaded through the model forward
(derived from ``<|doc_start|>``/``<|doc_end|>`` boundary tokens, mirroring corpus-reasoning's runtime
reconstruction). When ``chunk_ids`` is absent this falls back to normal landmark behaviour
(optionally with ``cu_doc_lens`` block-diagonal masking).

Eager-only and (for now) Ulysses-CP-free with chunk masking: the chunked mask is built on the full
local sequence, so context parallelism with ``chunk_ids`` is rejected until the mask is sharded.
"""

from typing import Optional, Tuple

import torch

from olmo_core.exceptions import OLMoConfigurationError

from . import LandmarkAttention
from .chunked_mask import CHUNKED_ATTENTION_PATTERNS, AttentionPattern, build_chunked_allowed_mask
from .landmark_kernel import has_landmark_kernel


class DocumentLandmarkAttention(LandmarkAttention):
    """
    Landmark attention with chunked-document masking (``AttentionType.document_landmark``). See the
    module docstring.

    Subclasses :class:`LandmarkAttention`; the projections, QK-norm, RoPE, output gate, and the eager
    grouped softmax are all inherited. The forward stashes the per-token ``chunk_ids`` so the
    overridden :meth:`_landmark_masks` can fold the chunked allowed-mask into the grouped-softmax
    additive mask.

    :param cross_doc_mode: The chunked attention pattern name (see
        :data:`~olmo_core.nn.attention.chunked_mask.CHUNKED_ATTENTION_PATTERNS`); the pluggable
        cross-document policy. Defaults to ``"chunked"``.
    :param doc_window_k: Window size for the ``"doc_window"`` pattern.
    :param token_window_w: Window width for the ``"token_window"`` pattern.

    See :class:`LandmarkAttention` / :class:`Attention` for the remaining parameters.
    """

    def __init__(
        self,
        *,
        mem_freq: int,
        cross_doc_mode: str = "chunked",
        doc_window_k: int = 0,
        token_window_w: int = 0,
        use_kernel: bool = False,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ):
        if cross_doc_mode not in CHUNKED_ATTENTION_PATTERNS:
            raise OLMoConfigurationError(
                f"Unknown cross_doc_mode {cross_doc_mode!r}; expected one of "
                f"{CHUNKED_ATTENTION_PATTERNS}"
            )
        # The base stays on the eager DISPATCH (use_kernel=False); when ``use_kernel`` is requested we
        # route to the FAST fused kernel with the per-token chunk mask from our ``_eager_forward``
        # override (validated numerically identical to the eager grouped softmax incl. gradients).
        super().__init__(
            mem_freq=mem_freq, use_kernel=False, softmax_scale=softmax_scale, **kwargs
        )
        self._use_chunk_kernel = bool(use_kernel)
        self.cross_doc_mode = cross_doc_mode
        self._pattern = AttentionPattern(
            name=cross_doc_mode, doc_window_k=doc_window_k, token_window_w=token_window_w
        )
        # Transient per-forward chunk roles, stashed by ``forward`` for ``_landmark_masks`` to read.
        self._chunk_ids: Optional[torch.Tensor] = None

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Route to the fast fused landmark kernel (per-token chunk mask) when enabled, else the eager
        grouped softmax. The kernel is numerically identical (fwd + grad) to the eager path and ~5x
        faster to train; it falls back to eager on CPU, without ``chunk_ids``, during top-k eval, or
        for ``block_size < 16`` (the kernel's ``tl.dot`` tile constraint).
        """
        if (
            self._use_chunk_kernel
            and self._chunk_ids is not None
            and self._eval_top_k is None
            and q.is_cuda
            and self.block_size >= 16
            and has_landmark_kernel()
        ):
            from .landmark_fast import fused_landmark_attention_fast

            T = q.shape[2]
            is_mem = (torch.arange(T, device=q.device) % self.block_size) == (self.block_size - 1)
            chunk_ids = self._chunk_ids
            if chunk_ids.dim() == 1:
                chunk_ids = chunk_ids.unsqueeze(0)
            if chunk_ids.shape[0] == 1 and q.shape[0] > 1:
                chunk_ids = chunk_ids.expand(q.shape[0], T)
            return fused_landmark_attention_fast(
                q, k, v, is_mem, self.softmax_scale, self.block_size, chunk_ids=chunk_ids
            )
        return super()._eager_forward(q, k, v, cu_doc_lens=cu_doc_lens)

    def forward(
        self,
        x: torch.Tensor,
        chunk_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply document-chunked landmark attention. ``chunk_ids`` is a per-token role tensor of shape
        ``(B, T)`` (or ``(T,)``); see the module docstring. All other arguments are forwarded to
        :meth:`LandmarkAttention.forward`.
        """
        if chunk_ids is not None and self.cp_enabled:
            raise OLMoConfigurationError(
                "DocumentLandmarkAttention does not support Ulysses CP together with chunk_ids yet "
                "(the chunked mask is built on the full local sequence)."
            )
        self._chunk_ids = chunk_ids
        try:
            return super().forward(x, **kwargs)
        finally:
            self._chunk_ids = None

    def _build_is_anchor(self, chunk_ids: torch.Tensor) -> torch.Tensor:
        """
        Derive the per-document anchor mask for the ``"last_token_anchor"`` pattern: the last token of
        each context chunk (the OLMo-core analogue of corpus-reasoning's ``<|doc_end|>`` token).

        :param chunk_ids: ``(B, T)`` per-token roles.
        :returns: ``(B, T)`` bool, ``True`` at the last position of each context chunk.
        """
        is_ctx = chunk_ids >= 0
        # A context token is a chunk's last token if the next token leaves the chunk (different id or
        # end of sequence).
        next_diff = torch.ones_like(chunk_ids, dtype=torch.bool)
        next_diff[:, :-1] = chunk_ids[:, 1:] != chunk_ids[:, :-1]
        return is_ctx & next_diff

    def _landmark_masks(
        self,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
        cu_doc_lens: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build the grouped-softmax masks, restricting visibility by the chunked-attention pattern.

        When no ``chunk_ids`` were provided to :meth:`forward`, defers to
        :meth:`LandmarkAttention._landmark_masks` (normal landmark, optionally with ``cu_doc_lens``
        block-diagonal masking). Otherwise the per-token chunked allowed-mask is folded into the
        additive mask before computing ``is_mem`` / ``last_section_mask`` for the grouped softmax.
        """
        chunk_ids = self._chunk_ids
        if chunk_ids is None:
            return super()._landmark_masks(
                T, device, dtype, cu_doc_lens=cu_doc_lens, batch_size=batch_size
            )

        block_size = self.block_size
        finfo_min = torch.finfo(dtype).min

        chunk_ids = chunk_ids.to(device=device)
        if chunk_ids.dim() == 1:
            chunk_ids = chunk_ids.unsqueeze(0)
        if chunk_ids.shape[-1] != T:
            raise OLMoConfigurationError(
                f"chunk_ids last dim ({chunk_ids.shape[-1]}) must equal the sequence length ({T})."
            )
        if chunk_ids.shape[0] == 1 and batch_size > 1:
            chunk_ids = chunk_ids.expand(batch_size, T)

        is_anchor = self._build_is_anchor(chunk_ids) if self._pattern.needs_anchor() else None
        # (B, T, T) boolean: True where the query may attend the key (causal + roles + pattern).
        allowed = build_chunked_allowed_mask(self._pattern, chunk_ids, is_anchor=is_anchor)
        # Additive mask (B, 1, T, T): 0 where allowed, -inf where masked.
        attn_mask = torch.where(
            allowed.unsqueeze(1),
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), finfo_min, dtype=dtype, device=device),
        )

        # Periodic landmark (block-end) positions, shape (1, 1, 1, T).
        is_mem = ((torch.arange(T, device=device) % block_size) == (block_size - 1)).view(
            1, 1, 1, T
        )
        mem_ids = torch.where(attn_mask < -1, -1, torch.cumsum(is_mem, -1) - is_mem.int())
        last_section_mask = torch.amax(mem_ids, -1, keepdim=True) == mem_ids
        # Mask landmark tokens that fall in the query's own (last) section.
        attn_mask.masked_fill_(last_section_mask & is_mem, finfo_min)
        last_section_mask = last_section_mask.logical_and(attn_mask > -1)
        is_mem = is_mem.logical_and(attn_mask > -1)

        return attn_mask, is_mem, last_section_mask
