"""
``DocumentChunkedAttention`` -- standard (dense) full attention restricted by the **chunked-document
attention logic** ported from corpus-reasoning. This is the *dense* analogue of
:class:`~olmo_core.nn.attention.landmark_document.DocumentLandmarkAttention`: same per-token
``chunk_id`` roles and same :class:`~olmo_core.nn.attention.chunked_mask.AttentionPattern`, but **no
landmark mechanism** -- attention is the ordinary scaled-dot-product over whatever keys the chunked
mask allows.

    allowed = causal & not_pad & (context_ok | q_free | kv_free)

The default pattern is ``"chunked"``: **context chunks are mutually isolated**, while **FREE tokens
(the trailing query/answer, plus any instruction) bridge across chunks** and see every context chunk's
content with full attention. Other patterns (``doc_window``, ``last_token_anchor``, ``token_window``,
``random_token``, ``standard``) select different context-context connectivity.

The chunk roles reach the attention as a ``chunk_ids`` tensor threaded through the model forward
(derived from the ``<|box_start|>``/``<|box_end|>`` document-boundary *special tokens*, see
:meth:`~olmo_core.nn.transformer.model.Transformer.enable_document_chunk_attention`). When
``chunk_ids`` is absent this falls back to ordinary causal attention, so the variant is a strict
superset of :class:`Attention`.

Eager-only (the chunked mask is materialized as a dense ``(B, 1, T, T)`` additive bias and fed to
``F.scaled_dot_product_attention``) and -- like the landmark document variant -- without KV-caching,
intra-document packing (``cu_doc_lens``), or Ulysses CP while ``chunk_ids`` is active.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from olmo_core.exceptions import OLMoConfigurationError

from . import Attention
from .backend import _repeat_kv
from .chunked_mask import (
    CHUNKED_ATTENTION_PATTERNS,
    AttentionPattern,
    build_chunked_allowed_mask,
    build_chunked_mask_mod,
)

log = logging.getLogger(__name__)

# FlexAttention path (block-sparse, fused, torch.compile-friendly). Because the chunked mask is
# sparse (documents isolated), a block-sparse kernel skips fully-masked blocks -> faster than dense
# attention at long context. Only used on CUDA; CPU (incl. tests) falls back to the materialized mask.
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    _flex_attention = torch.compile(flex_attention)
    _HAS_FLEX = True
except Exception:  # pragma: no cover - flex unavailable on old torch
    _HAS_FLEX = False

# FlexAttention only wins at long context: its block-sparse kernel + create_block_mask overhead make
# it ~1.3x SLOWER than the materialized (B,1,T,T) mask at T=4096, but 3-5x FASTER at T>=16k (where the
# materialized O(T^2) mask is also a memory problem). Below this length, use the materialized path.
# (Measured on H200, qwen3-4B attn dims: crossover between 4k and 16k.)
_FLEX_MIN_SEQ_LEN = 8192


class DocumentChunkedAttention(Attention):
    """
    Dense full attention with chunked-document masking (``AttentionType.document_chunked``). See the
    module docstring.

    Subclasses :class:`Attention`; the projections, QK-norm, RoPE, output gate, and output projection
    are all inherited. The forward stashes the per-token ``chunk_ids`` so the overridden :meth:`sdpa`
    can fold the chunked allowed-mask into the attention scores.

    :param cross_doc_mode: The chunked attention pattern name (see
        :data:`~olmo_core.nn.attention.chunked_mask.CHUNKED_ATTENTION_PATTERNS`); the pluggable
        cross-document policy. Defaults to ``"chunked"``.
    :param doc_window_k: Window size for the ``"doc_window"`` pattern.
    :param token_window_w: Window width for the ``"token_window"`` pattern.

    See :class:`Attention` for the remaining parameters.
    """

    def __init__(
        self,
        *,
        cross_doc_mode: str = "chunked",
        doc_window_k: int = 0,
        token_window_w: int = 0,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ):
        if kwargs.get("window_size") is not None:
            raise OLMoConfigurationError(
                "DocumentChunkedAttention does not support sliding-window attention (the chunked "
                "mask already governs which keys a query may see)."
            )
        if cross_doc_mode not in CHUNKED_ATTENTION_PATTERNS:
            raise OLMoConfigurationError(
                f"Unknown cross_doc_mode {cross_doc_mode!r}; expected one of "
                f"{CHUNKED_ATTENTION_PATTERNS}"
            )
        # ``dropout`` is forwarded to the (unused) backend by the base class; keep our own copy for
        # the direct ``F.scaled_dot_product_attention`` call below.
        self._dropout_p = float(kwargs.get("dropout") or 0.0)
        super().__init__(softmax_scale=softmax_scale, **kwargs)
        self.cross_doc_mode = cross_doc_mode
        self._pattern = AttentionPattern(
            name=cross_doc_mode, doc_window_k=doc_window_k, token_window_w=token_window_w
        )
        # The base ``Attention`` only forwards ``softmax_scale`` to the backend; store it for our own
        # SDPA call (mirrors :class:`LandmarkAttention`).
        self.softmax_scale = softmax_scale if softmax_scale is not None else self.head_dim**-0.5
        # Transient per-forward chunk roles, stashed by ``forward`` for ``sdpa`` to read.
        self._chunk_ids: Optional[torch.Tensor] = None
        # Sticky fallback: set if FlexAttention errors at runtime (then use the dense mask). Also
        # settable by tests to force the materialized path for flex-vs-dense parity checks.
        self._force_eager_mask: bool = False

    def forward(
        self,
        x: torch.Tensor,
        chunk_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply dense document-chunked attention. ``chunk_ids`` is a per-token role tensor of shape
        ``(B, T)`` (or ``(T,)``); see the module docstring. All other arguments are forwarded to
        :meth:`Attention.forward`.
        """
        if chunk_ids is not None and self.cp_enabled:
            raise OLMoConfigurationError(
                "DocumentChunkedAttention does not support Ulysses CP together with chunk_ids "
                "(the chunked mask is built on the full local sequence)."
            )
        self._chunk_ids = chunk_ids
        try:
            return super().forward(x, **kwargs)
        finally:
            self._chunk_ids = None

    def _build_is_anchor(self, chunk_ids: torch.Tensor) -> torch.Tensor:
        """
        Per-document anchor mask for the ``"last_token_anchor"`` pattern: the last token of each
        context chunk (the analogue of corpus-reasoning's ``<|doc_end|>`` token). Derived from the
        roles (the boundary token is the chunk's last position).
        """
        is_ctx = chunk_ids >= 0
        next_diff = torch.ones_like(chunk_ids, dtype=torch.bool)
        next_diff[:, :-1] = chunk_ids[:, 1:] != chunk_ids[:, :-1]
        return is_ctx & next_diff

    def _prep_chunk_ids(self, T: int, device: torch.device, batch_size: int) -> torch.Tensor:
        """Per-token roles as ``(B, T)`` on ``device`` (validated, batch-expanded)."""
        chunk_ids = self._chunk_ids.to(device=device)
        if chunk_ids.dim() == 1:
            chunk_ids = chunk_ids.unsqueeze(0)
        if chunk_ids.shape[-1] != T:
            raise OLMoConfigurationError(
                f"chunk_ids last dim ({chunk_ids.shape[-1]}) must equal the sequence length ({T})."
            )
        if chunk_ids.shape[0] == 1 and batch_size > 1:
            chunk_ids = chunk_ids.expand(batch_size, T)
        return chunk_ids

    def _build_additive_mask(self, chunk_ids: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        """Materialize the chunked allowed-mask as a ``(B, 1, T, T)`` additive bias (0 / finfo.min)."""
        device = chunk_ids.device
        is_anchor = self._build_is_anchor(chunk_ids) if self._pattern.needs_anchor() else None
        # (B, T, T) boolean: True where the query may attend the key (causal + roles + pattern).
        allowed = build_chunked_allowed_mask(self._pattern, chunk_ids, is_anchor=is_anchor)
        finfo_min = torch.finfo(dtype).min
        return torch.where(
            allowed.unsqueeze(1),
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), finfo_min, dtype=dtype, device=device),
        )

    def sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Dense scaled-dot-product attention restricted by the chunked allowed-mask. ``q``/``k``/``v``
        arrive as ``(B, T, H, D)`` / ``(B, T, H_kv, D)`` from :meth:`Attention._prepare_qkv`; the
        result is ``(B, T, H, D)`` (the layout :meth:`Attention.forward` expects).
        """
        if any(o is not None for o in (cu_doc_lens, cu_doc_lens_q, cu_doc_lens_k, local_k_slice)):
            raise NotImplementedError(
                "DocumentChunkedAttention does not support intra-document packing (cu_doc_lens)."
            )

        kvm = self.kv_cache_manager
        if kvm is None:
            # Training / eager eval: masked attention over the full sequence (no cache).
            return self._sdpa_masked(q, k, v)

        # ---- KV-cache inference (fast eval) ----
        # Every generated token is FREE and attends ALL cached keys causally, so the chunked mask is a
        # no-op at decode: PREFILL applies the mask + populates the cache; DECODE is plain causal over
        # the cache (delegated to the base flash path). Output is identical to the no-cache path, but
        # O(gen*n^2) -> O(n^2 + gen*n).
        if q.shape[1] == 1:  # decode: single FREE query over the cache
            return Attention.sdpa(self, q, k, v, cache_leftpad=cache_leftpad)
        # Prefill: cache the prompt's (pre-GQA-expand) K,V, then compute the masked attention.
        kvm.record_leftpad(cache_leftpad)
        pos = int(kvm.current_position())
        T_q = q.shape[1]
        kvm.k_cache[:, pos : pos + T_q].copy_(k)
        kvm.v_cache[:, pos : pos + T_q].copy_(v)
        out = self._sdpa_masked(q, k, v)
        kvm.update_seqlen(T_q)
        return out

    def _sdpa_masked(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Masked attention over the full ``(B, T, H, D)`` q/k/v (no cache): plain causal when there
        are no roles, else the chunked mask via FlexAttention (long ctx) or the dense materialization."""
        B, T, n_heads, _ = q.shape
        n_rep = n_heads // k.shape[2]
        # Expand GQA kv heads, mirroring TorchAttentionBackend, then go to (B, H, T, D) for SDPA.
        k = _repeat_kv(k, n_rep)
        v = _repeat_kv(v, n_rep)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self._chunk_ids is None:
            # No roles -> plain causal attention.
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self._dropout_p if self.training else 0.0,
                is_causal=True, scale=self.softmax_scale,
            )
            return out.transpose(1, 2).contiguous()

        chunk_ids = self._prep_chunk_ids(T, q.device, B)

        # Fast path: block-sparse FlexAttention on CUDA at long context (skips fully-masked blocks ->
        # 3-5x faster than the dense (B,1,T,T) materialization at T>=16k, and far less memory). At
        # short context the materialized path is faster, so it is gated by _FLEX_MIN_SEQ_LEN.
        if _HAS_FLEX and q.is_cuda and T >= _FLEX_MIN_SEQ_LEN and not self._force_eager_mask:
            mask_mod = build_chunked_mask_mod(self._pattern, chunk_ids)
            if mask_mod is not None:
                try:
                    block_mask = create_block_mask(mask_mod, B, None, T, T, device=q.device)
                    out = _flex_attention(
                        q.contiguous(), k.contiguous(), v.contiguous(),
                        block_mask=block_mask, scale=self.softmax_scale,
                    )
                    return out.transpose(1, 2).contiguous()
                except Exception as e:  # pragma: no cover - fall back if flex fails at runtime
                    self._force_eager_mask = True
                    log.warning(
                        f"FlexAttention failed ({e}); falling back to the dense chunked mask "
                        "for this DocumentChunkedAttention layer."
                    )

        # Fallback: dense materialized additive mask (CPU/tests, unsupported pattern, or flex error).
        attn_mask = self._build_additive_mask(chunk_ids, dtype=q.dtype)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self._dropout_p if self.training else 0.0,
            is_causal=False,  # the chunked mask already encodes causality
            scale=self.softmax_scale,
        )
        return out.transpose(1, 2).contiguous()
