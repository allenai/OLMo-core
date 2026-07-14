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

# Default FlexAttention block size (the granularity at which fully-masked blocks are skipped). Shrink
# it via ``flex_block_size`` to let sub-128-token chunks realize their block-sparsity.
_FLEX_DEFAULT_BLOCK_SIZE = 128

# Above this sequence length the dense materialized ``(B, 1, T, T)`` additive mask is too big to be a
# safe fallback: it needs ~``2 * B * T**2`` bytes (bf16) -- ~44 GiB at T=64k, ~176 GiB at T=128k -- so
# if FlexAttention itself OOMs at long context, materializing the dense mask would just OOM again
# (fatally). At/below this length the dense mask is small enough (<= ~2 GiB) to be a real fallback, so
# the historical <=32k behavior is preserved exactly; beyond it we raise a clear, actionable error
# instead of allocating the T x T tensor. See :meth:`DocumentChunkedAttention._sdpa_masked`.
_DENSE_MASK_MAX_SEQ_LEN = 32768

# Single-slot BlockMask cache shared across all attention layers within one forward pass. The chunked
# block mask depends only on ``chunk_ids`` (the SAME tensor threaded to every block), the sequence
# length, the pattern, and the flex block size -- all identical across layers -- so it is built once by
# the first layer and reused by the rest (and by AC recompute, which reuses the same ``chunk_ids``).
# Keyed on the chunk_ids tensor identity so a new forward (new chunk_ids) rebuilds it.
_BLOCK_MASK_CACHE: dict = {}


def _get_or_build_block_mask(mask_mod, key, *, B, T, device, block_size):
    """Return a cached :class:`BlockMask` for ``key`` or build (and cache) one. Single-slot cache."""
    cached = _BLOCK_MASK_CACHE.get("cur")
    if cached is not None and cached[0] == key:
        return cached[1]
    block_mask = create_block_mask(
        mask_mod, B, None, T, T, device=device, BLOCK_SIZE=(block_size, block_size)
    )
    _BLOCK_MASK_CACHE["cur"] = (key, block_mask)
    return block_mask


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
    :param dilation_n: Documents attended per layer for the ``"hierarchical_dilated"`` pattern.
    :param dilation_m: Dilation base for the ``"hierarchical_dilated"`` pattern. At this layer the
        stride is ``m**(layer_idx % dilation_cycle)`` (saturated within the cycle once the span covers
        all history).
    :param dilation_cycle: Rotation period ``L`` for the ``"hierarchical_dilated"`` pattern (the stride
        rotates with depth as ``layer_idx % L``). Defaults to 3.
    :param dilation_max_docs: Optional fixed saturation reference for ``"hierarchical_dilated"``
        (``None`` -> compute the cap per sequence from the actual chunk count).
    :param layer_idx: The transformer layer index this module lives at (0-based). The
        ``"hierarchical_dilated"`` pattern reads it to pick the per-layer dilation stride; other
        patterns ignore it.
    :param full_attention_layers: Optional list of layer indices that should use **plain full
        (causal) attention** instead of the chunked mask -- a hybrid where these ``N`` layers see the
        whole sequence and the rest stay document-chunked. Negative indices count from the end
        (``-1`` = last layer). When ``layer_idx`` is in this set the module drops ``chunk_ids`` and
        falls through to ordinary causal attention (identical to a ``"standard"``-pattern layer, and
        KV-cache-friendly at eval). ``None`` / empty -> every layer uses the chunked mask.

    See :class:`Attention` for the remaining parameters.
    """

    def __init__(
        self,
        *,
        cross_doc_mode: str = "chunked",
        doc_window_k: int = 0,
        token_window_w: int = 0,
        doc_keep_prob: float = 0.1,
        random_seed: int = 42,
        random_doc_per_example: bool = False,
        dilation_n: int = 2,
        dilation_m: int = 2,
        dilation_cycle: int = 3,
        dilation_max_docs: Optional[int] = None,
        flex_block_size: int = _FLEX_DEFAULT_BLOCK_SIZE,
        layer_idx: int = 0,
        n_layers: int = 1,
        full_attention_layers: Optional[list] = None,
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
        self.layer_idx = layer_idx
        self.n_layers = n_layers
        # Hybrid full/chunked: designated layers ignore the chunked mask and run plain causal
        # attention. Normalize negative indices from the end and validate against depth.
        normalized_full: set = set()
        for i in full_attention_layers or []:
            j = i + n_layers if i < 0 else i
            if not (0 <= j < n_layers):
                raise OLMoConfigurationError(
                    f"full_attention_layers index {i} is out of range for n_layers={n_layers}."
                )
            normalized_full.add(j)
        self.full_attention_layers = frozenset(normalized_full)
        self._is_full_attention_layer = layer_idx in self.full_attention_layers
        self.flex_block_size = int(flex_block_size)
        self._pattern = AttentionPattern(
            name=cross_doc_mode,
            doc_window_k=doc_window_k,
            token_window_w=token_window_w,
            doc_keep_prob=doc_keep_prob,
            random_seed=random_seed,
            random_doc_per_example=random_doc_per_example,
            dilation_n=dilation_n,
            dilation_m=dilation_m,
            dilation_cycle=dilation_cycle,
            dilation_max_docs=dilation_max_docs,
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
        # Hybrid: a full-attention layer ignores the roles entirely -> plain causal attention (and no
        # chunk_ids/CP conflict, since there are effectively no roles here).
        if self._is_full_attention_layer:
            chunk_ids = None
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
        # (B, T, T) boolean: True where the query may attend the key (causal + roles + pattern). The
        # layer index drives the per-layer dilation stride for the "hierarchical_dilated" pattern.
        allowed = build_chunked_allowed_mask(
            self._pattern, chunk_ids, is_anchor=is_anchor, layer_idx=self.layer_idx
        )
        finfo_min = torch.finfo(dtype).min
        return torch.where(
            allowed.unsqueeze(1),
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), finfo_min, dtype=dtype, device=device),
        )

    def _run_flex_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        chunk_ids: torch.Tensor,
        mask_mod,
        *,
        B: int,
        T: int,
    ) -> Optional[torch.Tensor]:
        """
        Run the block-sparse FlexAttention path once (building/reusing the shared BlockMask).

        On a CUDA OOM the failed BlockMask and any fragmentation are released
        (:func:`torch.cuda.empty_cache`) and the kernel is retried **once** -- long-context evals often
        OOM only because a slightly-too-large transient could not be carved out, and freeing the cached
        block mask recovers enough headroom to succeed. If it still OOMs, or a non-OOM flex error is
        raised, this returns ``None`` (the sticky ``_force_eager_mask`` flag is set so the rest of the
        forward skips flex) and the caller decides the fallback. Never materializes a dense mask itself.

        :returns: The attention output as ``(B, T, H, D)``, or ``None`` if flex could not run.
        """
        # Build the block mask once per forward and reuse it across all layers (they share the same
        # chunk_ids / T / pattern / block size). Key on the stashed chunk_ids identity so a new forward
        # rebuilds it.
        cids = self._chunk_ids
        key = (
            id(cids),
            cids.data_ptr(),
            cids._version,
            B,
            T,
            self.flex_block_size,
            self.cross_doc_mode,
            str(q.device),
        )
        for attempt in range(2):
            try:
                block_mask = _get_or_build_block_mask(
                    mask_mod, key, B=B, T=T, device=q.device, block_size=self.flex_block_size
                )
                out = _flex_attention(
                    q.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    block_mask=block_mask,
                    scale=self.softmax_scale,
                )
                return out.transpose(1, 2).contiguous()
            except torch.cuda.OutOfMemoryError:  # pragma: no cover - CUDA-only long-context path
                # Drop the (possibly half-built) cached block mask and free fragments, then retry once.
                _BLOCK_MASK_CACHE.pop("cur", None)
                torch.cuda.empty_cache()
                if attempt == 1:
                    self._force_eager_mask = True
                    log.warning(
                        f"FlexAttention OOM at seq_len={T} even after empty_cache; giving up on the "
                        "block-sparse path for this DocumentChunkedAttention forward."
                    )
                    return None
            except Exception as e:  # pragma: no cover - fall back if flex fails at runtime
                self._force_eager_mask = True
                log.warning(
                    f"FlexAttention failed ({e}); falling back to the dense chunked mask "
                    "for this DocumentChunkedAttention layer."
                )
                return None
        return None

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
        are no roles, else the chunked mask via FlexAttention (long ctx) or the dense materialization.
        """
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
                q,
                k,
                v,
                dropout_p=self._dropout_p if self.training else 0.0,
                is_causal=True,
                scale=self.softmax_scale,
            )
            return out.transpose(1, 2).contiguous()

        chunk_ids = self._prep_chunk_ids(T, q.device, B)

        # Fast path: block-sparse FlexAttention on CUDA at long context (skips fully-masked blocks ->
        # 3-5x faster than the dense (B,1,T,T) materialization at T>=16k, and far less memory). At
        # short context the materialized path is faster, so it is gated by _FLEX_MIN_SEQ_LEN.
        if _HAS_FLEX and q.is_cuda and T >= _FLEX_MIN_SEQ_LEN and not self._force_eager_mask:
            mask_mod = build_chunked_mask_mod(self._pattern, chunk_ids)
            if mask_mod is not None:
                out = self._run_flex_attention(q, k, v, chunk_ids, mask_mod, B=B, T=T)
                if out is not None:
                    return out
                # FlexAttention gave up (OOM even after a cache-clearing retry, or a non-OOM error).
                # At long context the dense (B,1,T,T) materialization below would allocate ~2*B*T**2
                # bytes and just OOM again fatally, so fail loudly with an actionable message instead
                # of the cryptic allocator OOM. At/below _DENSE_MASK_MAX_SEQ_LEN the dense mask is small
                # enough to be a genuine fallback (preserving the historical <=32k behavior).
                if T > _DENSE_MASK_MAX_SEQ_LEN:
                    gib = 2 * B * T * T / 2**30
                    raise RuntimeError(
                        f"DocumentChunkedAttention: FlexAttention could not run at seq_len={T} and "
                        f"the dense fallback mask would need ~{gib:.0f} GiB (B={B}). Reduce the eval "
                        f"max_length / KV-cache size to free GPU memory (the dense T x T fallback is "
                        f"only viable at seq_len <= {_DENSE_MASK_MAX_SEQ_LEN})."
                    )

        # Fallback: dense materialized additive mask (CPU/tests, unsupported pattern, or flex error at
        # short context). Guarded above so it is never reached with a fatally large T x T mask.
        attn_mask = self._build_additive_mask(chunk_ids, dtype=q.dtype)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self._dropout_p if self.training else 0.0,
            is_causal=False,  # the chunked mask already encodes causality
            scale=self.softmax_scale,
        )
        return out.transpose(1, 2).contiguous()
