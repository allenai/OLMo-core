"""
.. deprecated::
    ``DilatedSlidingWindowAttention`` (``AttentionType.dilated_sliding_window``) -- the *positional*
    "Hierarchical K" rotating-dilation window -- is **deprecated and no longer supported**;
    constructing it (or selecting it via config) raises :class:`NotImplementedError`. Use the
    document-chunked hierarchical-dilated pattern instead:
    :class:`~olmo_core.nn.attention.document_chunked.DocumentChunkedAttention` with
    ``cross_doc_mode="hierarchical_dilated"``, which ports the same rotating per-layer dilation
    schedule (via ``dilation_cycle``) to document granularity with FREE tokens and document
    boundaries. The functions/classes below are retained only for reference and back-compat imports.

``DilatedSlidingWindowAttention`` -- a **dilated causal sliding-window** mask whose dilation stride
*rotates with transformer depth* ("Hierarchical K"). Each layer attends to only ``K`` keys per query
(so attention stays ``O(N*K)`` per layer), but the spacing between those keys grows geometrically as
you go up the rotation, giving direct paths to progressively longer-distance dependencies without ever
paying the dense ``O(N^2)`` cost.

Let ``d`` be the per-layer dilation stride. A query at position ``t`` attends exactly the key
positions::

    {t - i*d : i = 0 .. K-1, and t - i*d >= 0}

i.e. its own position plus the ``K-1`` predecessors spaced ``d`` apart. The stride for a layer is
chosen from its position in a length-``L`` cycle::

    cycle_pos = layer_idx % L
    d = base ** cycle_pos

so with the defaults ``K=3``, ``L=3``, ``base=2`` the rotation is (newest token on the RIGHT,
``X`` = attend, ``O`` = masked):

===============  =========  =============  =====================
Layer (cycle)    stride d   offsets        6-wide pattern
===============  =========  =============  =====================
0, 3, 6, ...     1          {0, 1, 2}      ``...OOOXXX``
1, 4, 7, ...     2          {0, 2, 4}      ``...OXOXOX``
2, 5, 8, ...     4          {0, 4, 8}      ``...OXOOOX``
===============  =========  =============  =====================

The rotation resets every ``L`` layers (layer ``L`` uses the same stride as layer ``0``). This differs
from a plain sliding window (which would attend a *contiguous* ``K`` keys at every layer): here the
receptive field of a stack of ``L`` layers spans ``~ K * base**(L-1)`` tokens while each individual
layer still touches only ``K`` keys.

The mask is purely positional (it does not depend on token content or document boundaries), so unlike
:class:`~olmo_core.nn.attention.document_chunked.DocumentChunkedAttention` it needs no ``chunk_ids``
and always applies. It subclasses :class:`~olmo_core.nn.attention.Attention`, reusing all projections,
QK-norm, RoPE, and the output gate; only :meth:`sdpa` is overridden to fold in the dilated-window
mask.

During training / eager eval the mask is applied via ``F.scaled_dot_product_attention`` (a
FlexAttention block-sparse path is used on CUDA at long context). KV-cache decoding is also supported:
because the mask depends only on the offset ``q_pos - k_pos`` it generalizes unchanged to the
incremental ``(Tq, Tk)`` decode geometry -- the new query at absolute cache position ``P`` attends the
cached keys at offsets ``{0, d, 2d, ...}`` behind it -- so autoregressive generation runs a single
masked SDPA against the KV cache each step instead of an ``O(N^2)`` full-sequence recompute (see
:meth:`DilatedSlidingWindowAttention.sdpa`).
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from olmo_core.exceptions import OLMoConfigurationError

from . import Attention
from .backend import _repeat_kv
from .kv_cache import KVCacheManager

log = logging.getLogger(__name__)

# FlexAttention path (block-sparse, fused, torch.compile-friendly). The dilated window is very sparse
# (each query sees only K keys), so a block-sparse kernel skips fully-masked blocks. Only used on CUDA;
# CPU (incl. tests) falls back to the materialized additive mask.
try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    _flex_attention = torch.compile(flex_attention)
    _HAS_FLEX = True
except Exception:  # pragma: no cover - flex unavailable on old torch
    _HAS_FLEX = False

# FlexAttention only wins at long context (mirrors DocumentChunkedAttention's crossover): below this
# length the materialized mask path is faster.
_FLEX_MIN_SEQ_LEN = 8192

# Default FlexAttention block size (granularity at which fully-masked blocks are skipped).
_FLEX_DEFAULT_BLOCK_SIZE = 128


def layer_dilation(layer_idx: int, num_configs: int, base: int) -> int:
    """
    The dilation stride for a transformer layer under the rotating "Hierarchical K" schedule.

    :param layer_idx: 0-based transformer layer index.
    :param num_configs: Cycle length ``L`` (number of distinct dilation configs before the rotation
        resets).
    :param base: Dilation base ``m``. The stride at cycle position ``p`` is ``base ** p``.

    :returns: The dilation stride ``d = base ** (layer_idx % num_configs)``.
    """
    return int(base) ** (int(layer_idx) % int(num_configs))


def build_dilated_window_allowed_mask(
    q_positions: torch.Tensor,
    kv_positions: torch.Tensor,
    *,
    dilation: int,
    window: int,
) -> torch.Tensor:
    """
    Boolean ``(Tq, Tk)`` mask, ``True`` where a query may attend a key under the dilated causal
    sliding window: ``offset = q - k`` must be non-negative, a multiple of ``dilation``, and one of the
    first ``window`` such multiples (``offset // dilation < window``). Equivalent to the key offsets
    ``{0, dilation, 2*dilation, ..., (window-1)*dilation}``.

    :param q_positions: 1-D tensor of absolute query positions, shape ``(Tq,)``.
    :param kv_positions: 1-D tensor of absolute key positions, shape ``(Tk,)``.
    :param dilation: The dilation stride ``d >= 1``.
    :param window: The number of keys ``K >= 1`` each query attends.

    :returns: A ``(Tq, Tk)`` boolean tensor.
    """
    offset = q_positions.reshape(-1, 1) - kv_positions.reshape(1, -1)
    causal = offset >= 0
    on_stride = (offset % dilation) == 0
    in_window = (offset.div(dilation, rounding_mode="floor")) < window
    return causal & on_stride & in_window


def build_dilated_window_mask_mod(dilation: int, window: int):
    """
    Build a FlexAttention ``mask_mod`` for the dilated causal sliding window (see
    :func:`build_dilated_window_allowed_mask`). The returned callable takes ``(b, h, q_idx, kv_idx)``
    and returns a boolean tensor.

    :param dilation: The dilation stride ``d >= 1``.
    :param window: The number of keys ``K >= 1`` each query attends.
    """

    def mask_mod(b, h, q_idx, kv_idx):
        offset = q_idx - kv_idx
        causal = offset >= 0
        on_stride = (offset % dilation) == 0
        in_window = (offset // dilation) < window
        return causal & on_stride & in_window

    return mask_mod


class DilatedSlidingWindowAttention(Attention):
    """
    Dilated causal sliding-window attention with a per-layer rotating dilation
    (``AttentionType.dilated_sliding_window``). See the module docstring for the schedule.

    Subclasses :class:`Attention`; the projections, QK-norm, RoPE, output gate, and output projection
    are all inherited. Only :meth:`sdpa` is overridden to restrict attention to the dilated window,
    with the stride chosen from this layer's index.

    :param window: The number of keys ``K`` each query attends per layer (default 3).
    :param num_configs: The cycle length ``L`` -- the number of distinct dilation configs before the
        rotation resets (default 3).
    :param base: The dilation base ``m``; the stride at cycle position ``p`` is ``m ** p`` (default 2).
    :param layer_idx: The 0-based transformer layer index this module lives at; selects the per-layer
        dilation stride ``m ** (layer_idx % num_configs)``.
    :param n_layers: The total number of transformer layers (stored for reference; unused by the mask).
    :param flex_block_size: FlexAttention ``create_block_mask`` block size (long-context CUDA path).

    See :class:`Attention` for the remaining parameters.
    """

    def __init__(
        self,
        *,
        window: int = 3,
        num_configs: int = 3,
        base: int = 2,
        layer_idx: int = 0,
        n_layers: int = 1,
        flex_block_size: int = _FLEX_DEFAULT_BLOCK_SIZE,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ):
        raise NotImplementedError(
            "DilatedSlidingWindowAttention (AttentionType.dilated_sliding_window, the *positional* "
            "'Hierarchical K' rotating-dilation window) is DEPRECATED and no longer supported. Use the "
            "document-chunked hierarchical-dilated pattern instead: DocumentChunkedAttention with "
            "cross_doc_mode='hierarchical_dilated' (document-granularity, rotating per-layer dilation "
            "via 'dilation_cycle', with FREE tokens and document boundaries). See "
            "olmo_core.nn.attention.chunked_mask.build_chunked_allowed_mask."
        )
        if kwargs.get("window_size") is not None:
            raise OLMoConfigurationError(
                "DilatedSlidingWindowAttention does not support the flash sliding-window "
                "'window_size' (its own dilated-window mask governs which keys a query may see)."
            )
        if window < 1:
            raise OLMoConfigurationError(f"dilated-window 'window' (K) must be >= 1 (got {window})")
        if num_configs < 1:
            raise OLMoConfigurationError(
                f"dilated-window 'num_configs' (L) must be >= 1 (got {num_configs})"
            )
        if base < 1:
            raise OLMoConfigurationError(f"dilated-window 'base' (m) must be >= 1 (got {base})")
        # ``dropout`` is forwarded to the (unused) backend by the base class; keep our own copy for the
        # direct ``F.scaled_dot_product_attention`` call below.
        self._dropout_p = float(kwargs.get("dropout") or 0.0)
        super().__init__(softmax_scale=softmax_scale, **kwargs)
        self.window = int(window)
        self.num_configs = int(num_configs)
        self.base = int(base)
        self.layer_idx = int(layer_idx)
        self.n_layers = int(n_layers)
        self.flex_block_size = int(flex_block_size)
        self.dilation = layer_dilation(self.layer_idx, self.num_configs, self.base)
        # The base ``Attention`` only forwards ``softmax_scale`` to the backend; store it for our own
        # SDPA call (mirrors :class:`DocumentChunkedAttention`).
        self.softmax_scale = softmax_scale if softmax_scale is not None else self.head_dim**-0.5
        # Sticky fallback: set if FlexAttention errors at runtime (then use the dense mask). Also
        # settable by tests to force the materialized path for flex-vs-dense parity checks.
        self._force_eager_mask: bool = False

    def init_kv_cache_manager(self, batch_size: int, max_seq_len: int):
        """
        Allocate the KV cache for dilated-window decoding.

        Unlike :meth:`Attention.init_kv_cache_manager` this does *not* require the attention backend to
        support KV caching: :meth:`sdpa` manages the cache read/write and applies the dilated-window
        mask itself (a plain masked SDPA against the cached keys), so decoding is backend-agnostic.
        (The KV-cache backend kernels only implement plain-causal / contiguous flash sliding windows,
        not this per-layer dilated window, so they cannot be reused here.)

        :param batch_size: The batch size for the cache.
        :param max_seq_len: The maximum sequence length for the cache.
        """
        self.kv_cache_manager = KVCacheManager(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            device=self.w_k.weight.device,
        )

    def _additive_mask_from_positions(
        self,
        q_positions: torch.Tensor,
        kv_positions: torch.Tensor,
        *,
        dtype: torch.dtype,
        cache_leftpad: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Materialize the dilated-window mask as an additive bias (``0`` / ``finfo.min``).

        The result is ``(1, 1, Tq, Tk)`` -- or ``(B, 1, Tq, Tk)`` when ``cache_leftpad`` has a
        non-zero entry, in which case each row additionally forbids its ``[0, leftpad[b])`` padding
        columns. ``q_positions`` / ``kv_positions`` are the *absolute* (cache-column) positions of the
        queries and keys; the dilated-window rule depends only on ``q_pos - k_pos``, so the same
        builder serves the square prefill geometry (``q_positions == kv_positions``) and the
        rectangular decode geometry (a single new query against the cached prefix).

        :param q_positions: 1-D tensor of absolute query positions, shape ``(Tq,)``.
        :param kv_positions: 1-D tensor of absolute key positions, shape ``(Tk,)``.
        :param dtype: The dtype of the additive bias (matches the attention inputs).
        :param cache_leftpad: Optional per-row left-padding lengths, shape ``(B,)``; padded key
            columns are masked out for every query in that row.
        :param batch_size: The batch size ``B`` (only used to shape the per-row pad mask).
        """
        device = q_positions.device
        # shape: (1, 1, Tq, Tk)
        allowed = (
            build_dilated_window_allowed_mask(
                q_positions, kv_positions, dilation=self.dilation, window=self.window
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        if cache_leftpad is not None and bool((cache_leftpad > 0).any()):
            # Left-padding shifts the queries and keys of a row equally, so ``q_pos - k_pos`` (and
            # thus the dilated-window pattern) is unchanged; only the padded key columns must be
            # knocked out so no query ever attends padding. shape: (B, 1, 1, Tk)
            leftpad = cache_leftpad.to(device=device).reshape(batch_size, 1, 1, 1)
            not_pad = kv_positions.reshape(1, 1, 1, -1) >= leftpad
            allowed = allowed & not_pad
        finfo_min = torch.finfo(dtype).min
        return torch.where(
            allowed,
            torch.zeros((), dtype=dtype, device=device),
            torch.full((), finfo_min, dtype=dtype, device=device),
        )

    def _build_additive_mask(
        self, T: int, device: torch.device, *, dtype: torch.dtype
    ) -> torch.Tensor:
        """Materialize the square ``(1, 1, T, T)`` dilated-window additive bias (no KV cache)."""
        positions = torch.arange(T, device=device)
        return self._additive_mask_from_positions(positions, positions, dtype=dtype)

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
        Scaled-dot-product attention restricted to this layer's dilated causal window. ``q``/``k``/``v``
        arrive as ``(B, T, H, D)`` / ``(B, T, H_kv, D)`` from :meth:`Attention._prepare_qkv`; the
        result is ``(B, T, H, D)`` (the layout :meth:`Attention.forward` expects).
        """
        if any(o is not None for o in (cu_doc_lens, cu_doc_lens_q, cu_doc_lens_k, local_k_slice)):
            raise NotImplementedError(
                "DilatedSlidingWindowAttention does not support intra-document packing (cu_doc_lens)."
            )
        # KV-cache decoding (fast eval): manage the cache and apply the dilated-window mask against
        # the cached prefix. This is O(N*K) per generated token instead of the O(N^2) full recompute.
        if self.kv_cache_manager is not None:
            return self._sdpa_with_cache(q, k, v, cache_leftpad=cache_leftpad)

        # Training / eager eval: masked attention over the full sequence (no cache).
        return self._sdpa_full_sequence(q, k, v)

    def _sdpa_full_sequence(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Dilated-window attention over the full ``(B, T, H, D)`` q/k/v (no KV cache): block-sparse
        FlexAttention on CUDA at long context, else the dense materialized additive mask."""
        B, T, n_heads, _ = q.shape
        n_rep = n_heads // k.shape[2]
        # Expand GQA kv heads, mirroring TorchAttentionBackend, then go to (B, H, T, D) for SDPA.
        k = _repeat_kv(k, n_rep)
        v = _repeat_kv(v, n_rep)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Fast path: block-sparse FlexAttention on CUDA at long context (skips fully-masked blocks).
        if _HAS_FLEX and q.is_cuda and T >= _FLEX_MIN_SEQ_LEN and not self._force_eager_mask:
            mask_mod = build_dilated_window_mask_mod(self.dilation, self.window)
            try:
                block_mask = create_block_mask(
                    mask_mod,
                    B,
                    None,
                    T,
                    T,
                    device=q.device,
                    BLOCK_SIZE=(self.flex_block_size, self.flex_block_size),
                )
                out = _flex_attention(
                    q.contiguous(),
                    k.contiguous(),
                    v.contiguous(),
                    block_mask=block_mask,
                    scale=self.softmax_scale,
                )
                return out.transpose(1, 2).contiguous()
            except Exception as e:  # pragma: no cover - fall back if flex fails at runtime
                self._force_eager_mask = True
                log.warning(
                    f"FlexAttention failed ({e}); falling back to the dense dilated-window mask "
                    "for this DilatedSlidingWindowAttention layer."
                )

        # Fallback: dense materialized additive mask (CPU/tests, short context, or flex error).
        attn_mask = self._build_additive_mask(T, q.device, dtype=q.dtype)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self._dropout_p if self.training else 0.0,
            is_causal=False,  # the dilated-window mask already encodes causality
            scale=self.softmax_scale,
        )
        return out.transpose(1, 2).contiguous()

    def _sdpa_with_cache(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cache_leftpad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Dilated-window attention with a KV cache (prefill + incremental decode).

        The new step's ``T_q`` keys/values (pre-GQA-expand, ``(B, T_q, H_kv, D)``) are written into the
        cache at the current offset ``P = current_position()``; attention then runs as a single masked
        SDPA of the ``T_q`` queries against the full cached prefix of length ``T_k = P + T_q``. The
        additive mask is built from *absolute* cache-column positions -- queries at ``P .. P+T_q-1``,
        keys at ``0 .. T_k-1`` -- so the offset-based dilated-window rule is applied correctly for both
        the one-shot prefill (``P == 0``, square) and each single-token decode step (``T_q == 1``).
        With left-padding, the recorded ``cache_leftpad`` additionally masks each row's padding columns.
        """
        kvm = self.kv_cache_manager
        assert kvm is not None
        # Persist left-padding for the whole generation (record_leftpad is a no-op when ``None``, so a
        # decode step keeps the padding recorded at prefill).
        kvm.record_leftpad(cache_leftpad)

        B, T_q, n_heads, _ = q.shape
        pos = int(kvm.current_position())
        # Append this step's K/V (pre-GQA-expand) into the cache at the current offset.
        kvm.k_cache[:, pos : pos + T_q].copy_(k)
        kvm.v_cache[:, pos : pos + T_q].copy_(v)
        T_k = pos + T_q
        k_all = kvm.k_cache[:, :T_k]
        v_all = kvm.v_cache[:, :T_k]

        # Absolute cache-column positions of the queries and of every cached key.
        q_positions = torch.arange(pos, pos + T_q, device=q.device)
        kv_positions = torch.arange(T_k, device=q.device)

        # Expand GQA kv heads, mirroring TorchAttentionBackend, then go to (B, H, T, D) for SDPA.
        n_rep = n_heads // k_all.shape[2]
        k_all = _repeat_kv(k_all, n_rep).transpose(1, 2)
        v_all = _repeat_kv(v_all, n_rep).transpose(1, 2)
        qh = q.transpose(1, 2)

        attn_mask = self._additive_mask_from_positions(
            q_positions,
            kv_positions,
            dtype=q.dtype,
            cache_leftpad=kvm.cache_leftpad,
            batch_size=B,
        )
        out = F.scaled_dot_product_attention(
            qh,
            k_all,
            v_all,
            attn_mask=attn_mask,
            dropout_p=0.0,  # decoding is always in eval mode
            is_causal=False,  # the dilated-window mask already encodes causality
            scale=self.softmax_scale,
        )
        kvm.update_seqlen(T_q)
        return out.transpose(1, 2).contiguous()
