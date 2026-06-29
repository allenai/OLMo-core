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
from .chunked_mask import (
    CHUNKED_ATTENTION_PATTERNS,
    AttentionPattern,
    build_chunked_allowed_mask,
)
from .landmark import repeat_kv
from .landmark_fast import FastLandmarkAttention
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
        cross-document policy. Defaults to ``"chunked"``. ``cross_doc_mode`` is purely a cross-document
        *visibility* policy and is orthogonal to the landmark mechanism, so any pattern works --
        including the layer-dependent ``"hierarchical_dilated"`` (its dilated allowed-mask simply feeds
        the landmark grouped softmax instead of dense SDPA).
    :param doc_window_k: Window size for the ``"doc_window"`` pattern.
    :param token_window_w: Window width for the ``"token_window"`` pattern.
    :param dilation_n: Documents attended per layer for the ``"hierarchical_dilated"`` pattern.
    :param dilation_m: Dilation base for the ``"hierarchical_dilated"`` pattern (stride
        ``m**layer_idx``, saturated once the span covers all history).
    :param dilation_max_docs: Optional fixed saturation reference for ``"hierarchical_dilated"``
        (``None`` -> compute the cap per sequence from the actual chunk count).
    :param layer_idx: The transformer layer index this module lives at (0-based). The
        ``"hierarchical_dilated"`` pattern reads it to pick the per-layer dilation stride; other
        patterns ignore it.

    See :class:`LandmarkAttention` / :class:`Attention` for the remaining parameters.

    **Generation / KV cache.** Training stays the eager chunked grouped softmax (or the fused kernel);
    only an *additional* cached generation path is provided for fast eval. At PREFILL the prompt runs
    through the same chunked grouped-softmax forward (so the document mask is applied) while K,V are
    written into the :class:`~olmo_core.nn.attention.kv_cache.KVCacheManager`. At DECODE every generated
    token is FREE (``chunk_id == -1``) and attends *all* past blocks gated by their landmarks, so the
    chunked mask is a no-op -- decode reuses :class:`FastLandmarkAttention`'s incremental landmark
    decode (:meth:`~FastLandmarkAttention._decode_one`, hard top-k included), identical to a plain
    landmark decode. This turns eval from ``O(gen * T^2)`` eager re-feeding into ``O(T^2 + gen * T)``.
    """

    # Reuse FastLandmarkAttention's cached-decode machinery verbatim: the chunked mask is irrelevant at
    # decode (every generated query is FREE), so the incremental landmark decode is identical. These are
    # bound as plain functions so ``self`` resolves to the DocumentLandmarkAttention instance. Note the
    # decode top-k helper is the FastLandmark ``(scores, is_mem)`` variant under a distinct name -- the
    # inherited ``LandmarkAttention._apply_topk_landmark_retrieval(attn, T)`` is left untouched for the
    # eager prefill path.
    init_kv_cache_manager = FastLandmarkAttention.init_kv_cache_manager
    set_landmark_eval_decode = FastLandmarkAttention.set_landmark_eval_decode
    clear_landmark_eval_decode = FastLandmarkAttention.clear_landmark_eval_decode
    _decode_one = FastLandmarkAttention._decode_one
    _decode_one_eval = FastLandmarkAttention._decode_one_eval
    _decode_apply_topk_landmark_retrieval = (
        FastLandmarkAttention._decode_apply_topk_landmark_retrieval
    )

    def __init__(
        self,
        *,
        mem_freq: int,
        cross_doc_mode: str = "chunked",
        doc_window_k: int = 0,
        token_window_w: int = 0,
        dilation_n: int = 2,
        dilation_m: int = 2,
        dilation_max_docs: Optional[int] = None,
        layer_idx: int = 0,
        n_layers: int = 1,
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
        super().__init__(mem_freq=mem_freq, use_kernel=False, softmax_scale=softmax_scale, **kwargs)
        self._use_chunk_kernel = bool(use_kernel)
        self.cross_doc_mode = cross_doc_mode
        # The layer index drives the per-layer dilation stride for the "hierarchical_dilated" pattern;
        # all other patterns ignore it (so the default "chunked" behaviour is unchanged).
        self.layer_idx = layer_idx
        self.n_layers = n_layers
        self._pattern = AttentionPattern(
            name=cross_doc_mode,
            doc_window_k=doc_window_k,
            token_window_w=token_window_w,
            dilation_n=dilation_n,
            dilation_m=dilation_m,
            dilation_max_docs=dilation_max_docs,
        )
        # Transient per-forward chunk roles, stashed by ``forward`` for ``_landmark_masks`` to read.
        self._chunk_ids: Optional[torch.Tensor] = None
        # Eval-decode state for the reused FastLandmark decode (see ``set_landmark_eval_decode``). When
        # ``_eval_prompt_len`` is None the decode uses the per-block path (``_decode_one``), which
        # reproduces the periodic-landmark prefill structure step by step. ``_eval_top_k`` is inherited
        # from :class:`LandmarkAttention`.
        self._eval_prompt_len: Optional[int] = None
        self._eval_decode_mode: str = "extend_last_block"

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
            and self.cross_doc_mode == "chunked"
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
            # Cached generation path (fast eval): chunked-masked prefill + plain landmark decode.
            if self.kv_cache_manager is not None:
                if self.cp_enabled:
                    raise NotImplementedError(
                        "Context parallelism is not supported with landmark generation"
                    )
                return self._forward_generate(
                    x,
                    pos_sin=kwargs.get("pos_sin"),
                    pos_cos=kwargs.get("pos_cos"),
                    freqs_cis=kwargs.get("freqs_cis"),
                    cache_leftpad=kwargs.get("cache_leftpad"),
                )
            return super().forward(x, **kwargs)
        finally:
            self._chunk_ids = None

    def _forward_generate(
        self,
        x: torch.Tensor,
        pos_sin: Optional[torch.Tensor],
        pos_cos: Optional[torch.Tensor],
        freqs_cis: Optional[torch.Tensor],
        cache_leftpad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Cached generation: single-shot **prefill** (``T > 1``) or incremental **decode** (``T == 1``).

        Prefill runs the chunked grouped-softmax forward (:meth:`_eager_forward`, which reads the
        stashed ``self._chunk_ids``), so the document mask is applied exactly as in training, and also
        writes the prompt's pre-GQA-expand K,V into the cache. Decode reuses :class:`FastLandmarkAttention`'s
        :meth:`~FastLandmarkAttention._decode_one`: the single generated query is FREE, so it attends
        all cached keys gated only by the landmarks (the chunked mask is a no-op) and the output is
        identical to re-feeding the whole growing sequence through the eager forward.

        Blocks follow absolute position, so generation must be left-pad free (real tokens start at
        absolute position 0) -- i.e. ``batch_size == 1`` for these evals.
        """
        kvm = self.kv_cache_manager
        assert kvm is not None
        if cache_leftpad is not None and bool(cache_leftpad.ne(0).any()):
            raise NotImplementedError(
                "DocumentLandmark generation requires batch_size=1 / no left-padding "
                "(blocks are tied to absolute position)."
            )

        B, T, _ = x.shape
        start_pos = int(kvm.current_position())
        # ``_prepare_qkv`` reads ``kvm.current_position()`` for the RoPE offset, so it must run before
        # ``update_seqlen``. Single-document path (no intra-prompt packing during generation).
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
            # Decode: plain landmark decode over the full cache (chunk mask irrelevant for a FREE query).
            kh = repeat_kv(kvm.k_cache[:, :total].transpose(1, 2), n_rep)
            vh = repeat_kv(kvm.v_cache[:, :total].transpose(1, 2), n_rep)
            att = self._decode_one(qh, kh, vh, start_pos)
        else:
            if start_pos != 0:
                raise NotImplementedError(
                    "DocumentLandmark multi-token forward with a non-empty cache is not supported "
                    "(only single-shot prefill from position 0)."
                )
            # Prefill: chunked grouped-softmax over the prompt (uses the stashed self._chunk_ids).
            kh = repeat_kv(k.transpose(1, 2), n_rep)
            vh = repeat_kv(v.transpose(1, 2), n_rep)
            att = self._eager_forward(qh, kh, vh)

        att = att.transpose(1, 2).contiguous().view(B, T, -1)
        att = self._apply_gate(att, x)
        return self.w_out(att)

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
        # (B, T, T) boolean: True where the query may attend the key (causal + roles + pattern). The
        # layer index drives the per-layer dilation stride for the "hierarchical_dilated" pattern.
        allowed = build_chunked_allowed_mask(
            self._pattern, chunk_ids, is_anchor=is_anchor, layer_idx=self.layer_idx
        )
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
