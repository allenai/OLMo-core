"""
``SparseLandmarkAttention`` -- a sparse "landmark-only-across-chunks" attention sequence mixer.

Independent of (and not modifying) the original landmark attention. A query attends *fully* within
its own chunk (causal), but sees **past chunks only through their landmark tokens** (the last
``num_landmarks`` tokens of each chunk) -- the non-landmark content of past chunks is not attended.
Each past chunk is thus compressed to a few keys, so a query attends to about

    block_size (own chunk)  +  num_landmarks * num_past_chunks   keys

instead of the whole prefix. Score cost drops from ``O(T^2)`` to ``O(T*(L + num_landmarks*T/L))``
(``L = block_size``, ``C = T/L`` chunks) -- e.g. ~60x fewer score FLOPs at L=64, num_landmarks=1,
T=64k -- which is the source of the speedup over standard / dense-landmark attention.

Layout: the **last ``num_landmarks`` tokens of every chunk are landmarks**; ``T`` is a multiple of
``L`` and ``1 <= num_landmarks < L``.

  * :func:`sparse_landmark_attention_ref` -- dense O(T^2) masked reference (correctness).
  * :func:`sparse_landmark_attention`     -- efficient chunked form (block-local + landmark gather +
    one combined softmax); pure autograd-differentiable torch ops (backward for free).
  * :class:`SparseLandmarkAttention`       -- the :class:`Attention` sequence mixer wrapping it
    (``AttentionType.sparse_landmark``). A fused Triton kernel is the natural next step for peak speed.
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F

from olmo_core.exceptions import OLMoConfigurationError

from . import Attention
from . import landmark_gate_analysis as gate_log
from .kv_cache import KVCacheManager
from .landmark import repeat_kv
from .landmark_sparse_kernel import (
    has_sparse_kernel,
    sparse_landmark_attention_triton_train,
)


def sparse_landmark_attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    num_landmarks: int = 1,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Dense O(T^2) reference. ``q,k,v``: ``(B, H, T, D)``; ``T`` a multiple of ``block_size``.
    A query at position i (chunk ``c = i // L``) attends to key j iff:
      - same chunk and causal: ``j // L == c`` and ``j <= i``, OR
      - j is a landmark of a strictly-past chunk: ``(j % L) >= L - G`` and ``j // L < c``.
    """
    B, H, T, D = q.shape
    L, G = block_size, num_landmarks
    assert T % L == 0 and 1 <= G < L
    scale = scale if scale is not None else D**-0.5
    dev = q.device

    pos = torch.arange(T, device=dev)
    chunk = pos // L
    is_lm = (pos % L) >= (L - G)

    same_chunk_causal = (chunk[:, None] == chunk[None, :]) & (pos[None, :] <= pos[:, None])
    past_landmark = is_lm[None, :] & (chunk[None, :] < chunk[:, None])
    allowed = same_chunk_causal | past_landmark

    att = torch.matmul(q, k.transpose(-1, -2)) * scale
    att = att.masked_fill(~allowed[None, None], float("-inf"))
    return torch.matmul(torch.softmax(att, dim=-1), v)


def sparse_landmark_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    num_landmarks: int = 1,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Efficient chunked form, numerically equal to :func:`sparse_landmark_attention_ref`.
    ``q,k,v``: ``(B, H, T, D)``; ``T`` a multiple of ``block_size``; ``1 <= num_landmarks < block_size``.

    Own-chunk scores are ``(B,H,C,L,L)`` and landmark scores are ``(B,H,C,L,C,G)`` (``C=T/L``), so
    compute/peak-memory is ``O(B*H*T*(L + G*T/L))`` instead of ``O(B*H*T^2)``.
    """
    B, H, T, D = q.shape
    L, G = block_size, num_landmarks
    assert T % L == 0 and 1 <= G < L
    C = T // L
    scale = scale if scale is not None else D**-0.5
    dev = q.device
    neg_inf = float("-inf")

    qc = q.view(B, H, C, L, D)
    kc = k.view(B, H, C, L, D)
    vc = v.view(B, H, C, L, D)

    s_own = torch.matmul(qc, kc.transpose(-1, -2)) * scale
    causal = torch.tril(torch.ones(L, L, device=dev, dtype=torch.bool))
    s_own = s_own.masked_fill(~causal, neg_inf)

    k_lm = kc[:, :, :, L - G :, :]
    v_lm = vc[:, :, :, L - G :, :]

    s_lm = torch.einsum("bhcld,bhkgd->bhclkg", qc, k_lm) * scale
    cidx = torch.arange(C, device=dev)
    past = cidx[None, :] < cidx[:, None]
    s_lm = s_lm.masked_fill(~past[None, None, :, None, :, None], neg_inf)
    s_lm = s_lm.reshape(B, H, C, L, C * G)

    s = torch.cat([s_own, s_lm], dim=-1)
    p = torch.softmax(s, dim=-1)
    p_own = p[..., :L]
    p_lm = p[..., L:].reshape(B, H, C, L, C, G)

    out = torch.matmul(p_own, vc)
    out = out + torch.einsum("bhclkg,bhkgd->bhcld", p_lm, v_lm)
    return out.reshape(B, H, T, D)


def attended_key_count(T: int, block_size: int, num_landmarks: int = 1) -> float:
    """Average #keys a query attends (vs T for full causal) -- a FLOP proxy."""
    L, C, G = block_size, T // block_size, num_landmarks
    return (L + 1) / 2 + G * (C - 1) / 2


class SparseLandmarkAttention(Attention):
    """
    Sparse landmark-only-across-chunks attention as a drop-in :class:`Attention` variant
    (``AttentionType.sparse_landmark``). Each chunk is ``block_size = mem_freq + num_landmarks``
    tokens, the last ``num_landmarks`` of which are landmarks. Pure-torch (autograd) -- works on
    CPU/GPU; context parallelism is not yet supported. Supports the optional output gate inherited
    from :class:`Attention` (``att * sigmoid(w_g(x))``), so it drops into gated models like Qwen3.5.
    """

    def __init__(
        self,
        *,
        mem_freq: int,
        num_landmarks: int = 1,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ):
        if kwargs.get("window_size") is not None:
            raise OLMoConfigurationError(
                "SparseLandmarkAttention does not support sliding window attention"
            )
        super().__init__(softmax_scale=softmax_scale, **kwargs)
        if mem_freq is None or mem_freq < 1:
            raise OLMoConfigurationError(
                f"SparseLandmarkAttention requires mem_freq >= 1 (got {mem_freq})"
            )
        if num_landmarks < 1:
            raise OLMoConfigurationError(f"num_landmarks must be >= 1 (got {num_landmarks})")
        self.mem_freq = mem_freq
        self.num_landmarks = num_landmarks
        self.block_size = mem_freq + num_landmarks
        self.softmax_scale = softmax_scale if softmax_scale is not None else self.head_dim**-0.5
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
        :param top_k: If set, decode restricts past-chunk access to the ``top_k`` highest-scoring
            chunks per head (a chunk's score is the max over its landmark keys' scores); all other
            past chunks' landmarks are masked out and the softmax renormalizes over the local block
            plus the retrieved chunks' landmarks. ``None`` keeps all past chunks' landmarks visible.
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
        # Sparse landmark attention implements its own cached prefill/decode in ``_forward_generate``
        # (it does not route through the flash backend), so skip the backend KV-cache assertion.
        self.kv_cache_manager = KVCacheManager(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            num_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            device=self.w_k.weight.device,
            dtype=self.w_k.weight.dtype,  # eager decode matmuls q against the cache directly
        )

    def apply_cp(self, *args, **kwargs):
        raise NotImplementedError(
            "Context parallelism is not yet supported for SparseLandmarkAttention"
        )

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
                "Intra-document masking (cu_doc_lens) is not supported with sparse landmark attention"
            )
        # Generation path: incremental decode / prefill with a KV cache.
        if self.kv_cache_manager is not None:
            return self._forward_generate(x, pos_sin, pos_cos, freqs_cis, cache_leftpad)
        if cache_leftpad is not None:
            raise NotImplementedError(
                "cache_leftpad is only supported together with a KV cache manager"
            )

        B, T, _ = x.shape
        q, k, v = self._prepare_qkv(
            x, pos_sin=pos_sin, pos_cos=pos_cos, freqs_cis=freqs_cis, cu_doc_lens=None
        )
        if T % self.block_size != 0:
            raise OLMoConfigurationError(
                f"Sequence length ({T}) must be a multiple of block_size "
                f"(mem_freq + num_landmarks = {self.block_size})."
            )
        n_rep = q.shape[2] // k.shape[2]
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), n_rep)
        v = repeat_kv(v.transpose(1, 2), n_rep)

        att = self._attn_core(q, k, v)
        att = att.transpose(1, 2).contiguous().view(B, T, -1)
        att = self._apply_gate(att, x)
        return self.w_out(att)

    def _attn_core(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Sparse landmark self-attention on ``(B, H, T, D)`` with ``T`` a multiple of block_size.

        Fused Triton fwd+bwd kernel when available (much faster); eager torch fallback otherwise.
        Disable the kernel with ``LM_SPARSE_KERNEL=0``.
        """
        if q.is_cuda and has_sparse_kernel() and os.environ.get("LM_SPARSE_KERNEL", "1") != "0":
            return sparse_landmark_attention_triton_train(
                q, k, v, self.block_size, num_landmarks=self.num_landmarks, scale=self.softmax_scale
            )
        return sparse_landmark_attention(
            q, k, v, self.block_size, num_landmarks=self.num_landmarks, scale=self.softmax_scale
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

        Chunk boundaries follow absolute position, so generation must be left-pad free (the real
        tokens of every sequence must start at absolute position 0) to stay aligned with training --
        i.e. ``batch_size == 1`` for these evals. Batched left-padded generation is rejected.
        """
        kvm = self.kv_cache_manager
        assert kvm is not None
        if cache_leftpad is not None and bool(cache_leftpad.ne(0).any()):
            raise NotImplementedError(
                "Sparse landmark generation requires batch_size=1 / no left-padding "
                "(chunk boundaries are tied to absolute position)."
            )

        B, T, _ = x.shape
        start_pos = int(kvm.current_position())
        q, k, v = self._prepare_qkv(
            x, pos_sin=pos_sin, pos_cos=pos_cos, freqs_cis=freqs_cis, cu_doc_lens=None
        )

        # Append this step's K/V (un-repeated, n_kv_heads) to the cache and advance the length.
        kvm.k_cache[:, start_pos : start_pos + T].copy_(k)
        kvm.v_cache[:, start_pos : start_pos + T].copy_(v)
        kvm.update_seqlen(T)
        total = start_pos + T

        n_rep = q.shape[2] // k.shape[2]
        qh = q.transpose(1, 2)  # (B, H, T, D)

        if T == 1:
            kh = repeat_kv(kvm.k_cache[:, :total].transpose(1, 2), n_rep)  # (B, H, total, D)
            vh = repeat_kv(kvm.v_cache[:, :total].transpose(1, 2), n_rep)
            att = self._decode_one(qh, kh, vh, start_pos)
        else:
            if start_pos != 0:
                raise NotImplementedError(
                    "Sparse landmark multi-token forward with a non-empty cache is not supported "
                    "(only single-shot prefill from position 0)."
                )
            kh = repeat_kv(k.transpose(1, 2), n_rep)
            vh = repeat_kv(v.transpose(1, 2), n_rep)
            att = self._prefill(qh, kh, vh)

        att = att.transpose(1, 2).contiguous().view(B, T, -1)
        att = self._apply_gate(att, x)
        return self.w_out(att)

    def _prefill(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Prefill attention over an arbitrary-length prompt by right-padding to a multiple of
        block_size and reusing the self-attention core. Padding tokens are appended after the prompt,
        so causal/own-chunk masking never lets a real query attend to them; their outputs are sliced off.
        """
        T = q.shape[2]
        pad = (-T) % self.block_size
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        att = self._attn_core(q, k, v)
        return att[:, :, :T]

    def _decode_one(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qpos: int
    ) -> torch.Tensor:
        """Single-query decode: query at absolute position ``qpos`` attends to cached keys
        ``0..total-1`` under the sparse-landmark rule (own-chunk causal + past-chunk landmarks).
        Equivalent to row ``qpos`` of :func:`sparse_landmark_attention_ref`.
        """
        total = k.shape[2]
        L, G = self.block_size, self.num_landmarks
        j = torch.arange(total, device=q.device)
        is_lm = (j % L) >= (L - G)

        # Only *generated* queries (``qpos >= prompt_len``) use the "one long local block" eval decode.
        # Prompt-position queries keep the per-chunk rule so they reproduce prefill -- the final prompt
        # token is decoded first by the generation loop so top-k retrieval also gates the first
        # generated token (prefill itself never applies top-k).
        eval_mode = self._eval_prompt_len is not None and qpos >= self._eval_prompt_len
        if eval_mode:
            # Landmark eval mode: the generated query is part of "one long local block". It attends
            # directly to the growing block ``[section_start, qpos]`` and reaches earlier prompt
            # blocks only through their landmarks (generated positions are never landmarks).
            P = self._eval_prompt_len
            assert P is not None
            section_start = (P // L) * L if self._eval_decode_mode == "extend_last_block" else P
            allowed = ((j >= section_start) & (j <= qpos)) | (is_lm & (j < section_start))
        else:
            cq = qpos // L
            ck = j // L
            allowed = ((ck == cq) & (j <= qpos)) | (is_lm & (ck < cq))  # (total,)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.softmax_scale  # (B, H, 1, total)
        scores = scores.masked_fill(~allowed.view(1, 1, 1, total), float("-inf"))
        if eval_mode:
            retrievable = allowed & is_lm & (j < section_start)
        else:
            retrievable = allowed & is_lm & (ck < cq)
        scores = self._apply_topk_landmark_retrieval(scores, retrievable)
        p = torch.softmax(scores, dim=-1)
        return torch.matmul(p, v)

    def _apply_topk_landmark_retrieval(
        self, scores: torch.Tensor, retrievable: torch.Tensor
    ) -> torch.Tensor:
        """Hard top-k chunk retrieval at decode time (landmark-paper-style inference).

        Keeps only the ``top_k`` highest-scoring past chunks per batch/head -- a chunk's score is
        the max over its (already position-allowed) landmark keys' scores -- and masks the landmark
        keys of every other past chunk to ``-inf``. Since past chunks are reachable *only* through
        their landmark values here, masked chunks get exactly zero attention weight and the softmax
        renormalizes over the local block plus the retrieved chunks' landmarks.

        :param scores: Attention logits of shape ``(B, H, 1, total)`` (already position-masked).
        :param retrievable: Boolean mask of shape ``(total,)`` marking past-chunk landmark keys.
        """
        top_k = self._eval_top_k
        if top_k is None:
            return scores
        lm_idx = retrievable.nonzero(as_tuple=True)[0]  # (n_lm,)
        if lm_idx.numel() == 0:
            return scores
        chunk_ids = (lm_idx // self.block_size).view(1, 1, 1, -1)
        n_chunks = int(chunk_ids.max().item()) + 1
        recording = gate_log.is_enabled()
        if n_chunks <= top_k and not recording:
            # All past chunks are kept (nothing to mask); skip the work unless logging gates.
            return scores
        lm_scores = scores[..., lm_idx]  # (B, H, 1, n_lm)
        chunk_ids = chunk_ids.expand_as(lm_scores)
        chunk_scores = lm_scores.new_full((*lm_scores.shape[:-1], n_chunks), float("-inf"))
        chunk_scores.scatter_reduce_(-1, chunk_ids, lm_scores, reduce="amax", include_self=True)
        present = torch.isfinite(chunk_scores)  # chunk ordinals actually reachable this step
        if n_chunks <= top_k:
            keep_chunks = present
        else:
            keep_chunks = torch.zeros_like(chunk_scores, dtype=torch.bool)
            keep_chunks.scatter_(-1, chunk_scores.topk(top_k, dim=-1).indices, True)
            keep_chunks &= present
        if recording:
            # Per chunk one gate; block_ids are the chunk ordinals 0..n_chunks-1.
            gate_log.record_layer(
                getattr(self, "_gate_log_layer_idx", None),
                keep_chunks,
                torch.arange(n_chunks, device=scores.device),
            )
        if n_chunks <= top_k:
            return scores
        keep = keep_chunks.gather(-1, chunk_ids)
        scores = scores.clone()
        scores[..., lm_idx] = lm_scores.masked_fill(~keep, float("-inf"))
        return scores
