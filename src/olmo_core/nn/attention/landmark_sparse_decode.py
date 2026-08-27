"""
Genuinely sparse landmark decode -- O(n_blocks + k*block) per token instead of O(context).

The shipped landmark decode is top-k only *numerically*: :meth:`FastLandmarkAttention._decode_one`
scores the query against the **entire** KV cache, masks the non-selected blocks' landmarks to
``-inf``, and lets the grouped softmax zero them. Two costs follow, and this module removes both:

1. **The GQA expansion.** ``_forward_generate`` calls ``repeat_kv`` on the whole cache every step of
   every layer. ``repeat_kv`` is ``expand().reshape()``, and a reshape of a stride-0 expand cannot be
   a view -- so it *materializes* an ``n_rep``-fold copy of the full cache (for Qwen3-4B, 8 KV heads
   -> 32, i.e. 4x) before a single score is computed. Here the query is folded into the KV-head
   grouping instead, and the cache is read at its stored width.

2. **The dense scan.** Instead of scoring all ``total`` keys, score only the ``n_blocks`` landmark
   keys, take the top-k, and **gather** just those blocks' content KV plus the local section. The
   attention then runs over ``n_blocks + (k+1)*block`` keys.

Outputs are numerically equal to the shipped dense-masked decode (same selection, same grouped
softmax, same ``nonselected_landmark_mass`` reserve) -- this is a pure compute/memory optimization,
validated in ``debug/sparse_landmark_inference/test_sparse_decode.py``.

Standalone and opt-in, like :mod:`landmark_prefill_topk`: :func:`enable_sparse_decode` patches
``_forward_generate`` on a built model at eval time; nothing in the training or shipped decode path
changes.

**Two attention families, same two pathologies.** The discussion above is
:class:`~olmo_core.nn.attention.landmark_fast.FastLandmarkAttention`. The second half of this module
does the same job for :class:`~olmo_core.nn.attention.landmark_sparse.SparseLandmarkAttention`
(``AttentionType.sparse_landmark``), whose decode has an even worse ratio: a query there attends to
its own chunk plus ``num_landmarks`` keys per past chunk -- i.e. a few hundred keys at 32k -- yet
:meth:`SparseLandmarkAttention._decode_one` expands the cache with ``repeat_kv`` and scores all
``total`` of them, masking >99% to ``-inf``. See :func:`sparse_chunk_decode`.

**What the gather does and does not save.** Selection is per *query* head, while the cache is stored
at ``n_kv_heads``; a q-head gathers from its group's KV head, so heads in a group that pick different
blocks re-read those rows. Key-bytes touched are therefore
``n_rep * H_kv * (n_blocks + (k+1)*block)`` against a dense ``H_kv * total`` -- the win scales with
``total / (n_rep * k * block)``, which is why a *fixed small* ``k`` pays off far more than a
percentage budget (see ``debug/prefill_topk/README.md``).
"""

import types
from typing import Any, List, Optional, Tuple

import torch

__all__ = [
    "sparse_landmark_decode",
    "sparse_chunk_decode",
    "sparse_chunk_decode_ragged",
    "landmark_positions",
    "landmark_chunk_count",
    "enable_sparse_decode",
    "disable_sparse_decode",
    "reset_sparse_decode_cache",
]


def _gqa_scores(q: torch.Tensor, kc: torch.Tensor) -> torch.Tensor:
    """``q (B, H, 1, D)`` x ``kc (B, H_kv, N, D)`` -> ``(B, H, 1, N)`` without expanding ``kc``.

    Head ``h`` reads KV head ``h // n_rep``, matching :func:`~olmo_core.nn.attention.landmark.repeat_kv`.
    """
    B, H, _, D = q.shape
    H_kv, N = kc.shape[1], kc.shape[2]
    n_rep = H // H_kv
    s = torch.matmul(q.view(B, H_kv, n_rep, D), kc.transpose(-1, -2))  # (B, H_kv, n_rep, N)
    return s.view(B, H, 1, N)


def _gqa_av(probs: torch.Tensor, vc: torch.Tensor) -> torch.Tensor:
    """``probs (B, H, 1, N)`` x ``vc (B, H_kv, N, D)`` -> ``(B, H, 1, D)`` without expanding ``vc``."""
    B, H, _, N = probs.shape
    H_kv, D = vc.shape[1], vc.shape[3]
    n_rep = H // H_kv
    o = torch.matmul(probs.view(B, H_kv, n_rep, N), vc)  # (B, H_kv, n_rep, D)
    return o.view(B, H, 1, D)


def _gather_blocks(
    cache: torch.Tensor, sel: torch.Tensor, block_size: int, n_heads: int
) -> torch.Tensor:
    """Gather the selected blocks' rows per *query* head.

    :param cache: ``(B, H_kv, total, D)`` key or value cache.
    :param sel: ``(B, H, k)`` selected block indices, per query head.
    :param block_size: Landmark block size.
    :param n_heads: Number of query heads ``H``.

    :returns: ``(B, H, k * block_size, D)``.
    """
    B, H_kv, total, D = cache.shape
    k = sel.shape[-1]
    n_rep = n_heads // H_kv
    device = cache.device
    # token index within the flattened (H_kv * total) axis, so one gather serves every query head
    # even though heads in a group select different blocks.
    tok = sel[..., None] * block_size + torch.arange(block_size, device=device)  # (B, H, k, Lb)
    tok = tok.reshape(B, n_heads, k * block_size)
    kv_head = torch.arange(n_heads, device=device).div(n_rep, rounding_mode="floor")
    flat = kv_head.view(1, n_heads, 1) * total + tok  # (B, H, k*Lb)
    out = torch.gather(
        cache.reshape(B, H_kv * total, D),
        1,
        flat.reshape(B, n_heads * k * block_size, 1).expand(-1, -1, D),
    )
    return out.reshape(B, n_heads, k * block_size, D)


@torch.no_grad()
def sparse_landmark_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    block_size: int,
    softmax_scale: float,
    section_start: int,
    total: int,
    top_k: Optional[int],
    compressive: bool,
    nonselected_mass: float = 0.0,
    landmark_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    One landmark decode step that touches only the landmark keys, the top-k blocks, and the local
    section.

    :param q: ``(B, H, 1, D)`` query for the token being decoded.
    :param k_cache: ``(B, H_kv, >=total, D)`` key cache (NOT GQA-expanded).
    :param v_cache: ``(B, H_kv, >=total, D)`` value cache.
    :param block_size: Landmark block size (``mem_freq + 1``).
    :param softmax_scale: Score scale.
    :param section_start: Start of the local section; landmarks below it gate the past blocks.
    :param total: Number of valid cached keys (the query attends to ``0..total-1``).
    :param top_k: Blocks to retrieve, or ``None`` for dense gating over all past blocks.
    :param compressive: Include each block's landmark in the within-block softmax.
    :param nonselected_mass: ``alpha`` reserved for the non-selected blocks' landmarks (compressive).
    :param landmark_kv: Optional pre-extracted contiguous ``(lm_k, lm_v)`` for the prompt's landmarks
        -- they never change during a generation, so the caller can hoist the strided slice.

    :returns: ``(B, H, 1, D)`` attention output.
    """
    B, H, _, D = q.shape
    Lb = block_size
    n_lm = section_start // Lb
    kc = k_cache[:, :, :total]
    vc = v_cache[:, :, :total]

    if n_lm == 0:  # nothing but the local section
        s = _gqa_scores(q, kc) * softmax_scale
        return _gqa_av(torch.softmax(s, dim=-1).to(vc.dtype), vc)

    # ---- 1. score the landmark keys only: O(n_lm), not O(total) ----
    if landmark_kv is not None:
        lm_k, lm_v = landmark_kv
    else:
        lm_k = kc[:, :, Lb - 1 : section_start : Lb, :].contiguous()
        lm_v = vc[:, :, Lb - 1 : section_start : Lb, :].contiguous()
    lm_scores = (_gqa_scores(q, lm_k) * softmax_scale).squeeze(2)  # (B, H, n_lm)

    # ---- 2. select ----
    if top_k is None or top_k >= n_lm:
        sel = torch.arange(n_lm, device=q.device).view(1, 1, n_lm).expand(B, H, n_lm)
        has_nonselected = False
    else:
        sel = lm_scores.topk(top_k, dim=-1).indices  # (B, H, k)
        has_nonselected = True
    k_sel = sel.shape[-1]

    # ---- 3. gather only the selected blocks + the local section ----
    ks = _gather_blocks(kc, sel, Lb, H)  # (B, H, k*Lb, D)
    vs = _gather_blocks(vc, sel, Lb, H)
    sel_scores = (torch.matmul(q, ks.transpose(-1, -2)) * softmax_scale).view(B, H, k_sel, Lb)

    local_k = kc[:, :, section_start:total]
    local_v = vc[:, :, section_start:total]
    local_scores = (_gqa_scores(q, local_k) * softmax_scale).squeeze(2)  # (B, H, L)

    # ---- 4. grouped softmax over {selected block gates} U {local keys} ----
    block_gate_logits = sel_scores[..., Lb - 1]  # (B, H, k)
    gate = torch.softmax(torch.cat([block_gate_logits, local_scores], dim=-1), dim=-1)
    gate_blocks, gate_local = gate[..., :k_sel], gate[..., k_sel:]

    within_logits = sel_scores
    if not compressive:  # plain landmark: the landmark token contributes no value
        within_logits = within_logits.clone()
        within_logits[..., Lb - 1] = torch.finfo(within_logits.dtype).min
    within = torch.softmax(within_logits, dim=-1)  # (B, H, k, Lb)

    probs_sel = (gate_blocks.unsqueeze(-1) * within).reshape(B, H, 1, k_sel * Lb)
    out = torch.matmul(probs_sel.to(vs.dtype), vs)
    out = out + _gqa_av(gate_local.unsqueeze(2).to(local_v.dtype), local_v)

    # ---- 5. compressive alpha reserve over the NON-selected blocks' landmarks ----
    if nonselected_mass > 0.0 and has_nonselected:
        keep = torch.zeros_like(lm_scores, dtype=torch.bool).scatter_(-1, sel, True)
        ns_logits = lm_scores.masked_fill(keep, torch.finfo(lm_scores.dtype).min)
        ns_w = torch.softmax(ns_logits, dim=-1).unsqueeze(2)  # (B, H, 1, n_lm)
        ns_out = _gqa_av(ns_w.to(lm_v.dtype), lm_v)
        out = out * (1.0 - nonselected_mass) + nonselected_mass * ns_out

    return out


def _decode_sparse(self, q: torch.Tensor, qpos: int) -> torch.Tensor:
    """Sparse replacement for the ``_decode_one`` / ``_decode_one_eval`` pair.

    ``q`` is ``(B, H, 1, D)``; the KV cache is read unexpanded from the layer's cache manager.
    """
    cfg = self._sparse_decode_cfg
    Lb = self.block_size
    kvm = self.kv_cache_manager
    kc = kvm.k_cache.transpose(1, 2)  # (B, H_kv, max_len, D)
    vc = kvm.v_cache.transpose(1, 2)
    total = qpos + 1

    P = self._eval_prompt_len
    if P is not None and qpos >= P:
        # eval mode: one long local block covering everything from ``section_start`` on
        section_start = (P // Lb) * Lb if self._eval_decode_mode == "extend_last_block" else P
    else:
        # per-block decode (prompt positions). A landmark-position query does not attend to itself.
        if qpos % Lb == Lb - 1:
            total = qpos
        section_start = (qpos // Lb) * Lb

    # The prompt's landmark rows never change during a generation, so extract them once per example.
    cache = self._sparse_decode_lmkv
    if cache is None or cache[0] != section_start:
        lm_k = kc[:, :, Lb - 1 : section_start : Lb, :].contiguous()
        lm_v = vc[:, :, Lb - 1 : section_start : Lb, :].contiguous()
        cache = (section_start, (lm_k, lm_v))
        self._sparse_decode_lmkv = cache

    return sparse_landmark_decode(
        q,
        kc,
        vc,
        block_size=Lb,
        softmax_scale=self.softmax_scale,
        section_start=section_start,
        total=total,
        top_k=self._eval_top_k,
        compressive=cfg["compressive"],
        nonselected_mass=cfg["nonselected_mass"],
        landmark_kv=cache[1],
    )


def _forward_generate_sparse(
    self,
    x: torch.Tensor,
    pos_sin: Optional[torch.Tensor],
    pos_cos: Optional[torch.Tensor],
    freqs_cis: Optional[torch.Tensor],
    cache_leftpad: Optional[torch.Tensor],
) -> torch.Tensor:
    """``_forward_generate`` with the decode branch routed through :func:`sparse_landmark_decode`.

    The prefill branch is unchanged (it delegates to whatever ``_prefill`` is installed, so this
    composes with :mod:`landmark_prefill_topk`); only the ``T == 1`` step skips ``repeat_kv`` and the
    dense scan.
    """
    from .landmark import repeat_kv

    if self._ragged_qpos is not None and x.shape[1] == 1:
        return self._forward_generate_orig(x, pos_sin, pos_cos, freqs_cis, cache_leftpad)
    if cache_leftpad is not None and bool(cache_leftpad.ne(0).any()):
        raise NotImplementedError(
            "Landmark generation requires batch_size=1 / no left-padding "
            "(blocks are tied to absolute position)."
        )

    kvm = self.kv_cache_manager
    assert kvm is not None
    B, T, _ = x.shape
    start_pos = int(kvm.current_position())
    q, k, v = self._prepare_qkv(
        x, pos_sin=pos_sin, pos_cos=pos_cos, freqs_cis=freqs_cis, cu_doc_lens=None
    )

    kvm.k_cache[:, start_pos : start_pos + T].copy_(k)
    kvm.v_cache[:, start_pos : start_pos + T].copy_(v)
    kvm.update_seqlen(T)

    qh = q.transpose(1, 2)  # (B, H, T, D)
    if T == 1:
        att = _decode_sparse(self, qh, start_pos)
    else:
        if start_pos != 0:
            raise NotImplementedError(
                "Landmark multi-token forward with a non-empty cache is not supported "
                "(only single-shot prefill from position 0)."
            )
        self._sparse_decode_lmkv = None  # new example -> drop the cached landmark rows
        n_rep = q.shape[2] // k.shape[2]
        att = self._prefill(
            qh, repeat_kv(k.transpose(1, 2), n_rep), repeat_kv(v.transpose(1, 2), n_rep)
        )

    att = att.transpose(1, 2).contiguous().view(B, T, -1)
    att = self._apply_gate(att, x)
    return self.w_out(att)


# ---------------------------------------------------------------------------------------------
# SparseLandmarkAttention (``AttentionType.sparse_landmark``)
#
# A *different* architecture from the landmark variants above: a query attends fully (causally)
# within its own chunk and sees every past chunk ONLY through that chunk's last ``num_landmarks``
# tokens. There is no grouped softmax and no block content to gather -- the past is *already* just
# ``num_landmarks * n_past_chunks`` keys. Yet ``SparseLandmarkAttention._forward_generate`` still
# ``repeat_kv``-expands the whole cache and scores all ``total`` keys, masking ~99% of them to
# ``-inf``. So decode pays dense cost for an attention pattern that is inherently sparse; the
# functions below pay only for what is actually attended.
# ---------------------------------------------------------------------------------------------


def landmark_positions(
    section_start: int, block_size: int, num_landmarks: int, device: torch.device
) -> torch.Tensor:
    """Absolute positions ``j < section_start`` with ``(j % block_size) >= block_size - num_landmarks``.

    This is exactly the key set that :meth:`SparseLandmarkAttention._decode_one` marks
    ``retrievable``, in ascending order, but built in ``O(n_landmarks)`` instead of materializing an
    ``O(section_start)`` boolean mask.

    :param section_start: Start of the local section (everything below it is reachable only through
        landmarks).
    :param block_size: Chunk size (``mem_freq + num_landmarks``).
    :param num_landmarks: Landmark tokens at the end of each chunk.
    :param device: Device for the returned index tensor.

    :returns: ``(n_lm,)`` int64 positions.
    """
    L, G = block_size, num_landmarks
    n_full = section_start // L
    rem = section_start - n_full * L
    parts = []
    if n_full > 0:
        base = torch.arange(n_full, device=device, dtype=torch.long).view(n_full, 1) * L + (L - G)
        parts.append(
            (base + torch.arange(G, device=device, dtype=torch.long).view(1, G)).reshape(-1)
        )
    extra = rem - (L - G)  # a trailing partial chunk can expose some of its landmarks
    if extra > 0:
        parts.append(n_full * L + (L - G) + torch.arange(extra, device=device, dtype=torch.long))
    if not parts:
        return torch.empty(0, device=device, dtype=torch.long)
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def landmark_chunk_count(section_start: int, block_size: int, num_landmarks: int) -> int:
    """Number of chunks reachable through their landmarks below ``section_start``.

    This is ``max(chunk id of a landmark key) + 1``, i.e. exactly the ``n_chunks`` that
    :meth:`SparseLandmarkAttention._apply_topk_landmark_retrieval` derives from the key mask -- note
    a trailing *partial* chunk counts only if the section boundary actually reaches its landmarks
    (``rem > block_size - num_landmarks``), which is why this is not ``ceil(section_start / L)``.
    """
    n_full = section_start // block_size
    extra = section_start - n_full * block_size - (block_size - num_landmarks)
    return n_full + (1 if extra > 0 else 0)


@torch.no_grad()
def sparse_chunk_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    block_size: int,
    num_landmarks: int,
    softmax_scale: float,
    section_start: int,
    total: int,
    top_k: Optional[int],
    landmark_kv: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = None,
) -> torch.Tensor:
    """
    One :class:`SparseLandmarkAttention` decode step that touches only the past chunks' landmark
    keys plus the local section -- ``num_landmarks * n_chunks + local`` keys instead of ``total``.

    Numerically equal to :meth:`SparseLandmarkAttention._decode_one` (same key set, same top-k
    selection via :func:`~olmo_core.nn.attention.landmark_sparse.landmark_chunk_topk_keep`, same
    single flat softmax) up to floating-point reassociation of the softmax denominator and the
    ``A @ V`` reduction, which are split into a landmark term and a local term here.

    :param q: ``(B, H, 1, D)`` query for the token being decoded.
    :param k_cache: ``(B, H_kv, >=total, D)`` key cache (NOT GQA-expanded).
    :param v_cache: ``(B, H_kv, >=total, D)`` value cache.
    :param block_size: Chunk size (``mem_freq + num_landmarks``).
    :param num_landmarks: Landmark tokens at the end of each chunk.
    :param softmax_scale: Score scale.
    :param section_start: Start of the local section (the query's own chunk, or the "one long local
        block" in eval decode mode).
    :param total: Number of valid cached keys (the query attends to ``0..total-1``).
    :param top_k: Chunks to retrieve, or ``None`` for access to every past chunk's landmarks.
    :param landmark_kv: Optional pre-extracted ``(lm_k, lm_v, chunk_ids, n_chunks)`` for
        ``section_start`` -- the prompt's landmark rows never change during a generation, so the
        caller hoists the gather out of the decode loop.

    :returns: ``(B, H, 1, D)`` attention output.
    """
    from .landmark_sparse import landmark_chunk_topk_keep

    kc = k_cache[:, :, :total]
    vc = v_cache[:, :, :total]

    if landmark_kv is not None:
        lm_k, lm_v, chunk_ids, n_chunks = landmark_kv
    else:
        idx = landmark_positions(section_start, block_size, num_landmarks, q.device)
        lm_k = kc.index_select(2, idx).contiguous()
        lm_v = vc.index_select(2, idx).contiguous()
        chunk_ids = idx.div(block_size, rounding_mode="floor").view(1, 1, 1, -1)
        n_chunks = landmark_chunk_count(section_start, block_size, num_landmarks)

    local_k = kc[:, :, section_start:total]
    local_v = vc[:, :, section_start:total]
    local_scores = _gqa_scores(q, local_k) * softmax_scale  # (B, H, 1, n_local)

    n_lm = lm_k.shape[2]
    if n_lm == 0:  # first chunk -- nothing but the local section
        return _gqa_av(torch.softmax(local_scores, dim=-1).to(vc.dtype), local_v)

    lm_scores = _gqa_scores(q, lm_k) * softmax_scale  # (B, H, 1, n_lm)
    if top_k is not None and n_chunks > top_k:
        keep = landmark_chunk_topk_keep(lm_scores, chunk_ids, n_chunks, top_k)
        lm_scores = lm_scores.masked_fill(~keep, float("-inf"))

    probs = torch.softmax(torch.cat([lm_scores, local_scores], dim=-1), dim=-1)
    out = _gqa_av(probs[..., :n_lm].to(lm_v.dtype), lm_v)
    return out + _gqa_av(probs[..., n_lm:].to(local_v.dtype), local_v)


def _decode_sparse_chunk(self, q: torch.Tensor, qpos: int) -> torch.Tensor:
    """Sparse replacement for :meth:`SparseLandmarkAttention._decode_one`.

    ``q`` is ``(B, H, 1, D)``; the KV cache is read unexpanded from the layer's cache manager.
    """
    L, G = self.block_size, self.num_landmarks
    kvm = self.kv_cache_manager
    kc = kvm.k_cache.transpose(1, 2)  # (B, H_kv, max_len, D)
    vc = kvm.v_cache.transpose(1, 2)
    total = qpos + 1

    P = self._eval_prompt_len
    if P is not None and qpos >= P:
        # eval mode: one long local block covering everything from ``section_start`` on
        section_start = (P // L) * L if self._eval_decode_mode == "extend_last_block" else P
    else:
        section_start = (qpos // L) * L  # per-chunk decode (prompt positions)

    cache = self._sparse_decode_lmkv
    if cache is None or cache[0] != section_start:
        idx = landmark_positions(section_start, L, G, kc.device)
        lm_k = kc[:, :, :total].index_select(2, idx).contiguous()
        lm_v = vc[:, :, :total].index_select(2, idx).contiguous()
        chunk_ids = idx.div(L, rounding_mode="floor").view(1, 1, 1, -1)
        n_chunks = landmark_chunk_count(section_start, L, G)
        cache = (section_start, (lm_k, lm_v, chunk_ids, n_chunks))
        self._sparse_decode_lmkv = cache

    return sparse_chunk_decode(
        q,
        kc,
        vc,
        block_size=L,
        num_landmarks=G,
        softmax_scale=self.softmax_scale,
        section_start=section_start,
        total=total,
        top_k=self._eval_top_k,
        landmark_kv=cache[1],
    )


def _forward_generate_sparse_chunk(
    self,
    x: torch.Tensor,
    pos_sin: Optional[torch.Tensor],
    pos_cos: Optional[torch.Tensor],
    freqs_cis: Optional[torch.Tensor],
    cache_leftpad: Optional[torch.Tensor],
) -> torch.Tensor:
    """:meth:`SparseLandmarkAttention._forward_generate` with the ``T == 1`` branch routed through
    :func:`sparse_chunk_decode` -- no ``repeat_kv`` of the cache, no dense scan. The prefill branch
    is byte-for-byte the shipped one."""
    from .landmark import repeat_kv

    if getattr(self, "_ragged_qpos", None) is not None and x.shape[1] == 1:
        return _forward_generate_ragged_sparse(self, x, pos_sin, pos_cos, freqs_cis)
    if cache_leftpad is not None and bool(cache_leftpad.ne(0).any()):
        raise NotImplementedError(
            "Sparse landmark generation requires batch_size=1 / no left-padding "
            "(chunk boundaries are tied to absolute position)."
        )

    kvm = self.kv_cache_manager
    assert kvm is not None
    B, T, _ = x.shape
    start_pos = int(kvm.current_position())
    q, k, v = self._prepare_qkv(
        x, pos_sin=pos_sin, pos_cos=pos_cos, freqs_cis=freqs_cis, cu_doc_lens=None
    )

    kvm.k_cache[:, start_pos : start_pos + T].copy_(k)
    kvm.v_cache[:, start_pos : start_pos + T].copy_(v)
    kvm.update_seqlen(T)

    qh = q.transpose(1, 2)  # (B, H, T, D)
    if T == 1:
        att = _decode_sparse_chunk(self, qh, start_pos)
    else:
        if start_pos != 0:
            raise NotImplementedError(
                "Sparse landmark multi-token forward with a non-empty cache is not supported "
                "(only single-shot prefill from position 0)."
            )
        self._sparse_decode_lmkv = None  # new example -> drop the cached landmark rows
        n_rep = q.shape[2] // k.shape[2]
        att = self._prefill(
            qh, repeat_kv(k.transpose(1, 2), n_rep), repeat_kv(v.transpose(1, 2), n_rep)
        )

    att = att.transpose(1, 2).contiguous().view(B, T, -1)
    att = self._apply_gate(att, x)
    return self.w_out(att)


@torch.no_grad()
def sparse_chunk_decode_ragged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    block_size: int,
    num_landmarks: int,
    softmax_scale: float,
    section_start: torch.Tensor,
    qpos: torch.Tensor,
    top_k: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Ragged (right-padded, cross-length) analogue of :func:`sparse_chunk_decode`: every row decodes
    at its OWN absolute position with its own section start and chunk budget, and still touches only
    ``num_landmarks * n_chunks + local_window`` keys instead of the whole cache.

    Numerically equal to :meth:`SparseLandmarkAttention._decode_ragged` (and therefore, row for row,
    to the scalar :meth:`~olmo_core.nn.attention.landmark_sparse.SparseLandmarkAttention._decode_one`)
    up to floating-point reassociation.

    :param q: ``(B, H, 1, D)`` queries.
    :param k_cache: ``(B, H_kv, >=total, D)`` key cache (NOT GQA-expanded).
    :param v_cache: ``(B, H_kv, >=total, D)`` value cache.
    :param block_size: Chunk size (``mem_freq + num_landmarks``).
    :param num_landmarks: Landmark tokens at the end of each chunk.
    :param softmax_scale: Score scale.
    :param section_start: ``(B, 1)`` per-row local-section start.
    :param qpos: ``(B, 1)`` per-row absolute query position.
    :param top_k: ``(B,)`` per-row chunk budget, or ``None``.

    :returns: ``(B, H, 1, D)`` attention output.
    """
    L, G = block_size, num_landmarks
    B, H = q.shape[0], q.shape[1]
    dev = q.device
    neg_inf = float("-inf")

    sec_max = int(section_start.max().item())
    total = int(qpos.max().item()) + 1
    kc = k_cache[:, :, :total]
    vc = v_cache[:, :, :total]

    # ---- local window: [sec_b, sec_b + W) per row, gathered from the shared cache ----
    W = int((qpos - section_start).max().item()) + 1
    pos = section_start + torch.arange(W, device=dev)[None, :]  # (B, W)
    local_valid = pos <= qpos
    gidx = pos.clamp(max=total - 1)[:, None, :, None].expand(B, kc.shape[1], W, kc.shape[3])
    local_k = torch.gather(kc, 2, gidx)
    local_v = torch.gather(vc, 2, gidx)
    local_scores = _gqa_scores(q, local_k) * softmax_scale  # (B, H, 1, W)
    local_scores = local_scores.masked_fill(~local_valid[:, None, None, :], neg_inf)

    # ---- landmark keys: one shared superset (up to the largest section start), masked per row ----
    idx = landmark_positions(sec_max, L, G, dev)
    n_lm = int(idx.numel())
    if n_lm == 0:
        probs = torch.softmax(local_scores, dim=-1)
        return _gqa_av(probs.to(local_v.dtype), local_v)

    lm_k = kc.index_select(2, idx)
    lm_v = vc.index_select(2, idx)
    lm_valid = idx[None, :] < section_start  # (B, n_lm)
    lm_scores = _gqa_scores(q, lm_k) * softmax_scale  # (B, H, 1, n_lm)
    lm_scores = lm_scores.masked_fill(~lm_valid[:, None, None, :], neg_inf)

    if top_k is not None:
        chunk_ids_1d = idx.div(L, rounding_mode="floor")
        n_slots = landmark_chunk_count(sec_max, L, G)
        chunk_ids = chunk_ids_1d.view(1, 1, 1, n_lm).expand(B, H, 1, n_lm)
        k_b = top_k.to(dev).long().view(B, 1, 1, 1)
        neg = torch.finfo(lm_scores.dtype).min
        finite = torch.where(lm_scores.isneginf(), lm_scores.new_full((), neg), lm_scores)
        chunk_scores = lm_scores.new_full((B, H, 1, n_slots), neg_inf)
        chunk_scores.scatter_reduce_(-1, chunk_ids, finite, reduce="amax", include_self=True)
        order = chunk_scores.argsort(dim=-1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(-1, order, torch.arange(n_slots, device=dev).expand_as(order))
        n_full = section_start // L
        n_chunks = (n_full + ((section_start - n_full * L - (L - G)) > 0).long()).view(B, 1, 1, 1)
        drop_chunk = (ranks >= k_b) & (n_chunks > k_b)
        lm_scores = lm_scores.masked_fill(drop_chunk.gather(-1, chunk_ids), neg_inf)

    probs = torch.softmax(torch.cat([lm_scores, local_scores], dim=-1), dim=-1)
    out = _gqa_av(probs[..., :n_lm].to(lm_v.dtype), lm_v)
    return out + _gqa_av(probs[..., n_lm:].to(local_v.dtype), local_v)


def _forward_generate_ragged_sparse(
    self,
    x: torch.Tensor,
    pos_sin: Optional[torch.Tensor],
    pos_cos: Optional[torch.Tensor],
    freqs_cis: Optional[torch.Tensor],
) -> torch.Tensor:
    """:meth:`SparseLandmarkAttention._forward_generate_ragged` without the ``repeat_kv`` expansion
    or the dense scan -- so right-padded cross-length batching and the sparse decode compose."""
    kvm = self.kv_cache_manager
    assert kvm is not None
    qpos = self._ragged_qpos
    assert qpos is not None
    B = x.shape[0]
    q, _k, _v = self._prepare_qkv(
        x,
        pos_sin=pos_sin,
        pos_cos=pos_cos,
        freqs_cis=freqs_cis,
        cu_doc_lens=None,
        position_ids=qpos.view(B, 1),
    )
    bidx = torch.arange(B, device=x.device)
    kvm.k_cache[bidx, qpos] = _k[:, 0]
    kvm.v_cache[bidx, qpos] = _v[:, 0]

    att = sparse_chunk_decode_ragged(
        q.transpose(1, 2),
        kvm.k_cache.transpose(1, 2),
        kvm.v_cache.transpose(1, 2),
        block_size=self.block_size,
        num_landmarks=self.num_landmarks,
        softmax_scale=self.softmax_scale,
        section_start=self._ragged_section_start().to(x.device),
        qpos=qpos.to(x.device).long().view(B, 1),
        top_k=self._ragged_top_k,
    )
    att = att.transpose(1, 2).contiguous().view(B, 1, -1)
    att = self._apply_gate(att, x)
    return self.w_out(att)


def _landmark_layers(model: torch.nn.Module) -> List[Any]:
    """Every attention layer this module knows how to accelerate."""
    from .landmark_fast import FastLandmarkAttention
    from .landmark_sparse import SparseLandmarkAttention

    return [
        m
        for m in model.modules()
        if isinstance(m, (FastLandmarkAttention, SparseLandmarkAttention))
    ]


def enable_sparse_decode(model: torch.nn.Module, *, strict: bool = True) -> int:
    """
    Route every landmark layer's decode through the genuinely sparse path -- for
    :class:`~olmo_core.nn.attention.landmark_fast.FastLandmarkAttention` that is
    :func:`sparse_landmark_decode` (landmark-only scoring, top-k selection, gather of just the
    selected blocks); for :class:`~olmo_core.nn.attention.landmark_sparse.SparseLandmarkAttention`
    it is :func:`sparse_chunk_decode` (past chunks' landmark keys plus the local section). Both
    replace scoring the whole KV cache and masking. Outputs match the shipped decode; only the cost
    changes.

    Inference only. Composes with :func:`~olmo_core.nn.attention.landmark_prefill_topk.enable_prefill_topk`
    (the prefill branch still calls whatever ``_prefill`` is installed).

    :param model: A built model.
    :param strict: Raise if the model has no landmark attention layers. With ``strict=False`` this
        is a no-op returning ``0``, which is what callers that just want "make it fast if it
        applies" (e.g. the generation module) want.

    :returns: Number of layers patched.

    :raises RuntimeError: If ``strict`` and the model has no landmark attention layers.
    """
    from .landmark_compressive import FastCompressiveLandmarkAttention
    from .landmark_fast import FastLandmarkAttention

    layers = _landmark_layers(model)
    if not layers:
        if strict:
            raise RuntimeError("no landmark attention layers found; sparse decode does not apply")
        return 0
    for attn in layers:
        if isinstance(attn, FastLandmarkAttention):
            compressive = isinstance(attn, FastCompressiveLandmarkAttention)
            attn._sparse_decode_cfg = {
                "compressive": compressive,
                "nonselected_mass": (float(attn.nonselected_landmark_mass) if compressive else 0.0),
            }
            impl = _forward_generate_sparse
        else:  # SparseLandmarkAttention
            attn._sparse_decode_cfg = {}
            impl = _forward_generate_sparse_chunk
        attn._sparse_decode_lmkv = None
        if not hasattr(attn, "_forward_generate_orig"):
            attn._forward_generate_orig = attn._forward_generate
        attn._forward_generate = types.MethodType(impl, attn)
    return len(layers)


def reset_sparse_decode_cache(model: torch.nn.Module) -> None:
    """Drop the hoisted landmark-row cache on every patched layer.

    The cache is keyed by ``section_start`` and is refreshed automatically on each prefill, so this
    is belt-and-braces for callers that reuse a KV cache across examples without a fresh prefill.
    """
    for attn in _landmark_layers(model):
        if hasattr(attn, "_sparse_decode_lmkv"):
            attn._sparse_decode_lmkv = None


def disable_sparse_decode(model: torch.nn.Module) -> int:
    """Undo :func:`enable_sparse_decode`. Returns the number of layers restored."""
    n = 0
    for attn in _landmark_layers(model):
        if hasattr(attn, "_forward_generate_orig"):
            attn._forward_generate = attn._forward_generate_orig
            del attn._forward_generate_orig
            attn._sparse_decode_lmkv = None
            n += 1
    return n
