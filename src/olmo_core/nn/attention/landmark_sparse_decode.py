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

__all__ = ["sparse_landmark_decode", "enable_sparse_decode", "disable_sparse_decode"]


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


def _landmark_layers(model: torch.nn.Module) -> List[Any]:
    from .landmark_fast import FastLandmarkAttention

    return [m for m in model.modules() if isinstance(m, FastLandmarkAttention)]


def enable_sparse_decode(model: torch.nn.Module) -> int:
    """
    Route every landmark layer's decode through :func:`sparse_landmark_decode`: landmark-only
    scoring, top-k selection, and a gather of just the selected blocks -- instead of scoring the
    whole KV cache and masking. Outputs match the shipped decode; only the cost changes.

    Inference only. Composes with :func:`~olmo_core.nn.attention.landmark_prefill_topk.enable_prefill_topk`
    (the prefill branch still calls whatever ``_prefill`` is installed).

    :returns: Number of layers patched.

    :raises RuntimeError: If the model has no fast-landmark attention layers.
    """
    from .landmark_compressive import FastCompressiveLandmarkAttention

    layers = _landmark_layers(model)
    if not layers:
        raise RuntimeError("no landmark attention layers found; sparse decode does not apply")
    for attn in layers:
        compressive = isinstance(attn, FastCompressiveLandmarkAttention)
        attn._sparse_decode_cfg = {
            "compressive": compressive,
            "nonselected_mass": (float(attn.nonselected_landmark_mass) if compressive else 0.0),
        }
        attn._sparse_decode_lmkv = None
        if not hasattr(attn, "_forward_generate_orig"):
            attn._forward_generate_orig = attn._forward_generate
        attn._forward_generate = types.MethodType(_forward_generate_sparse, attn)
    return len(layers)


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
