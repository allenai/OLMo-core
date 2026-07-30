"""
Hard top-k landmark retrieval applied to the **whole prefill**, not just the decode steps.

Motivation
----------
Landmark / compressive-landmark eval today is top-k only at *decode*: each generated query (plus the
final prompt token, which the generation loop decodes rather than prefills) scores the cached
landmark keys, keeps the top ``k`` blocks and zeroes the rest
(:meth:`~olmo_core.nn.attention.landmark_fast.FastLandmarkAttention.set_landmark_eval_decode`). The
**prefill** is untouched: every prompt token still soft-gates over *all* past blocks, i.e. the prompt
representations the decode reads from were built with dense attention. So the reported numbers are
not what a genuinely sparse (O(k * block) per query) landmark model would produce.

This module implements the missing half: an eager, forward-only landmark attention where **every
query position** -- prompt tokens included -- keeps only its top-k past landmark blocks. Pointing an
eval at it answers "how much accuracy do we lose when top-k is applied everywhere, not just at
decode?".

It is deliberately standalone: nothing here modifies the training kernels or the existing decode
path. :func:`enable_prefill_topk` monkeypatches ``_prefill`` on the landmark layers of an
already-built model at eval time, and :func:`disable_prefill_topk` restores it.

Semantics
---------
For a query at absolute position ``p`` in block ``qb = p // block_size``:

* **past blocks** are blocks ``b < qb``; the block's gate score is its landmark key's score;
* the ``top_k`` highest-scoring past blocks are *selected*, all other past blocks get exactly zero
  weight (matching :meth:`FastLandmarkAttention._decode_apply_topk_landmark_retrieval`); a query with
  ``<= top_k`` past blocks keeps all of them;
* the **gate softmax** runs over the selected blocks' landmark keys plus the query's own (local)
  block, exactly as in training/decode;
* each selected block's gate weight is distributed over the block's tokens by a within-block
  softmax: content-only for the plain landmark variant, content **+ landmark** for the compressive
  variant (the landmark's value is the block's compressed summary);
* ``nonselected_mass`` (compressive only, ``alpha``) optionally reserves a fixed fraction of the mass
  for the *non-selected* blocks' landmark tokens, split by a softmax over their landmark scores --
  the prefill counterpart of
  :meth:`~olmo_core.nn.attention.landmark_compressive.FastCompressiveLandmarkAttention._compressive_decode_probs`.

With ``top_k=None`` and ``nonselected_mass=0`` this reduces to the exact dense grouped-softmax that
the fused Triton kernels compute, which is how the implementation is validated (see
``debug/prefill_topk/test_prefill_topk.py``).

Cost
----
This is an *accuracy* probe, not a speed one: it materializes ``(B, H, tile, T)`` score tiles, so it
is O(T^2) like dense attention (the kernel it replaces is too). ``query_tile`` bounds the peak
memory.
"""

import math
import types
from typing import Any, List, Optional

import torch

__all__ = [
    "landmark_topk_prefill_attention",
    "enable_prefill_topk",
    "disable_prefill_topk",
]


@torch.no_grad()
def landmark_topk_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int,
    softmax_scale: float,
    top_k: Optional[int] = None,
    compressive: bool = False,
    nonselected_mass: float = 0.0,
    query_tile: int = 128,
) -> torch.Tensor:
    """
    Landmark self-attention over a full prompt with per-query hard top-k block retrieval.

    :param q: Queries ``(B, H, T, D)``. ``T`` must be a multiple of ``block_size``.
    :param k: Keys ``(B, H, T, D)`` (GQA heads already expanded); no history KV.
    :param v: Values ``(B, H, T, D)``.
    :param block_size: Landmark block size (``mem_freq + 1``); the landmark token of a block is its
        last position.
    :param softmax_scale: Score scale (usually ``head_dim ** -0.5``).
    :param top_k: Number of past blocks each query may attend, or ``None`` for dense soft gating
        over all past blocks (the training/prefill default).
    :param compressive: Include each block's landmark token in the within-block softmax (the
        compressive variant), instead of gating content only.
    :param nonselected_mass: ``alpha`` in ``[0, 1)``: mass reserved for the non-selected blocks'
        landmark tokens. Only meaningful with ``top_k`` set and ``compressive=True``.
    :param query_tile: Number of query positions processed at once (peak memory knob).

    :returns: Attention output ``(B, H, T, D)``.
    """
    B, H, T, _ = q.shape
    if k.shape[2] != T or v.shape[2] != T:
        raise ValueError(
            f"prefill top-k expects self-attention with no history KV (got q T={T}, k T={k.shape[2]})"
        )
    Lb = block_size
    if T % Lb != 0:
        raise ValueError(f"sequence length {T} is not a multiple of block_size {Lb}")
    if not (0.0 <= nonselected_mass < 1.0):
        raise ValueError(f"nonselected_mass must be in [0, 1) (got {nonselected_mass})")
    n_blocks = T // Lb
    device = q.device

    key_pos = torch.arange(T, device=device)
    key_block = key_pos // Lb
    is_lm_key = (key_pos % Lb) == (Lb - 1)
    blk_idx = torch.arange(n_blocks, device=device)

    kf = k.float()
    out = torch.empty_like(q)

    for m0 in range(0, T, query_tile):
        m1 = min(m0 + query_tile, T)
        M = m1 - m0
        pos = torch.arange(m0, m1, device=device)
        q_block = pos // Lb
        # A landmark-position query does not attend to itself (the training kernel decrements the
        # causal bound on a block's last row), so its bound is p - 1.
        bound = torch.where((pos % Lb) == (Lb - 1), pos - 1, pos)

        causal = key_pos[None, :] <= bound[:, None]  # (M, T)
        past = key_block[None, :] < q_block[:, None]  # (M, T)
        local = (key_block[None, :] == q_block[:, None]) & causal  # (M, T)

        # (B, H, M, T) scores in fp32 (the kernel accumulates its dots in fp32 too).
        scores = torch.matmul(q[:, :, m0:m1].float(), kf.transpose(-1, -2)) * softmax_scale
        neg = torch.finfo(scores.dtype).min

        scores_blk = scores.view(B, H, M, n_blocks, Lb)
        lm_scores = scores_blk[..., Lb - 1]  # (B, H, M, n_blocks) block gate scores
        valid_blk = blk_idx[None, :] < q_block[:, None]  # (M, n_blocks) strictly-past blocks

        if top_k is not None and top_k < n_blocks:
            masked_lm = lm_scores.masked_fill(~valid_blk, neg)
            idx = masked_lm.topk(top_k, dim=-1).indices
            keep = torch.zeros_like(masked_lm, dtype=torch.bool).scatter_(-1, idx, True)
            # Rows with fewer than top_k past blocks: the padding picks belong to invalid blocks and
            # are dropped here, which is exactly "keep every past block".
            keep = keep & valid_blk
        else:
            keep = valid_blk.expand(B, H, M, n_blocks)

        keep_key = keep.repeat_interleave(Lb, dim=-1)  # (B, H, M, T)
        gate_set = (keep_key & is_lm_key[None, None, None, :] & past[None, None]) | local[
            None, None
        ]
        gate_w = torch.softmax(scores.masked_fill(~gate_set, neg), dim=-1)  # (B, H, M, T)

        # Local (own-block) keys keep their gate weight directly.
        probs = torch.where(local[None, None], gate_w, gate_w.new_zeros(()))

        # Past blocks: the block's gate weight spreads over its tokens by the within-block softmax.
        within_logits = scores_blk
        if not compressive:
            # Plain landmark: the landmark token itself contributes no value.
            within_logits = within_logits.clone()
            within_logits[..., Lb - 1] = neg
        within = torch.softmax(within_logits, dim=-1)  # (B, H, M, n_blocks, Lb)
        block_gate = gate_w.view(B, H, M, n_blocks, Lb)[..., Lb - 1]  # 0 for non-selected blocks
        contrib = (block_gate * keep).unsqueeze(-1) * within
        probs = probs + contrib.reshape(B, H, M, T) * past[None, None]

        if nonselected_mass > 0.0 and top_k is not None:
            nonsel = valid_blk & ~keep  # (B, H, M, n_blocks)
            has_ns = nonsel.any(-1)  # (B, H, M)
            ns_key = (
                nonsel.repeat_interleave(Lb, dim=-1)
                & is_lm_key[None, None, None, :]
                & past[None, None]
            )
            ns_w = torch.softmax(scores.masked_fill(~ns_key, neg), dim=-1)
            alpha = (has_ns.to(probs.dtype) * nonselected_mass).unsqueeze(-1)
            probs = probs * (1.0 - alpha) + alpha * ns_w

        out[:, :, m0:m1] = torch.matmul(probs.to(v.dtype), v)

    return out


def _prefill_topk(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Replacement for ``FastLandmarkAttention._prefill`` that applies per-query top-k retrieval."""
    cfg = self._prefill_topk_cfg
    Lb = self.block_size
    T = q.shape[2]
    pad = (-T) % Lb
    if pad:
        q = torch.nn.functional.pad(q, (0, 0, 0, pad))
        k = torch.nn.functional.pad(k, (0, 0, 0, pad))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad))
    n_blocks = q.shape[2] // Lb
    top_k = cfg["top_k"]
    if top_k is None and cfg["top_k_fraction"] is not None:
        top_k = max(1, math.ceil(cfg["top_k_fraction"] * n_blocks))
    att = landmark_topk_prefill_attention(
        q,
        k,
        v,
        block_size=Lb,
        softmax_scale=self.softmax_scale,
        top_k=top_k,
        compressive=cfg["compressive"],
        nonselected_mass=cfg["nonselected_mass"],
        query_tile=cfg["query_tile"],
    )
    return att[:, :, :T]


def _landmark_layers(model: torch.nn.Module) -> List[Any]:
    """All fast-landmark attention layers of ``model`` (``FastCompressiveLandmarkAttention``
    subclasses ``FastLandmarkAttention``, so one isinstance check covers both)."""
    from .landmark_fast import FastLandmarkAttention

    return [m for m in model.modules() if isinstance(m, FastLandmarkAttention)]


def enable_prefill_topk(
    model: torch.nn.Module,
    *,
    top_k: Optional[int] = None,
    top_k_fraction: Optional[float] = None,
    nonselected_mass: Optional[float] = None,
    query_tile: int = 128,
) -> int:
    """
    Route every landmark layer's prefill through :func:`landmark_topk_prefill_attention`, so *all*
    prompt positions -- not just the decoded ones -- see only their top-k landmark blocks.

    Inference only; the training path is untouched. Call :func:`disable_prefill_topk` to restore.

    :param model: A built (eval-mode) transformer with fast-landmark attention layers.
    :param top_k: Fixed number of past blocks per query. Takes precedence over ``top_k_fraction``.
    :param top_k_fraction: ``k = ceil(fraction * num_prompt_blocks)``, the same rule the decode uses.
    :param nonselected_mass: ``alpha`` for compressive layers; ``None`` reuses each layer's own
        :attr:`nonselected_landmark_mass`.
    :param query_tile: Query positions per tile (peak-memory knob).

    :returns: Number of layers patched.

    :raises RuntimeError: If the model has no fast-landmark attention layers.
    """
    from .landmark_compressive import FastCompressiveLandmarkAttention

    layers = _landmark_layers(model)
    if not layers:
        raise RuntimeError(
            "no FastLandmarkAttention / FastCompressiveLandmarkAttention layers found; "
            "prefill top-k only applies to landmark models"
        )
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1 or None (got {top_k})")
    for attn in layers:
        compressive = isinstance(attn, FastCompressiveLandmarkAttention)
        alpha = 0.0
        if compressive:
            alpha = (
                float(attn.nonselected_landmark_mass)
                if nonselected_mass is None
                else float(nonselected_mass)
            )
        attn._prefill_topk_cfg = {
            "top_k": top_k,
            "top_k_fraction": top_k_fraction,
            "compressive": compressive,
            "nonselected_mass": alpha,
            "query_tile": query_tile,
        }
        if not hasattr(attn, "_prefill_orig"):
            attn._prefill_orig = attn._prefill
        attn._prefill = types.MethodType(_prefill_topk, attn)
    return len(layers)


def disable_prefill_topk(model: torch.nn.Module) -> int:
    """Undo :func:`enable_prefill_topk`. Returns the number of layers restored."""
    n = 0
    for attn in _landmark_layers(model):
        if hasattr(attn, "_prefill_orig"):
            attn._prefill = attn._prefill_orig
            del attn._prefill_orig
            n += 1
    return n
