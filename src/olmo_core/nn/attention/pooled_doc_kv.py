"""
``PooledDocKVAttention`` -- **train-time KV compression for context documents, with off-the-shelf
full-attention inference**.

During training, each context document (``chunk_ids >= 0``) is either **kept** (real per-token K/V,
exactly like full attention) or **pooled**: for every query *outside* the document, the document's
per-token K/V entries are replaced by a single slot holding their (post-RoPE) **mean** key and mean
value, with ``+log(doc_len)`` added to the slot's attention logit. The keep set is gold documents
plus a small random subset of negatives (supplied per example by :func:`install_pooled_doc_keep`
from the gold sidecar, or a seeded random fraction as fallback).

**Why the mean + log-length bias**: a single slot with key ``k_mean``, value ``v_mean`` and logit
bias ``log(L)`` contributes ``L * exp(q . k_mean) * v_mean`` to the softmax numerator and
``L * exp(q . k_mean)`` to the denominator -- *identical* to attending ``L`` copies of
``(k_mean, v_mean)``. So training with this class is **exactly** standard full causal attention over
a perturbed sequence in which each pooled document's KV entries are all replaced by their average.
The attention *function* the model trains under is the test-time function; only the KV *inputs* for
pooled documents are lower-entropy. That is the in-distribution argument for zero-shot transfer to
ordinary full attention at inference (nothing to export, no special eval path). The bias also makes
a pooled document's total softmax mass scale with its length like a real document's diffuse mass
does (``sum_i exp(q.k_i) ~= L * exp(q.k_mean)`` when attention over the doc is diffuse -- the
regime a *negative* document should be in). Note ``exp(q . sum_i k_i)`` (sum-pooling the key) is NOT
that quantity -- it is the *product* of the per-token exponentials and diverges with ``L`` -- so
only mean-pooling is offered; ``len_bias=True`` is the principled "sum of attention mass" variant.

Semantics per query position ``p``:

* ``p`` in FREE / SINK region or in a **kept** document: causal attention over all earlier FREE /
  kept-doc tokens (real KV) + one pooled slot per **fully-preceding** pooled document.
* ``p`` inside a **pooled** document: real causal attention within its own document (so there is no
  future leakage through the pool -- the slot summarizes the *whole* document and is only visible
  once the document has ended) plus the same view of the earlier context as above. Its LM loss is
  therefore still well-defined.
* PAD is never attended; every position may attend itself (NaN guard).

Compute: attention cost drops from ``O(T^2)`` toward ``O(T * (keep_frac * T + n_docs + doc_len))``.
Realized speed requires the block-sparse FlexAttention path (CUDA, long context); the dense
materialized-mask fallback (CPU / short context / tests) is exact but not faster. QKV/MLP FLOPs are
unchanged -- every token still runs the full network to produce the K/V that get pooled -- so
end-to-end savings grow with sequence length.

Inherits the ``chunk_ids`` plumbing (and the hybrid ``full_attention_layers`` escape hatch) from
:class:`~olmo_core.nn.attention.document_chunked.DocumentChunkedAttention`; without ``chunk_ids``
it reduces to plain causal attention.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from .backend import _repeat_kv
from .chunked_mask import PAD_CHUNK_ID, build_chunk_ids_from_tokens
from .document_chunked import (
    _DENSE_MASK_MAX_SEQ_LEN,
    _FLEX_MIN_SEQ_LEN,
    _HAS_FLEX,
    DocumentChunkedAttention,
)

if _HAS_FLEX:  # pragma: no cover - CUDA/flex-only
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    _flex_attention = torch.compile(flex_attention)

log = logging.getLogger(__name__)

# Single-slot BlockMask cache shared across layers within one forward (same rationale as
# ``document_chunked._BLOCK_MASK_CACHE``, but a separate slot: the pooled mask has a different KV
# length and depends on the keep set).
_POOLED_BLOCK_MASK_CACHE: dict = {}


def _splitmix64(x: torch.Tensor) -> torch.Tensor:
    """Deterministic int64 avalanche hash (splitmix64 finalizer; wrap-around multiply is intended)."""
    x = x ^ (x >> 30)
    x = x * -0x40A7B892E31B1A47  # 0xBF58476D1CE4E5B9 as signed int64
    x = x ^ (x >> 27)
    x = x * -0x6B2FB644ECCEEE15  # 0x94D049BB133111EB as signed int64
    return x ^ (x >> 31)


def resolve_keep_docs(
    chunk_ids: torch.Tensor,
    n_docs: int,
    *,
    holder: Optional["PooledDocKeepHolder"],
    keep_prob: float,
    keep_seed: int,
) -> torch.Tensor:
    """
    The ``(B, n_docs)`` bool keep mask for one forward: the holder's gold-aware set when the
    :func:`install_pooled_doc_keep` hook is installed, else a deterministic seeded random fallback
    (a hash of the example's chunk layout and the doc index -- identical across layers,
    activation-checkpoint recompute, and epochs). Shared by :class:`PooledDocKVAttention` (per-layer
    KV pooling) and the soft-token pooling feature on
    :class:`~olmo_core.nn.transformer.model.Transformer`.
    """
    B = chunk_ids.shape[0]
    device = chunk_ids.device
    keep = None if holder is None else holder.keep_docs
    if keep is not None:
        keep = keep.to(device=device, dtype=torch.bool)
        if keep.shape[0] == 1 and B > 1:
            keep = keep.expand(B, -1)
        if keep.shape[0] != B:
            raise RuntimeError(
                f"pooled-KV keep holder batch ({keep.shape[0]}) does not match the forward's "
                f"batch ({B}); the pre-hook and the forward disagree about the batch."
            )
        # Reconcile widths: documents the holder never saw stay REAL (conservative).
        if keep.shape[1] < n_docs:
            pad = torch.ones(B, n_docs - keep.shape[1], dtype=torch.bool, device=device)
            keep = torch.cat([keep, pad], dim=1)
        return keep[:, :n_docs]
    sig = (
        (chunk_ids.to(torch.int64) + 5)
        * torch.arange(1, chunk_ids.shape[1] + 1, dtype=torch.int64, device=device)
    ).sum(dim=-1)
    d = torch.arange(n_docs, dtype=torch.int64, device=device)
    h = _splitmix64(sig[:, None] ^ _splitmix64(d[None, :] + keep_seed))
    u = (h & 0xFFFFFF).to(torch.float32) / float(1 << 24)
    return u < keep_prob


@dataclass
class PooledDocKeepHolder:
    """
    Shared per-step holder: the forward pre-hook installed by :func:`install_pooled_doc_keep` sets
    ``keep_docs`` each forward; every :class:`PooledDocKVAttention` layer reads it. ``keep_docs`` is
    ``(B, n_docs)`` bool -- ``True`` = the document keeps its real per-token KV.
    """

    keep_docs: Optional[torch.Tensor] = None
    n_attached: int = 0
    _hook_handle: object = field(default=None, repr=False)


class PooledDocKVAttention(DocumentChunkedAttention):
    """
    Full causal attention with per-document KV pooling (``AttentionType.pooled_doc_kv``). See the
    module docstring.

    :param keep_prob: Fallback Bernoulli probability that a context document keeps its real KV when
        no :func:`install_pooled_doc_keep` hook is active (a gold-blind control arm). The draw is a
        deterministic seeded hash of the example's chunk layout and the document index, so it is
        identical across layers, activation-checkpoint recompute, and epochs.
    :param keep_seed: Seed for the fallback keep draw.
    :param len_bias: Add ``log(doc_len)`` to each pooled slot's attention logit, making the slot
        exactly equivalent to ``doc_len`` copies of the mean KV entry (the "sum of attention mass"
        form). ``False`` gives the slot the mass of a *single* average token instead.

    See :class:`DocumentChunkedAttention` / :class:`Attention` for the remaining parameters
    (``cross_doc_mode`` and the chunked-pattern knobs are fixed: the non-pooled topology is plain
    causal, because the transfer target is standard full attention).
    """

    def __init__(
        self,
        *,
        keep_prob: float = 0.1,
        keep_seed: int = 42,
        len_bias: bool = True,
        **kwargs,
    ):
        if not (0.0 <= keep_prob <= 1.0):
            raise ValueError(f"keep_prob must be in [0, 1], got {keep_prob}")
        # The parent pattern is unused (this class builds its own mask); "standard" documents the
        # intent that the non-pooled topology is plain causal.
        super().__init__(cross_doc_mode="standard", **kwargs)
        self.keep_prob = float(keep_prob)
        self.keep_seed = int(keep_seed)
        self.len_bias = bool(len_bias)
        # Set by ``install_pooled_doc_keep``: the shared holder whose ``keep_docs`` a forward
        # pre-hook refreshes each forward with this batch's gold-aware keep set.
        self._pooled_keep_holder: Optional[PooledDocKeepHolder] = None

    # ------------------------------------------------------------------
    # Keep-set resolution
    # ------------------------------------------------------------------

    def _resolve_keep_docs(self, chunk_ids: torch.Tensor, n_docs: int) -> torch.Tensor:
        """See :func:`resolve_keep_docs` (shared with the soft-token pooling feature)."""
        return resolve_keep_docs(
            chunk_ids,
            n_docs,
            holder=self._pooled_keep_holder,
            keep_prob=self.keep_prob,
            keep_seed=self.keep_seed,
        )

    # ------------------------------------------------------------------
    # Pooling + mask construction
    # ------------------------------------------------------------------

    @staticmethod
    def _doc_stats(chunk_ids: torch.Tensor, n_docs: int):
        """Per-document flat index, token counts and end positions from ``(B, T)`` chunk ids.

        :returns: ``(flat_idx, is_ctx, counts, doc_end)`` where ``flat_idx`` is the ``(B*T,)``
            flattened ``b * n_docs + doc`` scatter index of the context tokens, ``counts`` is
            ``(B, n_docs)`` float, and ``doc_end`` is ``(B, n_docs)`` int64 (``-1`` if absent).
        """
        B, T = chunk_ids.shape
        device = chunk_ids.device
        is_ctx = chunk_ids >= 0
        doc = chunk_ids.clamp(min=0).to(torch.int64)
        flat = (torch.arange(B, dtype=torch.int64, device=device)[:, None] * n_docs + doc).reshape(
            -1
        )[is_ctx.reshape(-1)]
        ones = torch.ones(flat.shape[0], dtype=torch.float32, device=device)
        counts = torch.zeros(B * n_docs, dtype=torch.float32, device=device).index_add(
            0, flat, ones
        )
        pos = (
            torch.arange(T, dtype=torch.int64, device=device)
            .expand(B, T)
            .reshape(-1)[is_ctx.reshape(-1)]
        )
        doc_end = torch.full((B * n_docs,), -1, dtype=torch.int64, device=device).scatter_reduce(
            0, flat, pos, reduce="amax", include_self=True
        )
        return flat, is_ctx, counts.reshape(B, n_docs), doc_end.reshape(B, n_docs)

    @staticmethod
    def _pool_kv(
        t: torch.Tensor, flat_idx: torch.Tensor, is_ctx: torch.Tensor, counts: torch.Tensor
    ) -> torch.Tensor:
        """Mean-pool ``(B, T, H, D)`` per document -> ``(B, n_docs, H, D)`` (differentiable)."""
        B, T, H, D = t.shape
        n_docs = counts.shape[1]
        src = t.reshape(B * T, H * D)[is_ctx.reshape(-1)]
        summed = torch.zeros(B * n_docs, H * D, dtype=t.dtype, device=t.device).index_add(
            0, flat_idx, src
        )
        mean = summed / counts.reshape(B * n_docs, 1).clamp(min=1.0).to(t.dtype)
        return mean.reshape(B, n_docs, H, D)

    def _build_pooled_additive_mask(
        self,
        chunk_ids: torch.Tensor,
        keep: torch.Tensor,
        counts: torch.Tensor,
        doc_end: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        The ``(B, 1, T, T + n_docs)`` additive attention bias: real-KV columns first, one pooled
        slot per document appended. ``0`` / ``+log(doc_len)`` where allowed, ``finfo.min`` where not.
        """
        B, T = chunk_ids.shape
        n_docs = keep.shape[1]
        device = chunk_ids.device
        cid = chunk_ids.to(torch.int64)
        pos = torch.arange(T, device=device)

        notpad = cid != PAD_CHUNK_ID  # (B, T)
        kv_pooled = (cid >= 0) & ~torch.gather(keep, 1, cid.clamp(min=0))  # (B, T)
        causal = pos[:, None] >= pos[None, :]  # (T_q, T_kv)
        same_doc = (cid[:, :, None] == cid[:, None, :]) & (cid[:, None, :] >= 0)  # (B, T_q, T_kv)
        real_ok = causal[None] & notpad[:, None, :] & (~kv_pooled[:, None, :] | same_doc)
        # NaN guard: the self edge is always allowed (feeds PAD queries something to attend).
        real_ok |= torch.eye(T, dtype=torch.bool, device=device)[None]

        present_pooled = (counts > 0) & ~keep  # (B, n_docs)
        # A slot is visible only once its document has fully ended before the query.
        slot_ok = (
            present_pooled[:, None, :]
            & (pos[None, :, None] > doc_end[:, None, :])
            & notpad[:, :, None]
        )  # (B, T_q, n_docs)

        finfo_min = torch.finfo(dtype).min
        zero = torch.zeros((), dtype=dtype, device=device)
        neg = torch.full((), finfo_min, dtype=dtype, device=device)
        mask_real = torch.where(real_ok, zero, neg)
        slot_bias = (
            counts.clamp(min=1.0).log().to(dtype)[:, None, :]
            if self.len_bias
            else zero.expand(B, 1, n_docs)
        )
        mask_slot = torch.where(slot_ok, slot_bias, neg)
        return torch.cat([mask_real, mask_slot], dim=-1).unsqueeze(1)

    # ------------------------------------------------------------------
    # FlexAttention fast path (block-sparse; the source of the actual speedup)
    # ------------------------------------------------------------------

    def _pooled_flex(
        self,
        q: torch.Tensor,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        chunk_ids: torch.Tensor,
        keep: torch.Tensor,
        counts: torch.Tensor,
        doc_end: torch.Tensor,
    ) -> Optional[torch.Tensor]:  # pragma: no cover - CUDA-only
        """Block-sparse FlexAttention over ``(T_q, T + n_docs)``; ``None`` if flex cannot run."""
        B, _, T, _ = q.shape
        n_docs = keep.shape[1]
        cid = chunk_ids.to(torch.int64)
        keep_i = keep
        d_end = doc_end
        present_pooled = (counts > 0) & ~keep
        log_counts = counts.clamp(min=1.0).log().to(torch.float32)
        len_bias = self.len_bias

        def mask_mod(b, h, q_idx, kv_idx):
            qc = cid[b, q_idx]
            is_real = kv_idx < T
            kvr = torch.clamp(kv_idx, max=T - 1)
            kc = cid[b, kvr]
            kv_pooled = (kc >= 0) & ~keep_i[b, torch.clamp(kc, min=0)]
            real_ok = (
                (kv_idx <= q_idx) & (kc != PAD_CHUNK_ID) & (~kv_pooled | ((kc == qc) & (kc >= 0)))
            ) | (kv_idx == q_idx)
            d = torch.clamp(kv_idx - T, min=0)
            slot_ok = present_pooled[b, d] & (q_idx > d_end[b, d]) & (qc != PAD_CHUNK_ID)
            return torch.where(is_real, real_ok, slot_ok)

        def score_mod(score, b, h, q_idx, kv_idx):
            d = torch.clamp(kv_idx - T, min=0)
            return torch.where(kv_idx >= T, score + log_counts[b, d], score)

        key = (
            id(self._chunk_ids),
            self._chunk_ids.data_ptr(),
            self._chunk_ids._version,
            keep.data_ptr(),
            B,
            T,
            n_docs,
            self.flex_block_size,
            str(q.device),
        )
        try:
            cached = _POOLED_BLOCK_MASK_CACHE.get("cur")
            if cached is not None and cached[0] == key:
                block_mask = cached[1]
            else:
                block_mask = create_block_mask(
                    mask_mod,
                    B,
                    None,
                    T,
                    T + n_docs,
                    device=q.device,
                    BLOCK_SIZE=(self.flex_block_size, self.flex_block_size),
                )
                _POOLED_BLOCK_MASK_CACHE["cur"] = (key, block_mask)
            out = _flex_attention(
                q.contiguous(),
                k_all.contiguous(),
                v_all.contiguous(),
                score_mod=score_mod if len_bias else None,
                block_mask=block_mask,
                scale=self.softmax_scale,
            )
            return out.transpose(1, 2).contiguous()
        except torch.cuda.OutOfMemoryError:
            _POOLED_BLOCK_MASK_CACHE.pop("cur", None)
            torch.cuda.empty_cache()
            self._force_eager_mask = True
            log.warning(
                f"Pooled-KV FlexAttention OOM at seq_len={T}; falling back to the dense mask."
            )
            return None
        except Exception as e:
            self._force_eager_mask = True
            log.warning(f"Pooled-KV FlexAttention failed ({e}); falling back to the dense mask.")
            return None

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def _sdpa_masked(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Pooled-KV attention over the full ``(B, T, H, D)`` q/k/v; plain causal without roles."""
        if self._chunk_ids is None:
            return super()._sdpa_masked(q, k, v)

        B, T, n_heads, _ = q.shape
        chunk_ids = self._prep_chunk_ids(T, q.device, B)
        n_docs = int(chunk_ids.max().item()) + 1
        if n_docs <= 0:
            # No context documents in this batch (e.g. mask-mix collapsed the roles) -> the mask
            # below would be plain causal anyway; skip the pooling machinery.
            saved, self._chunk_ids = self._chunk_ids, None
            try:
                return super()._sdpa_masked(q, k, v)
            finally:
                self._chunk_ids = saved

        keep = self._resolve_keep_docs(chunk_ids, n_docs)
        flat_idx, is_ctx, counts, doc_end = self._doc_stats(chunk_ids, n_docs)

        # Pooled slots (computed for every doc, masked off for kept ones -- simpler and the waste is
        # tiny), appended after the real KV. Pool per KV head, then GQA-expand everything together.
        k_pool = self._pool_kv(k, flat_idx, is_ctx, counts)
        v_pool = self._pool_kv(v, flat_idx, is_ctx, counts)
        k_all = torch.cat([k, k_pool], dim=1)
        v_all = torch.cat([v, v_pool], dim=1)

        n_rep = n_heads // k.shape[2]
        k_all = _repeat_kv(k_all, n_rep).transpose(1, 2)  # (B, H, T + n_docs, D)
        v_all = _repeat_kv(v_all, n_rep).transpose(1, 2)
        q = q.transpose(1, 2)  # (B, H, T, D)

        if _HAS_FLEX and q.is_cuda and T >= _FLEX_MIN_SEQ_LEN and not self._force_eager_mask:
            out = self._pooled_flex(q, k_all, v_all, chunk_ids, keep, counts, doc_end)
            if out is not None:
                return out
            if T > _DENSE_MASK_MAX_SEQ_LEN:
                gib = 2 * B * T * (T + n_docs) / 2**30
                raise RuntimeError(
                    f"PooledDocKVAttention: FlexAttention could not run at seq_len={T} and the "
                    f"dense fallback mask would need ~{gib:.0f} GiB (B={B}). Reduce the sequence "
                    f"length or batch size (the dense fallback is only viable at seq_len <= "
                    f"{_DENSE_MASK_MAX_SEQ_LEN})."
                )

        attn_mask = self._build_pooled_additive_mask(
            chunk_ids, keep, counts, doc_end, dtype=q.dtype
        )
        out = F.scaled_dot_product_attention(
            q,
            k_all,
            v_all,
            attn_mask=attn_mask,
            dropout_p=self._dropout_p if self.training else 0.0,
            is_causal=False,  # the mask already encodes causality
            scale=self.softmax_scale,
        )
        return out.transpose(1, 2).contiguous()


# ---------------------------------------------------------------------------
# Gold-aware keep-set installer (mirrors gold_grad_mask's fingerprint machinery)
# ---------------------------------------------------------------------------

# A function mapping a batch's ``input_ids`` (B, S) -> ``(B, n_docs)`` bool keep-docs mask.
KeepDocsFn = Callable[[torch.Tensor], torch.Tensor]


def _flatten_gold(gold) -> list:
    """Gold sidecar values are either doc ids or id pairs; return the flat id list."""
    out = []
    for g in gold:
        if isinstance(g, (list, tuple)):
            out.extend(int(x) for x in g)
        else:
            out.append(int(g))
    return out


def make_fingerprint_keep_docs_fn(
    gold_table: Dict[str, Iterable[Any]],
    *,
    doc_start_id: int,
    doc_end_id: int,
    eos_id: int,
    n_random: int = 2,
    n_random_range: Optional[Tuple[int, int]] = None,
    n_random_frac: Optional[float] = None,
    mode: str = "gold_plus_random",
    seed: int = 0,
    n_gold: int = 0,
    n_pairs: int = 1,
    debug_calls: int = 12,
    mix_start_p: float = 0.0,
    mix_end_p: float = 0.0,
    mix_total_calls: int = 0,
) -> KeepDocsFn:
    """
    Build a ``keep_docs_fn(input_ids) -> (B, n_docs) bool`` for :func:`install_pooled_doc_keep`:
    look up each row's gold documents in ``gold_table`` (``{content_fingerprint: gold ids or
    pairs}``, the same sidecar format as
    :func:`~olmo_core.nn.attention.gold_grad_mask.make_fingerprint_gold_mask_fn`), apply the
    :func:`~olmo_core.nn.attention.gold_grad_mask.select_keep_docs` policy, and mark the selected
    documents as keeping their real per-token KV. A fingerprint miss (e.g. the trainer's synthetic
    warmup batch) keeps **all** documents real (degrades to full attention, never to a wrong pool).

    :param n_random: Random non-gold documents kept real per example (the "subset of negatives").
    :param n_random_frac: If given, the negative count is a FIXED FRACTION of the row's non-gold
        documents (``round(frac * n_non_gold)``, at least 1), so the kept share -- and hence the
        compression ratio -- is the same at every context length (overrides ``n_random`` and
        ``n_random_range``). This is the scale-invariant setting for a length-mix scaling study.
    :param n_random_range: If given, ``(lo, hi)`` — each (row, call) draws its own negative count
        log-uniformly in ``[lo, hi]`` (overrides ``n_random``). Training at varied candidate
        breadths teaches ranking that is scale-invariant, so the eval regime (every doc real) is
        not out-of-distribution. The draw is per (fingerprint, call): a row's breadth varies
        across epochs.
    :param mode: A :func:`select_keep_docs` policy (``"gold_plus_random"``, ``"gold_subsample"``,
        ``"random_only"``, ``"random_nongold"``, ``"gold_pair"``, ``"gold_halves"``).
    :param seed: Base seed; the per-example draw is seeded with ``f"{seed}:{fingerprint}"`` so it is
        stable across epochs and layers.
    :param mix_start_p: **Compression-mixing curriculum** (the pooled analogue of the proven
        mask-mixing curriculum): each row independently trains UNCOMPRESSED (keep every doc) with
        probability ``p``, annealed linearly from ``mix_start_p`` to ``mix_end_p`` over
        ``mix_total_calls`` invocations of this fn (== forwards on this rank). Zero-shot pooled ->
        full transfer collapses without it (f1 0.985 -> 0.08-0.16 measured); the curriculum keeps
        the model anchored to the full-attention task while the compressed examples make it cheap.
        The draw is seeded per (fingerprint, call index): deterministic given the data order, and a
        given example flips between compressed/full across epochs.
    """
    import random as _random

    from .gold_grad_mask import content_fingerprint_from_row, select_keep_docs

    table: Dict[str, Set[int]] = {}
    pair_table: Dict[str, List[List[int]]] = {}
    for fp, val in gold_table.items():
        items = list(val)
        if items and isinstance(items[0], (list, tuple)):
            pair_table[fp] = [[int(a) for a in p] for p in items]
            table[fp] = {int(a) for p in items for a in p}
        else:
            table[fp] = {int(i) for i in items}
    if mode in ("gold_pair", "gold_halves") and not pair_table:
        raise ValueError(
            f"mode {mode!r} needs a PAIR-preserving gold sidecar (values like [[6, 19], [18, 48]])."
        )
    state = {"calls": 0, "rows": 0, "hits": 0}

    def fn(input_ids: torch.Tensor) -> torch.Tensor:
        # Curriculum probability for THIS call (linear anneal; constant mix_start_p if no total).
        if mix_total_calls > 0:
            frac = min(1.0, state["calls"] / max(1, mix_total_calls))
            p_full = mix_start_p + (mix_end_p - mix_start_p) * frac
        else:
            p_full = mix_start_p
        ids_cpu = input_ids.detach().to("cpu")
        roles = build_chunk_ids_from_tokens(
            ids_cpu, doc_start_id=doc_start_id, doc_end_id=doc_end_id, eos_id=eos_id, mode="chunked"
        )
        B, _ = roles.shape
        n_docs = max(int(roles.max().item()) + 1, 1)
        keep = torch.ones(B, n_docs, dtype=torch.bool)
        ids2d = ids_cpu.tolist()
        n_found = 0
        n_mixed = 0
        for b in range(B):
            fp = content_fingerprint_from_row(ids2d[b], eos_id)
            gold = table.get(fp)
            if gold is None:
                continue  # unknown example -> all docs stay real
            n_found += 1
            # Compression-mixing curriculum: this row trains uncompressed with probability p_full.
            if (
                p_full > 0.0
                and _random.Random(f"mix:{seed}:{fp}:{state['calls']}").random() < p_full
            ):
                n_mixed += 1
                continue  # keep[b] stays all-True
            row_roles = roles[b]
            present = [int(d) for d in torch.unique(row_roles[row_roles >= 0]).tolist()]
            rng = _random.Random(f"{seed}:{fp}")
            n_rand_row = n_random
            if n_random_frac is not None:
                n_non_gold = max(0, len(present) - len(set(int(g) for g in _flatten_gold(gold))))
                n_rand_row = max(1, int(round(n_random_frac * n_non_gold)))
            elif n_random_range is not None:
                lo, hi = n_random_range
                u = _random.Random(f"nr:{seed}:{fp}:{state['calls']}").uniform(
                    math.log(max(1, lo)), math.log(max(1, hi))
                )
                n_rand_row = min(hi, max(lo, int(round(math.exp(u)))))
            keep_docs = select_keep_docs(
                present,
                gold,
                n_random=n_rand_row,
                mode=mode,
                rng=rng,
                n_gold=n_gold,
                gold_pairs=pair_table.get(fp),
                n_pairs=n_pairs,
            )
            keep[b] = False
            for d in keep_docs:
                if 0 <= d < n_docs:
                    keep[b, d] = True
        state["calls"] += 1
        state["rows"] += B
        state["hits"] += n_found
        if state["calls"] <= debug_calls:
            n_pool = int((~keep).sum().item())
            tag = " (warmup mock)" if n_found == 0 and state["calls"] == 1 else ""
            print(
                f"[pooled-kv] call#{state['calls']}{tag}: B={B} n_docs<={n_docs} mode={mode} "
                f"n_random={n_random} fp_hits={n_found}/{B} cum_hits={state['hits']}/{state['rows']} "
                f"pooled_docs={n_pool} p_full={p_full:.2f} mixed={n_mixed}",
                flush=True,
            )
        elif p_full > 0.0 and state["calls"] % 100 == 0:
            print(
                f"[pooled-kv] call#{state['calls']}: p_full={p_full:.3f} mixed={n_mixed}/{B}",
                flush=True,
            )
        return keep

    return fn


def install_pooled_doc_keep(
    model: torch.nn.Module, keep_docs_fn: KeepDocsFn
) -> PooledDocKeepHolder:
    """
    Install gold-aware keep-set selection for :class:`PooledDocKVAttention` on ``model`` in place:
    a ``forward_pre_hook`` computes the ``(B, n_docs)`` keep mask from the live batch's
    ``input_ids`` via ``keep_docs_fn`` once per forward, and every :class:`PooledDocKVAttention`
    layer reads it from the shared holder (so all layers pool the same documents).

    :returns: the :class:`PooledDocKeepHolder` (records how many layers were attached).
    """
    holder = PooledDocKeepHolder()

    def pre_hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None:
            return
        holder.keep_docs = keep_docs_fn(input_ids)

    holder._hook_handle = model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    n = 0
    for m in model.modules():
        if isinstance(m, PooledDocKVAttention):
            m._pooled_keep_holder = holder
            n += 1
        elif getattr(m, "_pooled_soft_tokens", None) is not None:
            # A Transformer with soft-token pooling enabled (duck-typed to avoid the circular
            # import); it reads the same holder in its compaction step.
            m._pooled_keep_holder = holder
            n += 1
    holder.n_attached = n
    if n == 0:
        log.warning(
            "install_pooled_doc_keep: no PooledDocKVAttention layers and no soft-token-enabled "
            "Transformer found on the model; the hook will compute keep masks nobody reads."
        )
    return holder
