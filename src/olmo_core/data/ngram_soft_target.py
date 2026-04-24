"""
Training-time soft-target lookup for ngram-train.

Reads per-n hash tables built by ``data_gen/arpa_to_table.py`` and, for any
given context, combines observed continuations across orders via **modified
Kneser-Ney cross-order backoff** using the per-prefix backoff weights
KenLM stored in the ARPA. Returns the top-K most probable continuations
with their renormalized probabilities.

Inputs (per position): a context of up to ``N_max - 1`` tokens (the
caller — an InstanceSource — is responsible for walking left from the
target position and stopping at the preceding EOS so the window doesn't
cross document boundaries).

Outputs (per position): ``(topk_ids: int32[K], topk_probs: float32[K])``
with ``topk_probs`` summing to 1.

Full-KN combine rule
--------------------
Given a context h of length L, we want P_KN(w | h) for all candidate
continuations w. KenLM's modified-KN semantics:

    P_KN(w | h) = P_observed(w | h)              if (h, w) was observed
                = α(h) · P_KN(w | h[1:])         otherwise

So in a loop from the longest-queryable order down to order 1:

    log10_cumulative_backoff = 0
    for order in [min(L+1, N_max), ..., 1]:
        h_suffix = context[-(order-1):]  (empty tuple when order=1)
        if probe(table[order], h_suffix) hits:
            for each observed continuation (w, log10_P_obs) at this order:
                if w is not already in candidates:
                    candidates[w] = log10_cumulative_backoff + log10_P_obs
            log10_cumulative_backoff += slot.backoff_log10
        # else: no cumulative-backoff update (α = 1 for ARPA-absent prefix)
    take top-K of candidates by log10 probability
    renormalize top-K linear probabilities to sum to 1

Unigram optimization
--------------------
When we fall through to the order-1 table (the empty-prefix unigram
distribution), only the top ``unigram_shortlist`` unigrams by unconditional
probability are considered as candidates. Any unigram outside the
shortlist would have negligible post-backoff probability even if added.
Avoids iterating all ~100K vocab tokens on every position lookup.

Status
------
- Pure-Python reference implementation for correctness + unit testing.
- numba-JIT'd batch path is a deferred optimization (TODO) for when we
  benchmark the training loop.
"""

from __future__ import annotations

import math
import os
import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# File format constants mirroring data_gen/arpa_to_table.py. We duplicate
# a handful of symbols rather than importing from there because this module
# lives in olmo-core and shouldn't reach outside the olmo-core src tree for
# a dependency on the build-side script.
_MAGIC = b"NGT1"
_VERSION = 1
_HEADER_SIZE = 64
_SLOT_DTYPE = np.dtype(
    [
        ("hash_key", "<u8"),
        ("digest", "<u4"),
        ("backoff_log10", "<f4"),
        ("cont_offset", "<u8"),
        ("num_cont", "<u4"),
        ("_pad", "<u4"),
    ],
    align=False,
)
assert _SLOT_DTYPE.itemsize == 32


# ---------------------------------------------------------------------------
# Hashing — must match what arpa_to_table.py used when packing.
# ---------------------------------------------------------------------------
# We prefer xxh3 when available (faster), fall back to SHA-256-truncated
# otherwise. The digest (collision second-hash) always uses SHA-256 bytes
# [8:12] so that digest and hash_key are independent even on the fallback
# path. This must match arpa_to_table.py's _hash_key / _digest.

try:
    import xxhash  # type: ignore

    _HAVE_XXHASH = True
except ImportError:
    _HAVE_XXHASH = False

import hashlib


def _prefix_to_bytes(prefix_tokens: tuple[int, ...]) -> bytes:
    if not prefix_tokens:
        return b""
    return struct.pack(f"<{len(prefix_tokens)}I", *prefix_tokens)


def _hash_key(prefix_bytes: bytes) -> int:
    if _HAVE_XXHASH:
        return int(xxhash.xxh3_64(prefix_bytes).intdigest())
    return int.from_bytes(hashlib.sha256(prefix_bytes).digest()[:8], "little")


def _digest(prefix_bytes: bytes) -> int:
    return int.from_bytes(hashlib.sha256(prefix_bytes).digest()[8:12], "little")


def _fixup_hash_key(h: int) -> int:
    return h | 1  # reserve 0 as "empty slot" sentinel


# ---------------------------------------------------------------------------
# Binary table reader
# ---------------------------------------------------------------------------


class _NgramTable:
    """Memory-mapped view of a single ``ngram_table_n<N>.bin`` file.

    Keeps three numpy views over regions of the same mmap:

    - ``slots[capacity]`` — the Robin-Hood-probed hash table
    - ``tokens[total_cont]`` — all continuation token IDs, flat
    - ``probs[total_cont]``  — their KenLM-smoothed log10 probabilities, flat

    A prefix's continuations live at ``tokens[off:off+n]`` and
    ``probs[off:off+n]`` where (off, n) come from its slot.
    """

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            header = f.read(_HEADER_SIZE)
        if len(header) < _HEADER_SIZE:
            raise ValueError(f"{path}: header short ({len(header)} bytes)")
        (
            magic,
            version,
            order,
            capacity,
            total_cont,
            load_factor,
            slots_off,
            tokens_off,
            probs_off,
        ) = struct.unpack("<4sIIQQfQQQ", header[:56])
        if magic != _MAGIC:
            raise ValueError(f"{path}: magic {magic!r} != expected {_MAGIC!r}")
        if version != _VERSION:
            raise ValueError(f"{path}: version {version} != expected {_VERSION}")
        self.order = int(order)
        self.capacity = int(capacity)
        self.capacity_mask = int(capacity - 1)
        self.total_continuations = int(total_cont)
        self.load_factor = float(load_factor)

        file_size = os.path.getsize(path)
        self._mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(file_size,))
        self.slots = np.frombuffer(
            self._mm, dtype=_SLOT_DTYPE, count=self.capacity, offset=slots_off
        )
        self.tokens = np.frombuffer(
            self._mm, dtype=np.uint32, count=self.total_continuations, offset=tokens_off
        )
        self.probs = np.frombuffer(
            self._mm, dtype=np.float16, count=self.total_continuations, offset=probs_off
        )

    def probe(self, prefix_tokens: tuple[int, ...]) -> Optional[Tuple[int, int, float]]:
        """Return (num_cont, cont_offset, backoff_log10), or None on miss.

        Linear-probe through slots starting at ``hash_key & capacity_mask``.
        A slot with ``hash_key == 0`` means "never occupied" — miss. A slot
        with matching hash_key + digest is the hit. Otherwise (bucket
        collision), advance one slot and keep probing.
        """
        pb = _prefix_to_bytes(prefix_tokens)
        h = _fixup_hash_key(_hash_key(pb))
        d = _digest(pb)
        slot = h & self.capacity_mask
        # Bounded probe: Robin Hood guarantees O(log n) worst case; cap at
        # capacity to be defensive against a malformed file.
        for _ in range(self.capacity):
            slot_hash = int(self.slots["hash_key"][slot])
            if slot_hash == 0:
                return None
            if slot_hash == h and int(self.slots["digest"][slot]) == d:
                return (
                    int(self.slots["num_cont"][slot]),
                    int(self.slots["cont_offset"][slot]),
                    float(self.slots["backoff_log10"][slot]),
                )
            slot = (slot + 1) & self.capacity_mask
        return None

    def continuations_of(self, prefix_tokens: tuple[int, ...]):
        """Return ``(tokens_view, log10_probs_view, backoff_log10)`` or ``(None, None, None)``."""
        hit = self.probe(prefix_tokens)
        if hit is None:
            return None, None, None
        n, off, b = hit
        return self.tokens[off : off + n], self.probs[off : off + n], b


# ---------------------------------------------------------------------------
# NgramTableSoftTargetSource — the training-time wrapper
# ---------------------------------------------------------------------------


class NgramTableSoftTargetSource:
    """Full-KN soft-target lookup over the five per-n mmap-able tables.

    At construction time: open all tables and precompute the top-N unigram
    shortlist so the order-1 backoff fall-through doesn't iterate the full
    vocab per position.

    At query time: for each position, probe all orders from longest-fitting
    down to 1, union observed continuations under KN-combined
    probabilities, take top-K, renormalize to sum to 1.

    This is the correctness-first reference implementation — a Python loop
    over positions. A numba-JIT'd batched path is a deferred optimization
    (see the docstring at the top of this file).
    """

    def __init__(
        self,
        table_dir: str | os.PathLike,
        K: int = 16,
        N_max: int = 5,
        unigram_shortlist: int = 100,
    ):
        self.K = int(K)
        self.N_max = int(N_max)
        table_dir = Path(table_dir)
        self.tables: dict[int, _NgramTable] = {}
        for n in range(1, self.N_max + 1):
            path = table_dir / f"ngram_table_n{n}.bin"
            if not path.is_file():
                raise FileNotFoundError(f"missing table file: {path}")
            self.tables[n] = _NgramTable(str(path))
            if self.tables[n].order != n:
                raise ValueError(
                    f"{path} header order={self.tables[n].order} but expected {n}"
                )

        # Precompute the top-`unigram_shortlist` unigrams by KN probability.
        # These are the *only* unigrams we'll ever consider as backoff
        # candidates; anything below is too small to survive a top-K after
        # an α-multiplier ≤ 1.
        uni_tokens, uni_log10, _ = self.tables[1].continuations_of(())
        if uni_tokens is None:
            raise RuntimeError(
                "order-1 table has no empty-prefix entry; cannot build unigram shortlist"
            )
        uni_log10_f32 = np.asarray(uni_log10, dtype=np.float32)
        uni_tokens_u32 = np.asarray(uni_tokens, dtype=np.uint32)
        # Sort descending by log10 prob; take top-N.
        order_idx = np.argsort(-uni_log10_f32)[: int(unigram_shortlist)]
        self.unigram_shortlist_ids = uni_tokens_u32[order_idx].copy()
        self.unigram_shortlist_log10 = uni_log10_f32[order_idx].copy()

    # ---- single-position lookup (reference implementation) ----

    def lookup_context(
        self, context: tuple[int, ...] | np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (topk_ids[K], topk_probs[K]) for one already-walked context.

        ``context`` is the caller-provided context window — tokens
        preceding the target position, walked left with the caller's own
        EOS-stopping rule. Length can be 0 (predicting first token of a
        document) up to any value; we use at most the last ``N_max - 1``
        tokens (the longest prefix our tables can resolve).
        """
        ctx = tuple(int(t) for t in context)[-(self.N_max - 1) :] if context else ()
        # Longest order we can probe at is the one whose prefix length (n-1)
        # matches the context length. E.g. context length 4 → query order 5.
        # If context is empty, we can still probe order 1 (empty prefix).
        max_order = min(len(ctx) + 1, self.N_max)

        candidates: dict[int, float] = {}
        log10_cumulative_backoff = 0.0

        for order in range(max_order, 0, -1):
            prefix_len = order - 1
            h_suffix = ctx[-prefix_len:] if prefix_len > 0 else ()
            table = self.tables[order]

            if order == 1:
                # Unigram level: use the precomputed top-N shortlist rather
                # than iterating the full vocab. The shortlist is already
                # sorted by log10 prob desc, so we just add anything not
                # already in `candidates` with the backoff multiplier baked in.
                for tok_u32, lp_f32 in zip(
                    self.unigram_shortlist_ids, self.unigram_shortlist_log10
                ):
                    tok = int(tok_u32)
                    if tok not in candidates:
                        candidates[tok] = log10_cumulative_backoff + float(lp_f32)
                # No backoff update at order 1 — there's no order 0 to fall through to.
            else:
                toks_view, probs_view, backoff_log10 = table.continuations_of(h_suffix)
                if toks_view is None:
                    # Prefix absent: convention α=1 → log10 α = 0 → no cumulative update
                    continue
                for tok_u32, lp_f16 in zip(toks_view.tolist(), probs_view.tolist()):
                    tok = int(tok_u32)
                    if tok not in candidates:
                        candidates[tok] = log10_cumulative_backoff + float(lp_f16)
                log10_cumulative_backoff += float(backoff_log10)

        if not candidates:
            # Pathological fallback: neither unigram shortlist nor any higher
            # order matched. Return the shortlist at uniform probs so the
            # loss still has something to work with.
            ids = np.zeros(self.K, dtype=np.int32)
            probs = np.full(self.K, 1.0 / self.K, dtype=np.float32)
            ids[: min(self.K, len(self.unigram_shortlist_ids))] = (
                self.unigram_shortlist_ids[: self.K].astype(np.int32)
            )
            return ids, probs

        # Top-K by log10 probability, descending.
        items = sorted(candidates.items(), key=lambda kv: -kv[1])[: self.K]
        topk_ids = np.zeros(self.K, dtype=np.int32)
        topk_log10 = np.full(self.K, -np.inf, dtype=np.float64)
        for i, (tok, lp) in enumerate(items):
            topk_ids[i] = tok
            topk_log10[i] = lp

        # Convert to linear probabilities and renormalize to sum to 1. Subtract
        # the max log10 first to avoid under/overflow in 10**x.
        max_log10 = topk_log10[0]
        # When fewer than K candidates exist, remaining slots have log10 = -inf → prob 0
        rel_log10 = topk_log10 - max_log10
        linear = np.power(10.0, rel_log10, where=np.isfinite(rel_log10), out=np.zeros_like(rel_log10))
        total = linear.sum()
        if total <= 0:
            # Shouldn't happen given we have at least one finite candidate, but defend
            probs = np.zeros(self.K, dtype=np.float32)
            probs[0] = 1.0
        else:
            probs = (linear / total).astype(np.float32)
        return topk_ids, probs

    # ---- batch lookup ----

    def lookup_batch(self, contexts) -> Tuple[np.ndarray, np.ndarray]:
        """Loop over contexts; return ``(ids[N,K], probs[N,K])``.

        ``contexts`` is a sequence of tuples/arrays, one per target position.
        For the training-time path we'll replace the Python loop with a
        numba-JIT'd kernel, but this reference version is what the unit
        tests exercise.
        """
        N = len(contexts)
        all_ids = np.zeros((N, self.K), dtype=np.int32)
        all_probs = np.zeros((N, self.K), dtype=np.float32)
        for i, ctx in enumerate(contexts):
            ids, probs = self.lookup_context(ctx)
            all_ids[i] = ids
            all_probs[i] = probs
        return all_ids, all_probs
