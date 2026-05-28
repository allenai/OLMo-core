"""Stupid-backoff ngram LM reader over a ``counts_index/`` directory built
by ``data_gen/build_counts_index.py`` (format_version 2).

This module is the runtime-side parallel of
:mod:`olmo_core.data.ngram_soft_target`. Where that one reads the
KN-smoothed top-K forward index (FXTK on disk), this one reads raw counts
plus a unigram-floor fallback and implements the Brants et al 2007
stupid-backoff scoring scheme.

Public surface
==============

- :class:`StupidBackoffNgramLM` — the reader. Used by:
    1. The standalone eval (``analysis/scripts/eval_nonparam_lm.py``):
       calls :meth:`target_logprobs` to score gold tokens for CE/PPL.
    2. The training-time instance source
       (:class:`olmo_core.data.composable.NgramStupidBackoffInstanceSource`):
       calls :meth:`compute_overrides_for_sequence` to produce ragged
       per-position SB overrides that get scattered onto LM logits at the
       train step. The shared length-V unigram floor is loaded
       independently at the train_module side from the same index dir.

ID space
========

Records on disk are in **kenlm-internal vocab space** — ids 3.. are dolma2
token ids encoded as decimal strings in ``pilot.counts.vocab``; ids 0/1/2
(``<unk>``/``<s>``/``</s>``) were filtered out at build time. The reader
constructs two small lookup arrays at startup so that callers can pass
dolma2-id-space contexts and receive dolma2-id-space outputs without
worrying about the kenlm-internal id space.

See ``plan.md → "Known structural limitation: doc-start blindness"`` for
what this filter costs at the first ``N_max - 1`` positions of every
training sequence.

Smoothing
=========

The unigram floor uses **Laplace +1 smoothing** so every dolma2 token
receives a finite score even if the corpus never observed it::

    unigram_floor[w] = (N_max - 1) · log α + log((C_1(w) + 1) / (T + V_dolma2))

where ``C_1(w)`` is the corpus count of dolma2 token w (0 if never
observed), ``T = total_corpus_tokens`` after filtering kenlm specials,
and ``V_dolma2`` is the LM tokenizer vocab size. The ``(N_max-1) · log α``
factor is the SB discount applied when a query backs off all the way
down to unigram from the maximum order.

Higher-order scores use the standard SB formula::

    score(w | h_k) = (N_max - k) · log α + log(C_k(h_k, w) / C_k(h_k))

— i.e. ``(N_max - k)`` discount steps from the maximum order (no
discount at order N_max where ``k = N_max``).
"""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def _sb_log(message: str) -> None:
    """Print a flush-safe SB debug line with enough process context for Beaker logs."""
    rank = os.environ.get("RANK", "?")
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(
        f"[SB pid={os.getpid()} rank={rank} local_rank={local_rank}] {message}",
        flush=True,
    )


def _sb_debug_enabled() -> bool:
    return os.environ.get("OLMO_SB_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _history_struct_dtype(plen: int) -> np.dtype:
    """Structured dtype whose ordering matches numeric lexicographic rows.

    A single subarray field, e.g. ``[("h", uint32, plen)]``, does not sort the
    same way as the row-wise uint32 histories written by build_counts_index.py
    once token ids exceed one byte. One field per history token makes NumPy's
    structured-array ``searchsorted`` compare h0, then h1, and so on.
    """
    return np.dtype([(f"h{i}", np.uint32) for i in range(plen)])


class PReadArray:
    """Small 1-D ndarray-like wrapper backed by explicit ``os.pread`` calls.

    This is intentionally narrower than ``np.memmap``. It supports the access
    shapes used by the SB reader for offsets, continuations, counts, and
    history totals while leaving the history table itself mmap-backed so
    NumPy's vectorized ``searchsorted`` path remains available.
    """

    def __init__(self, path: str, dtype):
        self.path = str(path)
        self.dtype = np.dtype(dtype)
        self.itemsize = int(self.dtype.itemsize)
        self.n_items = os.path.getsize(self.path) // self.itemsize
        self.shape = (self.n_items,)
        self.ndim = 1
        self._fd = os.open(self.path, os.O_RDONLY)

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._read_slice(key)
        if np.isscalar(key):
            return self._read_scalar(int(key))
        idx = np.asarray(key, dtype=np.int64)
        return self._read_indices(idx)

    def __array__(self, dtype=None):
        out = self._read_slice(slice(None))
        if dtype is not None:
            return out.astype(dtype, copy=False)
        return out

    def _read_scalar(self, idx: int):
        if idx < 0:
            idx += self.n_items
        if idx < 0 or idx >= self.n_items:
            raise IndexError(idx)
        data = os.pread(self._fd, self.itemsize, idx * self.itemsize)
        if len(data) != self.itemsize:
            raise OSError(f"short pread from {self.path} at item {idx}")
        return np.frombuffer(data, dtype=self.dtype, count=1)[0]

    def _read_slice(self, key: slice) -> np.ndarray:
        start, stop, step = key.indices(self.n_items)
        if step != 1:
            return self._read_slice(slice(start, stop, 1))[::step]
        if stop <= start:
            return np.zeros(0, dtype=self.dtype)
        n = stop - start
        data = os.pread(self._fd, n * self.itemsize, start * self.itemsize)
        if len(data) != n * self.itemsize:
            raise OSError(
                f"short pread from {self.path}: wanted {n * self.itemsize} bytes, "
                f"got {len(data)}"
            )
        return np.frombuffer(data, dtype=self.dtype, count=n).copy()

    def _read_indices(self, idx: np.ndarray) -> np.ndarray:
        original_shape = idx.shape
        flat = idx.reshape(-1)
        if flat.size == 0:
            return np.zeros(original_shape, dtype=self.dtype)
        flat = flat.copy()
        flat[flat < 0] += self.n_items
        if (flat < 0).any() or (flat >= self.n_items).any():
            raise IndexError("pread array index out of range")
        if flat.size == 1:
            return np.asarray([self._read_scalar(int(flat[0]))], dtype=self.dtype).reshape(
                original_shape
            )
        if np.all(flat[1:] == flat[:-1] + 1):
            return self._read_slice(slice(int(flat[0]), int(flat[-1]) + 1)).reshape(
                original_shape
            )

        unique, inverse = np.unique(flat, return_inverse=True)
        values = np.empty(unique.shape[0], dtype=self.dtype)
        run_start = 0
        while run_start < unique.shape[0]:
            run_end = run_start + 1
            while run_end < unique.shape[0] and unique[run_end] == unique[run_end - 1] + 1:
                run_end += 1
            chunk = self._read_slice(slice(int(unique[run_start]), int(unique[run_end - 1]) + 1))
            values[run_start:run_end] = chunk
            run_start = run_end
        return values[inverse].reshape(original_shape)


class StupidBackoffNgramLM:
    """Reader for a stupid-backoff ``counts_index/`` directory.

    :param table_dir: Path to either a pilot dir containing ``counts_index/``
        as a subdirectory, or that ``counts_index/`` dir directly.
    :param dolma2_vocab_size: The LM's tokenizer vocab size (V_dolma2). Used
        as the support of the unigram floor and the dimension of any
        bias vectors. For dolma2 this is 100_278.
    :param N_max: Highest ngram order to consult. Must equal the highest
        order present in the index (currently 5).
    :param alpha: Stupid-backoff discount factor per Brants et al 2007.
        Default 0.4. Lower means more weight on shorter histories.
    :param min_order_counts: Optional minimum raw-count thresholds by ngram
        order. When set for an order, continuations with counts below the
        threshold are omitted and may fall through to a lower order or the
        unigram floor. This is a post-hoc pruning approximation of rebuilding
        the counts index with low-count records removed.
    :param max_order_continuations: Optional hard caps by ngram order on the
        number of continuations emitted per matched history. When set for an
        order, the reader keeps the highest-count continuations for that
        history and lets omitted tokens fall through to the next lower order,
        or to the unigram floor if no lower order emits them.
    :param max_order2_continuations: Backward-compatible shortcut for
        ``max_order_continuations={2: value}``.
    :param index_access: ``"mmap"`` keeps the current memmap-backed access
        path. ``"pread"`` uses explicit ``os.pread`` calls for the large
        1-D offsets / continuation / count arrays while keeping histories
        mmap-backed for vectorized binary search.
    :param lookup_threads: Number of in-process threads used to split one
        long sequence into independent SB-override chunks. This is separate
        from PyTorch data-loader workers: workers prepare future batches,
        while lookup threads parallelize one sequence lookup.
    :param topk_uniform_residual_k: Optional KN-top-K-style mode for PoE
        experiments. When set, each position uses only the highest matching
        history row, keeps that row's top-K continuations, and emits logit
        deltas relative to a uniform residual over the rest of the vocab.
    :param recursive_topk_uniform_residual_k: Optional true-SB top-K mode for
        PoE experiments. When set, each position first computes recursive
        stupid-backoff scores from the pruned sparse index, includes high
        unigram fall-through candidates, then keeps the top-K final SB scores
        and emits deltas relative to a uniform residual.
    """

    def __init__(
        self,
        table_dir: str,
        *,
        dolma2_vocab_size: int,
        N_max: int = 5,
        alpha: float = 0.4,
        max_order2_continuations: Optional[int] = None,
        max_order_continuations: Optional[Dict[int, int]] = None,
        min_order_counts: Optional[Dict[int, int]] = None,
        mirror_to_shm: bool = True,
        index_access: str = "mmap",
        lookup_threads: int = 1,
        topk_uniform_residual_k: Optional[int] = None,
        recursive_topk_uniform_residual_k: Optional[int] = None,
    ):
        if index_access not in {"mmap", "pread"}:
            raise ValueError(f"index_access must be 'mmap' or 'pread'; got {index_access!r}")
        self.index_access = index_access
        self.lookup_threads = max(1, int(lookup_threads))
        if topk_uniform_residual_k is not None and topk_uniform_residual_k <= 0:
            raise ValueError(
                "topk_uniform_residual_k must be positive when set; "
                f"got {topk_uniform_residual_k}"
            )
        self.topk_uniform_residual_k = (
            None if topk_uniform_residual_k is None else int(topk_uniform_residual_k)
        )
        if (
            recursive_topk_uniform_residual_k is not None
            and recursive_topk_uniform_residual_k <= 0
        ):
            raise ValueError(
                "recursive_topk_uniform_residual_k must be positive when set; "
                f"got {recursive_topk_uniform_residual_k}"
            )
        self.recursive_topk_uniform_residual_k = (
            None
            if recursive_topk_uniform_residual_k is None
            else int(recursive_topk_uniform_residual_k)
        )
        if (
            self.topk_uniform_residual_k is not None
            and self.recursive_topk_uniform_residual_k is not None
        ):
            raise ValueError(
                "topk_uniform_residual_k and recursive_topk_uniform_residual_k "
                "are mutually exclusive"
            )
        self._lookup_pool: Optional[ThreadPoolExecutor] = None
        caps: Dict[int, int] = {}
        if max_order_continuations:
            caps.update({int(order): int(cap) for order, cap in max_order_continuations.items()})
        if max_order2_continuations is not None:
            caps[2] = int(max_order2_continuations)
        if max_order2_continuations is not None and max_order2_continuations <= 0:
            raise ValueError(
                "max_order2_continuations must be positive when set; "
                f"got {max_order2_continuations}"
            )
        for order, cap in caps.items():
            if order < 2:
                raise ValueError(f"continuation cap order must be >= 2; got {order}")
            if order > N_max:
                raise ValueError(
                    f"continuation cap order {order} exceeds N_max={N_max}"
                )
            if cap <= 0:
                raise ValueError(f"continuation cap for order {order} must be positive; got {cap}")
        self.max_order_continuations = caps
        self.max_order2_continuations = caps.get(2)
        min_counts: Dict[int, int] = {}
        if min_order_counts:
            min_counts.update({int(order): int(count) for order, count in min_order_counts.items()})
        for order, min_count in min_counts.items():
            if order < 2:
                raise ValueError(f"minimum-count order must be >= 2; got {order}")
            if order > N_max:
                raise ValueError(f"minimum-count order {order} exceeds N_max={N_max}")
            if min_count <= 0:
                raise ValueError(
                    f"minimum count for order {order} must be positive; got {min_count}"
                )
        self.min_order_counts = min_counts
        p = Path(table_dir)
        if (p / "counts_index").is_dir():
            index_dir = p / "counts_index"
            pilot_dir = p
        elif (p / "meta.json").is_file():
            index_dir = p
            pilot_dir = p.parent
        else:
            raise ValueError(
                f"table_dir must be a pilot dir containing counts_index/ "
                f"or a counts_index/ dir directly. Got: {p}"
            )
        self.index_dir = index_dir
        self.pilot_dir = pilot_dir
        t_init = time.perf_counter()
        _sb_log(
            f"reader init start table_dir={table_dir} index_dir={index_dir} "
            f"mirror_to_shm={mirror_to_shm} "
            f"index_access={self.index_access} "
            f"lookup_threads={self.lookup_threads} "
            f"topk_uniform_residual_k={self.topk_uniform_residual_k} "
            f"recursive_topk_uniform_residual_k={self.recursive_topk_uniform_residual_k} "
            f"max_order_continuations={self.max_order_continuations or None} "
            f"min_order_counts={self.min_order_counts or None}"
        )

        t_phase = time.perf_counter()
        with open(index_dir / "meta.json") as f:
            self.meta = json.load(f)
        if int(self.meta.get("format_version", 0)) != 2:
            raise ValueError(
                f"counts_index format_version={self.meta.get('format_version')}, "
                f"expected 2 — rebuild with the current build_counts_index.py"
            )
        if self.meta.get("id_space") != "kenlm_internal":
            raise ValueError(
                f"counts_index id_space={self.meta.get('id_space')}, "
                f"expected 'kenlm_internal'"
            )

        kenlm_vocab_size = int(self.meta["vocab_size"])
        total = int(self.meta["total_corpus_tokens"])
        self.N_max = int(N_max)
        self.alpha = float(alpha)
        self.log_alpha = math.log(self.alpha)
        self.vocab_size = int(dolma2_vocab_size)  # V_dolma2, NOT V_kenlm
        V = self.vocab_size
        _sb_log(
            f"meta loaded in {time.perf_counter() - t_phase:.2f}s "
            f"kenlm_vocab={kenlm_vocab_size:,} total_tokens={total:,} "
            f"N_max={self.N_max} alpha={self.alpha}"
        )

        # Build kenlm ↔ dolma2 id translation tables from pilot.counts.vocab.
        t_phase = time.perf_counter()
        vocab_path = pilot_dir / "pilot.counts.vocab"
        if not vocab_path.is_file():
            vocab_path = Path(self.meta.get("source_vocab_path", str(vocab_path)))
        with open(vocab_path, "rb") as f:
            vocab_strings = f.read().split(b"\x00")
        if vocab_strings and vocab_strings[-1] == b"":
            vocab_strings.pop()
        if len(vocab_strings) != kenlm_vocab_size:
            raise ValueError(
                f"vocab file {vocab_path} has {len(vocab_strings)} entries; "
                f"meta.json says vocab_size={kenlm_vocab_size}"
            )

        # kenlm_to_dolma2[kid] = dolma2 id for kenlm id `kid`. Slots 0/1/2
        # are kenlm specials and map to the sentinel V (out of dolma2 range).
        kenlm_to_dolma2 = np.full(kenlm_vocab_size, V, dtype=np.uint32)
        n_unparseable = 0
        for i, s in enumerate(vocab_strings):
            if i < 3:
                continue
            try:
                did = int(s.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                n_unparseable += 1
                continue
            if 0 <= did < V:
                kenlm_to_dolma2[i] = np.uint32(did)
            else:
                n_unparseable += 1
        self.kenlm_to_dolma2 = kenlm_to_dolma2
        self.n_unparseable_vocab_entries = n_unparseable

        # dolma2_to_kenlm[did] = kenlm id for dolma2 token `did`, or 0
        # (kUNK) for dolma2 tokens never seen in the corpus. kUNK has no
        # rows in any v2-filtered order, so the lookup silently misses
        # and we fall through to the unigram floor.
        dolma2_to_kenlm = np.zeros(V, dtype=np.uint32)
        for kid in range(3, kenlm_vocab_size):
            did = int(kenlm_to_dolma2[kid])
            if did < V:
                dolma2_to_kenlm[did] = np.uint32(kid)
        self.dolma2_to_kenlm = dolma2_to_kenlm
        _sb_log(
            f"vocab translation built in {time.perf_counter() - t_phase:.2f}s "
            f"unparseable_entries={n_unparseable:,}"
        )

        # Mmap the per-order index files. When ``mirror_to_shm=True``
        # (the default; required for training-time use with 16+ dataloader
        # workers per rank doing concurrent random-access lookups), mirror
        # each file from Weka to /dev/shm on first open per-node
        # (flock-protected; cooperative across workers). The same trick
        # the KN-smoothed ngram_soft_target reader uses for its
        # forward_index_topk.bin (see _mirror_to_shm in that module).
        # For 1e-4 the index is ~25 GiB total; sits comfortably in the
        # 100 GiB --shared-memory the launcher sets. For 1e-2 the index
        # is ~1.2 TB which doesn't fit in /dev/shm at all — single-process
        # standalone-eval callers should pass ``mirror_to_shm=False`` and
        # accept Weka I/O latency (page-cache makes the upper tree levels
        # fast enough single-process; the thrashing only matters with
        # many concurrent workers).
        from olmo_core.data.ngram_soft_target import _mirror_to_shm

        def _open_mmap(name: str, dtype) -> np.memmap:
            path = str(index_dir / name)
            if mirror_to_shm:
                path = _mirror_to_shm(path)
            return np.memmap(path, dtype=dtype, mode="r")

        def _open_1d(name: str, dtype):
            path = str(index_dir / name)
            if mirror_to_shm:
                path = _mirror_to_shm(path)
            if self.index_access == "pread":
                return PReadArray(path, dtype)
            return np.memmap(path, dtype=dtype, mode="r")

        self.orders: Dict[int, dict] = {}
        t_phase = time.perf_counter()
        total_hist = 0
        total_pairs = 0
        for n_str, info in self.meta["per_order"].items():
            n = int(n_str)
            n_hist = int(info["n_hist"])
            n_pairs = int(info["n_pairs"])
            total_hist += n_hist
            total_pairs += n_pairs
            plen = n - 1
            t_order = time.perf_counter()
            order_data: dict = {
                "n_hist": n_hist,
                "offsets": _open_1d(f"order{n}.offsets.bin", np.uint64),
                "continuations": _open_1d(f"order{n}.continuations.bin", np.uint32),
                "counts": _open_1d(f"order{n}.counts.bin", np.uint64),
                "history_totals": _open_1d(f"order{n}.history_totals.bin", np.uint64),
            }
            if plen == 0:
                order_data["histories"] = None
                order_data["histories_struct"] = None
            else:
                hist = _open_mmap(f"order{n}.histories.bin", np.uint32).reshape(
                    n_hist, plen
                )
                struct_dtype = _history_struct_dtype(plen)
                order_data["histories"] = hist
                order_data["histories_struct"] = hist.view(struct_dtype).reshape(-1)
            self.orders[n] = order_data
            _sb_log(
                f"order{n} mmap ready in {time.perf_counter() - t_order:.2f}s "
                f"n_hist={n_hist:,} n_pairs={n_pairs:,}"
            )
        _sb_log(
            f"all mmaps ready in {time.perf_counter() - t_phase:.2f}s "
            f"total_hist={total_hist:,} total_pairs={total_pairs:,}"
        )

        # Build the unigram floor in dolma2 vocab space (Laplace +1).
        t_phase = time.perf_counter()
        log_denom = math.log(total + V)
        unobserved_log_p = -log_denom  # log(1 / (total + V))
        discount = (self.N_max - 1) * self.log_alpha
        unigram_floor = np.full(V, unobserved_log_p + discount, dtype=np.float64)
        order1 = self.orders[1]
        kenlm_unigrams = np.asarray(order1["continuations"], dtype=np.uint32)
        unigram_counts = np.asarray(order1["counts"], dtype=np.uint64)
        for kid, cnt in zip(kenlm_unigrams.tolist(), unigram_counts.tolist()):
            did = int(kenlm_to_dolma2[kid])
            if did >= V:
                continue
            unigram_floor[did] = math.log(int(cnt) + 1) - log_denom + discount
        self.unigram_floor = unigram_floor
        self._unigram_desc_ids: Optional[np.ndarray] = None

        # Precompute log Z_unigram (the constant base normalizer).
        max_floor = float(unigram_floor.max())
        self.log_Z_unigram = max_floor + math.log(
            float(np.exp(unigram_floor - max_floor).sum())
        )

        self.kenlm_vocab_size = kenlm_vocab_size
        self.total_corpus_tokens = total
        self._override_debug_calls = 0
        _sb_log(
            f"unigram floor ready in {time.perf_counter() - t_phase:.2f}s; "
            f"reader init total {time.perf_counter() - t_init:.2f}s"
        )

    # ----- low-level helpers ----------------------------------------------

    def _lookup_history(self, n: int, history_kenlm: np.ndarray) -> Optional[int]:
        """Binary-search the order-n history table for `history_kenlm`
        (length n-1, kenlm ids). Returns the row index if found, else None."""
        order = self.orders.get(n)
        if order is None or order["n_hist"] == 0:
            return None
        struct = order["histories_struct"]
        if struct is None:
            return None
        plen = n - 1
        query = np.asarray(history_kenlm, dtype=np.uint32).reshape(1, plen)
        query_struct = query.view(struct.dtype).reshape(-1)
        idx = int(np.searchsorted(struct, query_struct[0]))
        if idx < struct.shape[0] and struct[idx] == query_struct[0]:
            return idx
        return None

    def _override_for_history(self, ctx_kenlm: np.ndarray) -> Dict[int, float]:
        """Build the per-position SB override dict for a single query.

        ``ctx_kenlm`` is the up-to-(N_max-1) history in kenlm-id space.
        Returns ``{dolma2_id: log_sb_score}`` for every token observed at
        *some* order > 1 for this exact history. Tokens not in the returned
        dict are assumed to use the unigram floor.

        Stupid-backoff semantics: walk orders from N_max down to 2, and
        the *first* (i.e. highest) order at which we see (h_k, w) wins for
        that w. Lower-order entries for the same w are ignored.
        """
        override: Dict[int, float] = {}
        for n in range(self.N_max, 1, -1):
            plen = n - 1
            if ctx_kenlm.shape[0] < plen:
                continue
            history = ctx_kenlm[-plen:]
            row_idx = self._lookup_history(n, history)
            if row_idx is None:
                continue
            order = self.orders[n]
            offsets = order["offsets"]
            lo = int(offsets[row_idx])
            hi = int(offsets[row_idx + 1])
            if hi == lo:
                continue
            conts_kenlm = np.asarray(order["continuations"][lo:hi], dtype=np.uint32)
            cnts = np.asarray(order["counts"][lo:hi], dtype=np.uint64)
            min_count = self.min_order_counts.get(n)
            if min_count is not None:
                keep_min_count = cnts >= np.uint64(min_count)
                if not keep_min_count.any():
                    continue
                conts_kenlm = conts_kenlm[keep_min_count]
                cnts = cnts[keep_min_count]
            order_cap = self.max_order_continuations.get(n)
            if order_cap is not None:
                k = int(order_cap)
                if cnts.shape[0] > k:
                    keep = np.argpartition(cnts, -k)[-k:]
                    conts_kenlm = conts_kenlm[keep]
                    cnts = cnts[keep]
            h_total = int(order["history_totals"][row_idx])
            log_h_total = math.log(h_total) if h_total > 0 else float("-inf")
            log_discount = (self.N_max - n) * self.log_alpha
            cont_dolma2 = self.kenlm_to_dolma2[conts_kenlm]
            for did_arr, cnt in zip(cont_dolma2.tolist(), cnts.tolist()):
                did = int(did_arr)
                if did >= self.vocab_size:
                    continue
                if did in override:
                    continue  # higher order already won
                if int(cnt) == 0:
                    continue
                override[did] = math.log(int(cnt)) - log_h_total + log_discount
        return override

    # ----- standalone-eval entry points -----------------------------------

    def _per_position_logp_gold(
        self, ctx_dolma2: np.ndarray, gold_dolma2: int
    ) -> Tuple[float, bool]:
        """Compute (log p_SB(gold | ctx), hit) for one scored position.

        ``hit`` is True iff the gold token was observed at *some* order > 1
        for this exact history (i.e. its score came from a real higher-order
        count, not the unigram floor).
        """
        ctx_kenlm = self.dolma2_to_kenlm[np.asarray(ctx_dolma2, dtype=np.int64)]
        override = self._override_for_history(ctx_kenlm)
        delta = 0.0
        for did, log_score in override.items():
            delta += math.exp(log_score) - math.exp(float(self.unigram_floor[did]))
        log_Z_position = self.log_Z_unigram + math.log1p(
            delta * math.exp(-self.log_Z_unigram)
        )
        if gold_dolma2 in override:
            log_score_gold = override[gold_dolma2]
            hit = True
        else:
            log_score_gold = float(self.unigram_floor[gold_dolma2])
            hit = False
        return log_score_gold - log_Z_position, hit

    def target_logprobs(
        self, input_ids: np.ndarray, score_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standalone-eval entry: (B, S) input_ids and (B, S-1) score_mask
        → per-position (log p(gold), hit) for the masked scored positions.

        ``log p`` is the natural-log probability of the gold next token
        under the SB distribution normalized over the full V_dolma2 vocab.
        Positions where ``score_mask`` is False get 0 in both outputs.
        """
        B, S = input_ids.shape
        assert score_mask.shape == (B, S - 1), score_mask.shape
        out = np.zeros((B, S - 1), dtype=np.float64)
        hit = np.zeros((B, S - 1), dtype=bool)
        if self.lookup_threads <= 1:
            prefix_len = self.N_max - 1
            for b in range(B):
                row = input_ids[b]
                mask_row = score_mask[b]
                for t in range(S - 1):
                    if not mask_row[t]:
                        continue
                    ctx_start = max(0, t + 1 - prefix_len)
                    ctx = row[ctx_start : t + 1]
                    gold = int(row[t + 1])
                    logp, h = self._per_position_logp_gold(ctx, gold)
                    out[b, t] = logp
                    hit[b, t] = h
            return out, hit

        for b in range(B):
            row = input_ids[b]
            mask_row = score_mask[b]
            score_positions = np.nonzero(mask_row)[0].astype(np.int64, copy=False)
            if score_positions.size == 0:
                continue
            z_delta_sum = np.zeros(S - 1, dtype=np.float64)
            gold_override_score = np.zeros(S - 1, dtype=np.float64)
            pos, tok, score = self.compute_overrides_for_sequence(
                row, score_positions=score_positions
            )
            if pos.size > 0:
                pos_i = pos.astype(np.int64, copy=False)
                tok_i = tok.astype(np.int64, copy=False)
                score_f = score.astype(np.float64, copy=False)
                valid = (pos_i >= 0) & (pos_i < S - 1)
                if valid.any():
                    valid_idx = np.flatnonzero(valid)
                    valid[valid_idx] = mask_row[pos_i[valid_idx]]
                if valid.any():
                    pos_v = pos_i[valid]
                    tok_v = tok_i[valid]
                    score_v = score_f[valid]
                    floor_at_tok = self.unigram_floor[tok_v]
                    delta = np.exp(score_v) - np.exp(floor_at_tok)
                    z_delta_sum += np.bincount(
                        pos_v, weights=delta, minlength=S - 1
                    )[: S - 1]
                    gold_matches = tok_v == row[pos_v + 1]
                    if gold_matches.any():
                        gold_pos = pos_v[gold_matches]
                        gold_override_score[gold_pos] = score_v[gold_matches]
                        hit[b, gold_pos] = True
            gold = row[score_positions + 1]
            log_z = self.log_Z_unigram + np.log1p(
                z_delta_sum[score_positions] * math.exp(-self.log_Z_unigram)
            )
            gold_score = self.unigram_floor[gold]
            gold_score = np.where(
                hit[b, score_positions],
                gold_override_score[score_positions],
                gold_score,
            )
            out[b, score_positions] = gold_score - log_z
        return out, hit

    # ----- training-time entry point --------------------------------------

    def compute_overrides_for_sequence(
        self, input_ids: np.ndarray, score_positions: np.ndarray | None = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build ragged SB overrides for one sequence, optionally using
        in-process chunk parallelism."""
        input_ids = np.asarray(input_ids, dtype=np.int64)
        S = int(input_ids.shape[0])
        if self.lookup_threads <= 1 or S <= 1:
            if self.recursive_topk_uniform_residual_k is not None:
                return self._compute_recursive_topk_uniform_residual_overrides_serial(
                    input_ids, score_positions=score_positions
                )
            if self.topk_uniform_residual_k is not None:
                return self._compute_topk_uniform_residual_overrides_serial(
                    input_ids, score_positions=score_positions
                )
            return self._compute_overrides_for_sequence_serial(
                input_ids, score_positions=score_positions
            )

        if score_positions is None:
            positions = np.arange(S - 1, dtype=np.int64)
        else:
            positions = np.asarray(score_positions, dtype=np.int64)
            positions = positions[(positions >= 0) & (positions < S - 1)]
            if positions.size == 0:
                return (
                    np.zeros(0, dtype=np.int32),
                    np.zeros(0, dtype=np.int32),
                    np.zeros(0, dtype=np.float32),
                )
            positions = np.unique(positions)

        n_positions = int(positions.shape[0])
        n_chunks = min(self.lookup_threads, n_positions)
        if n_chunks <= 1:
            if self.recursive_topk_uniform_residual_k is not None:
                return self._compute_recursive_topk_uniform_residual_overrides_serial(
                    input_ids, score_positions=positions
                )
            if self.topk_uniform_residual_k is not None:
                return self._compute_topk_uniform_residual_overrides_serial(
                    input_ids, score_positions=positions
                )
            return self._compute_overrides_for_sequence_serial(
                input_ids, score_positions=positions
            )

        t0 = time.perf_counter()
        chunks = np.array_split(positions, n_chunks)

        def compute_chunk(chunk_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            chunk_positions = chunks[chunk_idx]
            if chunk_positions.size == 0:
                return (
                    np.zeros(0, dtype=np.int32),
                    np.zeros(0, dtype=np.int32),
                    np.zeros(0, dtype=np.float32),
                )
            if self.recursive_topk_uniform_residual_k is not None:
                return self._compute_recursive_topk_uniform_residual_overrides_serial(
                    input_ids,
                    score_positions=chunk_positions,
                    log_timing=False,
                )
            if self.topk_uniform_residual_k is not None:
                return self._compute_topk_uniform_residual_overrides_serial(
                    input_ids,
                    score_positions=chunk_positions,
                    log_timing=False,
                )
            return self._compute_overrides_for_sequence_serial(
                input_ids,
                score_positions=chunk_positions,
                log_timing=False,
            )

        if self._lookup_pool is None:
            self._lookup_pool = ThreadPoolExecutor(
                max_workers=self.lookup_threads,
                thread_name_prefix="sb-lookup",
            )

        parts = list(self._lookup_pool.map(compute_chunk, range(n_chunks)))
        non_empty = [part for part in parts if part[0].size > 0]
        if not non_empty:
            result = (
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
            )
        else:
            result = (
                np.concatenate([part[0] for part in non_empty]).astype(np.int32, copy=False),
                np.concatenate([part[1] for part in non_empty]).astype(np.int32, copy=False),
                np.concatenate([part[2] for part in non_empty]).astype(np.float32, copy=False),
            )
        self._maybe_log_override_call(
            t0,
            S,
            int(result[0].shape[0]),
            [
                f"score_positions={n_positions:,}",
                f"thread_chunks={n_chunks}",
                f"lookup_threads={self.lookup_threads}",
            ],
        )
        return result

    def _compute_topk_uniform_residual_overrides_serial(
        self,
        input_ids: np.ndarray,
        *,
        score_positions: np.ndarray | None = None,
        log_timing: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Emit top-K continuation deltas relative to a uniform residual.

        This mode deliberately mirrors the KN-smoothed top-K PoE structure:
        for each position, choose the highest matching SB history row, keep
        the row's top-K continuations by count, and treat all other vocab
        items as sharing the remaining row mass uniformly. The emitted score
        is already a relative logit delta:

            log(top_score) - log(uniform_residual_score)

        so the train module should use a zero dense SB floor.
        """
        from numpy.lib.stride_tricks import sliding_window_view

        assert self.topk_uniform_residual_k is not None
        t0 = time.perf_counter()
        order_summaries: list[str] = []
        input_ids = np.asarray(input_ids, dtype=np.int64)
        S = int(input_ids.shape[0])
        EMPTY = (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float32),
        )
        if S <= 1:
            return EMPTY
        if score_positions is None:
            requested_positions = np.arange(S - 1, dtype=np.int64)
        else:
            requested_positions = np.asarray(score_positions, dtype=np.int64)
            requested_positions = requested_positions[
                (requested_positions >= 0) & (requested_positions < S - 1)
            ]
            if requested_positions.size == 0:
                return EMPTY
            requested_positions = np.unique(requested_positions)

        ctx_kenlm_full = self.dolma2_to_kenlm[input_ids].astype(np.uint32, copy=False)
        unresolved_positions = requested_positions
        out_pos: list[np.ndarray] = []
        out_did: list[np.ndarray] = []
        out_delta: list[np.ndarray] = []
        k = int(self.topk_uniform_residual_k)
        eps = np.finfo(np.float64).tiny

        for n in range(self.N_max, 1, -1):
            if unresolved_positions.size == 0:
                break
            plen = n - 1
            order_data = self.orders.get(n)
            if order_data is None or order_data["n_hist"] == 0:
                continue
            eligible = unresolved_positions[unresolved_positions >= plen - 1]
            if eligible.size == 0:
                continue
            windows = sliding_window_view(ctx_kenlm_full, plen)
            window_indices = eligible - (plen - 1)
            queries = np.ascontiguousarray(windows[window_indices])

            struct = order_data["histories_struct"]
            if struct is None:
                continue
            query_struct = queries.view(struct.dtype).reshape(-1)
            idx = np.searchsorted(struct, query_struct)
            clamped = np.minimum(idx, struct.shape[0] - 1)
            matches = (idx < struct.shape[0]) & (struct[clamped] == query_struct)
            if not matches.any():
                continue

            hit_positions = eligible[matches]
            hit_row_indices = idx[matches]
            selected_by_history_row: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            kept_for_order = 0
            rows_for_order = 0
            for pos_i, row_idx in zip(hit_positions.tolist(), hit_row_indices.tolist()):
                row_idx_i = int(row_idx)
                cached = selected_by_history_row.get(row_idx_i)
                if cached is None:
                    offsets = order_data["offsets"]
                    lo = int(offsets[row_idx_i])
                    hi = int(offsets[row_idx_i + 1])
                    if hi == lo:
                        cached = (
                            np.zeros(0, dtype=np.int64),
                            np.zeros(0, dtype=np.float64),
                        )
                    else:
                        row_counts = np.asarray(
                            order_data["counts"][lo:hi], dtype=np.uint64
                        )
                        positive = np.flatnonzero(row_counts > 0)
                        if positive.size == 0:
                            cached = (
                                np.zeros(0, dtype=np.int64),
                                np.zeros(0, dtype=np.float64),
                            )
                        else:
                            if positive.size > k:
                                top = np.argpartition(row_counts[positive], -k)[-k:]
                                keep = positive[top]
                            else:
                                keep = positive
                            arr_idx = int(lo) + keep.astype(np.int64, copy=False)
                            conts = np.asarray(
                                order_data["continuations"][arr_idx],
                                dtype=np.uint32,
                            )
                            did = self.kenlm_to_dolma2[conts].astype(
                                np.int64, copy=False
                            )
                            valid = did < self.vocab_size
                            if not valid.any():
                                cached = (
                                    np.zeros(0, dtype=np.int64),
                                    np.zeros(0, dtype=np.float64),
                                )
                            else:
                                did = did[valid]
                                counts = row_counts[keep][valid].astype(
                                    np.float64, copy=False
                                )
                                h_total = float(order_data["history_totals"][row_idx_i])
                                if h_total <= 0.0:
                                    cached = (
                                        np.zeros(0, dtype=np.int64),
                                        np.zeros(0, dtype=np.float64),
                                    )
                                else:
                                    log_discount = (self.N_max - n) * self.log_alpha
                                    discount = math.exp(log_discount)
                                    top_scores = discount * counts / h_total
                                    residual_mass = max(
                                        discount - float(top_scores.sum()), eps
                                    )
                                    residual_vocab = max(
                                        self.vocab_size - int(did.shape[0]), 1
                                    )
                                    log_residual = math.log(residual_mass) - math.log(
                                        residual_vocab
                                    )
                                    delta = np.log(top_scores) - log_residual
                                    cached = (did, delta.astype(np.float64, copy=False))
                    selected_by_history_row[row_idx_i] = cached
                did, delta = cached
                if did.size == 0:
                    continue
                out_pos.append(np.full(did.shape[0], pos_i, dtype=np.int32))
                out_did.append(did.astype(np.int32, copy=False))
                out_delta.append(delta.astype(np.float32, copy=False))
                kept_for_order += int(did.shape[0])
                rows_for_order += 1

            if rows_for_order:
                order_summaries.append(
                    f"n{n}:hist_hits={int(hit_positions.shape[0]):,},"
                    f"rows_used={rows_for_order:,},overrides={kept_for_order:,},"
                    f"topk_uniform_residual_k={k}"
                )
            matched_set = set(int(p) for p in hit_positions.tolist())
            if matched_set:
                unresolved_positions = np.asarray(
                    [p for p in unresolved_positions.tolist() if int(p) not in matched_set],
                    dtype=np.int64,
                )

        if not out_pos:
            if log_timing:
                self._maybe_log_override_call(t0, S, 0, order_summaries)
            return EMPTY

        result = (
            np.concatenate(out_pos).astype(np.int32, copy=False),
            np.concatenate(out_did).astype(np.int32, copy=False),
            np.concatenate(out_delta).astype(np.float32, copy=False),
        )
        if log_timing:
            self._maybe_log_override_call(t0, S, int(result[0].shape[0]), order_summaries)
        return result

    def _ensure_unigram_desc_ids(self) -> np.ndarray:
        """Dolma2 token ids sorted from highest to lowest unigram floor."""
        if self._unigram_desc_ids is None:
            self._unigram_desc_ids = np.argsort(self.unigram_floor)[::-1].astype(
                np.int64, copy=False
            )
        return self._unigram_desc_ids

    def _compute_recursive_topk_uniform_residual_overrides_serial(
        self,
        input_ids: np.ndarray,
        *,
        score_positions: np.ndarray | None = None,
        log_timing: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Emit true recursive-SB top-K deltas relative to a uniform residual.

        Unlike :meth:`_compute_topk_uniform_residual_overrides_serial`, this
        first applies the normal recursive stupid-backoff rule over the pruned
        sparse index: highest observed order wins per token, and unobserved
        tokens fall through to the unigram floor. It then keeps the top-K final
        SB scores from that full distribution, approximating the remaining
        mass uniformly over the rest of the vocabulary.
        """
        assert self.recursive_topk_uniform_residual_k is not None
        k = int(self.recursive_topk_uniform_residual_k)
        t0 = time.perf_counter()
        pos, did, score = self._compute_overrides_for_sequence_serial(
            input_ids,
            score_positions=score_positions,
            log_timing=False,
        )
        input_ids = np.asarray(input_ids, dtype=np.int64)
        S = int(input_ids.shape[0])
        EMPTY = (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float32),
        )
        if S <= 1:
            return EMPTY
        if score_positions is None:
            requested_positions = np.arange(S - 1, dtype=np.int64)
        else:
            requested_positions = np.asarray(score_positions, dtype=np.int64)
            requested_positions = requested_positions[
                (requested_positions >= 0) & (requested_positions < S - 1)
            ]
            if requested_positions.size == 0:
                return EMPTY
            requested_positions = np.unique(requested_positions)

        pos_i = pos.astype(np.int64, copy=False)
        did_i = did.astype(np.int64, copy=False)
        score_f = score.astype(np.float64, copy=False)
        if pos_i.size > 1 and np.any(pos_i[1:] < pos_i[:-1]):
            order = np.argsort(pos_i, kind="stable")
            pos_i = pos_i[order]
            did_i = did_i[order]
            score_f = score_f[order]
        if pos_i.size:
            unique_pos, group_start, group_count = np.unique(
                pos_i,
                return_index=True,
                return_counts=True,
            )
        else:
            unique_pos = np.zeros(0, dtype=np.int64)
            group_start = np.zeros(0, dtype=np.int64)
            group_count = np.zeros(0, dtype=np.int64)

        out_pos: list[np.ndarray] = []
        out_did: list[np.ndarray] = []
        out_delta: list[np.ndarray] = []
        total_recursive_candidates = int(did_i.shape[0])
        eps = np.finfo(np.float64).tiny
        z_unigram = math.exp(self.log_Z_unigram)
        unigram_desc_ids = self._ensure_unigram_desc_ids()
        unigram_excluded = np.zeros(self.vocab_size, dtype=bool)

        for p_i in requested_positions.tolist():
            group_idx = int(np.searchsorted(unique_pos, int(p_i)))
            if group_idx < unique_pos.shape[0] and int(unique_pos[group_idx]) == int(p_i):
                start = int(group_start[group_idx])
                stop = start + int(group_count[group_idx])
                rec_ids = did_i[start:stop]
                rec_scores_log = score_f[start:stop]
            else:
                rec_ids = np.zeros(0, dtype=np.int64)
                rec_scores_log = np.zeros(0, dtype=np.float64)

            if rec_ids.size:
                unigram_excluded[rec_ids] = True
            unigram_ids: list[int] = []
            for did_arr in unigram_desc_ids:
                did_uni = int(did_arr)
                if unigram_excluded[did_uni]:
                    continue
                unigram_ids.append(did_uni)
                if len(unigram_ids) >= k:
                    break
            if rec_ids.size:
                unigram_excluded[rec_ids] = False

            if unigram_ids:
                uni_ids = np.asarray(unigram_ids, dtype=np.int64)
                candidate_ids = np.concatenate([rec_ids, uni_ids])
                candidate_scores_log = np.concatenate(
                    [rec_scores_log, self.unigram_floor[uni_ids].astype(np.float64)]
                )
            else:
                candidate_ids = rec_ids
                candidate_scores_log = rec_scores_log
            if candidate_ids.size == 0:
                continue

            if candidate_ids.shape[0] > k:
                top_idx = np.argpartition(candidate_scores_log, -k)[-k:]
                top_ids = candidate_ids[top_idx]
                top_scores_log = candidate_scores_log[top_idx]
            else:
                top_ids = candidate_ids
                top_scores_log = candidate_scores_log
            top_scores = np.exp(top_scores_log)

            if rec_ids.size:
                z_delta = float(
                    np.exp(rec_scores_log).sum()
                    - np.exp(self.unigram_floor[rec_ids]).sum()
                )
            else:
                z_delta = 0.0
            z_total = max(z_unigram + z_delta, eps)
            residual_mass = max(z_total - float(top_scores.sum()), eps)
            residual_vocab = max(self.vocab_size - int(top_ids.shape[0]), 1)
            log_residual = math.log(residual_mass) - math.log(residual_vocab)
            delta = top_scores_log - log_residual

            out_pos.append(np.full(top_ids.shape[0], int(p_i), dtype=np.int32))
            out_did.append(top_ids.astype(np.int32, copy=False))
            out_delta.append(delta.astype(np.float32, copy=False))

        if not out_pos:
            if log_timing:
                self._maybe_log_override_call(
                    t0,
                    S,
                    0,
                    [f"recursive_topk_uniform_residual_k={k},recursive_candidates=0"],
                )
            return EMPTY

        result = (
            np.concatenate(out_pos).astype(np.int32, copy=False),
            np.concatenate(out_did).astype(np.int32, copy=False),
            np.concatenate(out_delta).astype(np.float32, copy=False),
        )
        if log_timing:
            self._maybe_log_override_call(
                t0,
                S,
                int(result[0].shape[0]),
                [
                    f"recursive_topk_uniform_residual_k={k}",
                    f"recursive_candidates={total_recursive_candidates:,}",
                ],
            )
        return result

    def _compute_overrides_for_sequence_serial(
        self,
        input_ids: np.ndarray,
        *,
        score_positions: np.ndarray | None = None,
        log_timing: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the ragged per-position SB overrides for one training
        instance.

        For each position ``i`` in ``[0, S-1]`` (where the LM is predicting
        ``input_ids[i+1]`` from ``input_ids[:i+1]``), compute the SB
        override dict for the history ``input_ids[max(0, i+1-N_max+1) : i+1]``
        and flatten the result across all positions.

        Returns three flat numpy arrays, each of length ``n_overrides``::

            override_position  (n_overrides,) int32 — seq position i
            override_token_id  (n_overrides,) int32 — dolma2 id w
            override_log_score (n_overrides,) float32 — natural-log SB score

        The training-time bias path adds ``λ · unigram_floor[w]`` to every
        logit via broadcast, then ``scatter_add_`` of
        ``λ · (override_log_score - unigram_floor[override_token_id])`` at
        ``(b, override_position, override_token_id)`` to override the
        unigram floor for observed (h_k, w) pairs. See
        :meth:`olmo_core.train.train_module.transformer.TransformerTrainModule._apply_poe_eval_bias`
        for the runtime structure (KN-smoothed version; the SB version
        is analogous).

        The last sequence position (``i = S-1``) has no next-token target
        in the LM's loss, so we skip it — the returned overrides cover
        ``i ∈ [0, S-2]``.
        """
        from numpy.lib.stride_tricks import sliding_window_view

        t0 = time.perf_counter()
        order_summaries: list[str] = []
        input_ids = np.asarray(input_ids, dtype=np.int64)
        S = int(input_ids.shape[0])
        EMPTY = (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float32),
        )
        if S <= 1:
            return EMPTY
        if score_positions is None:
            requested_positions = None
        else:
            requested_positions = np.asarray(score_positions, dtype=np.int64)
            requested_positions = requested_positions[
                (requested_positions >= 0) & (requested_positions < S - 1)
            ]
            if requested_positions.size == 0:
                return EMPTY
            requested_positions = np.unique(requested_positions)

        # Translate the full input_ids dolma2→kenlm once. Out-of-vocab dolma2
        # ids map to kenlm id 0 (kUNK) by the dolma2_to_kenlm table init;
        # kUNK has no rows in any v2-filtered order, so any history that
        # contains one will silently miss in searchsorted.
        ctx_kenlm_full = self.dolma2_to_kenlm[input_ids].astype(np.uint32, copy=False)

        # Per-order: batched binary search across all queries with enough left
        # context, then bulk-gather (continuation, count) rows for the hits.
        # Concatenate across orders at the end and resolve the
        # "highest-order-wins" semantic with a single lexsort + uniqueness pass.
        per_order: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []

        for n in range(2, self.N_max + 1):
            plen = n - 1
            order_data = self.orders.get(n)
            if order_data is None or order_data["n_hist"] == 0:
                continue
            # Positions with enough left context: i in [plen-1, S-2].
            # For position i, the history is ctx_kenlm_full[i+1-plen : i+1].
            # That's window index (i+1-plen) into sliding_window_view of width plen.
            windows = sliding_window_view(ctx_kenlm_full, plen)  # (S-plen+1, plen)
            if requested_positions is None:
                n_queries = S - plen
                if n_queries <= 0:
                    continue
                # Take the first n_queries windows — last position covered is S-2.
                queries = np.ascontiguousarray(windows[:n_queries])  # (n_queries, plen)
                position_for_query = np.arange(
                    plen - 1, plen - 1 + n_queries, dtype=np.int64
                )
            else:
                position_for_query = requested_positions[requested_positions >= plen - 1]
                if position_for_query.size == 0:
                    continue
                window_indices = position_for_query - (plen - 1)
                queries = np.ascontiguousarray(windows[window_indices])

            struct = order_data["histories_struct"]
            if struct is None:
                continue
            query_struct = queries.view(struct.dtype).reshape(-1)
            idx = np.searchsorted(struct, query_struct)
            # Clamp to a valid lookup index for the equality check (a query
            # past the end gets idx == struct.shape[0]).
            clamped = np.minimum(idx, struct.shape[0] - 1)
            matches = (idx < struct.shape[0]) & (struct[clamped] == query_struct)
            if not matches.any():
                continue
            hit_positions = position_for_query[matches]
            hit_row_indices = idx[matches]
            n_hits = int(hit_positions.shape[0])

            offsets_arr = order_data["offsets"]
            lo = np.asarray(offsets_arr[hit_row_indices], dtype=np.int64)
            hi = np.asarray(offsets_arr[hit_row_indices + 1], dtype=np.int64)
            lengths = hi - lo
            total = int(lengths.sum())
            if total == 0:
                continue
            h_totals = np.asarray(
                order_data["history_totals"][hit_row_indices], dtype=np.uint64
            )

            # Build flat indices into the order's continuations / counts arrays.
            # flat_row[k] = which hit (and thus which row + position) entry k
            # belongs to; within_row[k] = offset within that row's (continuation,
            # count) slice; flat_arr_idx[k] = the global index to read from.
            total_raw = total
            order_cap = self.max_order_continuations.get(n)
            min_count = self.min_order_counts.get(n)
            if order_cap is not None or min_count is not None:
                k = int(order_cap) if order_cap is not None else None
                selected_idx: list[np.ndarray] = []
                selected_row: list[np.ndarray] = []
                # The same one-token history often appears many times in a
                # sequence. Cache the selected global continuation indices per
                # history row so repeated contexts don't re-filter or
                # re-partition the same large count slice.
                selected_by_history_row: dict[int, np.ndarray] = {}
                for hit_i, (row_idx, start, end) in enumerate(zip(hit_row_indices, lo, hi)):
                    row_idx_i = int(row_idx)
                    arr_idx = selected_by_history_row.get(row_idx_i)
                    if arr_idx is None:
                        row_len = int(end - start)
                        if min_count is None and k is not None and row_len <= k:
                            arr_idx = np.arange(int(start), int(end), dtype=np.int64)
                        else:
                            row_counts = np.asarray(
                                order_data["counts"][int(start) : int(end)],
                                dtype=np.uint64,
                            )
                            if min_count is not None:
                                keep = np.flatnonzero(row_counts >= np.uint64(min_count))
                                if keep.size == 0:
                                    arr_idx = np.empty(0, dtype=np.int64)
                                else:
                                    kept_counts = row_counts[keep]
                                    if k is not None and keep.size > k:
                                        top = np.argpartition(kept_counts, -k)[-k:]
                                        keep = keep[top]
                                    arr_idx = int(start) + keep.astype(np.int64, copy=False)
                            elif k is not None:
                                keep = np.argpartition(row_counts, -k)[-k:].astype(
                                    np.int64, copy=False
                                )
                                arr_idx = int(start) + keep
                            else:
                                arr_idx = np.arange(int(start), int(end), dtype=np.int64)
                        selected_by_history_row[row_idx_i] = arr_idx
                    if arr_idx.size == 0:
                        continue
                    selected_idx.append(arr_idx)
                    selected_row.append(
                        np.full(arr_idx.size, hit_i, dtype=np.int64)
                    )
                if not selected_idx:
                    continue
                flat_arr_idx = np.concatenate(selected_idx)
                flat_row = np.concatenate(selected_row)
                total = int(flat_arr_idx.shape[0])
            else:
                flat_row = np.repeat(np.arange(n_hits, dtype=np.int64), lengths)
                row_start_in_flat = np.cumsum(lengths) - lengths  # (n_hits,)
                within_row = np.arange(total, dtype=np.int64) - row_start_in_flat[flat_row]
                flat_arr_idx = lo[flat_row] + within_row  # (total,)

            flat_conts_kenlm = np.asarray(
                order_data["continuations"][flat_arr_idx], dtype=np.uint32
            )
            flat_counts = np.asarray(
                order_data["counts"][flat_arr_idx], dtype=np.uint64
            )

            # Translate kenlm→dolma2 and drop sentinel / zero-count rows. The
            # raw-count threshold, when set, has already been applied before
            # flattening so pathological low-order rows don't create giant
            # temporary arrays.
            flat_did = self.kenlm_to_dolma2[flat_conts_kenlm].astype(np.int64, copy=False)
            valid_mask = (flat_did < self.vocab_size) & (flat_counts > 0)
            if not valid_mask.any():
                continue
            n_valid = int(valid_mask.sum())
            flat_row_v = flat_row[valid_mask]
            flat_did_v = flat_did[valid_mask]
            flat_counts_v = flat_counts[valid_mask]
            flat_h_total = h_totals[flat_row_v].astype(np.float64)
            log_discount = (self.N_max - n) * self.log_alpha
            flat_log_score = (
                np.log(flat_counts_v.astype(np.float64))
                - np.log(flat_h_total)
                + log_discount
            )
            flat_positions = hit_positions[flat_row_v].astype(np.int64)

            per_order.append((flat_positions, flat_did_v, flat_log_score, n))
            order_cap = self.max_order_continuations.get(n)
            min_count = self.min_order_counts.get(n)
            min_count_summary = (
                f",min_count={int(min_count):,}" if min_count is not None else ""
            )
            if order_cap is not None:
                order_summaries.append(
                    f"n{n}:hist_hits={n_hits:,},raw={total_raw:,},"
                    f"kept={total:,},overrides={n_valid:,},"
                    f"cap={int(order_cap):,}{min_count_summary}"
                )
            else:
                order_summaries.append(
                    f"n{n}:hist_hits={n_hits:,},overrides={n_valid:,}"
                    f"{min_count_summary}"
                )

        if not per_order:
            if log_timing:
                self._maybe_log_override_call(t0, S, 0, order_summaries)
            return EMPTY

        # Concatenate per-order outputs with a parallel order-id column, then
        # resolve "highest-order-wins" by sorting (position asc, did asc,
        # order desc) and taking the first entry per (position, did).
        all_pos = np.concatenate([p for p, _, _, _ in per_order])
        all_did = np.concatenate([d for _, d, _, _ in per_order])
        all_score = np.concatenate([s for _, _, s, _ in per_order])
        all_order = np.concatenate(
            [np.full(len(p), n, dtype=np.int8) for p, _, _, n in per_order]
        )

        # Negate order so the sort puts the highest order first within each
        # (position, did) group.
        sort_idx = np.lexsort((-all_order, all_did, all_pos))
        pos_s = all_pos[sort_idx]
        did_s = all_did[sort_idx]
        score_s = all_score[sort_idx]

        is_first = np.empty(pos_s.shape[0], dtype=bool)
        is_first[0] = True
        is_first[1:] = (pos_s[1:] != pos_s[:-1]) | (did_s[1:] != did_s[:-1])

        result = (
            pos_s[is_first].astype(np.int32, copy=False),
            did_s[is_first].astype(np.int32, copy=False),
            score_s[is_first].astype(np.float32, copy=False),
        )
        if log_timing:
            self._maybe_log_override_call(t0, S, int(result[0].shape[0]), order_summaries)
        return result

    def _maybe_log_override_call(
        self,
        t0: float,
        sequence_length: int,
        n_overrides: int,
        order_summaries: list[str],
    ) -> None:
        self._override_debug_calls += 1
        elapsed = time.perf_counter() - t0
        if (
            self._override_debug_calls <= 3
            or elapsed >= 1.0
            or _sb_debug_enabled()
        ):
            summary = "; ".join(order_summaries) if order_summaries else "no higher-order hits"
            _sb_log(
                f"compute_overrides call={self._override_debug_calls} "
                f"S={sequence_length:,} overrides={n_overrides:,} "
                f"elapsed={elapsed:.3f}s {summary}"
            )
