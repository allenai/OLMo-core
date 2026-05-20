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
        mirror_to_shm: bool = True,
        index_access: str = "mmap",
    ):
        if index_access not in {"mmap", "pread"}:
            raise ValueError(f"index_access must be 'mmap' or 'pread'; got {index_access!r}")
        self.index_access = index_access
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
            f"max_order_continuations={self.max_order_continuations or None}"
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

    # ----- training-time entry point --------------------------------------

    def compute_overrides_for_sequence(
        self, input_ids: np.ndarray
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
            n_queries = S - plen
            if n_queries <= 0:
                continue
            windows = sliding_window_view(ctx_kenlm_full, plen)  # (S-plen+1, plen)
            # Take the first n_queries windows — last position covered is S-2.
            queries = np.ascontiguousarray(windows[:n_queries])  # (n_queries, plen)
            position_for_query = np.arange(plen - 1, plen - 1 + n_queries, dtype=np.int64)

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
            if order_cap is not None:
                k = int(order_cap)
                selected_idx: list[np.ndarray] = []
                selected_row: list[np.ndarray] = []
                # The same one-token history often appears many times in a
                # sequence. Cache the selected global continuation indices per
                # history row so repeated contexts don't re-partition the same
                # large count slice.
                selected_by_history_row: dict[int, np.ndarray] = {}
                for hit_i, (row_idx, start, end) in enumerate(zip(hit_row_indices, lo, hi)):
                    row_idx_i = int(row_idx)
                    arr_idx = selected_by_history_row.get(row_idx_i)
                    if arr_idx is None:
                        row_len = int(end - start)
                        if row_len <= k:
                            arr_idx = np.arange(int(start), int(end), dtype=np.int64)
                        else:
                            row_counts = np.asarray(
                                order_data["counts"][int(start) : int(end)],
                                dtype=np.uint64,
                            )
                            keep = np.argpartition(row_counts, -k)[-k:].astype(
                                np.int64, copy=False
                            )
                            arr_idx = int(start) + keep
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

            # Translate kenlm→dolma2 and drop sentinel / zero-count rows.
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
            if order_cap is not None:
                order_summaries.append(
                    f"n{n}:hist_hits={n_hits:,},raw={total_raw:,},"
                    f"kept={total:,},overrides={n_valid:,},"
                    f"cap={int(order_cap):,}"
                )
            else:
                order_summaries.append(f"n{n}:hist_hits={n_hits:,},overrides={n_valid:,}")

        if not per_order:
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
