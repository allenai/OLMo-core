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
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


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
    """

    def __init__(
        self,
        table_dir: str,
        *,
        dolma2_vocab_size: int,
        N_max: int = 5,
        alpha: float = 0.4,
    ):
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

        # Build kenlm ↔ dolma2 id translation tables from pilot.counts.vocab.
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

        # Mmap the per-order index files.
        self.orders: Dict[int, dict] = {}
        for n_str, info in self.meta["per_order"].items():
            n = int(n_str)
            n_hist = int(info["n_hist"])
            plen = n - 1
            order_data: dict = {
                "n_hist": n_hist,
                "offsets": np.memmap(
                    index_dir / f"order{n}.offsets.bin",
                    dtype=np.uint64, mode="r",
                ),
                "continuations": np.memmap(
                    index_dir / f"order{n}.continuations.bin",
                    dtype=np.uint32, mode="r",
                ),
                "counts": np.memmap(
                    index_dir / f"order{n}.counts.bin",
                    dtype=np.uint64, mode="r",
                ),
                "history_totals": np.memmap(
                    index_dir / f"order{n}.history_totals.bin",
                    dtype=np.uint64, mode="r",
                ),
            }
            if plen == 0:
                order_data["histories"] = None
                order_data["histories_struct"] = None
            else:
                hist = np.memmap(
                    index_dir / f"order{n}.histories.bin",
                    dtype=np.uint32, mode="r",
                ).reshape(n_hist, plen)
                struct_dtype = np.dtype([("h", np.uint32, plen)])
                order_data["histories"] = hist
                order_data["histories_struct"] = hist.view(struct_dtype).reshape(-1)
            self.orders[n] = order_data

        # Build the unigram floor in dolma2 vocab space (Laplace +1).
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
        input_ids = np.asarray(input_ids, dtype=np.int64)
        S = int(input_ids.shape[0])
        prefix_len = self.N_max - 1

        positions = []
        token_ids = []
        log_scores = []

        for i in range(S - 1):
            ctx_start = max(0, i + 1 - prefix_len)
            ctx_dolma2 = input_ids[ctx_start : i + 1]
            ctx_kenlm = self.dolma2_to_kenlm[ctx_dolma2]
            override = self._override_for_history(ctx_kenlm)
            if not override:
                continue
            n_o = len(override)
            positions.append(np.full(n_o, i, dtype=np.int32))
            tk = np.empty(n_o, dtype=np.int32)
            sc = np.empty(n_o, dtype=np.float32)
            for j, (did, log_score) in enumerate(override.items()):
                tk[j] = did
                sc[j] = log_score
            token_ids.append(tk)
            log_scores.append(sc)

        if not positions:
            return (
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
            )
        return (
            np.concatenate(positions),
            np.concatenate(token_ids),
            np.concatenate(log_scores),
        )
