"""Wrap a base :class:`InstanceSource` with per-position stupid-backoff
ngram overrides.

For each instance, we add three flat numpy arrays of identical length
``n_overrides`` (variable per instance — typically a few times the
sequence length for typical pilots, since each position contributes
~2-50 observed continuations across orders 2..N_max)::

    sb_override_position   (n_overrides,) int32 — seq position i ∈ [0, S-2]
    sb_override_token_id   (n_overrides,) int32 — dolma2 token id w
    sb_override_log_score  (n_overrides,) float32 — natural-log SB score

These are the *additive* override the training step needs in order to
inject the stupid-backoff bias on top of a shared length-V_dolma2
unigram floor. The train_module loads the unigram floor itself from the
same counts_index/ directory at startup (it's constant across all
batches — no point shipping V floats through the dataloader at every
position), and at the train step it broadcast-adds ``λ · unigram_floor``
to logits then scatter-adds
``λ · (sb_override_log_score - unigram_floor[sb_override_token_id])``
at the (b, position, token_id) triples to override the unigram floor
with the higher-order SB score where one exists.

This is the parallel of :class:`NgramSoftTargetInstanceSource`, which
emits fixed-K dense per-position outputs from the KN-smoothed top-K
forward index. The SB analog has variable-K per position, which is
the whole point of switching to stupid backoff — no truncation, no
top-K cap by default. For training experiments that need a hard runtime
bound, the source can cap only the order-2 continuation set and leave
orders 3+ exact.

The reader (:class:`olmo_core.data.stupid_backoff_ngram.StupidBackoffNgramLM`)
is lazily constructed on first ``__getitem__`` call, so the source
pickles cleanly to spawn-mode dataloader workers.
"""

from __future__ import annotations

import functools as ft
import hashlib
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from olmo_core.aliases import PathOrStr

from .instance_source import Instance, InstanceSource, InstanceSourceConfig


class NgramStupidBackoffInstanceSource(InstanceSource):
    """An :class:`InstanceSource` that wraps another one and augments
    each instance with per-position stupid-backoff overrides.

    :param source: The wrapped instance source.
    :param table_dir: Directory containing a v2 ``counts_index/`` produced
        by ``data_gen/build_counts_index.py``. Either the pilot dir
        (which contains ``counts_index/`` as a subdir) or that subdir
        directly is accepted.
    :param dolma2_vocab_size: V_dolma2 — the LM tokenizer's vocab size,
        used as the support of the unigram floor and the dimension of the
        bias vectors. For dolma2 this is 100_278.
    :param N_max: Highest ngram order to probe. Must equal the highest
        order present in the index (currently 5).
    :param alpha: Stupid-backoff discount factor per Brants et al 2007.
        Default 0.4.
    :param max_order2_continuations: Optional hard cap on order-2
        continuations per one-token history. Orders 3+ remain exact; omitted
        order-2 continuations fall through to the unigram floor.
    """

    DISPLAY_ICON = "\U000f0d77"  # nf-md-graphql (same as the KN-smoothed source)

    def __init__(
        self,
        source: InstanceSource,
        *,
        table_dir: PathOrStr,
        dolma2_vocab_size: int,
        N_max: int = 5,
        alpha: float = 0.4,
        max_order2_continuations: Optional[int] = None,
        work_dir: PathOrStr,
        label: Optional[str] = None,
    ):
        super().__init__(
            work_dir=work_dir,
            sequence_length=source.sequence_length,
            max_sequence_length=source.max_sequence_length,
            label=label if label is not None else source.label,
        )
        self._source = source
        self._table_dir = str(table_dir)
        self._dolma2_vocab_size = int(dolma2_vocab_size)
        self._N_max = int(N_max)
        self._alpha = float(alpha)
        self._max_order2_continuations = max_order2_continuations
        # Lazy per-process init: don't mmap in the main process so the
        # source pickles cleanly to spawn workers; first lookup populates.
        self._reader = None
        self._getitem_debug_calls = 0

    @property
    def source(self) -> InstanceSource:
        return self._source

    @property
    def table_dir(self) -> str:
        return self._table_dir

    @property
    def N_max(self) -> int:
        return self._N_max

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def dolma2_vocab_size(self) -> int:
        return self._dolma2_vocab_size

    def _get_reader(self):
        if self._reader is None:
            from olmo_core.data.stupid_backoff_ngram import StupidBackoffNgramLM

            self._reader = StupidBackoffNgramLM(
                table_dir=self._table_dir,
                dolma2_vocab_size=self._dolma2_vocab_size,
                N_max=self._N_max,
                alpha=self._alpha,
                max_order2_continuations=self._max_order2_continuations,
            )
        return self._reader

    @ft.cached_property
    def fingerprint(self) -> str:
        sha = hashlib.sha256()
        sha.update(
            (
                f"class={self.__class__.__name__},"
                f"source={self._source.fingerprint},"
                f"table_dir={self._table_dir},"
                f"dolma2_vocab_size={self._dolma2_vocab_size},"
                f"N_max={self._N_max},"
                f"alpha={self._alpha},"
                f"max_order2_continuations={self._max_order2_continuations},"
            ).encode()
        )
        return sha.hexdigest()

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, idx: int) -> Instance:
        idx = self.validate_index(idx)
        t0 = time.perf_counter()
        inst = self._source[idx]
        t_source = time.perf_counter()

        input_ids = np.asarray(inst["input_ids"], dtype=np.int64)
        positions, token_ids, log_scores = self._get_reader().compute_overrides_for_sequence(
            input_ids
        )
        elapsed = time.perf_counter() - t0
        self._getitem_debug_calls += 1
        if self._getitem_debug_calls <= 4 or elapsed >= 1.0:
            print(
                f"[SB source pid={os.getpid()}] __getitem__ call={self._getitem_debug_calls} "
                f"idx={idx} S={input_ids.shape[0]:,} overrides={positions.shape[0]:,} "
                f"source={t_source - t0:.3f}s sb={time.perf_counter() - t_source:.3f}s "
                f"total={elapsed:.3f}s",
                flush=True,
            )

        out = dict(inst)
        out["sb_override_position"] = positions  # (n_overrides,) int32
        out["sb_override_token_id"] = token_ids  # (n_overrides,) int32
        out["sb_override_log_score"] = log_scores  # (n_overrides,) float32
        return out

    def children(self):
        return [self._source]


@dataclass
class NgramStupidBackoffInstanceSourceConfig(InstanceSourceConfig):
    """Config for :class:`NgramStupidBackoffInstanceSource`."""

    source: InstanceSourceConfig
    table_dir: str
    dolma2_vocab_size: int
    N_max: int = 5
    alpha: float = 0.4
    max_order2_continuations: Optional[int] = None
    label: Optional[str] = None

    def build(self, work_dir: PathOrStr) -> NgramStupidBackoffInstanceSource:
        base = self.source.build(work_dir)
        return NgramStupidBackoffInstanceSource(
            base,
            table_dir=self.table_dir,
            dolma2_vocab_size=self.dolma2_vocab_size,
            N_max=self.N_max,
            alpha=self.alpha,
            max_order2_continuations=self.max_order2_continuations,
            work_dir=work_dir,
            label=self.label,
        )


NgramStupidBackoffInstanceSource.Config = NgramStupidBackoffInstanceSourceConfig  # type: ignore[attr-defined]
