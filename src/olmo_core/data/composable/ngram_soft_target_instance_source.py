"""
Wrap a base :class:`InstanceSource` with per-position ngram soft targets.

For each instance, we add two (S, K) arrays:

- ``soft_target_token_ids[i]``: top-K next-token candidate IDs for
  predicting the token at position ``i+1`` given ``input_ids[.. i]``.
- ``soft_target_probs[i]``: the corresponding top-K probabilities,
  summing to 1 along the K axis.

The per-position probe uses
:class:`olmo_core.data.ngram_soft_target.NgramTableSoftTargetSource`,
which does full modified-KN cross-order backoff and falls through to a
precomputed unigram shortlist — so every position is guaranteed a full
top-K (no empty or padded slots).

We do *not* EOS-stop the context window: the n-gram tables were built
one-line-per-document, so high-order probes that straddle a document
boundary naturally miss and degrade through the backoff ladder to a
within-document top-K or the unigram floor. See the module docstring of
``ngram_soft_target.py`` for the combine rule.
"""

from __future__ import annotations

import functools as ft
import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from olmo_core.aliases import PathOrStr

from .instance_source import Instance, InstanceSource, InstanceSourceConfig


class NgramSoftTargetInstanceSource(InstanceSource):
    """
    An :class:`InstanceSource` that wraps another one and augments each
    instance with per-position ngram soft targets.

    :param source: The wrapped instance source.
    :param table_dir: Directory containing ``ngram_table_n<N>.bin`` files
        produced by ``data_gen/arpa_to_table.py``.
    :param K: Top-K size per position.
    :param N_max: Highest ngram order to probe. ``table_dir`` must contain
        ``ngram_table_n1.bin`` through ``ngram_table_n{N_max}.bin``.
    :param unigram_shortlist: Size of the precomputed unigram fallback
        shortlist used at the order-1 level of the backoff ladder.
    """

    DISPLAY_ICON = "\U000f0d77"  # nf-md-graphql

    def __init__(
        self,
        source: InstanceSource,
        *,
        table_dir: PathOrStr,
        K: int = 16,
        N_max: int = 5,
        unigram_shortlist: int = 100,
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
        self._K = int(K)
        self._N_max = int(N_max)
        self._unigram_shortlist = int(unigram_shortlist)
        if self._unigram_shortlist < self._K:
            # Otherwise the order-1 fallback path can't always fill K
            # candidates, and trailing slots end up zero-prob padding.
            # Harmless at loss time (0 * log p = 0) but indicates a
            # misconfiguration — flag it loudly.
            raise ValueError(
                f"unigram_shortlist ({self._unigram_shortlist}) must be >= K ({self._K})"
            )
        # Lazy per-process init: we don't open the mmap'd tables in the
        # main process, so each DataLoader worker ends up with its own
        # _NgramTable instances (OS page cache dedupes the underlying I/O).
        self._lookup = None

    @property
    def source(self) -> InstanceSource:
        return self._source

    @property
    def table_dir(self) -> str:
        return self._table_dir

    @property
    def K(self) -> int:
        return self._K

    @property
    def N_max(self) -> int:
        return self._N_max

    def _get_lookup(self):
        if self._lookup is None:
            # Imported lazily so instantiating this class in a process that
            # never calls __getitem__ (e.g. the main coordinator rank) does
            # not require numba/xxhash to be importable there.
            from olmo_core.data.ngram_soft_target import NgramTableSoftTargetSource

            self._lookup = NgramTableSoftTargetSource(
                table_dir=self._table_dir,
                K=self._K,
                N_max=self._N_max,
                unigram_shortlist=self._unigram_shortlist,
            )
        return self._lookup

    @ft.cached_property
    def fingerprint(self) -> str:
        sha = hashlib.sha256()
        sha.update(
            (
                f"class={self.__class__.__name__},"
                f"source={self._source.fingerprint},"
                f"table_dir={self._table_dir},"
                f"K={self._K},"
                f"N_max={self._N_max},"
                f"unigram_shortlist={self._unigram_shortlist},"
            ).encode()
        )
        return sha.hexdigest()

    def __len__(self) -> int:
        return len(self._source)

    def __getitem__(self, idx: int) -> Instance:
        idx = self.validate_index(idx)
        inst = self._source[idx]

        input_ids = np.asarray(inst["input_ids"], dtype=np.int64)
        S = int(input_ids.shape[0])
        prefix_len = self._N_max - 1  # longest prefix we can probe at order N_max

        # For position i (whose soft target predicts input_ids[i+1]) the
        # context is the last up-to-prefix_len tokens ending at i inclusive:
        #   context_i = input_ids[max(0, i+1-prefix_len) : i+1]
        contexts = [
            tuple(int(t) for t in input_ids[max(0, i + 1 - prefix_len) : i + 1])
            for i in range(S)
        ]
        ids, probs = self._get_lookup().lookup_batch(contexts)

        out = dict(inst)
        out["soft_target_token_ids"] = ids  # (S, K) int32
        out["soft_target_probs"] = probs  # (S, K) float32
        return out

    def children(self):
        return [self._source]


@dataclass
class NgramSoftTargetInstanceSourceConfig(InstanceSourceConfig):
    """Config for :class:`NgramSoftTargetInstanceSource`."""

    source: InstanceSourceConfig
    table_dir: str
    K: int = 16
    N_max: int = 5
    unigram_shortlist: int = 100
    label: Optional[str] = None

    def build(self, work_dir: PathOrStr) -> NgramSoftTargetInstanceSource:
        base = self.source.build(work_dir)
        return NgramSoftTargetInstanceSource(
            base,
            table_dir=self.table_dir,
            K=self.K,
            N_max=self.N_max,
            unigram_shortlist=self.unigram_shortlist,
            work_dir=work_dir,
            label=self.label,
        )


NgramSoftTargetInstanceSource.Config = NgramSoftTargetInstanceSourceConfig  # type: ignore[attr-defined]
