"""
Wrap a base :class:`InstanceSource` with per-position observed-context IDs.

For each instance, we add ``engram_context_ids`` (S). Each ID points to the
longest observed ngram context row in a context-key table derived from the
prefix blocks of the raw ``forward_index.bin``. The wrapper does not read KN
continuation token IDs, continuation counts, log probabilities, or backoff
weights, so it can feed learned Engram-style memory without leaking the
corpus-derived continuation distribution. It fails loudly instead of falling
back to K-specific top-k indexes.
"""

from __future__ import annotations

import functools as ft
import hashlib
from dataclasses import dataclass
from typing import Optional

import numpy as np

from olmo_core.aliases import PathOrStr

from .instance_source import Instance, InstanceSource, InstanceSourceConfig


class NgramContextInstanceSource(InstanceSource):
    """
    An :class:`InstanceSource` that wraps another one and augments each
    instance with per-position observed-context row IDs.

    :param source: The wrapped instance source.
    :param table_dir: Directory containing raw ``forward_index.bin`` (FIX1 v=2).
    :param N_max: Highest ngram order to probe. Must be <= the highest order
        present in the index.
    """

    DISPLAY_ICON = "\U000f0d77"  # nf-md-graphql

    def __init__(
        self,
        source: InstanceSource,
        *,
        table_dir: PathOrStr,
        N_max: int = 5,
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
        self._N_max = int(N_max)
        self._lookup = None

    @property
    def source(self) -> InstanceSource:
        return self._source

    @property
    def table_dir(self) -> str:
        return self._table_dir

    @property
    def N_max(self) -> int:
        return self._N_max

    def _get_lookup(self):
        if self._lookup is None:
            from olmo_core.data.ngram_topk import NgramContextSource

            self._lookup = NgramContextSource(
                table_dir=self._table_dir,
                N_max=self._N_max,
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
                f"N_max={self._N_max},"
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
        prefix_len = self._N_max - 1

        contexts = [
            tuple(int(t) for t in input_ids[max(0, i + 1 - prefix_len) : i + 1])
            for i in range(S)
        ]
        context_ids = self._get_lookup().lookup_batch(contexts)

        out = dict(inst)
        out["engram_context_ids"] = context_ids
        return out

    def children(self):
        return [self._source]


@dataclass
class NgramContextInstanceSourceConfig(InstanceSourceConfig):
    """Config for :class:`NgramContextInstanceSource`."""

    source: InstanceSourceConfig
    table_dir: str
    N_max: int = 5
    label: Optional[str] = None

    def build(self, work_dir: PathOrStr) -> NgramContextInstanceSource:
        base = self.source.build(work_dir)
        return NgramContextInstanceSource(
            base,
            table_dir=self.table_dir,
            N_max=self.N_max,
            work_dir=work_dir,
            label=self.label,
        )


NgramContextInstanceSource.Config = NgramContextInstanceSourceConfig  # type: ignore[attr-defined]
