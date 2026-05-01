"""
Wrap a base :class:`InstanceSource` with per-position ngram soft targets.

For each instance, we add a ``soft_target_token_ids`` (S, K) array of top-K
next-token candidate IDs plus one of two value arrays, controlled by the
``output_log_probs`` flag:

- ``soft_target_probs[i]`` (default): K linear probabilities renormalized
  to sum to 1 over the K candidates. Right shape for the soft-cross-entropy
  arm where the ngram is treated as a self-contained target distribution
  over those K tokens.
- ``soft_target_log_probs[i]`` (when ``output_log_probs=True``): K raw kenlm
  log-probabilities, in natural log (full-vocabulary log-probabilities for
  those K tokens). Right shape for the product-of-experts arm where we add
  ``λ * log_p_ngram`` to ``log_p_lm = log_softmax(LM_logits)`` at the K
  positions and rely on softmax to renormalize the joint.

The per-position probe uses
:class:`olmo_core.data.ngram_soft_target.NgramTableSoftTargetSource`,
which reads a precomputed top-K forward index built offline by
``data_gen/build_topk_forward_index.py``. Lookup is binary-search on the
longest-matching order's prefix table, then a single row read of K
(token, value) pairs — no kenlm at runtime.

We do *not* EOS-stop the context window: the n-gram tables were built
one-line-per-document, so high-order probes that straddle a document
boundary naturally miss and degrade through the order ladder.
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
    :param table_dir: Directory containing ``forward_index_topk.bin`` (FXTK
        v=1) produced by ``data_gen/build_topk_forward_index.py``.
    :param K: Top-K size per position. Must match the K the index was built
        with.
    :param N_max: Highest ngram order to probe. Must be ≤ the highest order
        present in the index.
    """

    DISPLAY_ICON = "\U000f0d77"  # nf-md-graphql

    def __init__(
        self,
        source: InstanceSource,
        *,
        table_dir: PathOrStr,
        K: int = 16,
        N_max: int = 5,
        output_log_probs: bool = False,
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
        self._output_log_probs = bool(output_log_probs)
        # Lazy per-process init: don't mmap in the main process so the source
        # pickles cleanly to spawn workers; first lookup populates.
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
                output_log_probs=self._output_log_probs,
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
                f"output_log_probs={self._output_log_probs},"
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
        ids, values = self._get_lookup().lookup_batch(contexts)

        out = dict(inst)
        out["soft_target_token_ids"] = ids  # (S, K) int32 / int64
        if self._output_log_probs:
            # Natural-log full-vocabulary log-probabilities for the K
            # candidates. PoE-arm signal — added directly to LM log-probs.
            out["soft_target_log_probs"] = values  # (S, K) float32
        else:
            # Linear probabilities renormalized to sum 1 over K candidates.
            # Soft-cross-entropy-arm signal.
            out["soft_target_probs"] = values  # (S, K) float32
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
    output_log_probs: bool = False
    label: Optional[str] = None

    def build(self, work_dir: PathOrStr) -> NgramSoftTargetInstanceSource:
        base = self.source.build(work_dir)
        return NgramSoftTargetInstanceSource(
            base,
            table_dir=self.table_dir,
            K=self.K,
            N_max=self.N_max,
            output_log_probs=self.output_log_probs,
            work_dir=work_dir,
            label=self.label,
        )


NgramSoftTargetInstanceSource.Config = NgramSoftTargetInstanceSourceConfig  # type: ignore[attr-defined]
