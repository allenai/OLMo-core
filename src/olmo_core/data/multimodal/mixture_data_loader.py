"""Weighted multi-source mixture data loader for Molmo2 stage-1.

Drives the :class:`~olmo_core.train.Trainer` over several map-style multimodal datasets
sampled by per-source weights — the OLMo-core analogue of mm_olmo's ``SubMixture`` /
``IterableDatasetMixture``. Used for the caption + pointing + NLP stage-1 mixture.

Each epoch interleaves examples by drawing a source per slot from ``weights`` (multinomial)
and cycling through a shuffled permutation of that source, so each source contributes
roughly ``weight`` of the examples. Batches are reported in *tokens*
(``instances × pad_sequence_length``) like :class:`MultimodalDataLoader`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from olmo_core.exceptions import OLMoConfigurationError

from ..data_loader import DataLoaderBase
from .collator import MultimodalCollator

__all__ = ["MixtureDataLoader"]


class MixtureDataLoader(DataLoaderBase):
    """Sample examples across multiple datasets by ``weights`` and collate into batches.

    :param datasets: the source datasets (each map-style, ``__getitem__`` -> example dict).
    :param weights: per-source sampling weights (need not sum to 1; normalized internally).
    :param collator: must have a fixed ``pad_sequence_length`` (token-based batching).
    :param global_batch_size: global batch size in *tokens* (= global instances × seq len).
    :param epoch_instances: number of (global) instances that make up one epoch; defaults to
        the sum of the source lengths.
    """

    def __init__(
        self,
        datasets: Sequence,
        weights: Sequence[float],
        collator: MultimodalCollator,
        *,
        work_dir,
        global_batch_size: int,
        seed: int = 0,
        epoch_instances: Optional[int] = None,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: Optional[int] = None,
    ):
        super().__init__(
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
        )
        if collator.pad_sequence_length is None:
            raise OLMoConfigurationError(
                "MixtureDataLoader requires the collator to have a fixed `pad_sequence_length`."
            )
        if len(datasets) != len(weights) or not datasets:
            raise OLMoConfigurationError(
                "datasets and weights must be non-empty and the same length"
            )
        self.datasets = list(datasets)
        w = np.asarray(weights, dtype=np.float64)
        self.weights = (w / w.sum()).tolist()
        self.collator = collator
        self.seed = seed
        self.seq_len = collator.pad_sequence_length
        self._sizes = [len(d) for d in self.datasets]
        self.epoch_instances = epoch_instances or sum(self._sizes)
        self._order: Optional[List] = None  # list of (src_idx, example_idx)

    @property
    def _global_instances(self) -> int:
        return self.global_batch_size // self.seq_len

    @property
    def _rank_instances(self) -> int:
        return self.rank_batch_size // self.seq_len

    @property
    def total_batches(self) -> Optional[int]:
        return self.epoch_instances // self._global_instances

    def reshuffle(self, epoch: Optional[int] = None, **kwargs):
        if epoch is not None:
            self._epoch = epoch
        epoch = self._epoch if self._epoch is not None else 1
        rng = np.random.RandomState(self.seed + epoch)
        n = (self.total_batches or 0) * self._global_instances
        # Per-source shuffled cycles (sampling within a source without replacement until
        # exhausted, then reshuffle — covers sources smaller than their sampled count).
        perms = [rng.permutation(s) if s else np.array([], dtype=int) for s in self._sizes]
        cursors = [0] * len(self.datasets)
        src_choices = rng.choice(len(self.datasets), size=n, p=self.weights)
        order: List = []
        for src in src_choices:
            size = self._sizes[src]
            if size == 0:
                continue
            if cursors[src] >= size:
                perms[src] = rng.permutation(size)
                cursors[src] = 0
            order.append((int(src), int(perms[src][cursors[src]])))
            cursors[src] += 1
        self._order = order

    def _iter_batches(self) -> Iterable[Dict[str, Any]]:
        if self._order is None:
            raise RuntimeError("call reshuffle() before iterating")
        gi, ri = self._global_instances, self._rank_instances
        n_batches = self.total_batches or 0
        for b in range(self.batches_processed, n_batches):
            global_slice = self._order[b * gi : (b + 1) * gi]
            rank_slice = global_slice[self.dp_rank * ri : (self.dp_rank + 1) * ri]
            examples = [self.datasets[src][idx] for src, idx in rank_slice]
            yield self.collator(examples)

    def get_mock_batch(self) -> Dict[str, Any]:
        ri = max(self._rank_instances, 1)
        # Pull from the first non-empty source.
        src = next((i for i, s in enumerate(self._sizes) if s), 0)
        examples = [self.datasets[src][i % max(self._sizes[src], 1)] for i in range(ri)]
        return self.collator(examples)

    def global_num_tokens_in_batch(self, batch: Dict[str, Any]) -> Optional[int]:
        del batch
        return self.global_batch_size

    def state_dict(self) -> Dict[str, Any]:
        return {
            "batches_processed": self.batches_processed,
            "epoch": self._epoch,
            "seed": self.seed,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.batches_processed = state_dict.get("batches_processed", 0)
        self._epoch = state_dict.get("epoch")
        self.seed = state_dict.get("seed", self.seed)
