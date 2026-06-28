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

import itertools
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

from olmo_core.exceptions import OLMoConfigurationError

from ..data_loader import DataLoaderBase
from .collator import MultimodalCollator
from .packing import iter_packs
from .prefetch import prefetch_map

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
        pack: bool = False,
        est_tokens_per_example: int = 1400,
        prefetch_workers: int = 0,
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
        self.pack = pack
        self.est_tokens_per_example = est_tokens_per_example
        self.prefetch_workers = prefetch_workers
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
        if self.pack:
            # Examples are packed several-per-sequence, so an epoch is fewer batches. Estimate
            # the pack count from the average real length (exact count is data-dependent; the
            # cycled packer keeps ranks in sync and the run is step-bounded regardless).
            est_packs = max(1, (self.epoch_instances * self.est_tokens_per_example) // self.seq_len)
            return est_packs // self._global_instances
        return self.epoch_instances // self._global_instances

    def reshuffle(self, epoch: Optional[int] = None, **kwargs):
        if epoch is not None:
            self._epoch = epoch
        epoch = self._epoch if self._epoch is not None else 1
        rng = np.random.RandomState(self.seed + epoch)
        # Number of example refs to draw. When packing, an epoch consumes ~all examples
        # (several per packed sequence), so draw a full epoch of examples; otherwise draw
        # exactly enough to fill ``total_batches`` of one-example-per-slot batches.
        n = (
            self.epoch_instances
            if self.pack
            else (self.total_batches or 0) * self._global_instances
        )
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
        ri = self._rank_instances
        n_batches = self.total_batches or 0
        if self.pack:
            rank_refs = self._order[self.dp_rank :: self.dp_world_size]
            gen = iter_packs(self._example_stream(rank_refs), self.seq_len)
            for _ in range(self.batches_processed * ri):  # resume: replay consumed packs
                next(gen)
            for _ in range(self.batches_processed, n_batches):
                yield self.collator([next(gen) for _ in range(ri)])
            return
        gi = self._global_instances
        for b in range(self.batches_processed, n_batches):
            global_slice = self._order[b * gi : (b + 1) * gi]
            rank_slice = global_slice[self.dp_rank * ri : (self.dp_rank + 1) * ri]
            examples = [self.datasets[src][idx] for src, idx in rank_slice]
            yield self.collator(examples)

    def _example_stream(self, rank_refs: Sequence) -> Iterator[Dict[str, Any]]:
        """Infinite stream of example dicts for this rank: cycle the refs, load each example
        (heavy image preprocessing) on a background thread pool when ``prefetch_workers > 0``
        so it overlaps the GPU step, yielding in order to keep packing deterministic."""
        return prefetch_map(
            lambda r: self.datasets[r[0]][r[1]],
            itertools.cycle(rank_refs),
            num_workers=self.prefetch_workers,
        )

    def get_mock_batch(self) -> Dict[str, Any]:
        ri = max(self._rank_instances, 1)
        # Pull from the first non-empty source.
        src = next((i for i, s in enumerate(self._sizes) if s), 0)
        size = max(self._sizes[src], 1)
        if self.pack:
            refs = [(src, i % size) for i in range(max(ri * 4, 4))]
            gen = iter_packs(self._example_stream(refs), self.seq_len)
            return self.collator([next(gen) for _ in range(ri)])
        examples = [self.datasets[src][i % size] for i in range(ri)]
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
