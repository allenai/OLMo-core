"""Multiprocess IterableDataset for packed Molmo2 stage-2 mixtures.

mm_olmo runs preprocess + dynamic packing + collation inside PyTorch ``DataLoader``
worker **processes** (``num_workers=16``, ``prefetch_factor=4``). The legacy
:class:`~olmo_core.data.multimodal.MixtureDataLoader` path prefetched examples on a
thread pool but still packed and collated on the training rank's main iterator thread.

This module yields **packed sequences** (one per ``__iter__`` step) from worker
processes so packing overlaps GPU steps. A standard ``DataLoader`` batches
``rank_instances`` packs and applies :class:`~olmo_core.data.multimodal.MultimodalCollator`.
"""

from __future__ import annotations

import itertools
import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .packing import iter_dynamic_packs

log = logging.getLogger(__name__)

DEFAULT_MAX_CONSECUTIVE_DATA_ERRORS = 10
DEFAULT_MAX_TOTAL_DATA_ERRORS = 1000
DEFAULT_SLOW_LOAD_THRESHOLD_S = 5.0

__all__ = [
    "PackedMixtureIterableDataset",
    "iter_rank_mixture_refs",
    "mixture_epoch_pairs",
    "worker_init_fn",
]


def _slow_load_threshold_s() -> float:
    """Log example loads slower than this (seconds). Set ``MM_DL_SLOW_LOAD_S=0`` to disable."""
    return float(os.environ.get("MM_DL_SLOW_LOAD_S", str(DEFAULT_SLOW_LOAD_THRESHOLD_S)))


_REF_CHUNK_SIZE = 8192
"""Bounded batch size for the ``rng.choice`` draws in :func:`mixture_epoch_pairs` — keeps
peak memory at O(chunk) instead of O(epoch_instances), which for full v9/v10 mixtures
(millions of examples across ~50 sources) previously multiplied across every DP rank and,
in the multiprocess packed path, every DataLoader worker, before the first batch."""


def mixture_epoch_pairs(
    seed: int,
    epoch: int,
    weights: Sequence[float],
    sizes: Sequence[int],
    n: int,
    *,
    chunk_size: int = _REF_CHUNK_SIZE,
) -> Iterator[Tuple[int, int]]:
    """Yield ``n`` ``(source_idx, example_idx)`` draws for one epoch (``seed + epoch`` fixes
    the sequence), weighted by ``weights`` over ``sizes``.

    Shared by :meth:`~olmo_core.data.multimodal.MixtureDataLoader.reshuffle` and
    :func:`iter_rank_mixture_refs` so both compute the exact same sequence. Draws source
    choices in bounded chunks and builds each source's permutation lazily, on first use,
    rather than allocating a full ``n``-length choice array and every source's permutation
    up front regardless of whether that source is ever drawn.
    """
    rng = np.random.RandomState(seed + epoch)
    num_sources = len(sizes)
    perms: List[Optional[np.ndarray]] = [None] * num_sources
    cursors = [0] * num_sources
    remaining = n
    while remaining > 0:
        m = min(chunk_size, remaining)
        for src in rng.choice(num_sources, size=m, p=weights):
            size = sizes[src]
            if size == 0:
                continue
            if perms[src] is None or cursors[src] >= size:
                perms[src] = rng.permutation(size)
                cursors[src] = 0
            yield (int(src), int(perms[src][cursors[src]]))
            cursors[src] += 1
        remaining -= m


def iter_rank_mixture_refs(
    seed: int,
    epoch: int,
    weights: Sequence[float],
    sizes: Sequence[int],
    dp_rank: int,
    dp_world_size: int,
    epoch_instances: int,
) -> Iterator[Tuple[int, int]]:
    """Yield one DP rank's strided slice of one epoch's mixture order, without
    materializing the full epoch.

    Matches :meth:`~olmo_core.data.multimodal.MixtureDataLoader.reshuffle` followed by
    ``rank_refs = order[dp_rank::dp_world_size]`` — both now share
    :func:`mixture_epoch_pairs`, so peak memory stays bounded regardless of how many
    ranks or DataLoader workers call this concurrently.
    """
    for global_i, ref in enumerate(
        mixture_epoch_pairs(seed, epoch, weights, sizes, epoch_instances)
    ):
        if global_i % dp_world_size == dp_rank:
            yield ref


def _repeat_deterministic(factory) -> Iterator:
    """Repeat a finite, deterministic generator forever by re-invoking its factory, instead
    of caching every item it has ever yielded (as ``itertools.cycle`` does)."""
    while True:
        yield from factory()


def worker_init_fn(worker_id: int) -> None:
    """Limit per-worker CPU threads — 16 dataloader workers × OpenMP defaults oversubscribes."""
    import os

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    log.info("PackedMixtureIterableDataset worker %d started (pid=%s)", worker_id, os.getpid())


class PackedMixtureIterableDataset(torch.utils.data.IterableDataset):
    """Yield packed training examples for one DP rank, partitioned across DataLoader workers.

    Each worker process owns a disjoint slice of the infinite ``rank_refs`` stream,
    loads examples (image preprocess + tokenize), and runs the dynamic 2D knapsack
    packer — matching mm_olmo's ``IterableDatasetMixture`` + packer layout.
    """

    def __init__(
        self,
        datasets: Sequence,
        dataset_names: Sequence[str],
        *,
        rank_refs: Optional[Sequence[Tuple[int, int]]] = None,
        mixture_seed: Optional[int] = None,
        mixture_epoch: Optional[int] = None,
        mixture_weights: Optional[Sequence[float]] = None,
        mixture_sizes: Optional[Sequence[int]] = None,
        dp_rank: int = 0,
        dp_world_size: int = 1,
        epoch_instances: Optional[int] = None,
        seq_len: int,
        pack_max_crops: int,
        pack_buffer_size: int = 48,
        pack_image_weight: float = 30.0,
        pack_shortcut_max_len_images: bool = False,
        max_consecutive_data_errors: int = DEFAULT_MAX_CONSECUTIVE_DATA_ERRORS,
        max_total_data_errors: int = DEFAULT_MAX_TOTAL_DATA_ERRORS,
    ):
        if rank_refs is None:
            if (
                mixture_seed is None
                or mixture_epoch is None
                or mixture_weights is None
                or mixture_sizes is None
                or epoch_instances is None
            ):
                raise ValueError(
                    "Provide either rank_refs or mixture_seed/epoch/weights/sizes/epoch_instances"
                )
        elif (
            mixture_seed is not None
            or mixture_epoch is not None
            or mixture_weights is not None
            or mixture_sizes is not None
            or epoch_instances is not None
        ):
            raise ValueError("rank_refs and mixture sampling params are mutually exclusive")
        if len(datasets) != len(dataset_names):
            raise ValueError("datasets and dataset_names must have the same length")
        self.datasets = list(datasets)
        self.dataset_names = list(dataset_names)
        self.rank_refs = list(rank_refs) if rank_refs is not None else None
        self.mixture_seed = mixture_seed
        self.mixture_epoch = mixture_epoch
        self.mixture_weights = list(mixture_weights) if mixture_weights is not None else None
        self.mixture_sizes = list(mixture_sizes) if mixture_sizes is not None else None
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.epoch_instances = epoch_instances
        self.seq_len = seq_len
        self.pack_max_crops = pack_max_crops
        self.pack_buffer_size = pack_buffer_size
        self.pack_image_weight = pack_image_weight
        self.pack_shortcut_max_len_images = pack_shortcut_max_len_images
        self.max_consecutive_data_errors = max_consecutive_data_errors
        self.max_total_data_errors = max_total_data_errors
        # Per-worker error counters (no threading.Lock — must be picklable for DataLoader workers).
        self._consecutive_data_errors = 0
        self._total_data_errors = 0

    def _try_load_example(self, ref: Tuple[int, int]) -> Dict[str, Any]:
        src_idx, example_idx = ref
        slow_threshold = _slow_load_threshold_s()
        t0 = time.perf_counter() if slow_threshold > 0 else 0.0
        ex = self.datasets[src_idx][example_idx]
        out = dict(ex)
        out["_source_name"] = self.dataset_names[src_idx]
        if slow_threshold > 0:
            elapsed = time.perf_counter() - t0
            if elapsed >= slow_threshold:
                log.warning(
                    "Slow example load %.2fs %s[%d] (worker pid may differ from rank)",
                    elapsed,
                    self.dataset_names[src_idx],
                    example_idx,
                )
        return out

    def _try_load_or_none(self, ref: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        try:
            out = self._try_load_example(ref)
        except Exception as e:
            self._consecutive_data_errors += 1
            self._total_data_errors += 1
            consecutive, total = self._consecutive_data_errors, self._total_data_errors
            src_idx, example_idx = ref
            if consecutive > self.max_consecutive_data_errors or total > self.max_total_data_errors:
                e.add_note(
                    f"Exceeded data error tolerance loading "
                    f"{self.dataset_names[src_idx]}[{example_idx}] "
                    f"(consecutive_data_errors={consecutive}, total_data_errors={total})"
                )
                raise
            log.warning(
                "Skipping %s[%d] after error "
                "(consecutive_data_errors=%d, total_data_errors=%d): %r",
                self.dataset_names[src_idx],
                example_idx,
                consecutive,
                total,
                e,
            )
            return None
        self._consecutive_data_errors = 0
        return out

    def _load_example(self, ref_iter: Iterator[Tuple[int, int]]) -> Dict[str, Any]:
        while True:
            out = self._try_load_or_none(next(ref_iter))
            if out is not None:
                return out

    def _example_stream(self, ref_iter: Iterator[Tuple[int, int]]) -> Iterator[Dict[str, Any]]:
        while True:
            yield self._load_example(ref_iter)

    def _ref_stream(self) -> Iterator[Tuple[int, int]]:
        if self.rank_refs is not None:
            # A finite, pre-supplied list — fine to cache/replay via itertools.cycle.
            return itertools.cycle(self.rank_refs)
        assert (
            self.mixture_seed is not None
            and self.mixture_epoch is not None
            and self.mixture_weights is not None
            and self.mixture_sizes is not None
            and self.epoch_instances is not None
        )
        # iter_rank_mixture_refs is finite (one epoch). Repeat it by re-invoking rather
        # than wrapping in itertools.cycle: cycle() caches every ref it has ever yielded to
        # replay it, which for a full v9/v10 epoch (millions of examples) would reintroduce
        # the same unbounded-memory-growth problem this generator exists to avoid. Since the
        # sequence is fully determined by ``seed + epoch``, re-invoking regenerates the
        # identical sequence each lap at the cost of some CPU instead of memory.
        mixture_seed, mixture_epoch = self.mixture_seed, self.mixture_epoch
        mixture_weights, mixture_sizes = self.mixture_weights, self.mixture_sizes
        epoch_instances = self.epoch_instances

        def factory() -> Iterator[Tuple[int, int]]:
            return iter_rank_mixture_refs(
                mixture_seed,
                mixture_epoch,
                mixture_weights,
                mixture_sizes,
                self.dp_rank,
                self.dp_world_size,
                epoch_instances,
            )

        return _repeat_deterministic(factory)

    def _partitioned_ref_iter(self, worker_id: int, num_workers: int) -> Iterator[Tuple[int, int]]:
        for i, ref in enumerate(self._ref_stream()):
            if i % num_workers == worker_id:
                yield ref

    def _packed_stream(self, worker_id: int, num_workers: int) -> Iterator[Dict[str, Any]]:
        ref_iter = self._partitioned_ref_iter(worker_id, num_workers)
        example_stream = self._example_stream(ref_iter)
        return iter_dynamic_packs(
            example_stream,
            self.seq_len,
            max_crops_per_pack=self.pack_max_crops,
            buffer_size=self.pack_buffer_size,
            image_weight=self.pack_image_weight,
            shortcut_max_len_images=self.pack_shortcut_max_len_images,
            flush=False,
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
        log.info(
            "PackedMixtureIterableDataset __iter__ worker=%d/%d refs=%s",
            worker_id,
            num_workers,
            f"on-the-fly(seed={self.mixture_seed},epoch={self.mixture_epoch})"
            if self.rank_refs is None
            else str(len(self.rank_refs)),
        )
        packs_yielded = 0
        for pack in self._packed_stream(worker_id, num_workers):
            packs_yielded += 1
            if packs_yielded == 1:
                log.info("PackedMixtureIterableDataset worker=%d yielded first pack", worker_id)
            yield pack
