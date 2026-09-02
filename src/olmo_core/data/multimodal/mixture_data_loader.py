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
import logging
import threading
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

from olmo_core.exceptions import OLMoConfigurationError

from ..data_loader import DataLoaderBase
from .collator import MultimodalCollator
from .packed_mixture_iterable import (
    PackedMixtureIterableDataset,
    mixture_epoch_pairs,
    worker_init_fn,
)
from .packing import iter_dynamic_packs, iter_packs
from .prefetch import prefetch_map

log = logging.getLogger(__name__)

DEFAULT_MAX_CONSECUTIVE_DATA_ERRORS = 10
DEFAULT_MAX_TOTAL_DATA_ERRORS = 1000

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

    # Resume replays a checkpointed (seed, epoch) through reshuffle()/mixture_epoch_pairs and
    # skips ahead by batches_processed, rather than persisting the actual shuffle order — so
    # that replay is only correct if the shuffle algorithm hasn't changed since the checkpoint
    # was written. Bump this whenever reshuffle()'s or mixture_epoch_pairs's RNG call sequence
    # changes (even if the seed/epoch semantics stay the same), so load_state_dict() can refuse
    # an unsafe resume instead of silently repeating or skipping examples.
    #
    # 1: the original reshuffle() built every source's full rng.permutation() up front (in
    #    source order), then drew all of src_choices in one rng.choice() call.
    # 2 (current): mixture_epoch_pairs draws rng.choice() in bounded chunks and builds each
    #    source's permutation lazily on first use — a different RNG call sequence for the same
    #    (seed, epoch), even though both are internally deterministic.
    SHUFFLE_ALGO_VERSION = 2

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
        ignore_shuffle_algo_version_mismatch: bool = False,
        pack: bool = False,
        pack_max_crops: Optional[int] = None,
        pack_buffer_size: int = 48,
        pack_image_weight: float = 30.0,
        pack_shortcut_max_len_images: bool = False,
        est_tokens_per_example: int = 1400,
        prefetch_workers: int = 0,
        dl_num_workers: int = 0,
        dl_prefetch_factor: int = 4,
        dl_persistent_workers: bool = True,
        dl_pin_memory: bool = True,
        max_consecutive_data_errors: int = DEFAULT_MAX_CONSECUTIVE_DATA_ERRORS,
        max_total_data_errors: int = DEFAULT_MAX_TOTAL_DATA_ERRORS,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: Optional[int] = None,
        dataset_names: Optional[Sequence[str]] = None,
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
        if dataset_names is None:
            self.dataset_names = [str(i) for i in range(len(datasets))]
        elif len(dataset_names) != len(datasets):
            raise OLMoConfigurationError(
                "dataset_names must be the same length as datasets when provided"
            )
        else:
            self.dataset_names = list(dataset_names)
        w = np.asarray(weights, dtype=np.float64)
        self.weights = (w / w.sum()).tolist()
        self.collator = collator
        self.seed = seed
        self.ignore_shuffle_algo_version_mismatch = ignore_shuffle_algo_version_mismatch
        self.seq_len = collator.pad_sequence_length
        self.pack = pack
        self.pack_max_crops = pack_max_crops
        self.pack_buffer_size = pack_buffer_size
        self.pack_image_weight = pack_image_weight
        self.pack_shortcut_max_len_images = pack_shortcut_max_len_images
        self.est_tokens_per_example = est_tokens_per_example
        if dl_num_workers > 0 and not pack:
            raise OLMoConfigurationError(
                "dl_num_workers > 0 requires pack=True (multiprocess workers run dynamic packing)."
            )
        if dl_num_workers > 0 and pack_max_crops is None:
            raise OLMoConfigurationError(
                "dl_num_workers > 0 requires pack_max_crops (dynamic 2D knapsack packer)."
            )
        if dl_num_workers > 0 and prefetch_workers > 0:
            log.info(
                "dl_num_workers=%d: disabling thread prefetch_workers (packing runs in worker processes)",
                dl_num_workers,
            )
            prefetch_workers = 0
        self.prefetch_workers = prefetch_workers
        self.dl_num_workers = dl_num_workers
        self.dl_prefetch_factor = dl_prefetch_factor
        self.dl_persistent_workers = dl_persistent_workers
        self.dl_pin_memory = dl_pin_memory
        self.max_consecutive_data_errors = max_consecutive_data_errors
        self.max_total_data_errors = max_total_data_errors
        self._consecutive_data_errors = 0
        self._total_data_errors = 0
        self._data_error_lock = threading.Lock()
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
        if self.pack and self.dl_num_workers > 0:
            # The multiprocess packed path (PackedMixtureIterableDataset /
            # iter_rank_mixture_refs) regenerates its own on-the-fly, per-worker,
            # per-DP-rank ref stream in bounded chunks and never reads self._order.
            # Materializing a full epoch here (millions of examples for the full v9/v10
            # mixtures) would just be wasted parent-process memory at startup.
            self._order = []
            return
        # Number of example refs to draw. When packing, an epoch consumes ~all examples
        # (several per packed sequence), so draw a full epoch of examples; otherwise draw
        # exactly enough to fill ``total_batches`` of one-example-per-slot batches.
        n = (
            self.epoch_instances
            if self.pack
            else (self.total_batches or 0) * self._global_instances
        )
        # Shares its chunked-choice / lazy-permutation implementation with
        # iter_rank_mixture_refs (used by the multiprocess packed path) so both compute the
        # same sequence and neither allocates a full-epoch choice array or every source's
        # permutation up front.
        self._order = list(mixture_epoch_pairs(self.seed, epoch, self.weights, self._sizes, n))

    def _iter_batches(self) -> Iterable[Dict[str, Any]]:
        ri = self._rank_instances
        n_batches = self.total_batches or 0
        if self.pack and self.dl_num_workers > 0:
            # Doesn't read self._order (see reshuffle()) — check before the None guard below,
            # since reshuffle() deliberately leaves it empty in this mode.
            yield from self._iter_multiprocess_packed_batches(n_batches)
            return
        if self._order is None:
            raise RuntimeError("call reshuffle() before iterating")
        if self.pack:
            rank_refs = self._order[self.dp_rank :: self.dp_world_size]
            gen = self._pack_stream(rank_refs)
            for _ in range(self.batches_processed * ri):  # resume: replay consumed packs
                next(gen)
            for _ in range(self.batches_processed, n_batches):
                yield self.collator([next(gen) for _ in range(ri)])
            return
        gi = self._global_instances
        for b in range(self.batches_processed, n_batches):
            global_slice = self._order[b * gi : (b + 1) * gi]
            rank_slice = global_slice[self.dp_rank * ri : (self.dp_rank + 1) * ri]
            ref_iter: Iterator = itertools.chain(rank_slice, itertools.cycle(self._order))
            examples = [self._load_example(ref_iter) for _ in range(ri)]
            yield self.collator(examples)

    def _try_load_example(self, ref) -> Dict[str, Any]:
        src_idx, example_idx = ref
        ex = self.datasets[src_idx][example_idx]
        out = dict(ex)
        out["_source_name"] = self.dataset_names[src_idx]
        return out

    def _iter_multiprocess_packed_batches(self, n_batches: int) -> Iterable[Dict[str, Any]]:
        """mm_olmo parity: preprocess + pack + collate in DataLoader worker processes."""
        import torch.utils.data

        dataset = PackedMixtureIterableDataset(
            self.datasets,
            self.dataset_names,
            mixture_seed=self.seed,
            mixture_epoch=self._epoch if self._epoch is not None else 1,
            mixture_weights=self.weights,
            mixture_sizes=self._sizes,
            dp_rank=self.dp_rank,
            dp_world_size=self.dp_world_size,
            epoch_instances=self.epoch_instances,
            seq_len=self.seq_len,
            pack_max_crops=self.pack_max_crops,
            pack_buffer_size=self.pack_buffer_size,
            pack_image_weight=self.pack_image_weight,
            pack_shortcut_max_len_images=self.pack_shortcut_max_len_images,
            max_consecutive_data_errors=self.max_consecutive_data_errors,
            max_total_data_errors=self.max_total_data_errors,
        )
        ri = self._rank_instances
        dl_kwargs: Dict[str, Any] = dict(
            batch_size=ri,
            collate_fn=self.collator,
            num_workers=self.dl_num_workers,
            pin_memory=self.dl_pin_memory,
            worker_init_fn=worker_init_fn,
        )
        if self.dl_num_workers > 0:
            import torch.multiprocessing as mp

            # Must use spawn, not fork: workers are created after the trainer's CUDA dry-run.
            dl_kwargs["multiprocessing_context"] = mp.get_context("spawn")
            dl_kwargs["prefetch_factor"] = self.dl_prefetch_factor
            dl_kwargs["persistent_workers"] = self.dl_persistent_workers
        inner_dl = torch.utils.data.DataLoader(dataset, **dl_kwargs)
        log.info(
            "Starting multiprocess packed DataLoader: workers=%d batch_packs=%d refs=on-the-fly(epoch=%s)",
            self.dl_num_workers,
            ri,
            self._epoch if self._epoch is not None else 1,
        )
        it = iter(inner_dl)
        try:
            for _ in range(self.batches_processed):
                next(it)
            for _ in range(self.batches_processed, n_batches):
                yield next(it)
        finally:
            # Drop the iterator so worker processes exit if the epoch ends early.
            # PyTorch's DataLoader __del__ method handles cleanup of worker processes.
            del it

    def _pack_stream(self, refs: Sequence) -> Iterator[Dict[str, Any]]:
        """Packed-example stream over cycled ``refs``.

        With ``pack_max_crops`` set this is mm_olmo's stage-2 packer: a buffer-48
        2D-knapsack over (text tokens, image crops) that freely mixes text-only and
        image examples in one pack. Without it, the legacy token-only next-fit
        (stage-1 behaviour) is used. ``flush=False`` because the ref stream is
        infinite (cycled).
        """
        stream = self._example_stream(refs)
        if self.pack_max_crops is not None:
            return iter_dynamic_packs(
                stream,
                self.seq_len,
                max_crops_per_pack=self.pack_max_crops,
                buffer_size=self.pack_buffer_size,
                image_weight=self.pack_image_weight,
                shortcut_max_len_images=self.pack_shortcut_max_len_images,
                flush=False,
            )
        return iter_packs(stream, self.seq_len)

    def _try_load_or_none(self, ref) -> Optional[Dict[str, Any]]:
        """Load one ref, returning ``None`` on a tolerated data error.

        Thread-safe: called from prefetch worker threads. The ref -> result mapping is
        1:1, so results stay in ref order regardless of worker count — broken refs are
        skipped downstream without perturbing the order of the surviving examples
        (deterministic packing and resume replay depend on this).
        """
        try:
            out = self._try_load_example(ref)
        except Exception as e:
            with self._data_error_lock:
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
        with self._data_error_lock:
            self._consecutive_data_errors = 0
        return out

    def _load_example(self, ref_iter: Iterator) -> Dict[str, Any]:
        """Load the next valid example, skipping refs that fail formatting (mm_olmo parity)."""
        while True:
            out = self._try_load_or_none(next(ref_iter))
            if out is not None:
                return out

    def _example_stream(self, rank_refs: Sequence) -> Iterator[Dict[str, Any]]:
        """Infinite stream of example dicts for this rank: cycle the refs, load each example
        (heavy image preprocessing) on a background thread pool when ``prefetch_workers > 0``
        so it overlaps the GPU step.

        Refs are pulled from the cycle on the *submitting* thread and results are yielded
        in ref order (``prefetch_map`` preserves input order), so the example stream —
        and therefore packing and resume replay — is deterministic for any worker count.
        """
        ref_iter = itertools.cycle(rank_refs)
        if self.prefetch_workers <= 0:
            while True:
                yield self._load_example(ref_iter)
        else:
            for out in prefetch_map(
                self._try_load_or_none,
                ref_iter,
                num_workers=self.prefetch_workers,
            ):
                if out is not None:
                    yield out

    def get_mock_batch(self) -> Dict[str, Any]:
        ri = max(self._rank_instances, 1)
        # Pull from the first non-empty source.
        src = next((i for i, s in enumerate(self._sizes) if s), 0)
        size = max(self._sizes[src], 1)
        if self.pack:
            refs = [(src, i % size) for i in range(max(ri * 4, 4))]
            gen = self._pack_stream(refs)
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
            "shuffle_algo_version": self.SHUFFLE_ALGO_VERSION,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        # Checkpoints only persist (seed, epoch, batches_processed) and replay reshuffle() to
        # reconstruct the reference order on resume, rather than persisting the order itself —
        # so a resume is only correct if reshuffle()'s RNG call sequence for that (seed, epoch)
        # hasn't changed since the checkpoint was written. Absence of the key means the
        # checkpoint predates this field, i.e. version 1 (see SHUFFLE_ALGO_VERSION).
        checkpoint_version = state_dict.get("shuffle_algo_version", 1)
        if checkpoint_version != self.SHUFFLE_ALGO_VERSION:
            msg = (
                f"Checkpoint was written with mixture shuffle algorithm version "
                f"{checkpoint_version}, but this code uses version {self.SHUFFLE_ALGO_VERSION}. "
                "Resuming would regenerate a different epoch and skip into it at the old batch "
                "offset, silently repeating some examples and omitting others."
            )
            if not self.ignore_shuffle_algo_version_mismatch:
                raise RuntimeError(
                    msg + " Set ignore_shuffle_algo_version_mismatch=True to resume anyway."
                )
            log.warning(msg + " Ignored since ignore_shuffle_algo_version_mismatch=True.")

        self.batches_processed = state_dict.get("batches_processed", 0)
        self._epoch = state_dict.get("epoch")
        self.seed = state_dict.get("seed", self.seed)
