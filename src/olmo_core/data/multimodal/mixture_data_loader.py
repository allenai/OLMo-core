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
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from olmo_core.exceptions import OLMoConfigurationError

from ..data_loader import DataLoaderBase
from .collator import MultimodalCollator
from .packing import (
    _select_buffered_pack_indices,
    example_crop_count,
    iter_packs,
    pack_examples,
)
from .prefetch import prefetch_map

log = logging.getLogger(__name__)

DEFAULT_MAX_CONSECUTIVE_DATA_ERRORS = 10
DEFAULT_MAX_TOTAL_DATA_ERRORS = 1000

__all__ = ["MixtureDataLoader"]

ExampleRef = Tuple[int, int]
LoadedExample = Tuple[ExampleRef, Optional[Dict[str, Any]], Optional[Exception]]


class _OrderedExampleStream(Iterator[Tuple[ExampleRef, Dict[str, Any]]]):
    """Load a cyclic reference stream in a deterministic order and track its cursor."""

    def __init__(
        self,
        loader: "MixtureDataLoader",
        rank_refs: Sequence[ExampleRef],
        *,
        refs_consumed: int = 0,
    ):
        if not rank_refs:
            raise OLMoConfigurationError("No examples are available for this data-parallel rank")
        if refs_consumed < 0:
            raise OLMoConfigurationError("refs_consumed must be non-negative")
        self.loader = loader
        self.rank_refs = rank_refs
        self.refs_consumed = refs_consumed
        self._results = iter(
            prefetch_map(
                loader._try_load_ref,
                self._refs_from_cursor(refs_consumed),
                num_workers=loader.prefetch_workers,
            )
        )

    def _refs_from_cursor(self, cursor: int) -> Iterator[ExampleRef]:
        index = cursor % len(self.rank_refs)
        while True:
            yield self.rank_refs[index]
            index += 1
            if index == len(self.rank_refs):
                index = 0

    @property
    def next_ref(self) -> ExampleRef:
        return self.rank_refs[self.refs_consumed % len(self.rank_refs)]

    def __iter__(self) -> "_OrderedExampleStream":
        return self

    def __next__(self) -> Tuple[ExampleRef, Dict[str, Any]]:
        while True:
            ref, example, error = next(self._results)
            # Count refs only as their ordered results are consumed. Thread-pool read-ahead is
            # intentionally excluded so a checkpoint can resume from this exact cursor.
            self.refs_consumed += 1
            if error is None:
                assert example is not None
                self.loader._consecutive_data_errors = 0
                return ref, example
            self.loader._handle_data_error(ref, error)

    def close(self):
        close = getattr(self._results, "close", None)
        if close is not None:
            close()


class _BufferedPackingIterator(Iterator[Dict[str, Any]]):
    """Stateful form of the buffered packer used by the mixture loader."""

    def __init__(
        self,
        example_stream: _OrderedExampleStream,
        *,
        seq_len: int,
        max_crops_per_pack: int,
        buffer_size: int,
        buffer: Optional[Sequence[Tuple[ExampleRef, Dict[str, Any]]]] = None,
        packs_emitted: int = 0,
    ):
        self.example_stream = example_stream
        self.seq_len = seq_len
        self.max_crops_per_pack = max_crops_per_pack
        self.buffer_size = buffer_size
        self.buffer = list(buffer or [])
        self.packs_emitted = packs_emitted

    def __iter__(self) -> "_BufferedPackingIterator":
        return self

    def __next__(self) -> Dict[str, Any]:
        while True:
            ref, example = next(self.example_stream)
            length = len(example["input_ids"])
            crops = example_crop_count(example)
            if length > self.seq_len or crops > self.max_crops_per_pack:
                self.packs_emitted += 1
                return pack_examples([example])
            if len(self.buffer) < self.buffer_size:
                self.buffer.append((ref, example))
                continue

            selected = _select_buffered_pack_indices(
                [len(buffered_example["input_ids"]) for _, buffered_example in self.buffer],
                [example_crop_count(buffered_example) for _, buffered_example in self.buffer],
                self.seq_len,
                self.max_crops_per_pack,
            )
            if not selected:
                raise RuntimeError(
                    "Buffered packer could not select an example within its constraints"
                )
            packed = [self.buffer[i][1] for i in selected]
            for i in sorted(selected, reverse=True):
                self.buffer.pop(i)
            self.buffer.append((ref, example))
            self.packs_emitted += 1
            return pack_examples(packed)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "refs_consumed": self.example_stream.refs_consumed,
            "rank_refs_len": len(self.example_stream.rank_refs),
            "next_ref": self.example_stream.next_ref,
            "buffer_refs": [ref for ref, _ in self.buffer],
            "packs_emitted": self.packs_emitted,
        }

    def close(self):
        self.example_stream.close()


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
        pack_max_crops: Optional[int] = None,
        pack_buffer_size: int = 0,
        est_tokens_per_example: int = 1400,
        prefetch_workers: int = 0,
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
        if pack_buffer_size < 0:
            raise OLMoConfigurationError("pack_buffer_size must be non-negative")
        if pack_buffer_size and not pack:
            raise OLMoConfigurationError("pack_buffer_size requires pack=True")
        if pack_buffer_size and pack_max_crops is None:
            raise OLMoConfigurationError(
                "pack_max_crops is required when pack_buffer_size is positive"
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
        self.seq_len = collator.pad_sequence_length
        self.pack = pack
        self.pack_max_crops = pack_max_crops
        self.pack_buffer_size = pack_buffer_size
        self.est_tokens_per_example = est_tokens_per_example
        self.prefetch_workers = prefetch_workers
        self.max_consecutive_data_errors = max_consecutive_data_errors
        self.max_total_data_errors = max_total_data_errors
        self._consecutive_data_errors = 0
        self._total_data_errors = 0
        self._sizes = [len(d) for d in self.datasets]
        self.epoch_instances = epoch_instances or sum(self._sizes)
        self._order: Optional[List[ExampleRef]] = None
        self._active_packer: Optional[_BufferedPackingIterator] = None
        self._packing_state: Optional[Dict[str, Any]] = None

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
            if self.pack_buffer_size:
                packer = self._build_buffered_packer(rank_refs)
                self._active_packer = packer
                try:
                    if self._packing_state is None:
                        packs_to_replay = self.batches_processed * ri
                        if packs_to_replay:
                            log.warning(
                                "Packed-loader checkpoint has no cursor state; replaying %d "
                                "previously consumed packs once for backwards compatibility",
                                packs_to_replay,
                            )
                        for _ in range(packs_to_replay):
                            next(packer)
                    for _ in range(self.batches_processed, n_batches):
                        yield self.collator([next(packer) for _ in range(ri)])
                finally:
                    self._packing_state = self._buffered_packing_state(packer)
                    self._active_packer = None
                    packer.close()
                return

            gen = iter_packs(
                self._example_stream(rank_refs),
                self.seq_len,
                max_crops_per_pack=self.pack_max_crops,
                buffer_size=self.pack_buffer_size,
            )
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

    def _try_load_ref(self, ref: ExampleRef) -> LoadedExample:
        """Load one specific ref without mutating error counters in a worker thread."""
        try:
            return ref, self._try_load_example(ref), None
        except Exception as error:
            return ref, None, error

    def _handle_data_error(self, ref: ExampleRef, error: Exception):
        """Apply data-error tolerance in reference order on the consumer thread."""
        self._consecutive_data_errors += 1
        self._total_data_errors += 1
        src_idx, example_idx = ref
        if (
            self._consecutive_data_errors > self.max_consecutive_data_errors
            or self._total_data_errors > self.max_total_data_errors
        ):
            error.add_note(
                f"Exceeded data error tolerance loading "
                f"{self.dataset_names[src_idx]}[{example_idx}] "
                f"(consecutive_data_errors={self._consecutive_data_errors}, "
                f"total_data_errors={self._total_data_errors})"
            )
            raise error
        log.warning(
            "Skipping %s[%d] after error " "(consecutive_data_errors=%d, total_data_errors=%d): %r",
            self.dataset_names[src_idx],
            example_idx,
            self._consecutive_data_errors,
            self._total_data_errors,
            error,
        )

    def _load_example(self, ref_iter: Iterator) -> Dict[str, Any]:
        """Load the next valid example, skipping refs that fail formatting (mm_olmo parity)."""
        while True:
            ref = next(ref_iter)
            try:
                out = self._try_load_example(ref)
                self._consecutive_data_errors = 0
                return out
            except Exception as error:
                self._handle_data_error(ref, error)

    def _example_stream(self, rank_refs: Sequence) -> Iterator[Dict[str, Any]]:
        """Infinite stream of example dicts for this rank: cycle the refs, load each example
        (heavy image preprocessing) on a background thread pool when ``prefetch_workers > 0``
        so it overlaps the GPU step, yielding in order to keep packing deterministic."""
        stream = _OrderedExampleStream(self, rank_refs)
        try:
            while True:
                _, example = next(stream)
                yield example
        finally:
            stream.close()

    def _build_buffered_packer(self, rank_refs: Sequence[ExampleRef]) -> _BufferedPackingIterator:
        if not rank_refs:
            raise OLMoConfigurationError("No examples are available for this data-parallel rank")
        if self.pack_max_crops is None:
            raise OLMoConfigurationError("Buffered packing requires pack_max_crops")
        if self._packing_state is None:
            return _BufferedPackingIterator(
                _OrderedExampleStream(self, rank_refs),
                seq_len=self.seq_len,
                max_crops_per_pack=self.pack_max_crops,
                buffer_size=self.pack_buffer_size,
            )

        state = self._packing_state
        expected = {
            "version": 1,
            "epoch": self._epoch,
            "seed": self.seed,
            "dp_world_size": self.dp_world_size,
            "dp_rank": self.dp_rank,
            "rank_instances": self._rank_instances,
            "seq_len": self.seq_len,
            "pack_buffer_size": self.pack_buffer_size,
            "pack_max_crops": self.pack_max_crops,
            "rank_refs_len": len(rank_refs),
            "dataset_sizes": self._sizes,
            "dataset_names": self.dataset_names,
            "weights": self.weights,
        }
        for key, expected_value in expected.items():
            if state.get(key) != expected_value:
                raise OLMoConfigurationError(
                    f"Packed-loader resume state has {key}={state.get(key)!r}, "
                    f"but the current loader requires {expected_value!r}"
                )

        refs_consumed = int(state["refs_consumed"])
        if refs_consumed < 0:
            raise OLMoConfigurationError(
                "Packed-loader resume state contains a negative reference cursor"
            )
        packs_emitted = int(state["packs_emitted"])
        expected_packs = self.batches_processed * self._rank_instances
        if packs_emitted != expected_packs:
            raise OLMoConfigurationError(
                f"Packed-loader resume state contains {packs_emitted} packs, but "
                f"batches_processed requires {expected_packs}"
            )
        next_ref = self._ref_from_state(state["next_ref"])
        if next_ref != rank_refs[refs_consumed % len(rank_refs)]:
            raise OLMoConfigurationError(
                "Packed-loader reference order changed since the checkpoint was written"
            )

        buffer_refs = [self._ref_from_state(ref) for ref in state["buffer_refs"]]
        if len(buffer_refs) > self.pack_buffer_size:
            raise OLMoConfigurationError(
                "Packed-loader checkpoint buffer exceeds the configured buffer size"
            )
        buffer: List[Tuple[ExampleRef, Dict[str, Any]]] = []
        for ref in buffer_refs:
            try:
                buffer.append((ref, self._try_load_example(ref)))
            except Exception as error:
                error.add_note(
                    f"Failed to restore buffered example {ref}; exact packed-loader "
                    "resume requires every saved buffer ref to remain loadable"
                )
                raise

        return _BufferedPackingIterator(
            _OrderedExampleStream(self, rank_refs, refs_consumed=refs_consumed),
            seq_len=self.seq_len,
            max_crops_per_pack=self.pack_max_crops,
            buffer_size=self.pack_buffer_size,
            buffer=buffer,
            packs_emitted=packs_emitted,
        )

    @staticmethod
    def _ref_from_state(value: Any) -> ExampleRef:
        try:
            src_idx, example_idx = value
        except (TypeError, ValueError) as error:
            raise OLMoConfigurationError(
                f"Invalid example reference in packed-loader state: {value!r}"
            ) from error
        return int(src_idx), int(example_idx)

    def _buffered_packing_state(self, packer: _BufferedPackingIterator) -> Dict[str, Any]:
        state = packer.state_dict()
        state.update(
            {
                "version": 1,
                "epoch": self._epoch,
                "seed": self.seed,
                "dp_world_size": self.dp_world_size,
                "dp_rank": self.dp_rank,
                "rank_instances": self._rank_instances,
                "seq_len": self.seq_len,
                "pack_buffer_size": self.pack_buffer_size,
                "pack_max_crops": self.pack_max_crops,
                "dataset_sizes": self._sizes,
                "dataset_names": self.dataset_names,
                "weights": self.weights,
            }
        )
        return state

    def get_mock_batch(self) -> Dict[str, Any]:
        ri = max(self._rank_instances, 1)
        # Pull from the first non-empty source.
        src = next((i for i, s in enumerate(self._sizes) if s), 0)
        size = max(self._sizes[src], 1)
        if self.pack:
            refs = [(src, i % size) for i in range(max(ri * 4, 4))]
            gen = iter_packs(
                self._example_stream(refs),
                self.seq_len,
                max_crops_per_pack=self.pack_max_crops,
                buffer_size=self.pack_buffer_size,
            )
            return self.collator([next(gen) for _ in range(ri)])
        examples = [self.datasets[src][i % size] for i in range(ri)]
        return self.collator(examples)

    def global_num_tokens_in_batch(self, batch: Dict[str, Any]) -> Optional[int]:
        del batch
        return self.global_batch_size

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "batches_processed": self.batches_processed,
            "epoch": self._epoch,
            "seed": self.seed,
            "consecutive_data_errors": self._consecutive_data_errors,
            "total_data_errors": self._total_data_errors,
        }
        if self._active_packer is not None:
            state["packing_state"] = self._buffered_packing_state(self._active_packer)
        elif self._packing_state is not None:
            state["packing_state"] = self._packing_state
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.batches_processed = state_dict.get("batches_processed", 0)
        self._epoch = state_dict.get("epoch")
        self.seed = state_dict.get("seed", self.seed)
        self._consecutive_data_errors = state_dict.get("consecutive_data_errors", 0)
        self._total_data_errors = state_dict.get("total_data_errors", 0)
        self._packing_state = state_dict.get("packing_state")

    def reset(self):
        super().reset()
        self._packing_state = None
