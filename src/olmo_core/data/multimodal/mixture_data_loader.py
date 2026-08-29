"""Weighted multi-source mixture data loader for Molmo2 stage-1.

Drives the :class:`~olmo_core.train.Trainer` over several map-style multimodal datasets
sampled by per-source weights — the OLMo-core analogue of mm_olmo's ``SubMixture`` /
``IterableDatasetMixture``. Used for the caption + pointing + NLP stage-1 mixture.

The production buffered-packing path follows Molmo2's continuous multinomial source stream
and per-source shuffled epochs. Step-bounded training therefore never resets source RNG state
at an artificial OLMo-core epoch boundary. Batches are reported in *tokens* (``instances ×
pad_sequence_length``) like :class:`MultimodalDataLoader`.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

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
from .rng import make_random_state

log = logging.getLogger(__name__)

DEFAULT_MAX_CONSECUTIVE_DATA_ERRORS = 10
DEFAULT_MAX_TOTAL_DATA_ERRORS = 1000
PACKED_LOADER_STATE_VERSION = 5
LEGACY_PACKED_LOADER_STATE_VERSIONS = (3, 4)

__all__ = ["MixtureDataLoader"]

ExampleRef = Tuple[int, int, int]
LoadedExample = Tuple[ExampleRef, Optional[Dict[str, Any]], Optional[Exception]]
AllowedDataErrorSignatures = Mapping[Tuple[str, int, int], Tuple[type[Exception], str]]


class _OrderedExampleStream(Iterator[Tuple[ExampleRef, Dict[str, Any]]]):
    """Load the exact Molmo2 mixture stream in order and track its consumed cursor."""

    def __init__(
        self,
        loader: "MixtureDataLoader",
        *,
        refs_consumed: int = 0,
    ):
        if refs_consumed < 0:
            raise OLMoConfigurationError("refs_consumed must be non-negative")
        self.loader = loader
        self.refs_consumed = refs_consumed
        self._results = iter(
            prefetch_map(
                loader._try_load_ref,
                loader._rank_refs_from_cursor(refs_consumed),
                num_workers=loader.prefetch_workers,
            )
        )

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
        image_weight: float = 1.0,
        buffer: Optional[Sequence[Tuple[ExampleRef, Dict[str, Any]]]] = None,
        packs_emitted: int = 0,
    ):
        self.example_stream = example_stream
        self.seq_len = seq_len
        self.max_crops_per_pack = max_crops_per_pack
        self.buffer_size = buffer_size
        self.image_weight = image_weight
        self.buffer = list(buffer or [])
        self.packs_emitted = packs_emitted

    def __iter__(self) -> "_BufferedPackingIterator":
        return self

    def __next__(self) -> Dict[str, Any]:
        while True:
            ref, example = next(self.example_stream)
            length = len(example["input_ids"])
            crops = example_crop_count(example)
            token_granularity = max(1, self.seq_len // 512)
            at_token_capacity = (length + token_granularity - 1) // token_granularity >= (
                self.seq_len + token_granularity - 1
            ) // token_granularity
            if at_token_capacity or crops > self.max_crops_per_pack:
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
                self.image_weight,
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
    :param allowed_data_error_signatures: Exact data errors that may be skipped even when they
        exceed the generic error limits. Keys are ``(dataset_name, example_index,
        source_epoch)`` and values are ``(exception_type, exact_message)``. This is intended
        only for narrowly audited, immutable source defects.
    :param allow_legacy_state_without_dataset_fingerprints: Allow restoring version 3 or 4
        buffered-packing cursor state, which predates per-source content fingerprints. This
        remains enabled by default for existing recipes and emits a warning. New recipes that
        require exact content validation should disable it explicitly. State from before
        buffered cursor support is still replayed from the beginning for backwards
        compatibility; that fallback is not an exact content-validated resume.
    """

    _epoch: Optional[int]

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
        pack_image_weight: float = 1.0,
        est_tokens_per_example: int = 1400,
        prefetch_workers: int = 0,
        max_consecutive_data_errors: int = DEFAULT_MAX_CONSECUTIVE_DATA_ERRORS,
        max_total_data_errors: int = DEFAULT_MAX_TOTAL_DATA_ERRORS,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: Optional[int] = None,
        dataset_names: Optional[Sequence[str]] = None,
        allow_legacy_state_without_dataset_fingerprints: bool = True,
        allowed_data_error_signatures: Optional[AllowedDataErrorSignatures] = None,
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
        if not math.isfinite(pack_image_weight) or pack_image_weight < 0:
            raise OLMoConfigurationError("pack_image_weight must be finite and non-negative")
        self.datasets = list(datasets)
        if dataset_names is None:
            self.dataset_names = [str(i) for i in range(len(datasets))]
        elif len(dataset_names) != len(datasets):
            raise OLMoConfigurationError(
                "dataset_names must be the same length as datasets when provided"
            )
        else:
            self.dataset_names = list(dataset_names)
        self.allowed_data_error_signatures = self._validate_allowed_data_error_signatures(
            allowed_data_error_signatures or {}
        )
        self.dataset_fingerprints = [
            self._dataset_fingerprint(dataset, name)
            for dataset, name in zip(self.datasets, self.dataset_names)
        ]
        self.allow_legacy_state_without_dataset_fingerprints = (
            allow_legacy_state_without_dataset_fingerprints
        )
        w = np.asarray(weights, dtype=np.float64)
        if not np.isfinite(w).all() or (w <= 0).any() or not math.isfinite(float(w.sum())):
            raise OLMoConfigurationError("Mixture weights must be finite and strictly positive")
        self.weights = (w / w.sum()).tolist()
        # Molmo2 casts the normalized rates to float32 in IterableDatasetMixture. Keep that
        # exact dtype because it determines the multinomial source-choice boundaries.
        self._sampling_weights = np.asarray(self.weights, dtype=np.float32)
        self.collator = collator
        self.seed = seed
        self.seq_len = collator.pad_sequence_length
        self.pack = pack
        self.pack_max_crops = pack_max_crops
        self.pack_buffer_size = pack_buffer_size
        self.pack_image_weight = pack_image_weight
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

    def _validate_allowed_data_error_signatures(
        self, signatures: AllowedDataErrorSignatures
    ) -> Dict[Tuple[str, int, int], Tuple[type[Exception], str]]:
        """Validate and freeze exact data-error exceptions before iteration."""
        validated: Dict[Tuple[str, int, int], Tuple[type[Exception], str]] = {}
        for key, signature in signatures.items():
            if (
                not isinstance(key, tuple)
                or len(key) != 3
                or not isinstance(key[0], str)
                or key[0] not in self.dataset_names
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in key[1:]
                )
            ):
                raise OLMoConfigurationError(
                    "Allowed data-error keys must be (known_dataset_name, "
                    f"non_negative_index, non_negative_source_epoch); got {key!r}"
                )
            if (
                not isinstance(signature, tuple)
                or len(signature) != 2
                or not isinstance(signature[0], type)
                or not issubclass(signature[0], Exception)
                or not isinstance(signature[1], str)
                or not signature[1]
            ):
                raise OLMoConfigurationError(
                    "Allowed data-error signatures must be (Exception subclass, "
                    f"non-empty exact message); got {signature!r}"
                )
            validated[key] = signature
        return validated

    @staticmethod
    def _dataset_fingerprint(dataset: Any, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Return a stable content identity advertised by a mixture source, if any.

        ``content_fingerprint`` is the preferred protocol. ``fingerprint`` is accepted for
        existing OLMo datasets, including ``NativeTextReplayDataset``. Fingerprint values
        must be non-empty strings so an accidentally unstable or unserializable value cannot
        silently weaken checkpoint validation.
        """
        fingerprint = None
        for attribute in ("content_fingerprint", "fingerprint"):
            fingerprint = getattr(dataset, attribute, None)
            if fingerprint is not None:
                break
        if fingerprint is None:
            return None
        if callable(fingerprint):
            fingerprint = fingerprint()
        if not isinstance(fingerprint, str) or not fingerprint:
            raise OLMoConfigurationError(
                f"Mixture dataset {dataset_name!r} advertises an invalid content fingerprint: "
                f"{fingerprint!r}"
            )

        version = getattr(dataset, "content_fingerprint_version", None)
        if version is None:
            version = getattr(dataset, "fingerprint_version", None)
        if callable(version):
            version = version()
        if version is not None and (not isinstance(version, str) or not version):
            raise OLMoConfigurationError(
                f"Mixture dataset {dataset_name!r} advertises an invalid fingerprint version: "
                f"{version!r}"
            )
        dataset_type = f"{type(dataset).__module__}.{type(dataset).__qualname__}"
        return {"type": dataset_type, "version": version, "value": fingerprint}

    def _validate_dataset_fingerprints(self, saved_fingerprints: Any) -> None:
        """Require every version-5 source identity to match the current mixture exactly."""
        if not isinstance(saved_fingerprints, (list, tuple)):
            raise OLMoConfigurationError(
                "Packed-loader version-5 state is missing its dataset_fingerprints list"
            )
        if len(saved_fingerprints) != len(self.dataset_fingerprints):
            raise OLMoConfigurationError(
                "Packed-loader resume state contains "
                f"{len(saved_fingerprints)} dataset fingerprints, but the current loader "
                f"has {len(self.dataset_fingerprints)} sources"
            )
        for dataset_name, saved, current in zip(
            self.dataset_names, saved_fingerprints, self.dataset_fingerprints
        ):
            if saved != current:
                raise OLMoConfigurationError(
                    f"Packed-loader dataset content fingerprint changed for source "
                    f"{dataset_name!r}: checkpoint has {saved!r}, current dataset has "
                    f"{current!r}"
                )

    @property
    def _global_instances(self) -> int:
        return self.global_batch_size // self.seq_len

    @property
    def _rank_instances(self) -> int:
        return self.rank_batch_size // self.seq_len

    @property
    def total_batches(self) -> Optional[int]:
        if self.pack and self.pack_buffer_size:
            # Molmo2's packed IterableDatasetMixture is infinite and the trainer is bounded
            # by steps. Returning None keeps one continuous source RNG/count stream for the
            # whole run instead of introducing artificial OLMo-core epoch boundaries.
            return None
        if self.pack:
            # Examples are packed several-per-sequence, so an epoch is fewer batches. Estimate
            # the pack count from the average real length (exact count is data-dependent; the
            # cycled packer keeps ranks in sync and the run is step-bounded regardless).
            est_packs = max(1, (self.epoch_instances * self.est_tokens_per_example) // self.seq_len)
            return est_packs // self._global_instances
        return self.epoch_instances // self._global_instances

    @property
    def total_data_errors(self) -> int:
        """Return the cumulative number of data errors consumed by this rank."""
        return self._total_data_errors

    def reshuffle(self, epoch: Optional[int] = None, **kwargs):
        if epoch is not None:
            self._epoch = epoch
        epoch = self._epoch if self._epoch is not None else 1
        if self.pack and self.pack_buffer_size:
            self._order = None
            return
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
        # exhausted, then reshuffle). The third reference coordinate is the source epoch,
        # which drives deterministic per-example augmentation without freezing it forever.
        perms = [rng.permutation(s) if s else np.array([], dtype=int) for s in self._sizes]
        cursors = [0] * len(self.datasets)
        source_epochs = [
            (epoch - 1) * ((self.epoch_instances + size - 1) // size + 1) if size else 0
            for size in self._sizes
        ]
        src_choices = rng.choice(len(self.datasets), size=n, p=self.weights)
        order: List[ExampleRef] = []
        for src in src_choices:
            size = self._sizes[src]
            if size == 0:
                continue
            if cursors[src] >= size:
                perms[src] = rng.permutation(size)
                cursors[src] = 0
                source_epochs[src] += 1
            order.append((int(src), int(perms[src][cursors[src]]), int(source_epochs[src])))
            cursors[src] += 1
        self._order = order

    def _iter_batches(self) -> Iterable[Dict[str, Any]]:
        ri = self._rank_instances
        if self.pack and self.pack_buffer_size:
            packer = self._build_buffered_packer()
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
                while True:
                    yield self.collator([next(packer) for _ in range(ri)])
            finally:
                self._packing_state = self._buffered_packing_state(packer)
                self._active_packer = None
                packer.close()
            return

        if self._order is None:
            raise RuntimeError("call reshuffle() before iterating")
        order = self._order
        n_batches = self.total_batches or 0
        if self.pack:
            rank_refs = order[self.dp_rank :: self.dp_world_size]
            gen = iter_packs(
                self._example_stream(rank_refs),
                self.seq_len,
                max_crops_per_pack=self.pack_max_crops,
                buffer_size=self.pack_buffer_size,
                image_weight=self.pack_image_weight,
            )
            for _ in range(self.batches_processed * ri):  # resume: replay consumed packs
                next(gen)
            for _ in range(self.batches_processed, n_batches):
                yield self.collator([next(gen) for _ in range(ri)])
            return
        gi = self._global_instances
        start_batch = self.batches_processed

        def remaining_rank_refs() -> Iterator[ExampleRef]:
            for batch_idx in range(start_batch, n_batches):
                global_slice = order[batch_idx * gi : (batch_idx + 1) * gi]
                yield from global_slice[self.dp_rank * ri : (self.dp_rank + 1) * ri]

        # Keep one ordered stream alive across batches. Its bounded read-ahead overlaps image
        # preprocessing for the next batch with the current GPU step, while results and data
        # errors are still consumed in precisely the original reference order.
        results = iter(
            prefetch_map(
                self._try_load_ref,
                remaining_rank_refs(),
                num_workers=self.prefetch_workers,
            )
        )
        try:
            for _ in range(start_batch, n_batches):
                examples = []
                for _ in range(ri):
                    ref, example, error = next(results)
                    if error is None:
                        assert example is not None
                        self._consecutive_data_errors = 0
                        examples.append(example)
                    else:
                        self._handle_data_error(ref, error)

                # Match the previous chain(rank_slice, cycle(order)) behavior exactly: any
                # invalid scheduled refs are replaced from a fresh cycle of the global order.
                fallback_refs = itertools.cycle(order)
                while len(examples) < ri:
                    examples.append(self._load_example(fallback_refs))
                yield self.collator(examples)
        finally:
            close = getattr(results, "close", None)
            if close is not None:
                close()

    def _try_load_example(self, ref) -> Dict[str, Any]:
        src_idx, example_idx, source_epoch = ref
        dataset = self.datasets[src_idx]
        get = getattr(dataset, "get", None)
        ex = get(example_idx, source_epoch) if get is not None else dataset[example_idx]
        out = dict(ex)
        out["_source_name"] = self.dataset_names[src_idx]
        return out

    def _try_load_ref(self, ref: ExampleRef) -> LoadedExample:
        """Load one specific ref without mutating error counters in a worker thread."""
        try:
            return ref, self._try_load_example(ref), None
        except Exception as error:
            return ref, None, error

    def _rank_refs_from_cursor(self, refs_consumed: int = 0) -> Iterator[ExampleRef]:
        """Yield this rank's slice of Molmo2's continuous global reference stream.

        Source selection exactly follows ``IterableDatasetMixture``: MT19937 seeded once,
        float32 multinomial rates, a global per-source count, and a source permutation from
        ``make_random_state(seed, source_epoch, 1)``. Vectorized fast-forwarding makes exact
        checkpoint resume practical without storing or replaying preprocessed examples.
        """
        if refs_consumed < 0:
            raise OLMoConfigurationError("refs_consumed must be non-negative")
        if any(size <= 0 for size in self._sizes):
            raise OLMoConfigurationError("Every sampled mixture source must be non-empty")

        rng = np.random.RandomState(self.seed)
        counts = np.zeros(len(self.datasets), dtype=np.int64)
        global_cursor = self.dp_rank + refs_consumed * self.dp_world_size
        remaining = global_cursor
        while remaining:
            chunk_size = min(remaining, 1_000_000)
            choices = rng.choice(len(self.datasets), size=chunk_size, p=self._sampling_weights)
            counts += np.bincount(choices, minlength=len(self.datasets))
            remaining -= chunk_size

        shuffled_orders: List[Tuple[Optional[int], Optional[np.ndarray]]] = [
            (None, None) for _ in self.datasets
        ]
        while True:
            # Generate this rank's global slot plus the intervening slots assigned to the
            # other ranks. Batched choice is bit-identical to repeated scalar choice for
            # RandomState and leaves the MT19937 state at the same position.
            choices = np.asarray(
                rng.choice(
                    len(self.datasets),
                    size=self.dp_world_size,
                    p=self._sampling_weights,
                ),
                dtype=np.int64,
            )
            source = int(choices[0])
            source_count = int(counts[source])
            counts += np.bincount(choices, minlength=len(self.datasets))

            size = self._sizes[source]
            source_epoch = source_count // size
            shuffled_for, shuffled_order = shuffled_orders[source]
            if shuffled_for != source_epoch:
                shuffled_order = np.arange(size, dtype=np.int32)
                make_random_state(self.seed, source_epoch, 1).shuffle(shuffled_order)
                shuffled_orders[source] = (source_epoch, shuffled_order)
            assert shuffled_order is not None
            yield source, int(shuffled_order[source_count % size]), source_epoch

    def _handle_data_error(self, ref: ExampleRef, error: Exception):
        """Apply data-error tolerance in reference order on the consumer thread."""
        self._consecutive_data_errors += 1
        self._total_data_errors += 1
        src_idx, example_idx, source_epoch = ref
        context = (
            f"{self.dataset_names[src_idx]}[{example_idx}] at source epoch {source_epoch} "
            f"(consecutive_data_errors={self._consecutive_data_errors}, "
            f"total_data_errors={self._total_data_errors})"
        )
        allowed_signature = self.allowed_data_error_signatures.get(
            (self.dataset_names[src_idx], example_idx, source_epoch)
        )
        if allowed_signature is not None and (
            type(error) is allowed_signature[0] and str(error) == allowed_signature[1]
        ):
            log.warning("Skipping explicitly allowlisted data error loading %s: %r", context, error)
            return
        if (
            self._consecutive_data_errors > self.max_consecutive_data_errors
            or self._total_data_errors > self.max_total_data_errors
        ):
            message = f"Exceeded data error tolerance loading {context}: {error!r}"
            error.add_note(message)
            log.error(message)
            raise error
        log.warning(
            "Skipping %s[%d] at source epoch %d after error "
            "(consecutive_data_errors=%d, total_data_errors=%d): %r",
            self.dataset_names[src_idx],
            example_idx,
            source_epoch,
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
        if not rank_refs:
            raise OLMoConfigurationError("No examples are available for this data-parallel rank")
        results = iter(
            prefetch_map(
                self._try_load_ref,
                itertools.cycle(rank_refs),
                num_workers=self.prefetch_workers,
            )
        )
        try:
            while True:
                ref, example, error = next(results)
                if error is None:
                    assert example is not None
                    self._consecutive_data_errors = 0
                    yield example
                else:
                    self._handle_data_error(ref, error)
        finally:
            close = getattr(results, "close", None)
            if close is not None:
                close()

    def _build_buffered_packer(self) -> _BufferedPackingIterator:
        if self.pack_max_crops is None:
            raise OLMoConfigurationError("Buffered packing requires pack_max_crops")
        if self._packing_state is None:
            return _BufferedPackingIterator(
                _OrderedExampleStream(self),
                seq_len=self.seq_len,
                max_crops_per_pack=self.pack_max_crops,
                buffer_size=self.pack_buffer_size,
                image_weight=self.pack_image_weight,
            )

        state = self._packing_state
        state_version = int(state.get("version", 0))
        if state_version in LEGACY_PACKED_LOADER_STATE_VERSIONS:
            if not self.allow_legacy_state_without_dataset_fingerprints:
                raise OLMoConfigurationError(
                    f"Packed-loader state version {state_version} predates dataset content "
                    "fingerprints. Set "
                    "allow_legacy_state_without_dataset_fingerprints=True to resume it "
                    "without content validation."
                )
            log.warning(
                "Restoring legacy packed-loader state version %d without validating dataset "
                "content fingerprints because "
                "allow_legacy_state_without_dataset_fingerprints=True",
                state_version,
            )
        if state_version == 3:
            if self.pack_image_weight != 1.0:
                raise OLMoConfigurationError(
                    "Version-3 packed-loader state implies pack_image_weight=1.0"
                )
        elif state_version not in (4, PACKED_LOADER_STATE_VERSION):
            raise OLMoConfigurationError(f"Unsupported packed-loader state version {state_version}")
        expected = {
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
        if state_version >= 4:
            expected["pack_image_weight"] = self.pack_image_weight
        if state_version >= PACKED_LOADER_STATE_VERSION:
            self._validate_dataset_fingerprints(state.get("dataset_fingerprints"))
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
            _OrderedExampleStream(self, refs_consumed=refs_consumed),
            seq_len=self.seq_len,
            max_crops_per_pack=self.pack_max_crops,
            buffer_size=self.pack_buffer_size,
            image_weight=self.pack_image_weight,
            buffer=buffer,
            packs_emitted=packs_emitted,
        )

    @staticmethod
    def _ref_from_state(value: Any) -> ExampleRef:
        try:
            src_idx, example_idx, source_epoch = value
        except (TypeError, ValueError) as error:
            raise OLMoConfigurationError(
                f"Invalid example reference in packed-loader state: {value!r}"
            ) from error
        return int(src_idx), int(example_idx), int(source_epoch)

    def _buffered_packing_state(self, packer: _BufferedPackingIterator) -> Dict[str, Any]:
        state = packer.state_dict()
        state.update(
            {
                "version": PACKED_LOADER_STATE_VERSION,
                "epoch": self._epoch,
                "seed": self.seed,
                "dp_world_size": self.dp_world_size,
                "dp_rank": self.dp_rank,
                "rank_instances": self._rank_instances,
                "seq_len": self.seq_len,
                "pack_buffer_size": self.pack_buffer_size,
                "pack_max_crops": self.pack_max_crops,
                "pack_image_weight": self.pack_image_weight,
                "dataset_sizes": self._sizes,
                "dataset_names": self.dataset_names,
                "dataset_fingerprints": self.dataset_fingerprints,
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
            refs = [(src, i % size, 0) for i in range(max(ri * 4, 4))]
            gen = iter_packs(
                self._example_stream(refs),
                self.seq_len,
                max_crops_per_pack=self.pack_max_crops,
                buffer_size=self.pack_buffer_size,
                image_weight=self.pack_image_weight,
            )
            return self.collator([next(gen) for _ in range(ri)])
        examples = [self._try_load_example((src, i % size, 0)) for i in range(ri)]
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
        epoch = state_dict.get("epoch")
        self._epoch = None if epoch is None else int(epoch)
        self.seed = state_dict.get("seed", self.seed)
        self._consecutive_data_errors = state_dict.get("consecutive_data_errors", 0)
        self._total_data_errors = state_dict.get("total_data_errors", 0)
        self._packing_state = state_dict.get("packing_state")

    def reset(self):
        super().reset()
        self._packing_state = None
