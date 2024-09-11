import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.utils.data

from ..aliases import PathOrStr
from ..config import Config
from ..distributed.utils import barrier
from ..utils import roundrobin, threaded_generator
from .collator import DataCollator
from .numpy_dataset import NumpyDatasetBase, NumpyFSLDataset, NumpyVSLDataset
from .utils import iter_batched, load_array_slice

__all__ = [
    "IterableDatasetBase",
    "IterableFSLDataset",
    "IterableVSLDataset",
    "VSLCurriculum",
    "VSLNaturalCurriculum",
    "VSLGrowP2Curriculum",
]

log = logging.getLogger(__name__)


class IterableDatasetBase(ABC, torch.utils.data.IterableDataset[Dict[str, Any]]):
    """
    Adapted from PyTorch's ``DistributedSampler``, this is a base class for iterable datasets
    that wrap a :class:`~olmo_core.data.NumpyDatasetBase`
    and can be deterministically restarted.

    .. warning::
        This is used internally by the :class:`~olmo_core.train.Trainer`.
        In general you shouldn't be using these classes directly unless you really know what you're
        doing! It's easy to misuse, resulting in incorrect data order.
    """

    def __init__(
        self,
        dataset: NumpyDatasetBase,
        *,
        rank_batch_size: int,
        collator: DataCollator,
        work_dir: PathOrStr,
        seed: int = 0,
        epoch: int = 0,
        shuffle: bool = True,
        num_threads: Optional[int] = None,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: int = 0,
        start_index: int = 0,
    ):
        self.dataset = dataset
        self.rank_batch_size = rank_batch_size
        self.collator = collator
        self.seed = seed
        self.epoch = epoch
        self.shuffle = shuffle
        self.num_threads = num_threads
        self.work_dir = work_dir
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.fs_local_rank = fs_local_rank
        # NOTE: the semantic of 'start_index' depend on the implementation.
        # It could be an instance index, batch index, or something else.
        self.start_index = start_index

    @property
    @abstractmethod
    def total_batches(self) -> int:
        """
        The total number of batches that the dataset will produce over the course of an epoch.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _global_indices_file(self) -> Path:
        raise NotImplementedError

    @abstractmethod
    def _build_global_indices(self) -> np.ndarray:
        """
        Construct the global shuffled instance indices.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_local_instance_indices(self, indices: np.ndarray) -> Iterable[int]:
        """
        Filter global instance indices down to the local worker indices.
        """
        raise NotImplementedError

    @property
    def global_batch_size(self) -> int:
        return self.rank_batch_size * self.dp_world_size

    @property
    def worker_info(self):
        return torch.utils.data.get_worker_info()

    def _get_global_indices(self) -> np.ndarray:
        """
        Get the global shuffled instance indices.
        """
        if not self._global_indices_file.is_file():
            raise RuntimeError(
                "Missing global indices file, did you forget to call 'build_and_save_global_indices()' "
                "or 'reshuffle()'?"
            )
        return np.memmap(self._global_indices_file, mode="r", dtype=np.uint32)  # type: ignore

    def build_and_save_global_indices(self):
        """
        Construct and save the global shuffled instance indices.
        Should be called from all ranks after initialization.
        """
        if self.fs_local_rank == 0:
            if self._global_indices_file.is_file():
                log.info(
                    f"Using existing global indices file for seed {self.seed} and epoch {self.epoch} "
                    f"at '{self._global_indices_file}'"
                )
            else:
                log.info(
                    f"Saving global data order indices for seed {self.seed} and epoch {self.epoch} "
                    f"to '{self._global_indices_file}'..."
                )
                self._global_indices_file.parent.mkdir(parents=True, exist_ok=True)
                global_indices = self._build_global_indices()
                global_indices_mmap = np.memmap(
                    self._global_indices_file,
                    dtype=np.uint32,
                    mode="w+",
                    shape=(len(global_indices),),
                )
                global_indices_mmap[:] = global_indices
                global_indices_mmap.flush()
                del global_indices_mmap
                log.info(f"Global data order indices saved to '{self._global_indices_file}'")
        barrier()

    def state_dict(self, *, batches_processed: int, tokens_processed: int) -> Dict[str, Any]:
        return {
            "dataset_fingerprint_version": self.dataset.fingerprint_version,
            "dataset_fingerprint": self.dataset.fingerprint,
            "batches_processed": batches_processed,
            "tokens_processed": tokens_processed,
            "seed": self.seed,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if state_dict["dataset_fingerprint_version"] != self.dataset.fingerprint_version:
            log.warning(
                "Dataset fingerprint version does not match the version in the checkpoint, "
                "this could mean the data has changed"
            )
        elif state_dict["dataset_fingerprint"] != self.dataset.fingerprint:
            raise RuntimeError(
                "Restoring state from a different dataset is not supported! (fingerprint doesn't match)"
            )

        if state_dict["seed"] != self.seed:
            log.warning(
                "Restoring data loading state with a different data seed, "
                "will use data seed from state dict for data order consistency."
            )
            self.seed = state_dict["seed"]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over the local rank+worker instances.
        """
        indices = iter(self._get_local_instance_indices(self._get_global_indices()))

        num_threads = self.num_threads
        if self.worker_info is None and self.num_threads is None:
            # If `num_threads` hasn't been specified and we're not using multiprocessing we'll
            # try to guess a good number of threads.
            num_threads = 4

        # Potentially slice by threads.
        instance_iterator: Iterator[Dict[str, Any]]
        if num_threads:
            # In order to stay ahead of training the total queue size (sum across all threads)
            # should be bigger than the maximum number of instances per batch locally.
            max_instances_per_rank: int
            if isinstance(self.dataset, NumpyFSLDataset):
                max_instances_per_rank = self.rank_batch_size // self.dataset.sequence_length
            elif isinstance(self.dataset, NumpyVSLDataset):
                max_instances_per_rank = self.rank_batch_size // self.dataset.min_sequence_length
            else:
                raise NotImplementedError

            queue_size = math.ceil(max_instances_per_rank * 2 / num_threads)

            thread_generators = []
            for i in range(num_threads):
                generator = (
                    self._get_dataset_item(int(idx))
                    for idx in islice(indices, i, None, num_threads)
                )
                thread_generators.append(
                    threaded_generator(
                        generator, maxsize=queue_size, thread_name=f"data thread {i}"
                    )
                )

            instance_iterator = (x for x in roundrobin(*thread_generators))
        else:
            instance_iterator = (self._get_dataset_item(int(idx)) for idx in indices)

        return (
            self.collator(batch) for batch in iter_batched(instance_iterator, self.rank_batch_size)
        )

    def reshuffle(self, epoch: int):
        """
        Reshuffle for the given epoch. Should be called at the beginning of an epoch.

        :param epoch: The epoch number.
        """
        self.epoch = epoch
        self.build_and_save_global_indices()

    def reset(self):
        """
        Reset epoch bookkeeping. Should be called at the end of an epoch.
        """
        self.start_index = 0

    def _get_dataset_item(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        if isinstance(item, dict):
            return dict(**item, index=idx)
        else:
            return {"input_ids": item, "index": idx}


class IterableFSLDataset(IterableDatasetBase):
    """
    An iterable dataset that wraps a :class:`~olmo_core.data.NumpyFSLDataset`.
    """

    def __init__(
        self,
        dataset: NumpyFSLDataset,
        *,
        chunk_size: int = 1,
        **kwargs,
    ):
        assert chunk_size >= 1
        super().__init__(dataset, **kwargs)
        self.chunk_size = chunk_size

    @property
    def total_size(self) -> int:
        """
        The total number of instances that the dataset will produce over the course of an epoch.
        """
        return self.dp_world_size * (len(self.dataset) // self.dp_world_size)

    @property
    def total_batches(self) -> int:
        assert isinstance(self.dataset, NumpyFSLDataset)
        return self.total_size // (self.global_batch_size // self.dataset.sequence_length)

    @property
    def _global_indices_file(self) -> Path:
        global_indices_fname = (
            f"global_indices_seed{self.seed}_epoch{self.epoch}_size{self.total_size}"
        )
        if self.chunk_size > 1:
            global_indices_fname += f"_chunk{self.chunk_size}"
        return Path(self.work_dir) / f"{global_indices_fname}.npy"

    def _build_global_indices(self) -> np.ndarray:
        assert len(self.dataset) < np.iinfo(np.uint32).max

        rng: Optional[np.random.Generator] = None
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            rng = _get_rng(self.seed + self.epoch)

        indices: np.ndarray
        if self.chunk_size == 1:
            indices = np.arange(len(self.dataset), dtype=np.uint32)
            if rng is not None:
                rng.shuffle(indices)
        else:
            chunk_indices = np.arange(len(self.dataset) // self.chunk_size)
            if rng is not None:
                rng.shuffle(chunk_indices)
            indices = np.repeat(chunk_indices * self.chunk_size, self.chunk_size)
            indices = indices.reshape((-1, self.chunk_size)) + np.arange(
                0, self.chunk_size
            ).reshape((1, -1))
            indices = indices.reshape(-1)

        # Remove tail of data to make it evenly divisible.
        indices = indices[: self.total_size]
        return indices

    def _get_local_instance_indices(self, indices: np.ndarray) -> Iterable[int]:
        # NOTE: 'indices' are global instance indices.

        # Start at the specified global instance index.
        indices = indices[self.start_index : self.total_size]

        # Slice up by batch.
        assert isinstance(self.dataset, NumpyFSLDataset)
        instances_per_batch = self.global_batch_size // self.dataset.sequence_length
        # shape: (global num batches, global num instances per batch)
        indices = indices.reshape(-1, instances_per_batch)

        # Slice batches by data loader worker rank to avoid duplicates.
        if (worker_info := self.worker_info) is not None:
            # Note that each data loading worker gathers a whole batch at a time, and the workers
            # are called round-robin by rank. So to slice these up in a way that preserves order, regardless
            # of the number of workers, we give worker 0 the first batch, worker 1 the second batch, etc.
            indices = indices[worker_info.id :: worker_info.num_workers]

        # Finally slice batches into micro batches for the local DP rank.
        indices = indices[:, self.dp_rank :: self.dp_world_size].reshape((-1,))

        return indices

    def state_dict(self, *, batches_processed: int, tokens_processed: int) -> Dict[str, Any]:
        state_dict = super().state_dict(
            batches_processed=batches_processed,
            tokens_processed=tokens_processed,
        )
        assert isinstance(self.dataset, NumpyFSLDataset)
        state_dict["sequence_length"] = self.dataset.sequence_length
        state_dict["max_target_sequence_length"] = self.dataset.max_target_sequence_length
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        super().load_state_dict(state_dict)

        assert isinstance(self.dataset, NumpyFSLDataset)

        if state_dict["max_target_sequence_length"] != self.dataset.max_target_sequence_length:
            raise RuntimeError(
                "Restoring an FSL dataset state with a different 'max_target_sequence_length' "
                "is not supported"
            )

        # Set 'start_index' (which indexes into instances).
        self.start_index = state_dict["tokens_processed"] // self.global_batch_size


@dataclass
class VSLCurriculum(Config):
    """
    Base class for variable sequence length curriculums.
    """

    @abstractmethod
    def batches_per_bucket(
        self, dataset: NumpyVSLDataset, global_batch_size: int
    ) -> List[Tuple[int, int]]:
        raise NotImplementedError

    @abstractmethod
    def get_batch_indices(
        self, batches_per_bucket: Sequence[Tuple[int, int]], seed: int
    ) -> np.ndarray:
        raise NotImplementedError

    def get_total_batches(self, batches_per_bucket: Sequence[Tuple[int, int]]) -> int:
        return sum([batches for _, batches in batches_per_bucket])


@dataclass
class VSLNaturalCurriculum(VSLCurriculum):
    """
    Implements the natural curriculum from
    `Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum
    <https://arxiv.org/pdf/2405.13226>`_.
    """

    def batches_per_bucket(
        self, dataset: NumpyVSLDataset, global_batch_size: int
    ) -> List[Tuple[int, int]]:
        batches_per_bucket = []
        for seq_len, num_instances in dataset.instances_per_bucket:
            instances_per_batch = global_batch_size // seq_len
            batches = num_instances // instances_per_batch
            batches_per_bucket.append((seq_len, batches))
        return batches_per_bucket

    def get_batch_indices(
        self, batches_per_bucket: Sequence[Tuple[int, int]], seed: int
    ) -> np.ndarray:
        batch_indices = np.arange(self.get_total_batches(batches_per_bucket), dtype=np.uint32)
        rng = _get_rng(seed)
        rng.shuffle(batch_indices)
        return batch_indices


@dataclass
class VSLGrowP2Curriculum(VSLCurriculum):
    """
    Implements the "Grow-P2" curriculum from
    `Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum
    <https://arxiv.org/pdf/2405.13226>`_.
    """

    num_cycles: int = 8

    def batches_per_bucket(
        self, dataset: NumpyVSLDataset, global_batch_size: int
    ) -> List[Tuple[int, int]]:
        actual_batches_per_bucket = VSLNaturalCurriculum().batches_per_bucket(
            dataset, global_batch_size
        )
        batches_per_bucket = min([batches for _, batches in actual_batches_per_bucket])
        batches_per_bucket = self.num_cycles * (batches_per_bucket // self.num_cycles)
        return [(seq_len, batches_per_bucket) for seq_len, _ in actual_batches_per_bucket]

    def get_batch_indices(
        self, batches_per_bucket: Sequence[Tuple[int, int]], seed: int
    ) -> np.ndarray:
        raise NotImplementedError


class IterableVSLDataset(IterableDatasetBase):
    """
    An iterable dataset that wraps a :class:`~olmo_core.data.NumpyVSLDataset` and implements
    a sequence length-based curriculum as introduced in
    `Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum
    <https://arxiv.org/pdf/2405.13226>`_.
    """

    def __init__(
        self,
        dataset: NumpyVSLDataset,
        *,
        curriculum: Optional[VSLCurriculum] = None,
        **kwargs,
    ):
        super().__init__(dataset, **kwargs)
        self._batches_per_bucket: Optional[Tuple[Tuple[int, int], ...]] = None
        self._buckets: Optional[Tuple[int, ...]] = None
        self.curriculum = curriculum or VSLNaturalCurriculum()
        self.start_index = 0
        if not self.shuffle:
            log.warning("VSL curriculum will be ignored since shuffle=False")

    @property
    def buckets(self) -> Tuple[int, ...]:
        if self._buckets is None:
            self._buckets = tuple([seq_len for seq_len, _ in self.batches_per_bucket])
        return self._buckets

    @property
    def batches_per_bucket(self) -> Tuple[Tuple[int, int], ...]:
        if self._batches_per_bucket is None:
            assert isinstance(self.dataset, NumpyVSLDataset)
            self._batches_per_bucket = tuple(
                self.curriculum.batches_per_bucket(self.dataset, self.global_batch_size)
            )
        return self._batches_per_bucket

    @property
    def total_batches(self) -> int:
        return sum([batches for _, batches in self.batches_per_bucket])

    @property
    def _global_indices_file(self) -> Path:
        global_indices_fname = f"global_batch_indices_seed{self.seed}_epoch{self.epoch}.npy"
        return Path(self.work_dir) / f"dataset-{self.dataset.fingerprint}" / global_indices_fname

    def _bucket_indices_file(self, seq_len: int) -> Path:
        return (
            Path(self.work_dir)
            / f"dataset-{self.dataset.fingerprint}"
            / f"bucket{seq_len}_seed{self.seed}_epoch{self.epoch}_instance_indices.npy"
        )

    def _build_bucket_indices(self, seq_len: int, num_instances: int) -> np.ndarray:
        instance_indices = np.arange(num_instances, dtype=np.uint32)
        if self.shuffle:
            rng = _get_rng(self.seed + self.epoch + seq_len)
            rng.shuffle(instance_indices)
        return instance_indices

    def build_and_save_global_indices(self):
        # We also need to build and save the bucket instance indices.
        if self.fs_local_rank == 0:
            assert isinstance(self.dataset, NumpyVSLDataset)
            for seq_len, num_instances in self.dataset.instances_per_bucket:
                bucket_indices_file = self._bucket_indices_file(seq_len)
                if bucket_indices_file.is_file():
                    log.info(
                        f"Using existing bucket indices file for bucket {seq_len}, seed {self.seed}, "
                        f"and epoch {self.epoch} at '{bucket_indices_file}'"
                    )
                    continue

                log.info(
                    f"Saving bucket indices for bucket {seq_len}, seed {self.seed}, and epoch {self.epoch} "
                    f"to '{bucket_indices_file}'..."
                )
                bucket_indices_file.parent.mkdir(parents=True, exist_ok=True)
                bucket_indices = self._build_bucket_indices(seq_len, num_instances)
                bucket_indices_mmap = np.memmap(
                    bucket_indices_file,
                    dtype=np.uint32,
                    mode="w+",
                    shape=(num_instances,),
                )
                bucket_indices_mmap[:] = bucket_indices
                bucket_indices_mmap.flush()
                del bucket_indices_mmap
                log.info(f"Bucket indices saved to '{bucket_indices_file}'")
        super().build_and_save_global_indices()

    def _build_global_indices(self) -> np.ndarray:
        if self.shuffle:
            return self.curriculum.get_batch_indices(
                self.batches_per_bucket, self.seed + self.epoch
            )
        else:
            return np.arange(self.total_batches, dtype=np.uint32)

    def _get_local_instance_indices(self, indices: np.ndarray) -> Iterable[int]:
        # NOTE: 'indices' are *batch* indices at this point.

        # Start at the specified batch index.
        if self.start_index > 0:
            indices = indices[self.start_index :]

        # Slice the batch indices by data loader worker rank to avoid duplicates.
        if (worker_info := self.worker_info) is not None:
            # Note that each data loading worker gathers a whole batch at a time, and the workers
            # are called round-robin by rank. So to slice these up in a way that preserves order, regardless
            # of the number of workers, we give worker 0 the first batch, worker 1 the second batch, etc.
            indices = indices[worker_info.id :: worker_info.num_workers]

        for batch_index in indices:
            for instance_index in self._batch_index_to_local_instance_indices(batch_index):
                yield instance_index

    def _batch_index_to_local_instance_indices(self, batch_index: int) -> np.ndarray:
        bucket, bucket_batch_index = self._batch_index_to_bucket_batch_index(batch_index)
        instances_per_batch = self.global_batch_size // bucket
        bucket_indices_file = self._bucket_indices_file(bucket)
        instance_start_index = bucket_batch_index * instances_per_batch
        # Slice up by rank.
        instances_per_rank = instances_per_batch // self.dp_world_size
        instance_start_index += self.dp_rank * instances_per_rank
        instance_end_index = instance_start_index + instances_per_rank
        return load_array_slice(
            bucket_indices_file, instance_start_index, instance_end_index, np.uint32
        )

    def _batch_index_to_bucket_batch_index(self, batch_index: int) -> Tuple[int, int]:
        bucket_start_offset = 0
        bucket_end_offset = 0
        for seq_len, num_batches in self.batches_per_bucket:
            bucket_end_offset += num_batches
            if batch_index < bucket_end_offset:
                return seq_len, batch_index - bucket_start_offset
            bucket_start_offset += num_batches
        raise IndexError(f"Batch index '{batch_index}' out of bounds")

    def state_dict(self, *, batches_processed: int, tokens_processed: int) -> Dict[str, Any]:
        state_dict = super().state_dict(
            batches_processed=batches_processed,
            tokens_processed=tokens_processed,
        )
        assert isinstance(self.dataset, NumpyVSLDataset)
        state_dict["max_sequence_length"] = self.dataset.max_sequence_length
        state_dict["min_sequence_length"] = self.dataset.min_sequence_length
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        super().load_state_dict(state_dict)

        assert isinstance(self.dataset, NumpyVSLDataset)

        if state_dict["max_sequence_length"] != self.dataset.max_sequence_length:
            raise RuntimeError(
                "Restoring dataset state with a different 'max_sequence_length' is not supported"
            )
        elif state_dict["min_sequence_length"] != self.dataset.min_sequence_length:
            raise RuntimeError(
                "Restoring dataset state with a different 'min_sequence_length' is not supported"
            )

        # Set 'start_index' (which indexes into batches).
        self.start_index = state_dict["batches_processed"]


def _get_rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed=seed))
