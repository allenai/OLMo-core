"""
Distributed, deterministic, stateful data loaders used by the :class:`~olmo_core.train.Trainer`.

Overview
--------

Construct a data loader from a :class:`~olmo_core.data.numpy_dataset.NumpyDatasetBase` instance
using :meth:`NumpyDataLoaderBase.wrap_numpy_dataset()`::

    data_loader = NumpyDataLoaderBase.wrap_numpy_dataset(dataset, ...)

Then load batches for an epoch like this::

    # Prepare for the epoch.
    data_loader.reshuffle(epoch=1)

    for batch in data_loader:
        # process batch
        pass

    # Reset internal bookkeeping.
    data_loader.reset()

"""

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data

from ..aliases import PathOrStr
from ..config import Config
from ..distributed.utils import barrier, get_fs_local_rank, get_rank, get_world_size
from ..exceptions import OLMoConfigurationError
from ..utils import get_default_device, roundrobin, threaded_generator
from .collator import DataCollator
from .numpy_dataset import (
    NumpyDatasetBase,
    NumpyDatasetType,
    NumpyFSLDataset,
    NumpyVSLDataset,
)
from .utils import get_rng, iter_batched, load_array_slice, memmap_to_write

__all__ = [
    "DataLoaderBase",
    "NumpyDataLoaderBase",
    "NumpyFSLDataLoader",
    "NumpyVSLDataLoader",
    "NumpyDataLoaderConfig",
]

log = logging.getLogger(__name__)


class DataLoaderBase(ABC):
    """
    An abstract base class for data loaders used by the :class:`~olmo_core.train.Trainer`.

    .. warning::
        When using a :class:`DataLoaderBase` directly (outside of the :class:`~olmo_core.train.Trainer`),
        you must call :meth:`reshuffle()` *before* starting a new epoch
        (i.e. before calling :meth:`__iter__`) and you must call :meth:`reset()` *after* each
        epoch (i.e. after the iterator returned from :meth:`__iter__` has been exhausted).
        Failure to do so will result in incorrect data order.

    :param collator: The data collator to use to create batches from instances.
    :param work_dir: The working directory. Should be shared among local ranks.
    :param global_batch_size: The global batch size *in tokens*.
    :param dp_world_size: The data parallel world size.
    :param dp_rank: The local data parallel rank.
    :param fs_local_rank: The filesystem-local rank.
    """

    def __init__(
        self,
        *,
        collator: DataCollator,
        work_dir: PathOrStr,
        global_batch_size: int,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: int = 0,
    ):
        self.collator = collator
        self.work_dir = work_dir
        self.global_batch_size = global_batch_size
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.fs_local_rank = fs_local_rank

        self.batches_processed = 0
        """
        The total number of batches processed so far in the current epoch.
        """

        self.tokens_processed = 0
        """
        The total number of tokens processed globally so far in the current epoch.
        """

        self._epoch: Optional[int] = None

    @property
    def epoch(self) -> int:
        """
        Get the current epoch (1-based).

        .. warning::
            Accessing this before :meth:`reshuffle()` is called will raise an error.
        """
        if self._epoch is None:
            raise RuntimeError(
                "The data loader's epoch has not been set yet, did you forget to call 'reshuffle()'?"
            )
        return self._epoch

    @property
    @abstractmethod
    def total_batches(self) -> Optional[int]:
        """
        The total number of batches that the dataset will produce over the course of an epoch, if known.
        Otherwise this should return ``None``.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """
        Returns the total number of batches in an epoch (same as :data:`total_batches`) if known,
        otherwise a :class:`TypeError` is raised.
        """
        if self.total_batches is not None:
            return self.total_batches
        else:
            raise TypeError("data loader length (number of batches) is unknown")

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over the local rank batches.
        """
        for batch in self._iter_batches():
            if batch["input_ids"].numel() != self.rank_batch_size:
                raise RuntimeError(
                    f"Expected batch size of {self.rank_batch_size:,d} tokens on rank {self.dp_rank}, "
                    f"got input IDs with shape {tuple(batch['input_ids'].shape)} = {batch['input_ids'].numel():,d} tokens"
                )
            self.batches_processed += 1
            self.tokens_processed += self.global_batch_size
            yield batch

    @property
    def rank_batch_size(self) -> int:
        """
        The batch size, per rank, in tokens.
        """
        return self.global_batch_size // self.dp_world_size

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """
        Get a state dictionary for checkpointing.
        """
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load a state dict from :meth:`state_dict()` to restore the data loader's state.
        """
        raise NotImplementedError

    @abstractmethod
    def reshuffle(self, epoch: Optional[int] = None, **kwargs):
        """
        Reshuffle for a new epoch. Should be called before starting the epoch, regardless
        of whether or not you've called :meth:`load_state_dict()`.

        :param epoch: The epoch number.
        """
        raise NotImplementedError

    @abstractmethod
    def _iter_batches(self) -> Iterable[Dict[str, Any]]:
        """
        Returns an iterable over all batches in the epoch.

        .. important::
            This should account for data parallelism in that only the local rank's portion of each
            batch should be generated from this method.

        :returns: All batches in the epoch, where each batch just contains the local rank's portion
            of the batch, which should have exactly :data:`rank_batch_size` tokens.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset epoch bookkeeping. Should be called at the end of an epoch.
        """
        self.batches_processed = 0
        self.tokens_processed = 0

    @abstractmethod
    def get_mock_batch(self) -> Dict[str, Any]:
        """
        Return a batch with arbitrary data. This can just be random data as it's only used by the
        trainer to do a dry-run of the forward and backward pass before training officially starts.
        """
        raise NotImplementedError


class NumpyDataLoaderBase(DataLoaderBase):
    """
    A distributed, deterministic, stateful data loader base class for use with
    :class:`~olmo_core.data.numpy_dataset.NumpyDatasetBase` dataset classes.

    .. seealso::
        - :class:`NumpyFSLDataLoader`
        - :class:`NumpyVSLDataLoader`

    :param dataset: The dataset to wrap / load from.
    :param collator: The data collator to use to create batches from instances.
    :param global_batch_size: The global batch size *in tokens*.
    :param work_dir: The working directory. Should be shared among local ranks.
    :param seed: The seed to use for shuffling / sampling data.
    :param epoch: The epoch to start from.
    :param shuffle: Whether or not to shuffle the data instances.
    :param num_threads: The number of threads to use when loading instances.
    :param num_workers: The number of workers to use when loading batches.
    :param prefetch_factor: The number of batches to prefetch from each worker.
    :param target_device_type: The target device type, i.e. the type of the device where the data
        will ultimately end up on. Note that this data loader does not move batches any device,
        it just uses this to optimize certain settings.
    :param dp_world_size: The data parallel world size.
    :param dp_rank: The local data parallel rank.
    :param fs_local_rank: The filesystem-local rank.
    """

    def __init__(
        self,
        dataset: NumpyDatasetBase,
        *,
        collator: DataCollator,
        global_batch_size: int,
        work_dir: PathOrStr,
        seed: int = 0,
        shuffle: bool = True,
        num_threads: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
        target_device_type: str = "cpu",
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: int = 0,
    ):
        super().__init__(
            collator=collator,
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
        )
        self.dataset = dataset
        self.seed = seed
        self.shuffle = shuffle
        self.num_threads = num_threads
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.target_device_type = target_device_type
        self._global_indices: Optional[np.ndarray] = None

    @classmethod
    def wrap_numpy_dataset(
        cls,
        dataset: NumpyDatasetBase,
        *,
        global_batch_size: int,
        collator: DataCollator,
        work_dir: Optional[PathOrStr] = None,
        seed: int = 0,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: int = 0,
        num_threads: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
        target_device_type: str = "cpu",
    ) -> "NumpyDataLoaderBase":
        """
        Construct the corresponding :class:`NumpyDataLoaderBase` instance for the given :class:`NumpyDatasetBase`.

        :param dataset: The dataset to wrap.
        """
        kwargs = dict(
            global_batch_size=global_batch_size,
            collator=collator,
            work_dir=work_dir or dataset.work_dir,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
            seed=seed,
            num_threads=num_threads,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            target_device_type=target_device_type,
        )
        data_loader: DataLoaderBase
        if isinstance(dataset, NumpyFSLDataset):
            data_loader = NumpyFSLDataLoader(
                dataset,
                **kwargs,  # type: ignore
            )
            if dataset.max_target_sequence_length is not None:
                data_loader.chunk_size = (
                    dataset.max_target_sequence_length // dataset.sequence_length
                )
        elif isinstance(dataset, NumpyVSLDataset):
            data_loader = NumpyVSLDataLoader(
                dataset,
                **kwargs,  # type: ignore
            )
        else:
            raise NotImplementedError

        return data_loader

    def state_dict(self) -> Dict[str, Any]:
        return {
            "dataset_fingerprint_version": self.dataset.fingerprint_version,
            "dataset_fingerprint": self.dataset.fingerprint,
            "batches_processed": self.batches_processed,
            "tokens_processed": self.tokens_processed,
            "seed": self.seed,
            "epoch": self._epoch,
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

        self.batches_processed = state_dict["batches_processed"]
        self.tokens_processed = state_dict["tokens_processed"]
        self._epoch = state_dict["epoch"] or self._epoch

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
    def worker_info(self):
        return torch.utils.data.get_worker_info()

    def get_global_indices(self) -> np.ndarray:
        if self._global_indices is not None:
            return self._global_indices
        if not self._global_indices_file.is_file():
            raise RuntimeError("Missing global indices file, did you forget to call 'reshuffle()'?")
        return np.memmap(self._global_indices_file, mode="r", dtype=np.uint32)  # type: ignore

    def build_and_save_global_indices(self, in_memory: bool = False):
        if in_memory:
            self._global_indices = self._build_global_indices()
        else:
            self._global_indices = None
            if self.fs_local_rank == 0:
                if self._global_indices_file.is_file():
                    log.info(
                        f"Using existing global indices file for seed {self.seed} and epoch {self.epoch} "
                        f"at:\n'{self._global_indices_file}'"
                    )
                else:
                    log.info(
                        f"Saving global data order indices for seed {self.seed} and epoch {self.epoch} "
                        f"to:\n'{self._global_indices_file}'..."
                    )
                    global_indices = self._build_global_indices()
                    assert len(global_indices) < np.iinfo(np.uint32).max
                    with memmap_to_write(
                        self._global_indices_file, shape=global_indices.shape, dtype=np.uint32
                    ) as global_indices_mmap:
                        global_indices_mmap[:] = global_indices
                    log.info(f"Global data order indices saved to:\n'{self._global_indices_file}'")
        barrier()

    def reshuffle(self, epoch: Optional[int] = None, in_memory: bool = False, **kwargs):
        del kwargs
        if epoch is None:
            epoch = 1 if self._epoch is None else self._epoch + 1
        if epoch <= 0:
            raise ValueError(f"'epoch' must be at least 1, got {epoch}")
        self._epoch = epoch
        self.build_and_save_global_indices(in_memory=in_memory)

    def get_mock_batch(self) -> Dict[str, Any]:
        num_instances = self.rank_batch_size // self.dataset.max_sequence_length
        input_ids = torch.randint(
            0, self.dataset.vocab_size, (num_instances, self.dataset.max_sequence_length)
        )
        return {"input_ids": input_ids}

    def _iter_batches(self) -> Iterable[Dict[str, Any]]:
        return torch.utils.data.DataLoader(
            _IterableDatasetWrapper(self),
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=self.target_device_type == "cuda" and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=False,
            timeout=0,
        )

    def _get_dataset_item(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        if isinstance(item, dict):
            return dict(**item, index=idx)
        else:
            return {"input_ids": item, "index": idx}

    def _format_fname_from_fields(self, prefix: str, **fields) -> str:
        parts = [prefix]
        for key in sorted(fields):
            value = fields[key]
            if value is not None:
                parts.append(f"{key}{value}")
        return "_".join(parts)


class NumpyFSLDataLoader(NumpyDataLoaderBase):
    """
    A fixed sequence length :class:`DataLoaderBase` for use with a
    :class:`~olmo_core.data.numpy_dataset.NumpyFSLDataset`.
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
        assert isinstance(self.dataset, NumpyFSLDataset)
        if self.rank_batch_size % self.dataset.sequence_length != 0:
            raise OLMoConfigurationError(
                "rank batch size (in tokens) must be divisible by sequence length"
            )

    @property
    def total_size(self) -> int:
        """
        The total number of instances that the dataset will produce over the course of an epoch.
        """
        assert isinstance(self.dataset, NumpyFSLDataset)
        instances_per_batch = self.global_batch_size // self.dataset.sequence_length
        return instances_per_batch * (len(self.dataset) // instances_per_batch)

    @property
    def total_batches(self) -> int:
        assert isinstance(self.dataset, NumpyFSLDataset)
        return self.total_size // (self.global_batch_size // self.dataset.sequence_length)

    @property
    def _global_indices_file(self) -> Path:
        global_indices_fname = self._format_fname_from_fields(
            "global_indices",
            seed=self.seed if self.shuffle else None,
            epoch=self.epoch if self.shuffle else None,
            size=self.total_size,
            chunk=self.chunk_size if self.chunk_size > 1 else None,
        )
        return Path(self.work_dir) / f"{global_indices_fname}.npy"

    def _build_global_indices(self) -> np.ndarray:
        assert len(self.dataset) < np.iinfo(np.uint32).max

        rng: Optional[np.random.Generator] = None
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            rng = get_rng(self.seed + self.epoch)

        indices: np.ndarray
        if self.chunk_size == 1:
            indices = np.arange(len(self.dataset), dtype=np.uint32)
            if rng is not None:
                rng.shuffle(indices)
        else:
            chunk_indices = np.arange(len(self.dataset) // self.chunk_size, dtype=np.uint32)
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

        # Slice up by batch.
        assert isinstance(self.dataset, NumpyFSLDataset)
        instances_per_batch = self.global_batch_size // self.dataset.sequence_length
        # shape: (global num batches, global num instances per batch)
        indices = indices.reshape(-1, instances_per_batch)

        # Offset by the number of batches already processed.
        if self.batches_processed > 0:
            indices = indices[self.batches_processed :]

        # Slice batches by data loader worker rank to avoid duplicates.
        if (worker_info := self.worker_info) is not None:
            # Note that each data loading worker gathers a whole batch at a time, and the workers
            # are called round-robin by rank. So to slice these up in a way that preserves order, regardless
            # of the number of workers, we give worker 0 the first batch, worker 1 the second batch, etc.
            indices = indices[worker_info.id :: worker_info.num_workers]

        # Finally slice batches into micro batches for the local DP rank.
        indices = indices[:, self.dp_rank :: self.dp_world_size].reshape((-1,))

        return indices

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        assert isinstance(self.dataset, NumpyFSLDataset)
        state_dict["dataset_type"] = str(NumpyDatasetType.fsl)
        state_dict["sequence_length"] = self.dataset.sequence_length
        state_dict["max_target_sequence_length"] = self.dataset.max_target_sequence_length
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        super().load_state_dict(state_dict)
        # Account for change in batch size / sequence length.
        self.batches_processed = self.tokens_processed // self.global_batch_size
        log.info(
            f"Data loader will resume from batch {self.batches_processed}/{self.total_batches}"
        )

        assert isinstance(self.dataset, NumpyFSLDataset)
        if state_dict["dataset_type"] != NumpyDatasetType.fsl:
            raise RuntimeError(
                "Dataset type mismatch: attempting to restore state from a variable sequence length dataset "
                "into a fixed sequence length dataset"
            )

        if state_dict["max_target_sequence_length"] != self.dataset.max_target_sequence_length:
            raise RuntimeError(
                "Restoring an FSL dataset state with a different 'max_target_sequence_length' "
                "is not supported"
            )


class NumpyVSLDataLoader(NumpyDataLoaderBase):
    """
    A variable sequence length :class:`DataLoaderBase` for use with a :class:`~olmo_core.data.numpy_dataset.NumpyVSLDataset`.

    This implements a sequence length-based curriculum as introduced in
    `Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum
    <https://arxiv.org/pdf/2405.13226>`_.
    """

    def __init__(
        self,
        dataset: NumpyVSLDataset,
        **kwargs,
    ):
        super().__init__(dataset, **kwargs)
        self._batches_per_bucket: Optional[Tuple[Tuple[int, int], ...]] = None
        self._buckets: Optional[Tuple[int, ...]] = None
        self._bucket_indices: Optional[Dict[int, np.ndarray]] = None
        if not self.shuffle:
            log.warning("VSL curriculum will be ignored since shuffle=False")
        if self.rank_batch_size < max(self.buckets):
            raise OLMoConfigurationError(
                f"rank batch size ({self.rank_batch_size:,d}) is too small relative to "
                f"maximum sequence length ({max(self.buckets):,d})"
            )

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
                self.dataset.curriculum.batches_per_bucket(self.dataset, self.global_batch_size)
            )
        return self._batches_per_bucket

    @property
    def total_batches(self) -> int:
        return sum([batches for _, batches in self.batches_per_bucket])

    @property
    def _global_indices_file(self) -> Path:
        assert isinstance(self.dataset, NumpyVSLDataset)
        global_indices_fname = self._format_fname_from_fields(
            "global_batch_indices",
            seed=self.seed if self.shuffle else None,
            epoch=self.epoch if self.shuffle else None,
            bz=self.global_batch_size,
        )
        return (
            Path(self.work_dir)
            / f"dataset-{self.dataset.fingerprint}"
            / self.dataset.curriculum.short_str
            / f"{global_indices_fname}.npy"
        )

    def _bucket_indices_file(self, seq_len: int) -> Path:
        assert isinstance(self.dataset, NumpyVSLDataset)
        bucket_indices_fname = self._format_fname_from_fields(
            "instance_indices",
            bucket=seq_len,
            seed=self.seed if self.shuffle else None,
            epoch=self.epoch if self.shuffle else None,
        )
        return (
            Path(self.work_dir)
            / f"dataset-{self.dataset.fingerprint}"
            / self.dataset.curriculum.short_str
            / f"{bucket_indices_fname}.npy"
        )

    def _get_bucket_indices(self, seq_len: int) -> np.ndarray:
        assert isinstance(self.dataset, NumpyVSLDataset)
        bucket_indices = self.dataset.get_instance_bucket(seq_len)
        if self.shuffle:
            rng = get_rng(self.seed + self.epoch + seq_len)
            return rng.permutation(bucket_indices)
        else:
            return bucket_indices

    def build_and_save_global_indices(self, in_memory: bool = False):
        assert isinstance(self.dataset, NumpyVSLDataset)

        if in_memory:
            self._bucket_indices = {}
            for seq_len, num_instances in self.dataset.instances_per_bucket:
                bucket_indices = self._get_bucket_indices(seq_len)
                assert bucket_indices.shape[0] == num_instances
                self._bucket_indices[seq_len] = bucket_indices
        else:
            self._bucket_indices = None

        # We also need to build and save the bucket instance indices.
        if not in_memory and self.fs_local_rank == 0:
            for seq_len, num_instances in self.dataset.instances_per_bucket:
                bucket_indices_file = self._bucket_indices_file(seq_len)
                if bucket_indices_file.is_file():
                    log.info(
                        f"Using existing bucket indices file for bucket {seq_len}, seed {self.seed}, "
                        f"and epoch {self.epoch} at:\n'{bucket_indices_file}'"
                    )
                    continue

                log.info(
                    f"Saving bucket indices for bucket {seq_len}, seed {self.seed}, and epoch {self.epoch} "
                    f"to:\n'{bucket_indices_file}'..."
                )
                bucket_indices = self._get_bucket_indices(seq_len)
                assert bucket_indices.shape[0] == num_instances
                with memmap_to_write(
                    bucket_indices_file, shape=(num_instances,), dtype=np.uint32
                ) as bucket_indices_mmap:
                    bucket_indices_mmap[:] = bucket_indices

                log.info(f"Bucket indices saved to:\n'{bucket_indices_file}'")

            log.info(f"Using {self.dataset.curriculum} with {len(self.buckets)} buckets:")
            self.dataset.curriculum.log_buckets(
                self.dataset, self.global_batch_size, self.batches_per_bucket
            )

        super().build_and_save_global_indices(in_memory=in_memory)

    def _build_global_indices(self) -> np.ndarray:
        if self.shuffle:
            assert isinstance(self.dataset, NumpyVSLDataset)
            return self.dataset.curriculum.get_batch_indices(
                self.batches_per_bucket, self.seed + self.epoch
            )
        else:
            return np.arange(self.total_batches, dtype=np.uint32)

    def _get_local_instance_indices(self, indices: np.ndarray) -> Iterable[int]:
        # NOTE: 'indices' are *batch* indices at this point.

        # Start at the specified batch index.
        if self.batches_processed > 0:
            indices = indices[self.batches_processed :]

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
        bucket_seq_len, bucket_batch_index = self._batch_index_to_bucket_batch_index(batch_index)
        instances_per_batch = self.global_batch_size // bucket_seq_len
        instance_start_index = bucket_batch_index * instances_per_batch

        # Slice up by rank.
        instances_per_rank = instances_per_batch // self.dp_world_size
        instance_start_index += self.dp_rank * instances_per_rank
        instance_end_index = instance_start_index + instances_per_rank

        local_instance_indices: np.ndarray
        if self._bucket_indices is not None:
            local_instance_indices = self._bucket_indices[bucket_seq_len][
                instance_start_index:instance_end_index
            ]
        else:
            bucket_indices_file = self._bucket_indices_file(bucket_seq_len)
            local_instance_indices = load_array_slice(
                bucket_indices_file, instance_start_index, instance_end_index, np.uint32
            )

        assert (
            local_instance_indices.shape[0] == instances_per_rank
        ), f"Expected {instances_per_rank} instances, got {local_instance_indices.shape[0]}"

        return local_instance_indices

    def _batch_index_to_bucket_batch_index(self, batch_index: int) -> Tuple[int, int]:
        bucket_start_offset = 0
        bucket_end_offset = 0
        for seq_len, num_batches in self.batches_per_bucket:
            bucket_end_offset += num_batches
            if bucket_start_offset <= batch_index < bucket_end_offset:
                return seq_len, batch_index - bucket_start_offset
            bucket_start_offset += num_batches
        raise IndexError(f"Batch index '{batch_index}' out of bounds")

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        assert isinstance(self.dataset, NumpyVSLDataset)
        state_dict["dataset_type"] = str(NumpyDatasetType.vsl)
        state_dict["vsl_curriculum"] = self.dataset.curriculum.short_str
        state_dict["max_sequence_length"] = self.dataset.max_sequence_length
        state_dict["min_sequence_length"] = self.dataset.min_sequence_length
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        super().load_state_dict(state_dict)
        log.info(
            f"Data loader will resume from batch {self.batches_processed}/{self.total_batches}"
        )

        assert isinstance(self.dataset, NumpyVSLDataset)
        if state_dict["dataset_type"] != NumpyDatasetType.vsl:
            raise RuntimeError(
                "Dataset type mismatch: attempting to restore state from a fixed sequence length dataset "
                "into a variable sequence length dataset"
            )
        elif state_dict["vsl_curriculum"] != self.dataset.curriculum.short_str:
            raise RuntimeError(
                "Restoring dataset state with a different VSL curriculum is not supported"
            )
        elif state_dict["max_sequence_length"] != self.dataset.max_sequence_length:
            raise RuntimeError(
                "Restoring dataset state with a different 'max_sequence_length' is not supported"
            )
        elif state_dict["min_sequence_length"] != self.dataset.min_sequence_length:
            raise RuntimeError(
                "Restoring dataset state with a different 'min_sequence_length' is not supported"
            )


class _IterableDatasetWrapper(torch.utils.data.IterableDataset[Dict[str, Any]]):
    def __init__(self, data_loader: NumpyDataLoaderBase):
        self.data_loader = data_loader

    @property
    def dataset(self) -> NumpyDatasetBase:
        return self.data_loader.dataset

    @property
    def worker_info(self):
        return torch.utils.data.get_worker_info()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over the local rank+worker instances.
        """
        global_indices = self.data_loader.get_global_indices()

        num_threads = self.data_loader.num_threads
        if self.worker_info is None and self.data_loader.num_threads is None:
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
                max_instances_per_rank = (
                    self.data_loader.rank_batch_size // self.dataset.sequence_length
                )
            elif isinstance(self.dataset, NumpyVSLDataset):
                max_instances_per_rank = (
                    self.data_loader.rank_batch_size // self.dataset.min_sequence_length
                )
            else:
                raise NotImplementedError

            queue_size = math.ceil(max_instances_per_rank * 2 / num_threads)

            thread_generators = []
            for i in range(num_threads):
                # NOTE: `_get_local_instance_indices` might return an iterator, so we have to
                # create a unique one for each thread otherwise it would be exhausted prematurely
                # and give the wrong order.
                indices = self.data_loader._get_local_instance_indices(global_indices)
                generator = (
                    self.data_loader._get_dataset_item(int(idx))
                    for idx in islice(indices, i, None, num_threads)
                )
                thread_generators.append(
                    threaded_generator(
                        generator, maxsize=queue_size, thread_name=f"data thread {i}"
                    )
                )

            instance_iterator = roundrobin(*thread_generators)
        else:
            indices = self.data_loader._get_local_instance_indices(global_indices)
            instance_iterator = (self.data_loader._get_dataset_item(int(idx)) for idx in indices)

        return (
            self.data_loader.collator(batch)
            for batch in iter_batched(instance_iterator, self.data_loader.rank_batch_size)
        )


@dataclass
class NumpyDataLoaderConfig(Config):
    """
    A configuration class for building :class:`NumpyDataLoaderBase` data loaders.
    """

    global_batch_size: int
    seed: int
    work_dir: Optional[str] = None
    num_threads: Optional[int] = None
    num_workers: int = 0
    prefetch_factor: Optional[int] = None
    target_device_type: Optional[str] = None

    def build(
        self,
        dataset: NumpyDatasetBase,
        *,
        collator: Optional[DataCollator] = None,
        dp_process_group: Optional[dist.ProcessGroup] = None,
    ) -> NumpyDataLoaderBase:
        """
        Construct the :class:`NumpyDataLoaderBase`.
        """
        if self.work_dir is not None and not dataset.work_dir_set:
            dataset.work_dir = Path(self.work_dir)

        dataset.prepare()

        data_loader = NumpyDataLoaderBase.wrap_numpy_dataset(
            dataset,
            global_batch_size=self.global_batch_size,
            collator=collator or DataCollator(pad_token_id=dataset.pad_token_id),
            work_dir=self.work_dir or dataset.work_dir,
            dp_world_size=get_world_size(dp_process_group),
            dp_rank=get_rank(dp_process_group),
            fs_local_rank=get_fs_local_rank(),
            seed=self.seed,
            num_threads=self.num_threads,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            target_device_type=self.target_device_type or get_default_device().type,
        )
        return data_loader
