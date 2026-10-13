"""
Distributed, deterministic, stateful data loaders used by the :class:`~olmo_core.train.Trainer`.

"""
import functools
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    Optional,
    Tuple,
    TypeVar,
)

import bettermap
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
from torch.distributed import DeviceMesh

from ..aliases import PathOrStr
from ..config import Config, Registrable
from ..distributed.parallel import get_dp_process_group
from ..distributed.utils import barrier, get_fs_local_rank, get_rank, get_world_size
from ..exceptions import OLMoConfigurationError
from ..io import is_url, normalize_path
from ..utils import get_default_device, roundrobin, threaded_generator
from .collator import DataCollator
from .numpy_dataset import (
    NumpyDatasetBase,
    NumpyFSLDatasetBase,
    NumpyFSLDynamicRepackedDataset,
    NumpyVSLDataset,
)
from .utils import get_rng, iter_batched, load_array_slice, memmap_to_write

__all__ = [
    "DataLoaderBase",
    "TextDataLoaderBase",
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
        For example::

            # Prepare for the epoch.
            data_loader.reshuffle(epoch=1)

            for batch in data_loader:
                # process batch
                pass

            # Reset internal bookkeeping.
            data_loader.reset()

    :param work_dir: The working directory. Should be a local directory shared among local ranks.
    :param global_batch_size: The global batch size. The units for this depend on the data loader
        implementation.
    :param dp_world_size: The data parallel world size.
    :param dp_rank: The local data parallel rank.
    :param fs_local_rank: The filesystem-local rank relative to the working directory.
    """

    def __init__(
        self,
        *,
        work_dir: PathOrStr,
        global_batch_size: int,
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: Optional[int] = None,
    ):
        if is_url(work_dir):
            raise OLMoConfigurationError(
                f"'work_dir' should be a local path, not a URL ('{work_dir}')."
            )
        self.work_dir = Path(normalize_path(work_dir))
        self._global_batch_size = global_batch_size
        assert dp_rank < dp_world_size
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.fs_local_rank = fs_local_rank if fs_local_rank is not None else get_fs_local_rank()

        self.batches_processed = 0
        """
        The total number of batches processed so far in the current epoch.
        """

        self._epoch: Optional[int] = None

    @property
    def global_batch_size(self) -> int:
        return self._global_batch_size

    @global_batch_size.setter
    def global_batch_size(self, new_global_batch_size: int):
        if not isinstance(new_global_batch_size, int):
            raise TypeError(f"expected int, got {type(new_global_batch_size)}")

        if new_global_batch_size > self.global_batch_size:
            if new_global_batch_size % self.global_batch_size != 0:
                raise RuntimeError("global batch size can only be increased by an integer factor")

            ratio = new_global_batch_size // self.global_batch_size
            if self.batches_processed % ratio != 0:
                raise RuntimeError(
                    f"increasing global batch size by a factor of {ratio} can only be done after processing "
                    f"N x {ratio} batches to avoid duplicating or losing data. Currently {self.batches_processed:d} "
                    f"batches have been processed, which is not divisible by {ratio}."
                )

            self.batches_processed //= ratio
        elif new_global_batch_size < self.global_batch_size:
            if self.global_batch_size % new_global_batch_size != 0:
                raise RuntimeError("global batch size can only be decreased by an integer factor")

            ratio = self.global_batch_size // new_global_batch_size
            self.batches_processed *= ratio

        self._global_batch_size = new_global_batch_size

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
        The total number of batches that the dataset will produce over the course of the current epoch, if known.
        Otherwise this should return ``None``.
        """
        raise NotImplementedError

    def batches_in_epoch(self, epoch: int) -> Optional[int]:
        """
        By default this is the same as :meth:`total_batches`, though some data loaders might generate
        a different number of batches per epoch.
        """
        del epoch
        return self.total_batches

    def __len__(self) -> int:
        """
        Returns the total number of batches in the current epoch (same as :data:`total_batches`) if known,
        otherwise a :class:`TypeError` is raised.
        """
        if self.total_batches is not None:
            return self.total_batches
        else:
            raise TypeError(
                f"total length (number of batches) is unknown for {self.__class__.__name__}"
            )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over the local rank batches.
        """
        for batch in self._iter_batches():
            self.batches_processed += 1
            yield batch

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a single batch if possible, otherwise :class:`TypeError` is raised.
        """
        del index
        raise TypeError(f"__getitem__ is not implemented for {self.__class__.__name__}")

    @property
    def rank_batch_size(self) -> int:
        """
        The batch size per rank.
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
            of the batch, which should have size exactly :data:`rank_batch_size`.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset epoch bookkeeping. Should be called at the end of an epoch.
        """
        self.batches_processed = 0

    @abstractmethod
    def get_mock_batch(self) -> Dict[str, Any]:
        """
        Return a batch with arbitrary data. This can just be random data as it's only used by the
        trainer to do a dry-run of the forward and backward pass before training officially starts.
        """
        raise NotImplementedError

    def global_num_tokens_in_batch(self, batch: Dict[str, Any]) -> Optional[int]:
        """
        For text-based data loaders this should return the total (global) number of tokens
        in the batch. This is used by the trainer for bookkeeping.
        """
        del batch
        return None


class TextDataLoaderBase(DataLoaderBase):
    """
    An abstract base class for text-based data loaders.

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
        fs_local_rank: Optional[int] = None,
    ):
        super().__init__(
            work_dir=work_dir,
            global_batch_size=global_batch_size,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            fs_local_rank=fs_local_rank,
        )
        self.collator = collator
        """
        The data collator.
        """
        self.tokens_processed: int = 0
        """
        The total number of tokens processed so far in the current epoch.
        """

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for batch in super().__iter__():
            if batch["input_ids"].numel() != self.rank_batch_size:
                raise RuntimeError(
                    f"Expected batch size of {self.rank_batch_size:,d} tokens on rank {self.dp_rank}, "
                    f"got input IDs with shape {tuple(batch['input_ids'].shape)} = {batch['input_ids'].numel():,d} tokens"
                )
            self.tokens_processed += self.global_batch_size
            yield batch

    def reset(self):
        super().reset()
        self.tokens_processed = 0

    def global_num_tokens_in_batch(self, batch: Dict[str, Any]) -> Optional[int]:
        del batch
        return self.global_batch_size


class NumpyDataLoaderBase(TextDataLoaderBase):
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
    :param reshuffle_each_epoch: When ``shuffle`` is enabled, whether to generate a new
        deterministic permutation every epoch. If ``False``, the epoch-1 permutation is reused
        for every epoch, preserving both sequence order and batch membership.
    :param batch_shuffling: Preserve the ordinary epoch-1 global batches and shuffle only their
        order in later epochs. Disabled by default.
    :param restore_data_order_from_state: Restore ``shuffle``, ``reshuffle_each_epoch``, and
        ``batch_shuffling`` from a checkpoint's data loader state. Disable only when
        intentionally branching from a checkpoint into a different data-order experiment.
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
        reshuffle_each_epoch: bool = True,
        batch_shuffling: bool = False,
        restore_data_order_from_state: bool = True,
        num_threads: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
        target_device_type: str = "cpu",
        dp_world_size: int = 1,
        dp_rank: int = 0,
        fs_local_rank: Optional[int] = None,
        ignore_fingerprint_mismatch: bool = False,
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
        self.reshuffle_each_epoch = reshuffle_each_epoch
        self.batch_shuffling = batch_shuffling
        self._validate_data_order_settings()
        self.restore_data_order_from_state = restore_data_order_from_state
        self.num_threads = num_threads
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.target_device_type = target_device_type
        self._global_indices: Optional[np.ndarray] = None
        self.ignore_fingerprint_mismatch = ignore_fingerprint_mismatch

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
        fs_local_rank: Optional[int] = None,
        num_threads: Optional[int] = None,
        num_workers: int = 0,
        prefetch_factor: Optional[int] = None,
        target_device_type: str = "cpu",
        shuffle: bool = True,
        reshuffle_each_epoch: bool = True,
        batch_shuffling: bool = False,
        restore_data_order_from_state: bool = True,
        ignore_fingerprint_mismatch: bool = False,
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
            shuffle=shuffle,
            reshuffle_each_epoch=reshuffle_each_epoch,
            batch_shuffling=batch_shuffling,
            restore_data_order_from_state=restore_data_order_from_state,
            ignore_fingerprint_mismatch=ignore_fingerprint_mismatch,
        )
        data_loader: DataLoaderBase
        if isinstance(dataset, NumpyFSLDatasetBase):
            data_loader = NumpyFSLDataLoader(dataset, **kwargs)  # type: ignore
            if dataset.max_target_sequence_length is not None:
                data_loader.chunk_size = (
                    dataset.max_target_sequence_length // dataset.sequence_length
                )
        elif isinstance(dataset, NumpyVSLDataset):
            data_loader = NumpyVSLDataLoader(dataset, **kwargs)  # type: ignore
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
            "shuffle": self.shuffle,
            "reshuffle_each_epoch": self.reshuffle_each_epoch,
            "batch_shuffling": self.batch_shuffling,
            "epoch": self._epoch,
        }

    def _validate_data_order_settings(self):
        if self.batch_shuffling and not self.shuffle:
            raise OLMoConfigurationError("'batch_shuffling' requires 'shuffle=True'")
        if self.batch_shuffling and not self.reshuffle_each_epoch:
            raise OLMoConfigurationError("'batch_shuffling' requires 'reshuffle_each_epoch=True'")
        if self.batch_shuffling and isinstance(self.dataset, NumpyFSLDynamicRepackedDataset):
            raise OLMoConfigurationError(
                "'batch_shuffling' cannot be combined with dynamic document repacking"
            )
        if self.batch_shuffling and not isinstance(self.dataset, NumpyFSLDatasetBase):
            raise OLMoConfigurationError(
                "'batch_shuffling' currently supports only fixed-sequence-length datasets"
            )

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if state_dict["dataset_fingerprint_version"] != self.dataset.fingerprint_version:
            log.warning(
                "Dataset fingerprint version does not match the version in the checkpoint, "
                "this could mean the data has changed"
            )
        elif state_dict["dataset_fingerprint"] != self.dataset.fingerprint:
            if not self.ignore_fingerprint_mismatch:
                raise RuntimeError(
                    "Dataset fingerprint does not match the fingerprint in the checkpoint, "
                    "set ignore_fingerprint_mismatch=True to ignore this error. "
                    "This will probably result in a different data order!"
                )
            log.warning(
                "Fingerprint mismatch ignored since ignore_fingerprint_mismatch=True. "
                "This will probably result in a different data order!"
            )

        if state_dict["seed"] != self.seed:
            log.warning(
                "Restoring data loading state with a different data seed, "
                "will use data seed from state dict for data order consistency."
            )
            self.seed = state_dict["seed"]

        # These keys are absent from checkpoints created before the data-order controls were
        # added. Such checkpoints used the historical defaults: shuffle enabled with a fresh
        # deterministic permutation every epoch.
        saved_shuffle = state_dict.get("shuffle", True)
        saved_reshuffle_each_epoch = state_dict.get("reshuffle_each_epoch", True)
        saved_batch_shuffling = state_dict.get("batch_shuffling", False)
        if self.restore_data_order_from_state:
            if saved_shuffle != self.shuffle:
                log.warning(
                    "Restoring data loading state with a different 'shuffle' setting; "
                    "will use the setting from the state dict for data order consistency."
                )
                self.shuffle = saved_shuffle
            if saved_reshuffle_each_epoch != self.reshuffle_each_epoch:
                log.warning(
                    "Restoring data loading state with a different 'reshuffle_each_epoch' setting; "
                    "will use the setting from the state dict for data order consistency."
                )
                self.reshuffle_each_epoch = saved_reshuffle_each_epoch
            if saved_batch_shuffling != self.batch_shuffling:
                log.warning(
                    "Restoring data loading state with a different 'batch_shuffling' setting; "
                    "will use the setting from the state dict for data order consistency."
                )
                self.batch_shuffling = saved_batch_shuffling
        elif (
            saved_shuffle != self.shuffle
            or saved_reshuffle_each_epoch != self.reshuffle_each_epoch
            or saved_batch_shuffling != self.batch_shuffling
        ):
            log.warning(
                "Ignoring checkpoint data-order settings because "
                "restore_data_order_from_state=False; keeping the configured data-order mode."
            )
        self._validate_data_order_settings()

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
        if isinstance(self.dataset, NumpyFSLDynamicRepackedDataset):
            self.dataset.repack_for_epoch(epoch, seed=self.seed)
        self.build_and_save_global_indices(in_memory=in_memory)

    def get_mock_batch(self) -> Dict[str, Any]:
        device = torch.device("cpu")
        rng = torch.Generator(device=device)
        rng.manual_seed(self.seed + self.dp_rank)
        num_instances = self.rank_batch_size // self.dataset.max_sequence_length
        input_ids = torch.randint(
            0,
            self.dataset.vocab_size,
            (num_instances, self.dataset.max_sequence_length),
            generator=rng,
            device=device,
        )
        out: Dict[str, Any] = {"input_ids": input_ids}
        if isinstance(self.dataset, NumpyFSLDatasetBase) and self.dataset._generate_doc_lengths:
            splits = torch.randint(
                2,
                self.dataset.max_sequence_length - 2,
                (num_instances,),
                generator=rng,
                device=device,
            )
            out["doc_lens"] = torch.stack(
                [splits, self.dataset.max_sequence_length - splits], dim=1
            )
            out["max_doc_lens"] = torch.max(out["doc_lens"], dim=-1).values.tolist()
        return out

    def _iter_batches(self) -> Iterable[Dict[str, Any]]:
        # If we're already at the end of epoch we can skip creating the iterator.
        if self.total_batches is not None and self.batches_processed >= self.total_batches:
            yield from ()

        def _build_batch_iterator():
            return iter(
                torch.utils.data.DataLoader(
                    _IterableDatasetWrapper(self),
                    batch_size=None,
                    num_workers=self.num_workers,
                    pin_memory=self.target_device_type == "cuda" and self.num_workers > 0,
                    prefetch_factor=self.prefetch_factor,
                    persistent_workers=False,
                    timeout=0,
                ),
            )

        current_global_batch_size = self.global_batch_size
        batch_iterator = _build_batch_iterator()
        while (batch := next(batch_iterator, None)) is not None:
            yield batch

            # If batch size has changed, re-initialize the workers.
            # NOTE: base class handles the logic of adjusting `self.batches_processed` when
            # `self.global_batch_size` is changed through the property setter.
            if current_global_batch_size != self.global_batch_size:
                if self.num_workers > 0:
                    log.info("Batch size has changed, reinitializing data loading workers...")
                current_global_batch_size = self.global_batch_size
                batch_iterator = _build_batch_iterator()

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

    @property
    def data_order_epoch(self) -> Optional[int]:
        """
        The epoch used to construct the deterministic data order.

        Fixed-order training deliberately reuses the ordinary epoch-1 permutation. This makes
        the first epoch identical between fixed-order and per-epoch-reshuffling experiments, so
        the conditions differ only once the dataset is repeated.
        """
        if not self.shuffle:
            return None
        return self.epoch if self.reshuffle_each_epoch else 1

    @property
    def data_order_seed(self) -> int:
        """The deterministic RNG seed for the current data order."""
        data_order_epoch = self.data_order_epoch
        return self.seed + (data_order_epoch if data_order_epoch is not None else 0)


class NumpyFSLDataLoader(NumpyDataLoaderBase):
    """
    A fixed sequence length :class:`DataLoaderBase` for use with a
    :class:`~olmo_core.data.numpy_dataset.NumpyFSLDataset`.
    """

    _BATCH_SHUFFLING_SEED_DOMAIN = 0x42415443  # "BATC"

    def __init__(
        self,
        dataset: NumpyFSLDatasetBase,
        *,
        chunk_size: int = 1,
        **kwargs,
    ):
        assert chunk_size >= 1
        super().__init__(dataset, **kwargs)
        self.chunk_size = chunk_size
        assert isinstance(self.dataset, NumpyFSLDatasetBase)
        if self.rank_batch_size % self.dataset.sequence_length != 0:
            raise OLMoConfigurationError(
                f"rank batch size (in tokens) {self.rank_batch_size} must be divisible by sequence length {self.dataset.sequence_length}"
            )

    @property
    def total_size(self) -> int:
        """
        The total number of instances that the dataset will produce over the course of an epoch.
        """
        assert isinstance(self.dataset, NumpyFSLDatasetBase)
        instances_per_batch = self.global_batch_size // self.dataset.sequence_length
        return instances_per_batch * (len(self.dataset) // instances_per_batch)

    @property
    def total_batches(self) -> int:
        assert isinstance(self.dataset, NumpyFSLDatasetBase)
        return self.total_size // (self.global_batch_size // self.dataset.sequence_length)

    @property
    def _global_indices_file(self) -> Path:
        global_indices_fname = self._format_fname_from_fields(
            "global_indices",
            seed=self.seed if self.shuffle else None,
            epoch=self.data_order_epoch,
            dataset_size=len(self.dataset),
            chunk=self.chunk_size if self.chunk_size > 1 else None,
            batch_shuffling=self.batch_shuffling or None,
            global_batch_size=self.global_batch_size if self.batch_shuffling else None,
            v=2 if self.batch_shuffling else 1,  # tick if logic changes
        )
        return self.work_dir / f"{global_indices_fname}.npy"

    def _build_global_indices(self) -> np.ndarray:
        assert len(self.dataset) < np.iinfo(np.uint32).max

        if self.batch_shuffling:
            # Fix the ordinary epoch-1 sequence permutation and its exact global-batch
            # membership. Later epochs permute only those complete global-batch rows.
            indices = self._build_instance_indices(get_rng(self.seed + 1))
            if self.epoch > 1:
                assert isinstance(self.dataset, NumpyFSLDatasetBase)
                instances_per_batch = self.global_batch_size // self.dataset.sequence_length
                batches = indices[: self.total_size].reshape(-1, instances_per_batch).copy()
                rng = get_rng(self.seed + self.epoch + self._BATCH_SHUFFLING_SEED_DOMAIN)
                rng.shuffle(batches)
                indices[: self.total_size] = batches.reshape(-1)
            return indices

        rng = get_rng(self.data_order_seed) if self.shuffle else None
        return self._build_instance_indices(rng)

    def _build_instance_indices(self, rng: Optional[np.random.Generator]) -> np.ndarray:
        """Build instance indices using the historical sequence/chunk-shuffle behavior."""
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

        return indices

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # NOTE: Make sure the logic here matches that in '_get_local_instance_indices()'

        # NOTE: 'indices' are global instance indices.
        indices = self.get_global_indices()[: self.total_size]

        # Slice up by batch.
        assert isinstance(self.dataset, NumpyFSLDatasetBase)
        instances_per_batch = self.global_batch_size // self.dataset.sequence_length
        # shape: (global num batches, global num instances per batch)
        indices = indices.reshape(-1, instances_per_batch)

        # Slice batches into micro batches for the local DP rank.
        if self.dp_world_size > 1:
            indices = indices[:, self.dp_rank :: self.dp_world_size]

        # Get instances for the batch.
        map_fn: Callable
        if self.worker_info is None and self.num_threads is not None and self.num_threads > 1:
            map_fn = functools.partial(
                bettermap.ordered_map_per_thread, parallelism=self.num_threads
            )
        else:
            map_fn = map
        instances = list(map_fn(lambda idx: self._get_dataset_item(int(idx)), indices[index]))

        return self.collator(instances)

    def _get_local_instance_indices(self, indices: np.ndarray) -> Iterable[int]:
        # NOTE: 'indices' are global instance indices.
        # Make sure the logic here matches that in '__getitem__()'

        # Remove tail of data to make it evenly divisible.
        indices = indices[: self.total_size]

        # Slice up by batch.
        assert isinstance(self.dataset, NumpyFSLDatasetBase)
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
        assert isinstance(self.dataset, NumpyFSLDatasetBase)
        state_dict["dataset_type"] = "fsl"
        state_dict["sequence_length"] = self.dataset.sequence_length
        state_dict["max_target_sequence_length"] = self.dataset.max_target_sequence_length
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        super().load_state_dict(state_dict)
        # Account for change in batch size / sequence length.
        self.batches_processed = self.tokens_processed // self.global_batch_size
        log.info(
            f"Data loader will resume from batch {self.batches_processed:,d}/{self.total_batches:,d} "
            f"based on batch size of {self.global_batch_size:,d} tokens"
        )

        assert isinstance(self.dataset, NumpyFSLDatasetBase)
        if state_dict["dataset_type"] != "fsl":
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
            epoch=self.data_order_epoch,
            bz=self.global_batch_size,
        )
        return (
            self.work_dir
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
            epoch=self.data_order_epoch,
        )
        return (
            self.work_dir
            / f"dataset-{self.dataset.fingerprint}"
            / self.dataset.curriculum.short_str
            / f"{bucket_indices_fname}.npy"
        )

    def _get_bucket_indices(self, seq_len: int) -> np.ndarray:
        assert isinstance(self.dataset, NumpyVSLDataset)
        bucket_indices = self.dataset.get_instance_bucket(seq_len)
        if self.shuffle:
            rng = get_rng(self.data_order_seed + seq_len)
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
                self.batches_per_bucket, self.data_order_seed
            )
        else:
            return np.arange(self.total_batches, dtype=np.uint32)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # NOTE: Make sure the logic here matches that in '_get_local_instance_indices()'

        # NOTE: 'indices' are global *batch* indices.
        indices = self.get_global_indices()

        # Map to instance indices.
        instance_indices = self._batch_index_to_local_instance_indices(indices[index])

        # Get instances for the batch.
        instances = [self._get_dataset_item(int(idx)) for idx in instance_indices]

        return self.collator(instances)

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
        state_dict["dataset_type"] = "vsl"
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
        if state_dict["dataset_type"] != "vsl":
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
            if isinstance(self.dataset, NumpyFSLDatasetBase):
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


L = TypeVar("L", bound="DataLoaderBase")


@dataclass
class DataLoaderConfig(Config, Registrable, Generic[L]):
    """
    Registrable base class for data loader configs.
    """

    @abstractmethod
    def build(self, *args, **kwargs) -> L:
        raise NotImplementedError


@DataLoaderConfig.register("numpy")
@dataclass
class NumpyDataLoaderConfig(DataLoaderConfig[NumpyDataLoaderBase]):
    """
    A configuration class for building :class:`NumpyDataLoaderBase` data loaders.
    """

    global_batch_size: int
    seed: int
    shuffle: bool = True
    """Whether to shuffle sequence instances before forming batches."""
    reshuffle_each_epoch: bool = True
    """
    When shuffling, generate a new deterministic sequence permutation each epoch. Set to
    ``False`` to reuse the epoch-1 permutation and exact batch grouping across all epochs.
    """
    batch_shuffling: bool = False
    """
    Preserve the ordinary epoch-1 fixed-length sequence permutation and exact global-batch
    membership, while generating a new deterministic permutation of intact global batches each
    epoch. This is opt-in and mutually exclusive with disabling shuffling, fixed data order,
    dynamic document repacking, and variable-sequence-length datasets.
    """
    restore_data_order_from_state: bool = True
    """
    Restore ``shuffle``, ``reshuffle_each_epoch``, and ``batch_shuffling`` from a trainer
    checkpoint. Disable only when intentionally branching from a checkpoint into a different
    data-order experiment.
    """
    work_dir: Optional[str] = None
    num_threads: Optional[int] = None
    num_workers: int = 0
    prefetch_factor: Optional[int] = None
    target_device_type: Optional[str] = None
    ignore_fingerprint_mismatch: bool = False

    def build(
        self,
        dataset: NumpyDatasetBase,
        *,
        collator: Optional[DataCollator] = None,
        mesh: Optional[DeviceMesh] = None,
        dp_process_group: Optional[dist.ProcessGroup] = None,
    ) -> NumpyDataLoaderBase:
        """
        Construct the :class:`NumpyDataLoaderBase`.

        :param dataset: The dataset.
        :param mesh: An optional ``DeviceMesh`` that defines the data parallel dimensions. Ideally
            you should create this mesh using :func:`~olmo_core.distributed.parallel.build_world_mesh()`.
            Alternatively you can pass the ``dp_process_group`` instead.
        :param dp_process_group: The data parallel process group.
        """
        if self.work_dir is not None:
            if is_url(self.work_dir):
                raise OLMoConfigurationError(
                    f"'work_dir' should be a local path, not a URL ('{self.work_dir}')."
                )
            if not dataset.work_dir_set:
                dataset.work_dir = Path(normalize_path(self.work_dir))

        if dp_process_group is None and mesh is not None:
            dp_process_group = get_dp_process_group(mesh)

        dataset.prepare()

        data_loader = NumpyDataLoaderBase.wrap_numpy_dataset(
            dataset,
            global_batch_size=self.global_batch_size,
            collator=collator
            or DataCollator(pad_token_id=dataset.pad_token_id, vocab_size=dataset.vocab_size),
            work_dir=self.work_dir or dataset.work_dir,
            dp_world_size=get_world_size(dp_process_group),
            dp_rank=get_rank(dp_process_group),
            fs_local_rank=get_fs_local_rank(),
            seed=self.seed,
            shuffle=self.shuffle,
            reshuffle_each_epoch=self.reshuffle_each_epoch,
            batch_shuffling=self.batch_shuffling,
            restore_data_order_from_state=self.restore_data_order_from_state,
            num_threads=self.num_threads,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            target_device_type=self.target_device_type or get_default_device().type,
            ignore_fingerprint_mismatch=self.ignore_fingerprint_mismatch,
        )
        return data_loader
