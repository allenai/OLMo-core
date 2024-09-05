import contextlib
import logging
import math
import signal
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from ..aliases import PathOrStr
from ..config import StrEnum
from ..data import DataCollator, IterableDataset, MemMapDataset
from ..distributed.utils import (
    all_reduce_value,
    backend_supports_cpu,
    barrier,
    get_fs_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    scatter_object,
)
from ..exceptions import OLMoConfigurationError
from ..io import copy_file, file_exists, is_url, join_path, normalize_path
from ..nn.functional.cross_entropy_loss import (
    cross_entropy_loss,
    fused_cross_entropy_loss,
)
from ..utils import move_to_device
from .callbacks import (
    Callback,
    CheckpointerCallback,
    ConsoleLoggerCallback,
    GarbageCollectorCallback,
    SpeedMonitorCallback,
)
from .checkpoint import Checkpointer
from .utils import (
    Duration,
    DurationUnit,
    EnvRngStates,
    ReduceType,
    get_local_tensor,
    move_metrics,
    reduce_metrics,
)

log = logging.getLogger(__name__)

TRAIN_CE_LOSS_METRIC = "train/CE loss"
TRAIN_PPL_METRIC = "train/PPL"
TRAIN_Z_LOSS_METRIC = "train/Z loss"


class LoadStrategy(StrEnum):
    """
    Determines the strategy for loading checkpoints prior to training.
    """

    if_available = "if_available"
    """
    Only load from the load path if a checkpoint exists there.
    """

    always = "always"
    """
    Always try loading from the load path.
    """

    never = "never"
    """
    Never load from the load path.
    """


@dataclass
class Trainer:
    """
    Language model trainer.

    .. tip::
        Use :class:`TrainerConfig` instead of constructing this class directly.
    """

    work_dir: Path
    """
    A local working directory to use for temporary files needed during training.
    Files added to this working directory can be persisted to the :data:`save_folder` via
    :meth:`persist_working_file()`.
    """

    model: nn.Module
    """
    The model to fit.
    """

    optim: Optimizer
    """
    The optimizer to use.
    """

    dataset: MemMapDataset
    """
    The training dataset.
    """

    collator: DataCollator
    """
    The data collator.
    """

    device: torch.device
    """
    The default device to use. Should match the device the model is on and be appropriate for the
    distributed backend, if there is one.
    """

    save_folder: str
    """
    The folder to save all checkpoints to. Could be a local directory (if using a shared filesytem)
    or a URL.
    """

    checkpointer: Checkpointer
    """
    The checkpointer. This is a wrapper around the functionality in
    :mod:`olmo_core.distributed.checkpoint`, which means you can use
    :func:`~olmo_core.distributed.checkpoint.unshard_checkpoint` to unshard the model and optimizer
    state from a trainer checkpoint.
    """

    callbacks: Dict[str, Callback]
    """
    Trainer callbacks.
    """

    max_duration: Duration
    """
    The duration to train for.
    """

    train_sequence_length: int
    """
    Training sequence length.
    """

    global_batch_size: int
    """
    Global training batch size (in terms of instances, not tokens).
    """

    microbatch_size: int
    """
    Microbatch size per rank, i.e. the number of instances to process at a time from each rank.
    """

    save_overwrite: bool = False
    """
    Whether to overwrite existing files/checkpoints in the :data:`save_folder`.
    """

    load_path: Optional[PathOrStr] = None
    """
    Where to load a checkpoint from prior to training.
    Defaults to ``save_folder``.
    """

    load_strategy: LoadStrategy = LoadStrategy.if_available
    """
    The strategy for loading a checkpoint prior to training.
    """

    metrics_collect_interval: int = 5
    """
    How often (in steps) to collect, reduce, and pass on metrics to the
    :meth:`Callback.log_metrics <olmo_core.train.callbacks.Callback.log_metrics>` method on callbacks.

    .. note::
        Regardless of what this is set to, the
        :meth:`Callback.log_metrics <olmo_core.train.callbacks.Callback.log_metrics>` methods are still
        called with the metrics for every single step, but will be delayed according to this value.

        For example, if this is set to 5, then every 5 steps the metrics from the past 5 steps
        are collected and reduced together, then passed on to 
        :meth:`Callback.log_metrics <olmo_core.train.callbacks.Callback.log_metrics>` altogether.

    .. tip::
        Increasing this can improve throughput since logging metrics always requires a host-device
        sync.
    """

    fused_loss: bool = False
    """
    Used the fused cross-entropy loss function.
    """

    z_loss_multiplier: Optional[float] = None
    """
    Use Z-loss with this multiplier.
    """

    autocast_precision: Optional[torch.dtype] = None
    """
    Enable AMP with this data type.
    """

    dp_process_group: Optional[dist.ProcessGroup] = None
    """
    The distributed process group for all data parallel ranks.
    """

    data_seed: int = 0
    """
    The seed to use to shuffle the dataset.
    """

    data_loader_workers: int = 0
    """
    The number of data loading workers to use.
    """

    data_loader_prefetch_factor: Optional[int] = None
    """
    The number of batches to prefetch.
    """

    # Bookkeeping

    global_step: int = 0
    """
    The current step (1-based), though it's initialized to 0 before the first step.
    """

    global_train_tokens_seen: int = 0
    """
    The total number of training tokens seen.
    """

    global_train_tokens_seen_this_epoch: int = 0
    """
    The total number of training tokens seen in the current epoch.
    """

    global_train_examples_seen_this_epoch: int = 0
    """
    The total number of examples seen in the current epoch.
    """

    epoch: int = 1
    """
    The current epoch (1-based).
    """

    cancel_check_interval: int = 25
    """
    The interval (in steps) to check if the run is canceled. Checking requires distributed comms.
    """

    hard_stop: Optional[Duration] = None
    """
    Set a hard stopping point for the trainer. This is useful for ablations when you you don't
    want to do a complete training run, but you don't want to change :data:`max_duration` as to
    not affect the learning rate schedule.
    """

    _metrics: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=OrderedDict)
    _metrics_reduce_type: Dict[str, Optional[ReduceType]] = field(default_factory=dict)
    _canceled: bool = False
    _cancel_reason: Optional[str] = None
    _canceling_rank: Optional[int] = None
    _error: Optional[BaseException] = None
    _rank_batch_size: Optional[int] = None
    _thread_pool: Optional[ThreadPoolExecutor] = None
    _bookkeeping_pg: Optional[dist.ProcessGroup] = None
    _checkpoint_loaded: bool = False

    def __post_init__(self):
        self.save_folder = normalize_path(self.save_folder)
        if self.load_path is not None:
            self.load_path = normalize_path(self.load_path)

        # If save folder is a local directory, make sure we're using a shared filesystem.
        if not is_url(self.save_folder) and get_fs_local_rank() != get_rank():
            raise OLMoConfigurationError(
                "Checkpointing to a local directory requires a shared filesystem. "
                "If you do have a shared filesystem please set the env var 'OLMO_SHARED_FS=1' "
                "or set 'FS_LOCAL_RANK' to the global rank for each process."
            )

        # Configure working directory.
        self.work_dir = Path(self.work_dir)

        # Ensure save folder and working directory exist.
        if get_fs_local_rank() == 0:
            self.work_dir.mkdir(exist_ok=True, parents=True)
            if not is_url(self.save_folder):
                Path(self.save_folder).mkdir(exist_ok=True, parents=True)

        # Ensure we have necessary callbacks.
        has_console_logger_callback = False
        has_checkpointer_callback = False
        has_speed_monitor_callback = False
        has_gc_collector_callback = False
        for callback in self.callbacks.values():
            if isinstance(callback, ConsoleLoggerCallback):
                has_console_logger_callback = True
            elif isinstance(callback, CheckpointerCallback):
                has_checkpointer_callback = True
            elif isinstance(callback, SpeedMonitorCallback):
                has_speed_monitor_callback = True
            elif isinstance(callback, GarbageCollectorCallback):
                has_gc_collector_callback = True
        if not has_console_logger_callback:
            self.callbacks.setdefault(
                "console_logger",
                ConsoleLoggerCallback(
                    log_interval=1, metrics_log_interval=self.metrics_collect_interval
                ),
            )
        if not has_checkpointer_callback:
            self.callbacks.setdefault("checkpointer", CheckpointerCallback())
        if not has_speed_monitor_callback:
            self.callbacks.setdefault("speed_monitor", SpeedMonitorCallback())
        if is_distributed() and not has_gc_collector_callback:
            self.callbacks.setdefault("garbage_collector", GarbageCollectorCallback())

        # Set pointer to self in all callbacks.
        for callback in self.callbacks.values():
            callback.trainer = self

        # Sort callbacks by (priority, name).
        # We do this for 2 reasons: (1) to respect the priority, and (2) to ensure the callback
        # order is consistent across the process group since some callbacks make distributed
        # synchronization/communication calls.
        self.callbacks = OrderedDict(
            (
                (k, cb)
                for k, cb in sorted(
                    self.callbacks.items(), key=lambda x: (x[1].priority, x[0]), reverse=True
                )
            )
        )

        # Maybe create separate process group for bookkeeping.
        if self._bookkeeping_pg is None and is_distributed():
            if backend_supports_cpu():
                log.info("Creating new process group for bookkeeping")
                self._bookkeeping_pg = dist.new_group()
            else:
                log.warning(
                    "No CPU backend configured, bookkeeping collectives will occur on the default "
                    "backend and will be blocking. This may result in slower training throughput."
                )

        # Make sure global batch size is divisible by microbatch size times world size
        if (
            self.global_batch_size
            % (self.microbatch_size * (ws := get_world_size(self.dp_process_group)))
            != 0
        ):
            raise OLMoConfigurationError(
                f"global batch size ({self.global_batch_size}) must be divisible by "
                f"micro-batch size ({self.microbatch_size}) x DP world size ({ws})"
            )

        # Other validation.
        if isinstance(self.dataset, MemMapDataset):
            if self.dataset.sequence_length != self.train_sequence_length:
                raise OLMoConfigurationError("trainer and dataset sequence length does not match")

    @property
    def rank_batch_size(self) -> int:
        if self._rank_batch_size is None:
            assert self.global_batch_size % get_world_size(self.dp_process_group) == 0
            self._rank_batch_size = self.global_batch_size // get_world_size(self.dp_process_group)
        return self._rank_batch_size

    @property
    def training_complete(self) -> bool:
        if self._error is not None:
            raise RuntimeError("An error occurred") from self._error

        if (
            not self._canceled
            and self.global_step > 0
            and self.global_step % self.cancel_check_interval == 0
        ):
            # NOTE: any collective operations done in a separate thread should use the bookkeeping
            # process group to avoid race conditions.
            self.thread_pool.submit(self._check_if_canceled)

        if self._canceled:
            return True
        elif self._duration_due(self.max_duration):
            return True
        elif self.hard_stop is not None and self._duration_due(self.hard_stop):
            return True
        else:
            return False

    @property
    def tokens_per_batch(self) -> int:
        """
        The number of tokens in each training batch.
        """
        return self.global_batch_size * self.train_sequence_length

    @property
    def steps_per_epoch(self) -> int:
        """
        The total number of training steps in an epoch.
        """
        return self.dataset_total_size // self.global_batch_size

    @property
    def tokens_per_epoch(self) -> int:
        """
        The total number of tokens in the training dataset, minus left-overs.
        """
        return self.dataset_total_size * self.train_sequence_length

    @property
    def dataset_total_size(self) -> int:
        """
        The total number of complete examples in the training dataset.
        """
        dp_world_size = get_world_size(self.dp_process_group)
        if len(self.dataset) % dp_world_size == 0:
            return len(self.dataset)
        else:
            num_samples = math.ceil((len(self.dataset) - dp_world_size) / dp_world_size)
            return num_samples * dp_world_size

    @property
    def max_steps(self) -> int:
        """
        The maximum number of steps to train for, as determined by :data:`max_duration`.
        """
        if self.max_duration.unit == DurationUnit.steps:
            return self.max_duration.value
        elif self.max_duration.unit == DurationUnit.epochs:
            # Need to account for a change in batch size.
            max_epochs = self.max_duration.value
            complete_epochs_remaining = max(max_epochs - self.epoch, 0)
            steps_remaining = complete_epochs_remaining * self.steps_per_epoch
            tokens_remaining_this_epoch = max(
                self.tokens_per_epoch - self.global_train_tokens_seen_this_epoch, 0
            )
            steps_remaining += math.ceil(tokens_remaining_this_epoch / self.tokens_per_batch)
            return self.global_step + steps_remaining
        elif self.max_duration.unit == DurationUnit.tokens:
            # Need to account for a change in batch size.
            max_tokens = self.max_duration.value
            tokens_remaining = max(max_tokens - self.global_train_tokens_seen, 0)
            steps_remaining = math.ceil(tokens_remaining / self.tokens_per_batch)
            return self.global_step + steps_remaining
        else:
            raise NotImplementedError

    @property
    def bookkeeping_device(self) -> torch.device:
        """
        The device used for collective bookkeeping (non-training) operations that can potentially.
        use a different backend.
        """
        if backend_supports_cpu():
            return torch.device("cpu")
        else:
            return self.device

    @property
    def bookkeeping_pg(self) -> Optional[dist.ProcessGroup]:
        """
        The process group used for bookkeeping collectives.

        Since bookkeeping collectives might be done in a separate thread, we need a separate process
        group to avoid potential race conditions.
        """
        return self._bookkeeping_pg

    @property
    def thread_pool(self) -> ThreadPoolExecutor:
        """
        A thread that can be used by callbacks to run bookkeeping tasks without blocking training.
        """
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="trainer")
        return self._thread_pool

    @property
    def checkpoint_loaded(self) -> bool:
        """
        If a checkpoint has been loaded.
        """
        return self._checkpoint_loaded

    def cancel_run(self, reason: str):
        """
        Mark the run canceled.

        :param reason: The reason for canceling.
        """
        #  self._canceled = True  # NOTE: important not to set this!! Leads to distributed hang.
        self._canceling_rank = get_rank()
        self._cancel_reason = reason

    def fit(self):
        """
        Fit the model, potentially loading a checkpoint before hand depending on the
        :data:`load_strategy`.
        """
        self._canceled = False
        self._cancel_reason = None
        self._canceling_rank = None

        # Maybe load a checkpoint.
        if not self.checkpoint_loaded:
            load_path = self.load_path if self.load_path is not None else self.save_folder
            if self.load_strategy == LoadStrategy.always:
                self.load_checkpoint(load_path)
            elif self.load_strategy == LoadStrategy.if_available:
                self.maybe_load_checkpoint(load_path)

        log.info(f"Training for {self.max_steps:,d} steps")

        self.model.train()

        for callback in self.callbacks.values():
            callback.pre_train()

        barrier()

        # Install SIGTERM handler.
        og_handler = signal.signal(signal.SIGTERM, self._handle_sigterm)

        try:
            while not self.training_complete:
                self._fit_epoch()
        except BaseException as exc:
            for callback in self.callbacks.values():
                callback.on_error(exc)
            raise
        finally:
            # Restore original SIGTERM handler.
            signal.signal(signal.SIGTERM, og_handler)

        for callback in self.callbacks.values():
            callback.post_train()

        log.info("Training complete")

    def state_dict(self) -> Dict[str, Any]:
        """
        Get the trainer state to save.
        """
        return dict(
            global_step=self.global_step,
            global_train_tokens_seen=self.global_train_tokens_seen,
            global_train_tokens_seen_this_epoch=self.global_train_tokens_seen_this_epoch,
            global_train_examples_seen_this_epoch=self.global_train_examples_seen_this_epoch,
            dataset_fingerprint=self.dataset.fingerprint,
            data_seed=self.data_seed,
            epoch=self.epoch,
            world_size=get_world_size(),  # global world size here on purpose
            train_sequence_length=self.train_sequence_length,
            rng=EnvRngStates.current_state().as_dict(),
        )

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load trainer state (not model or optimizer state).
        """
        if state_dict["train_sequence_length"] != self.train_sequence_length:
            raise NotImplementedError(
                "Restoring trainer state with a different sequence length is not supported"
            )

        if (
            state_dict.get("dataset_fingerprint", self.dataset.fingerprint)
            != self.dataset.fingerprint
        ):
            raise RuntimeError(
                "Restoring trainer state with a different dataset is not supported! (fingerprint doesn't match)"
            )

        if (data_seed := state_dict.get("data_seed", self.data_seed)) != self.data_seed:
            log.warning(
                "Restoring from trainer state with a different data seed, "
                "will use data seed from state dict for data order consistency."
            )
            self.data_seed = data_seed

        self.train_sequence_length = state_dict["train_sequence_length"]
        self.global_step = state_dict["global_step"]
        self.global_train_tokens_seen = state_dict["global_train_tokens_seen"]
        self.global_train_tokens_seen_this_epoch = state_dict["global_train_tokens_seen_this_epoch"]
        self.global_train_examples_seen_this_epoch = state_dict[
            "global_train_examples_seen_this_epoch"
        ]
        self.epoch = state_dict["epoch"]

        log.info(f"Will resume training from step {self.global_step}, epoch {self.epoch}")

        if state_dict["world_size"] == get_world_size():  # global world size here on purpose
            rng_state = EnvRngStates.from_dict(state_dict["rng"])
            if not rng_state.restore():
                log.warning(
                    "Some RNG states were not restored due to differences in library versions"
                )
        else:
            log.warning(
                "Trainer will not restore rank RNG states since the RNG states in the checkpoint "
                "were saved with a different world size."
            )

    def load_checkpoint(
        self, dir: PathOrStr, *, load_optimizer_state: bool = True, load_trainer_state: bool = True
    ):
        """
        Load a checkpoint.

        .. note::
            :meth:`fit()` may call this method automatically depending on the :data:`load_strategy`.

        :param dir: The path/URL to a checkpoint or a folder of checkpoints.
        :param load_optimizer_state: Load optimizer state.
        :param load_trainer_state: Load trainer state.
        """
        dir = normalize_path(dir)

        # NOTE: to avoid making a ton of client requests (S3 or otherwise) we only make those
        # requests from rank 0 then scatter the result to the other ranks.
        if get_rank() == 0 and not self.checkpointer.dir_is_checkpoint(dir):
            # Try to find the latest checkpoint in the directory.
            dir = self.checkpointer.latest_checkpoint(dir)
        dir = scatter_object(dir)

        log.info(f"Loading checkpoint from '{dir}'...")
        trainer_state = self.checkpointer.load(
            dir,
            self.model,
            self.optim,
            load_optimizer_state=load_optimizer_state,
            load_trainer_state=load_trainer_state,
        )
        if load_trainer_state:
            assert trainer_state is not None
            self.load_state_dict(trainer_state)

        for callback in self.callbacks.values():
            callback.post_checkpoint_loaded(dir)

        self._checkpoint_loaded = True
        log.info("Checkpoint successfully loaded")

    def maybe_load_checkpoint(
        self, dir: PathOrStr, *, load_optimizer_state: bool = True, load_trainer_state: bool = True
    ) -> bool:
        """
        Like :meth:`load_checkpoint()` but is a no-op if there is no checkpoint in the ``dir`` provided.

        .. note::
            :meth:`fit()` may call this method automatically depending on the :data:`load_strategy`.

        :returns: If a checkpoint was loaded.
        """
        should_load: bool = True
        if get_rank() == 0:
            should_load = self.checkpointer.contains_checkpoint(dir)
        should_load = scatter_object(should_load)
        if should_load:
            self.load_checkpoint(
                dir,
                load_optimizer_state=load_optimizer_state,
                load_trainer_state=load_trainer_state,
            )
        else:
            log.warning(f"No checkpoint found in '{dir}', will train from scratch...")
        return should_load

    def save_checkpoint(self) -> PathOrStr:
        """
        Save a checkpoint for the current step to the :data:`save_folder`.

        :returns: The path/URL to the checkpoint.
        """
        dirname = self.checkpointer.checkpoint_dirname(self.global_step)
        path = join_path(self.save_folder, dirname)
        log.info(f"Saving checkpoint for step {self.global_step} to '{path}'...")
        self.checkpointer.save(path, self.model, self.optim, self.state_dict())
        for callback in self.callbacks.values():
            callback.post_checkpoint_saved(path)
        log.info("Checkpoint saved")
        return path

    def save_checkpoint_async(self) -> Tuple[PathOrStr, Future]:
        """
        Save a checkpoint for the current step to the :data:`save_folder` asynchronously.

        :returns: The path/URL to the checkpoint and a future which will complete when the
            checkpoint is successfully saved.
        """
        step = self.global_step
        dirname = self.checkpointer.checkpoint_dirname(step)
        path = join_path(self.save_folder, dirname)

        log.info(f"Saving checkpoint for step {step} to '{path}' asynchronously...")
        fut = self.checkpointer.save_async(path, self.model, self.optim, self.state_dict())

        def callback(future: Future):
            future.result()  # ensure it finished successfully
            for callback in self.callbacks.values():
                callback.post_checkpoint_saved(path)
            log.info(f"Checkpoint for step {step} saved successfully")

        fut.add_done_callback(callback)

        return path, fut

    def record_metric(
        self, name: str, value: Union[float, torch.Tensor], reduce_type: Optional[ReduceType] = None
    ):
        """
        Record a new metric for the current step.

        .. important::
            Metrics added with a ``reduce_type`` are reduced across the data parallel process group,
            which is not necessarily the default process group.

        :param name: The name of the metric.
        :param value: The value of the metric.
        :param reduce_type: Specifies how to reduce the metric across the distributed process group.
            ``None`` means no reduction.
        """
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        else:
            value = get_local_tensor(value).float()
        if self.global_step not in self._metrics:
            self._metrics[self.global_step] = OrderedDict()
        self._metrics[self.global_step][name] = value
        self._metrics_reduce_type[name] = reduce_type

    def get_metric(self, name: str) -> Optional[torch.Tensor]:
        """
        Get the value of a metric recorded during the current step.

        .. seealso::
            - :meth:`get_loss()`
            - :meth:`get_zloss()`

        .. warning::
            Metrics will only be available from the time they're recorded until the end of the
            current step.

        .. warning::
            Accessing a metric can inadvertently trigger a host-device sync, which slows down
            training.

        :param name: The name of the metric.
        """
        if self.global_step not in self._metrics:
            return None
        return self._metrics[self.global_step].get(name)

    def get_loss(self) -> Optional[torch.Tensor]:
        """
        Get the value of the cross-entropy loss for the current step.

        .. important::
            If you're trying to access the loss from a callback, it may only be accessible
            within the following callback methods:

            - :meth:`~olmo_core.train.callbacks.Callback.pre_optim_step()`
            - :meth:`~olmo_core.train.callbacks.Callback.post_train_batch()`
            - :meth:`~olmo_core.train.callbacks.Callback.post_step()`
        """
        return self.get_metric(TRAIN_CE_LOSS_METRIC)

    def get_zloss(self) -> Optional[torch.Tensor]:
        """
        Get the value of the Z-loss for the current step.

        .. important::
            If you're trying to access the Z-loss from a callback, it may only be accessible
            within the following callback methods:

            - :meth:`~olmo_core.train.callbacks.Callback.pre_optim_step()`
            - :meth:`~olmo_core.train.callbacks.Callback.post_train_batch()`
            - :meth:`~olmo_core.train.callbacks.Callback.post_step()`
        """
        return self.get_metric(TRAIN_Z_LOSS_METRIC)

    def write_file(
        self, name: str, contents: Union[str, bytes], dir: Optional[PathOrStr] = None
    ) -> PathOrStr:
        """
        Write a file to the :data:`save_folder` or ``dir``, if provided.

        :param fname: The name of the file to write, relative to the :data:`save_folder` or ``dir``.
        :param contents: The contents of the file to write.
        :param dir: The path/URL to a directory to write the file to. Defaults to :data:`save_folder`.

        :returns: The path/URL of the file.
        """
        return self.checkpointer.write_file(dir or self.save_folder, name, contents)

    def persist_working_file(self, name: PathOrStr) -> PathOrStr:
        """
        Persist a file in the :data:`work_dir` by saving/uploading it to the :data:`save_folder`.

        :param name: The name/path of the file *relative* to the :data:`work_dir`.

        :returns: The full path/URL to the saved file.

        :raises FileNotFoundError: If the file can't be found.
        :raises FileExistsError: If the file already exists in the save folder and :data:`save_overwrite`
            is ``False``.
        """
        if Path(name).is_relative_to(self.work_dir):
            name = Path(name).relative_to(self.work_dir)
        source = join_path(self.work_dir, name)
        target = join_path(self.save_folder, name)
        if source != target:
            copy_file(source, target, save_overwrite=self.save_overwrite)
        elif not file_exists(source):
            raise FileNotFoundError(source)
        return target

    def _duration_due(self, duration: Duration) -> bool:
        if duration.unit == DurationUnit.steps:
            return self.global_step >= duration.value
        elif duration.unit == DurationUnit.tokens:
            return self.global_train_tokens_seen >= duration.value
        elif duration.unit == DurationUnit.epochs:
            return self.epoch > duration.value
        else:
            raise NotImplementedError

    def _handle_sigterm(self, *args):
        del args
        log.warning("SIGTERM received")
        self.cancel_run("SIGTERM received")

    def _check_if_canceled(self):
        if self._canceled:
            return

        canceling_rank = self._canceling_rank if self._canceling_rank is not None else -1
        canceling_rank = all_reduce_value(
            canceling_rank, self.bookkeeping_device, op=dist.ReduceOp.MAX, group=self.bookkeeping_pg
        )
        if canceling_rank >= 0:
            cancel_reason = scatter_object(self._cancel_reason, src=canceling_rank)
            assert cancel_reason is not None
            self._canceled = True
            self._canceling_rank = canceling_rank
            self._cancel_reason = cancel_reason
            log.warning(f"Run canceled from rank {canceling_rank}. Reason: {cancel_reason}")

    def _log_metrics(self):
        if not self._metrics:
            return

        # Prep metrics to reduce by moving to bookkeeping device all at once.
        # NOTE: if training on GPU and `bookkeeping_device` is CPU, this triggers
        # host-device sync. It's unavoidable to have a host-device at some point, but we
        # prefer to do that early and then finish processing the metrics in a separate thread
        # so CUDA training can continue.
        metrics_to_reduce = move_metrics(self._metrics, self.bookkeeping_device)
        self._metrics.clear()

        if self.bookkeeping_device.type == "cpu" and self.bookkeeping_pg is not None:
            # If we have a separate CPU backend and process group we can safely reduce
            # metrics on CPU in a thread.
            future = self.thread_pool.submit(
                reduce_metrics,
                metrics_to_reduce,
                self._metrics_reduce_type,
                self.bookkeeping_device,
                process_group=self.bookkeeping_pg,
            )

            def callback(fut):
                try:
                    self._check_and_pass_on_metrics(fut.result())
                except BaseException as e:
                    log.exception(e)
                    self._error = e

            future.add_done_callback(callback)
        else:
            # Otherwise we have to reduce them now in the main thread.
            # NOTE: if we're training on GPU and didn't have a host device sync above, this will
            # trigger a host-device sync as we transfer the metrics back to CPU post-reducing.
            metrics = reduce_metrics(
                metrics_to_reduce,
                self._metrics_reduce_type,
                self.bookkeeping_device,
                process_group=self.bookkeeping_pg,
            )
            self._check_and_pass_on_metrics(metrics)

    def _check_and_pass_on_metrics(self, metrics: Dict[int, Dict[str, float]]):
        for step in sorted(metrics.keys()):
            # Check for nan/inf loss and add perplexity.
            if (ce_loss := metrics[step].get(TRAIN_CE_LOSS_METRIC)) is not None:
                if not math.isfinite(ce_loss):
                    raise RuntimeError(f"{ce_loss} loss encountered at step {step}")
                if ce_loss < 10:
                    metrics[step][TRAIN_PPL_METRIC] = math.exp(ce_loss)
            for callback in self.callbacks.values():
                callback.log_metrics(step, metrics[step])

    def _get_labels(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, label_mask, attention_mask, instance_mask = (
            batch["input_ids"].clone(),
            batch.get("label_mask"),
            batch.get("attention_mask"),
            batch.get("instance_mask"),
        )
        if label_mask is not None:
            labels.masked_fill_(~label_mask, -100)
        if attention_mask is not None:
            labels.masked_fill_(attention_mask == 0.0, -100)
        if instance_mask is not None:
            labels.masked_fill_(~instance_mask.unsqueeze(-1), value=-100)
        return labels[..., 1:].contiguous()

    @contextlib.contextmanager
    def _model_forward_context(self) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if self.autocast_precision is not None:
                stack.enter_context(torch.autocast(self.device.type, dtype=self.autocast_precision))
            yield

    def _model_forward(
        self,
        batch: Dict[str, Any],
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        compute_z_loss: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        with self._model_forward_context():
            # shape: (batch_size, seq_len, vocab_size)
            logits = self.model(
                input_ids=batch["input_ids"],
                #  attention_mask=batch.get("attention_mask"),
                #  attention_bias=batch.get("attention_bias"),
                doc_lens=batch.get("doc_lens"),
                max_doc_lens=batch.get("max_doc_lens"),
            )

        # shape: (batch_size, seq_len - 1, vocab_size)
        logits_for_loss = logits[..., :-1, :].contiguous()
        # shape: (batch_size * (seq_len - 1), vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
        # shape: (batch_size, seq_len - 1)
        labels = batch.get("labels", self._get_labels(batch))
        # shape: (batch_size * (seq_len - 1),)
        labels = labels.view(-1)

        loss_fn = cross_entropy_loss if not self.fused_loss else fused_cross_entropy_loss
        ce_loss, z_loss = loss_fn(
            logits_for_loss,
            labels,
            ignore_index=-100,
            reduction=loss_reduction,
            compute_z_loss=compute_z_loss,
            z_loss_multiplier=self.z_loss_multiplier or 1e-4,
        )

        if loss_reduction == "none":
            # Reshape (batch_size * (seq_len - 1),) -> (batch_size, seq_len - 1)
            ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1)
            if z_loss is not None:
                z_loss = z_loss.view(batch["input_ids"].shape[0], -1)

        return ce_loss, z_loss, logits

    def _get_microbatch_loss(
        self, micro_batch: Dict[str, Any], batch_num_tokens_for_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # NOTE: we use the "sum" loss reduction and then divide by 'batch_num_tokens_for_loss'
        # (the total number of tokens used in the loss across the whole batch, not just the micro batch)
        # to avoid biasing the loss in the case where micro-batches might not be the same size.
        ce_loss, z_loss, logits = self._model_forward(
            micro_batch, compute_z_loss=self.z_loss_multiplier is not None, loss_reduction="sum"
        )
        ce_loss = ce_loss / batch_num_tokens_for_loss

        # In case this helps with memory utilization.
        del micro_batch

        # Get loss to optimize for.
        if self.z_loss_multiplier is not None:
            assert z_loss is not None
            z_loss = z_loss / batch_num_tokens_for_loss
            loss = ce_loss + z_loss
        else:
            loss = ce_loss

        del logits

        return loss, ce_loss, z_loss

    @contextlib.contextmanager
    def _train_microbatch_context(
        self, micro_batch_idx: int, num_micro_batches: int
    ) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if isinstance(self.model, DDP) and micro_batch_idx != num_micro_batches - 1:
                # For DDP, only sync gradients on the final micro batch.
                stack.enter_context(self.model.no_sync())
            yield

    def _train_batch(self, batch: Dict[str, Any]):
        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None:
            self.record_metric("train/masked instances", (~instance_mask).sum(), ReduceType.sum)

        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        # Move tensors to the right device.
        batch = move_to_device(batch, self.device, non_blocking=True)

        # Generate labels, calculate how many tokens are going to be use in the loss.
        if "labels" not in batch:
            batch["labels"] = self._get_labels(batch)
        batch_num_tokens_for_loss = (batch["labels"] != -100).sum()

        # Split into micro-batches.
        micro_batches = self._split_batch(batch)
        num_micro_batches = len(micro_batches)

        # In case this helps with memory utilization.
        del batch

        ce_batch_loss = torch.tensor(0.0, device=self.device)
        z_batch_loss = (
            None if self.z_loss_multiplier is None else torch.tensor(0.0, device=self.device)
        )

        # Train one micro-batch at a time.
        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                # Run forward pass.
                loss, ce_loss, z_loss = self._get_microbatch_loss(
                    micro_batch, batch_num_tokens_for_loss
                )

                # Update overall CE batch loss.
                ce_batch_loss += ce_loss.detach()

                # Update overall Z batch loss.
                if z_loss is not None:
                    assert z_batch_loss is not None
                    z_batch_loss += z_loss.detach()

                # Run backward pass.
                loss.backward()

        self.record_metric(TRAIN_CE_LOSS_METRIC, ce_batch_loss, ReduceType.mean)
        if z_batch_loss is not None:
            self.record_metric(TRAIN_Z_LOSS_METRIC, z_batch_loss, ReduceType.mean)

        # Run through callbacks.
        for callback in self.callbacks.values():
            callback.pre_optim_step()

        # Optimizer step.
        self.optim.step()

    def _iter_batches(self) -> Generator[Dict[str, Any], None, None]:
        iterable_dataset = IterableDataset(
            self.dataset,
            seed=self.data_seed,
            epoch=self.epoch,
            start_index=self.global_train_examples_seen_this_epoch,
            rank_batch_size=self.rank_batch_size,
            dp_world_size=get_world_size(self.dp_process_group),
            dp_rank=get_rank(self.dp_process_group),
            fs_local_rank=get_fs_local_rank(),
            drop_last=True,
            work_dir=self.work_dir,
        )
        iterable_dataset.build_and_save_global_indices()
        barrier()
        data_loader = DataLoader(
            iterable_dataset,
            batch_size=self.rank_batch_size,
            drop_last=True,
            collate_fn=self.collator,
            num_workers=self.data_loader_workers,
            pin_memory=True,
            prefetch_factor=self.data_loader_prefetch_factor,
            persistent_workers=False,
            timeout=0,
        )
        data_iterator = iter(data_loader)

        while True:
            for callback in self.callbacks.values():
                callback.pre_load_batch()

            try:
                batch = next(data_iterator)
                yield batch
            except StopIteration:
                break

    def _fit_epoch(self):
        log.info(f"Starting epoch {self.epoch}...")

        for callback in self.callbacks.values():
            callback.pre_epoch()

        first_batch = True
        for batch in self._iter_batches():
            # Bookkeeping.
            # NOTE: To track the global batch size / number of tokens per batch we make the
            # assumption that all batches see the same number of tokens, which should always be
            # the case except for potentially the last batch in an epoch if drop_last=False.
            # Alternatively we'd have to use a distributed collective which isn't worth it.
            batch_size, seq_len = batch["input_ids"].shape
            assert seq_len == self.train_sequence_length
            assert batch_size == self.rank_batch_size
            self.global_step += 1
            self.global_train_tokens_seen_this_epoch += self.global_batch_size * seq_len
            self.global_train_examples_seen_this_epoch += self.global_batch_size
            self.global_train_tokens_seen += self.global_batch_size * seq_len

            for callback in self.callbacks.values():
                callback.pre_step(batch)

            self._train_batch(batch)

            for callback in self.callbacks.values():
                callback.post_train_batch()

            # TODO: evals

            for callback in self.callbacks.values():
                callback.post_step()

            if first_batch or self.global_step % self.metrics_collect_interval == 0:
                self._log_metrics()

            first_batch = False

            if self.training_complete:
                # Finishing before the epoch is complete.
                # Log any remaining metrics.
                self._log_metrics()
                return

        # Log any remaining metrics.
        self._log_metrics()

        log.info("Epoch complete")

        for callback in self.callbacks.values():
            callback.post_epoch()

        # Bookkeeping
        self.epoch += 1
        self.global_train_tokens_seen_this_epoch = 0
        self.global_train_examples_seen_this_epoch = 0

    def _split_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        batch_size = batch["input_ids"].shape[0]
        if batch_size <= self.microbatch_size:
            return [batch]
        else:
            micro_batches = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    micro_batches[key] = value.split(self.microbatch_size, dim=0)
                elif isinstance(value, list):
                    micro_batches[key] = [
                        value[
                            self.microbatch_size * i : self.microbatch_size * i
                            + self.microbatch_size
                        ]
                        for i in range(math.ceil(batch_size / self.microbatch_size))
                    ]
                else:
                    raise ValueError(f"unexpected item in batch: '{key}={value}'")
            return [
                {key: value[i] for key, value in micro_batches.items()}  # type: ignore
                for i in range(len(micro_batches["input_ids"]))
            ]
