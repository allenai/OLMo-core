import contextlib
import logging
import math
import signal
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Literal,
    Optional,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer

from ..aliases import PathOrStr
from ..data import DataLoaderBase
from ..data.utils import get_labels, split_batch
from ..distributed.utils import (
    all_reduce_value,
    backend_supports_cpu,
    barrier,
    get_fs_local_rank,
    get_local_tensor,
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
from ..optim import SkipStepOptimizer
from ..utils import cuda_sync_debug_mode, move_to_device
from .callbacks import (
    Callback,
    CheckpointerCallback,
    ConsoleLoggerCallback,
    GarbageCollectorCallback,
    SpeedMonitorCallback,
)
from .checkpoint import Checkpointer
from .common import Duration, DurationUnit, LoadStrategy, ReduceType
from .utils import EnvRngStates, move_metrics, reduce_metrics

log = logging.getLogger(__name__)

TRAIN_CE_LOSS_METRIC = "train/CE loss"
TRAIN_PPL_METRIC = "train/PPL"
TRAIN_Z_LOSS_METRIC = "train/Z loss"
OPTIM_STEP_SKIPPED_METRIC = "optim/step skipped"
SEQ_LEN_METRIC = "data/sequence length"

T = TypeVar("T")


class TrainerStateDict(TypedDict):
    global_step: int
    global_train_tokens_seen: int
    data_loader: Dict[str, Any]
    epoch: int
    world_size: int
    rng: Dict[str, Any]


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

    .. note::
        When constructing your trainer through a :class:`TrainerConfig` this will default to
        the :data:`save_folder` if it's a local directory.
    """

    model: nn.Module
    """
    The model to fit.
    """

    optim: Optimizer
    """
    The optimizer to use.
    """

    data_loader: DataLoaderBase
    """
    The train data loader.
    """

    device: torch.device
    """
    The default device to use. Should match the device the model is on and be appropriate for the
    main distributed backend.
    """

    save_folder: str
    """
    The folder to save all checkpoints to. Could be a local directory (if using a shared filesytem)
    or a URL.

    .. warning::
        If you try to use a local directory without a globally shared filesystem across all ranks
        you will get an error.
    """

    checkpointer: Checkpointer
    """
    The checkpointer. This is a wrapper around the functionality in
    :mod:`olmo_core.distributed.checkpoint`, which means you can use
    :func:`~olmo_core.distributed.checkpoint.unshard_checkpoint` to unshard the model and optimizer
    state from a train checkpoint after the fact.
    """

    callbacks: Dict[str, Callback]
    """
    Trainer callbacks.
    """

    max_duration: Duration
    """
    The duration to train for.

    .. important::
        The total number of training steps must be known ahead of time for various reasons such
        as setting a learning rate schedule. Therefore if your data loader's number of batches
        (:data:`~olmo_core.data.data_loader.DataLoaderBase.total_batches`) is unknown ahead of time,
        you must set the ``max_duration`` in terms of :meth:`tokens <Duration.tokens>`
        or :meth:`steps <Duration.steps>`, but not epochs.
    """

    rank_microbatch_size: int
    """
    Microbatch size *in tokens* per rank, i.e. the number of tokens to process at a time from each rank.

    This must evently divide into :data:`global_batch_size` by a factor of the data parallel world size.
    If this is less than :data:`global_batch_size` divided by the data parallel world size then
    gradient accumulation is used.
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
    Use the fused cross-entropy loss function (:func:`~olmo_core.nn.functional.fused_cross_entropy_loss`)
    instead the PyTorch built-in. This can help reduce GPU memory usage. Relative performance will
    depend on the input sizes.
    """

    compile_loss: bool = False
    """
    Compile the loss function.
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

    # Bookkeeping

    global_step: int = 0
    """
    The current step/batch. 1-based, though it's initialized to 0 before the first step.
    This does *not* reset after an epoch.
    """

    global_train_tokens_seen: int = 0
    """
    The total number of training tokens seen.
    """

    epoch: int = 1
    """
    The current epoch (1-based).
    """

    cancel_check_interval: int = 25
    """
    The interval (in steps) to check if the run is canceled. Checking requires distributed comms,
    but if you've configured a separate CPU-only backend (like "gloo") then this shouldn't impact
    training throughput.
    """

    hard_stop: Optional[Duration] = None
    """
    Set a hard stopping point for the trainer. This is useful for ablations when you you don't
    want to do a complete training run, but you don't want to change :data:`max_duration` as to
    not affect the learning rate schedule.
    """

    async_bookkeeping: Optional[bool] = None
    """
    Do collective bookkeeping operations like reducing metrics asynchronously.
    This requires a separate CPU-only backend, and will default to ``True`` if one is available.
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
    # NOTE: do not assign a default here or it will become a bound method due to the way
    # dataclasses work.
    _loss_fn = None

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
        self.callbacks.setdefault(
            "console_logger",
            ConsoleLoggerCallback(
                log_interval=1, metrics_log_interval=self.metrics_collect_interval
            ),
        )
        self.callbacks.setdefault("checkpointer", CheckpointerCallback())
        self.callbacks.setdefault("speed_monitor", SpeedMonitorCallback())
        if is_distributed():
            self.callbacks.setdefault("garbage_collector", GarbageCollectorCallback())

        # Set pointer to self in all callbacks.
        for callback in self.callbacks.values():
            callback.trainer = self

        # Sort callbacks by (priority, name).
        # We do this for 2 reasons: (1) to respect the priority, and (2) to ensure the callback
        # order is consistent across the process group since some callbacks make distributed
        # synchronization/communication calls.
        self._sort_callbacks()

        # Maybe create separate process group for bookkeeping.
        if self._bookkeeping_pg is None and is_distributed():
            if self.async_bookkeeping is None:
                self.async_bookkeeping = backend_supports_cpu()
            if self.async_bookkeeping:
                if not backend_supports_cpu():
                    raise OLMoConfigurationError(
                        "A CPU-only backend is required for async bookkeeping"
                    )
                log.info("Creating new process group for async bookkeeping")
                self._bookkeeping_pg = dist.new_group(
                    ranks=None
                    if self.dp_process_group is None
                    else dist.get_process_group_ranks(self.dp_process_group)
                )

        # Check data loader configuration.
        if self.data_loader.dp_world_size != get_world_size(self.dp_process_group):
            raise OLMoConfigurationError(
                "data loader's DP world size appears to be configured incorrectly, "
                f"got {self.data_loader.dp_world_size}, expected {get_world_size(self.dp_process_group)}."
            )
        if self.data_loader.dp_rank != get_rank(self.dp_process_group):
            raise OLMoConfigurationError(
                "data loader's DP rank appears to be configured incorrectly, "
                f"got {self.data_loader.dp_rank}, expected {get_rank(self.dp_process_group)}."
            )
        if self.data_loader.fs_local_rank != get_fs_local_rank():
            raise OLMoConfigurationError(
                "data loader's FS local rank appears to be configured incorrectly, "
                f"got {self.data_loader.fs_local_rank}, expected {get_fs_local_rank()}."
            )

        # Make sure global batch size is divisible by microbatch size times world size
        if (
            self.global_batch_size
            % (self.rank_microbatch_size * (ws := get_world_size(self.dp_process_group)))
            != 0
        ):
            raise OLMoConfigurationError(
                f"global batch size ({self.global_batch_size}) must be divisible by "
                f"micro-batch size ({self.rank_microbatch_size}) x DP world size ({ws})"
            )

        for callback in self.callbacks.values():
            callback.post_attach()

        # Set loss function.
        if self.fused_loss:
            self._loss_fn = fused_cross_entropy_loss
        else:
            self._loss_fn = cross_entropy_loss
        if self.compile_loss:
            self._loss_fn = torch.compile(self._loss_fn)

    @property
    def global_batch_size(self) -> int:
        """
        Global training batch size *in tokens*.
        """
        return self.data_loader.global_batch_size

    @property
    def rank_batch_size(self) -> int:
        """
        The number of tokens in each training batch per rank.
        """
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
            self.check_if_canceled()

        if self.is_canceled:
            return True
        elif self._duration_due(self.max_duration):
            return True
        elif self.hard_stop is not None and self._duration_due(self.hard_stop):
            return True
        else:
            return False

    @property
    def is_canceled(self) -> bool:
        if self._error is not None:
            raise RuntimeError("An error occurred") from self._error
        return self._canceled

    @property
    def tokens_per_batch(self) -> int:
        """
        The number of tokens in each training batch.
        """
        return self.global_batch_size

    @property
    def steps_per_epoch(self) -> Optional[int]:
        """
        The total number of training steps in an epoch, if known.
        """
        return self.data_loader.total_batches

    @property
    def tokens_per_epoch(self) -> Optional[int]:
        """
        The total number of tokens in the training dataset, minus left-overs.
        """
        if self.steps_per_epoch is not None:
            return self.steps_per_epoch * self.tokens_per_batch
        else:
            return None

    @property
    def max_steps(self) -> int:
        """
        The maximum number of steps to train for, as determined by :data:`max_duration`.
        """
        if self.max_duration.unit == DurationUnit.steps:
            return self.max_duration.value
        elif self.max_duration.unit == DurationUnit.epochs:
            if self.data_loader.total_batches is None:
                raise RuntimeError(
                    "the number of steps cannot be determined from an 'epochs' duration since "
                    "the data loader's number of batches is unknown"
                )
            max_epochs = self.max_duration.value
            complete_epochs_remaining = max(max_epochs - self.epoch, 0)
            steps_remaining_this_epoch = max(
                self.data_loader.total_batches - self.data_loader.batches_processed, 0
            )
            steps_remaining = (
                complete_epochs_remaining * self.data_loader.total_batches
                + steps_remaining_this_epoch
            )
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
        if self.async_bookkeeping and backend_supports_cpu():
            return torch.device("cpu")
        else:
            return self.device

    @property
    def bookkeeping_pg(self) -> Optional[dist.ProcessGroup]:
        """
        The process group used for bookkeeping collectives. This should include the same ranks
        as the :data:`dp_process_group`.

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

    def check_if_canceled(self):
        """
        Asynchronously check if the run is canceled. Use :data:`is_canceled` to see the result.
        This needs to be called by all ranks at the same point in the training loop.
        """
        self._run_bookkeeping_op(self._check_if_canceled)

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

        # Install SIGTERM + SIGINT handlers.
        og_sigterm_handler = signal.signal(signal.SIGTERM, self._handle_os_signal)
        og_sigint_handler = signal.signal(signal.SIGINT, self._handle_os_signal)

        # Do a dry-run for compiling and catch OOMs.
        self._dry_run_batch()

        try:
            while not self.training_complete:
                self._fit_epoch()
        except BaseException as exc:
            log.error(f"Training failed due to:\n{exc}")
            for callback in self.callbacks.values():
                callback.on_error(exc)
            raise
        finally:
            # Restore original signal handlers.
            signal.signal(signal.SIGTERM, og_sigterm_handler)
            signal.signal(signal.SIGINT, og_sigint_handler)

        for callback in self.callbacks.values():
            callback.post_train()

        log.info("Training complete")

    def state_dict(self) -> TrainerStateDict:
        """
        Get the trainer state to save.
        """
        return {
            "global_step": self.global_step,
            "global_train_tokens_seen": self.global_train_tokens_seen,
            "data_loader": self.data_loader.state_dict(),
            "epoch": self.epoch,
            "world_size": get_world_size(),  # global world size here on purpose
            "rng": EnvRngStates.current_state().as_dict(),
        }

    def load_state_dict(self, state_dict: TrainerStateDict):
        """
        Load trainer state (not model or optimizer state).
        """
        # For backwards compatibility.
        if "data_loader" not in state_dict:
            if "dataset" in state_dict:
                state_dict["data_loader"] = state_dict.pop("dataset")
                state_dict["data_loader"]["epoch"] = state_dict["epoch"]
            else:
                state_dict["data_loader"] = {
                    "dataset_type": "fsl",
                    "dataset_fingerprint_version": state_dict.pop("dataset_fingerprint_version"),
                    "dataset_fingerprint": state_dict.pop("dataset_fingerprint"),
                    "tokens_processed": state_dict["global_train_tokens_seen_this_epoch"],
                    "batches_processed": state_dict["global_train_tokens_seen_this_epoch"]
                    // self.global_batch_size,
                    "sequence_length": state_dict.pop("train_sequence_length"),
                    "max_target_sequence_length": state_dict.pop("max_train_sequence_length"),
                    "seed": state_dict["data_seed"],
                    "epoch": state_dict["epoch"],
                }

        self.data_loader.load_state_dict(state_dict["data_loader"])
        self.global_step = state_dict["global_step"]
        self.global_train_tokens_seen = state_dict["global_train_tokens_seen"]
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
            self.load_state_dict(cast(TrainerStateDict, trainer_state))

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
        self.checkpointer.save(
            path, self.model, self.optim, cast(Dict[str, Any], self.state_dict())
        )
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
        fut = self.checkpointer.save_async(
            path, self.model, self.optim, cast(Dict[str, Any], self.state_dict())
        )

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

        :returns: The loss scalar tensor, typically on GPU.
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

        :returns: The Z-loss scalar tensor, typically on GPU.
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

    def add_callback(self, name: str, callback: Callback):
        if name in self.callbacks:
            raise OLMoConfigurationError(f"A callback with name '{name}' already exists!")
        callback.trainer = self
        self.callbacks[name] = callback
        self._sort_callbacks()
        callback.post_attach()

    def model_forward(self, micro_batch: Dict[str, Any]) -> torch.Tensor:
        """
        Run a forward pass on a micro-batch, returning the logits.
        """
        with self._model_forward_context():
            # shape: (batch_size, seq_len, vocab_size)
            logits = self.model(
                input_ids=micro_batch["input_ids"],
                #  attention_mask=micro_batch.get("attention_mask"),
                #  attention_bias=micro_batch.get("attention_bias"),
                doc_lens=micro_batch.get("doc_lens"),
                max_doc_lens=micro_batch.get("max_doc_lens"),
            )
        return logits

    def get_losses(
        self,
        micro_batch: Dict[str, Any],
        logits: torch.Tensor,
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        compute_z_loss: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute the cross-entropy loss and optionally the Z-loss from a micro-batch and the
        corresponding logits returned from :meth:`model_forward()`.

        :param micro_batch: The micro-batch to evaluate.
        :param logits: The logits from the forward pass.
        :param loss_reduction: The (local) reduction to apply to the loss(es).
        :param compute_z_loss: Whether or not to compute and return the Z-loss.

        :returns: The cross entropy and optional Z-loss, respectively.
        """
        if compute_z_loss is None:
            compute_z_loss = self.z_loss_multiplier is not None

        # shape: (batch_size, seq_len - 1, vocab_size)
        logits_for_loss = logits[..., :-1, :].contiguous()
        # shape: (batch_size * (seq_len - 1), vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))

        # shape: (batch_size, seq_len - 1)
        labels = micro_batch.get("labels", self._get_labels(micro_batch))
        # shape: (batch_size * (seq_len - 1),)
        labels = labels.view(-1)

        ce_loss, z_loss = self._loss_fn(  # type: ignore
            logits_for_loss,
            labels,
            ignore_index=self.data_loader.collator.label_ignore_index,
            reduction=loss_reduction,
            compute_z_loss=compute_z_loss,
            z_loss_multiplier=self.z_loss_multiplier or 1e-4,
        )

        if loss_reduction == "none":
            # Reshape (batch_size * (seq_len - 1),) -> (batch_size, seq_len - 1)
            ce_loss = ce_loss.view(micro_batch["input_ids"].shape[0], -1)
            if z_loss is not None:
                z_loss = z_loss.view(micro_batch["input_ids"].shape[0], -1)

        return ce_loss, z_loss

    def eval_batch(
        self,
        batch: Dict[str, Any],
        loss_reduction: Literal["mean", "sum", "none"] = "mean",
        compute_z_loss: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Get the loss for an eval batch.

        .. important::
            You are responsible for ensuring the model is in ``.eval()`` mode before calling this.

        :param batch: The batch to evaluate.
        :param loss_reduction: The (local) reduction to apply to the loss(es).
        :param compute_z_loss: Whether or not to compute and return the Z-loss.

        :returns: The logits, cross-entropy loss, and Z-loss, respectively.
        """
        batch = move_to_device(batch, self.device)
        for callback in self.callbacks.values():
            callback.pre_eval_batch(batch)
        with torch.no_grad():
            logits = self.model_forward(batch)
            ce_loss, z_loss = self.get_losses(
                batch, logits, loss_reduction=loss_reduction, compute_z_loss=compute_z_loss
            )
        return logits, ce_loss, z_loss

    def _sort_callbacks(self):
        self.callbacks = OrderedDict(
            (
                (k, cb)
                for k, cb in sorted(
                    self.callbacks.items(), key=lambda x: (x[1].priority, x[0]), reverse=True
                )
            )
        )

    def _duration_due(self, duration: Duration) -> bool:
        return duration.due(
            step=self.global_step, tokens=self.global_train_tokens_seen, epoch=self.epoch
        )

    def _handle_os_signal(self, signalnum, stack_frame):
        del stack_frame

        signame: Optional[str] = None
        if signalnum == signal.SIGTERM:
            signame = "SIGTERM"
        elif signalnum == signal.SIGINT:
            signame = "SIGINT"

        msg: str
        if signame is not None:
            msg = f"{signame} received"
        else:
            msg = f"Sig({signalnum}) received"

        log.warning(msg)
        self.cancel_run(msg)

    def _run_bookkeeping_op(
        self, op: Callable[..., T], *args, cb: Optional[Callable[[T], None]] = None, **kwargs
    ):
        if (
            self.async_bookkeeping
            and self.bookkeeping_device.type == "cpu"
            and self.bookkeeping_pg is not None
        ):
            # Can safely run in the thread pool.
            future = self.thread_pool.submit(op, *args, **kwargs)
            if cb is not None:

                def callback(fut: Future[T]):
                    try:
                        cb(fut.result())  # type: ignore[misc]
                    except BaseException as e:
                        log.exception(e)
                        self._error = e

                future.add_done_callback(callback)
        else:
            result = op(*args, **kwargs)
            if cb is not None:
                cb(result)

    def _check_if_canceled(self):
        if self._canceled:
            return

        canceling_rank = self._canceling_rank if self._canceling_rank is not None else -1
        # NOTE: this is a known host-device sync (potentially) so we don't need the warning
        with cuda_sync_debug_mode(0):
            canceling_rank = all_reduce_value(
                canceling_rank,
                self.bookkeeping_device,
                op=dist.ReduceOp.MAX,
                group=self.bookkeeping_pg,
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
        self._run_bookkeeping_op(
            reduce_metrics,
            metrics_to_reduce,
            self._metrics_reduce_type,
            self.bookkeeping_device,
            process_group=self.bookkeeping_pg,
            cb=self._check_and_pass_on_metrics,
        )

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
        return get_labels(batch, label_ignore_index=self.data_loader.collator.label_ignore_index)

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
        # NOTE: keep this method for backwards compatibility.
        logits = self.model_forward(batch)
        ce_loss, z_loss = self.get_losses(
            batch, logits, loss_reduction=loss_reduction, compute_z_loss=compute_z_loss
        )
        return ce_loss, z_loss, logits

    @contextlib.contextmanager
    def _train_microbatch_context(
        self, micro_batch_idx: int, num_micro_batches: int
    ) -> Generator[None, None, None]:
        with contextlib.ExitStack() as stack:
            if isinstance(self.model, DDP) and micro_batch_idx != num_micro_batches - 1:
                # For DDP, only sync gradients on the final micro batch.
                stack.enter_context(self.model.no_sync())
            yield

    def _train_batch(self, batch: Dict[str, Any], dry_run: bool = False):
        # Record how many instances are going to be skipped (masked out).
        if (instance_mask := batch.get("instance_mask")) is not None and not dry_run:
            self.record_metric("train/masked instances", (~instance_mask).sum(), ReduceType.sum)

        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Generate labels, calculate how many tokens are going to be use in the loss.
        if "labels" not in batch:
            batch["labels"] = self._get_labels(batch)
        batch_num_tokens_for_loss = (
            batch["labels"] != self.data_loader.collator.label_ignore_index
        ).sum()

        # Split into micro-batches.
        if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
            )
        micro_batches = split_batch(batch, self.rank_microbatch_size // seq_len)
        num_micro_batches = len(micro_batches)

        ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        z_batch_loss = (
            None
            if self.z_loss_multiplier is None
            else move_to_device(torch.tensor(0.0), self.device)
        )

        # Train one micro-batch at a time.
        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self._train_microbatch_context(micro_batch_idx, num_micro_batches):
                # Run forward pass.
                logits = self.model_forward(micro_batch)

                # NOTE: we use the "sum" loss reduction and then divide by 'batch_num_tokens_for_loss'
                # (the total number of tokens used in the loss across the whole batch, not just the micro batch)
                # to avoid biasing the loss in the case where micro-batches might not be the same size.
                ce_loss, z_loss = self.get_losses(micro_batch, logits, loss_reduction="sum")
                ce_loss.div_(batch_num_tokens_for_loss)
                if z_loss is not None:
                    z_loss.div_(batch_num_tokens_for_loss)

                # Get loss to optimize for.
                loss: torch.Tensor
                if z_loss is not None:
                    loss = ce_loss + z_loss
                else:
                    loss = ce_loss

                # Update overall CE batch loss.
                ce_batch_loss += get_local_tensor(ce_loss.detach())

                # Update overall Z batch loss.
                if z_loss is not None:
                    assert z_batch_loss is not None
                    z_batch_loss += get_local_tensor(z_loss.detach())

                # Run through callbacks.
                for callback in self.callbacks.values():
                    callback.pre_backward(batch=batch, micro_batch=micro_batch, loss=loss)

                # Run backward pass.
                loss.backward()

        # In case this helps with memory utilization.
        del batch

        if dry_run:
            # Zero-gradients again.
            self.optim.zero_grad(set_to_none=True)
            return

        self.record_metric(TRAIN_CE_LOSS_METRIC, ce_batch_loss, ReduceType.mean)
        if z_batch_loss is not None:
            self.record_metric(TRAIN_Z_LOSS_METRIC, z_batch_loss, ReduceType.mean)

        if isinstance(self.optim, SkipStepOptimizer):
            self.optim.latest_loss = ce_batch_loss

        # Run through callbacks.
        for callback in self.callbacks.values():
            callback.pre_optim_step()

        # Optimizer step.
        self.optim.step()
        if isinstance(self.optim, SkipStepOptimizer):
            self.record_metric(OPTIM_STEP_SKIPPED_METRIC, self.optim.step_skipped)

        # Run through callbacks.
        for callback in self.callbacks.values():
            callback.post_train_batch()

    def _iter_batches(self) -> Generator[Dict[str, Any], None, None]:
        data_iterator = iter(self.data_loader)

        while True:
            for callback in self.callbacks.values():
                callback.pre_load_batch()

            try:
                batch = next(data_iterator)
                yield batch
            except StopIteration:
                break

    def _validate_batch(self, batch: Dict[str, Any]) -> int:
        """
        Validate the data in a batch and return the global total number of tokens in the batch.
        """
        # NOTE: To track the global number of tokens seen per batch we make the
        # assumption that all ranks see the same number batch size in tokens per step,
        # which should always be the case for training efficiency at least.
        # Alternatively we'd have to use a distributed collective which isn't worth it.
        if batch["input_ids"].numel() != self.rank_batch_size:
            raise RuntimeError(
                f"Expected batch size of {self.rank_batch_size:,d} tokens on rank {get_rank()}, "
                f"got input IDs with shape {tuple(batch['input_ids'].shape)} = {batch['input_ids'].numel():,d} tokens"
            )
        return self.global_batch_size

    def _dry_run_batch(self):
        try:
            batch = self.data_loader.get_mock_batch()
        except NotImplementedError:
            return  # for backwards compatibility

        log.info("Starting forward/backward dry-run batch...")
        self._validate_batch(batch)
        self._train_batch(batch, dry_run=True)
        log.info("Dry-run complete")

    def _fit_epoch(self):
        self.data_loader.reshuffle(self.epoch)

        log.info(f"Starting epoch {self.epoch}...")

        for callback in self.callbacks.values():
            callback.pre_epoch()

        first_batch = True
        for batch in self._iter_batches():
            # Bookkeeping.
            self.global_step += 1
            self.global_train_tokens_seen += self._validate_batch(batch)

            self.record_metric(SEQ_LEN_METRIC, float(batch["input_ids"].shape[1]))

            for callback in self.callbacks.values():
                callback.pre_step(batch)

            self._train_batch(batch)

            for callback in self.callbacks.values():
                callback.post_step()

            if first_batch or self.global_step % self.metrics_collect_interval == 0:
                self._log_metrics()
                torch.cuda.set_sync_debug_mode("warn")

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
        self.data_loader.reset()
