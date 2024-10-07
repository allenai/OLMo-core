import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from olmo_core.data import NumpyFSLDataLoader, NumpyFSLDataset
from olmo_core.data.utils import melt_batch, truncate_batch
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import gc_cuda

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class SequenceLengthSchedulerCallback(Callback):
    """
    A :class:`Callback` for introducing a linear sequence-length warm-up schedule
    over the course of :data:`warmup_steps` starting from :data:`min_sequence_length`
    and ending at the configured training sequence length
    (:data:`NumpyFSLDataset.sequence_length <olmo_core.data.NumpyFSLDataset.sequence_length`).

    When :data:`truncate` is ``False`` the scheduler works by splitting each instance in a batch
    into more shorter instances while maintaining the same number of tokens in each batch and micro-batch.
    In this case the sequence length set during the warm-up will always be a multiple of
    :data:`min_sequence_length` by a power of 2, and therefore
    the train sequence length must be a multiple of :data:`min_sequence_length` by a power of 2.

    Otherwise the scheduler simply truncates the instances in the batch to the desired sequence
    length, throwing out the extra tokens. The scheduler will ensure the sequence length
    during the warm-up is always a multiple of :data:`keep_multiple_of`.

    .. important::
        This callback is only compatible with a :class:`~olmo_core.data.data_loader.NumpyFSLDataLoader`
        training :data:`~olmo_core.train.Trainer.data_loader`.

    .. note::
        The "total tokens" recorded by the trainer and :class:`SpeedMonitorCallback` will
        still include tokens truncated by this callback for bookkeeping purposes.
    """

    min_sequence_length: int = 128
    warmup_steps: int = 2000
    truncate: bool = False
    keep_multiple_of: int = 128
    enabled: bool = True

    _og_rank_microbatch_size: Optional[int] = None
    _last_seq_len: Optional[int] = None

    def pre_train(self):
        if not self.enabled:
            return

        if not isinstance(self.trainer.data_loader, NumpyFSLDataLoader):
            raise OLMoConfigurationError(
                "The sequence length scheduler callback requires a 'NumpyFSLDataLoader', "
                f"got '{type(self.trainer.data_loader)}' instead"
            )

        dataset = self.trainer.data_loader.dataset
        assert isinstance(dataset, NumpyFSLDataset)

        if self.truncate and (
            dataset.sequence_length % self.min_sequence_length != 0
            or (math.log(dataset.sequence_length // self.min_sequence_length, 2) % 1 != 0)
        ):
            raise OLMoConfigurationError(
                "train sequence length must be a multiple of 'min_sequence_length' by a power of 2 "
                "when 'truncate=False'."
            )
        elif dataset.sequence_length <= self.min_sequence_length:
            raise OLMoConfigurationError(
                "train sequence length must be greater than 'min_sequence_length'"
            )

        self._og_rank_microbatch_size = self.trainer.rank_microbatch_size

    def pre_step(self, batch: Dict[str, Any]):
        if not self.enabled:
            return

        if self.step > self.warmup_steps:
            return

        assert isinstance(self.trainer.data_loader, NumpyFSLDataLoader)
        dataset = self.trainer.data_loader.dataset
        assert isinstance(dataset, NumpyFSLDataset)

        new_seq_len: int
        if self.truncate:
            new_seq_len = _get_truncated_sequence_length(
                self.min_sequence_length,
                dataset.sequence_length,
                self.step,
                self.warmup_steps,
                self.keep_multiple_of,
            )

            for key, value in truncate_batch(
                batch,
                new_seq_len,
            ).items():
                batch[key] = value
        else:
            new_seq_len = _get_split_sequence_length(
                self.min_sequence_length,
                dataset.sequence_length,
                self.step,
                self.warmup_steps,
            )

            for key, value in melt_batch(
                batch,
                new_seq_len,
            ).items():
                batch[key] = value

            # Increase micro-batch size proportionally to maintain the same number of tokens
            # in each micro-batch.
            assert self._og_rank_microbatch_size is not None
            new_rank_microbatch_size = self._og_rank_microbatch_size * (
                dataset.sequence_length // new_seq_len
            )
            self.trainer.rank_microbatch_size = new_rank_microbatch_size

        if new_seq_len != self._last_seq_len:
            log.info(f"Changing sequence length to {new_seq_len} per warm-up schedule")
            self._last_seq_len = new_seq_len
            # Empty CUDA cache since shapes have now changed.
            gc_cuda()

    def post_train_batch(self):
        if not self.enabled or self.step > self.warmup_steps + 1:
            return

        assert self._og_rank_microbatch_size is not None
        self.trainer.rank_microbatch_size = self._og_rank_microbatch_size


def _get_split_sequence_length(
    min_sequence_length: int, max_sequence_length: int, step: int, warmup_steps: int
) -> int:
    seq_len = (
        min_sequence_length
        + (max_sequence_length - min_sequence_length) * min(step, warmup_steps) / warmup_steps
    )

    n = math.floor(math.log(seq_len // min_sequence_length, 2))
    return min_sequence_length * 2**n


def _get_truncated_sequence_length(
    min_sequence_length: int,
    max_sequence_length: int,
    step: int,
    warmup_steps: int,
    keep_multiple_of: int,
) -> int:
    seq_len = (
        min_sequence_length
        + (max_sequence_length - min_sequence_length) * min(step, warmup_steps) / warmup_steps
    )

    seq_len = keep_multiple_of * (seq_len // keep_multiple_of)
    return int(seq_len)
