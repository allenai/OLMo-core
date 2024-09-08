import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from olmo_core.data import melt_batch
from olmo_core.exceptions import OLMoConfigurationError

from .callback import Callback

log = logging.getLogger(__name__)


@dataclass
class SequenceLengthSchedulerCallback(Callback):
    """
    A :class:`Callback` for introducing a linear sequence-length warm-up schedule
    over the course of :data:`warmup_steps` starting from :data:`min_sequence_length`
    and ending at
    :data:`Trainer.train_sequence_length <olmo_core.train.Trainer.train_sequence_length>`.
    The sequence length set during the warm-up will always be a multiple of :data:`min_sequence_length`
    by a power of 2.

    .. important::
        :data:`Trainer.train_sequence_length <olmo_core.train.Trainer.train_sequence_length>`
        must be a multiple of :data:`min_sequence_length` by a power of 2.
    """

    min_sequence_length: int = 128
    warmup_steps: int = 2000

    _og_microbatch_size: Optional[int] = None
    _last_seq_len: Optional[int] = None

    def pre_train(self):
        if self.trainer.train_sequence_length % self.min_sequence_length != 0 or (
            math.log(self.trainer.train_sequence_length // self.min_sequence_length, 2) % 1 != 0
        ):
            raise OLMoConfigurationError(
                "'train_sequence_length' must be a multiple of 'min_sequence_length' by a power of 2"
            )
        self._og_microbatch_size = self.trainer.microbatch_size

    def pre_step(self, batch: Dict[str, Any]):
        if self.step <= self.warmup_steps:
            new_seq_len = _get_sequence_length(
                self.min_sequence_length,
                self.trainer.train_sequence_length,
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
            assert self._og_microbatch_size is not None
            new_microbatch_size = self._og_microbatch_size * (
                self.trainer.train_sequence_length // new_seq_len
            )
            self.trainer.microbatch_size = new_microbatch_size

            if new_seq_len != self._last_seq_len:
                log.info(f"Changing sequence length to {new_seq_len} per warm-up schedule")

    def post_train_batch(self):
        assert self._og_microbatch_size is not None
        self.trainer.microbatch_size = self._og_microbatch_size


def _get_sequence_length(
    min_sequence_length: int, max_sequence_length: int, step: int, warmup_steps: int
) -> int:
    seq_len = (
        min_sequence_length
        + (max_sequence_length - min_sequence_length) * min(step, warmup_steps) / warmup_steps
    )

    n = math.floor(math.log(seq_len // min_sequence_length, 2))
    return min_sequence_length * 2**n
