from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Iterable, Iterator, Optional

import torch
import torch.distributed as dist

from ..data import DataLoaderBase


class Evaluator(metaclass=ABCMeta):
    """
    Base class for in-loop evaluators.

    .. seealso::
        This can be used with an :class:`~olmo_core.train.callbacks.EvaluatorCallback` to run an
        evaluator within the training loop.

    :param name: A name to assign to the evaluator.
    :param batches: Generates batches for the evaluator. These should at least include the
        "input_ids" field, but can contain any other arbitrary fields as well.
    :param device: The device to compute/reduce metrics on.
    :param dp_process_group: The data parallel process group to reduce metrics over.
    """

    def __init__(
        self,
        *,
        name: str,
        batches: Iterable[Dict[str, Any]],
        device: Optional[torch.device] = None,
        dp_process_group: Optional[dist.ProcessGroup] = None,
    ):
        self.name = name
        self.batches = batches
        self.device = device
        self.dp_process_group = dp_process_group

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterator over the evaluator's batches.
        """
        if isinstance(self.batches, DataLoaderBase):
            self.batches.reshuffle(in_memory=True)
        for batch in self.batches:
            yield batch
        if isinstance(self.batches, DataLoaderBase):
            self.batches.reset()

    @property
    def total_batches(self) -> Optional[int]:
        """
        Get the total number of batches in an eval loop if it's known ahead of time.
        """
        try:
            return len(self.batches)  # type: ignore
        except TypeError:
            return None

    @abstractmethod
    def update_metrics(
        self, batch: Dict[str, Any], ce_loss: torch.Tensor, logits: torch.Tensor
    ) -> None:
        """
        Update metrics with from the ``batch`` just processed and the corresponding ``logits``.

        :param batch: A batch generated from :data:`batches`.
        :param ce_loss: The cross-entropy loss per token (un-reduced) of the batch. This will
            have shape ``(batch_size, (seq_len - 1))``.
        :param logits: The logits generated from the forward pass of the model.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self) -> Dict[str, torch.Tensor]:
        """
        Compute the final value of the metrics for the current evaluation loop.
        The metrics returned should already be reduced, if needed.
        """
        raise NotImplementedError

    @abstractmethod
    def reset_metrics(self) -> None:
        """
        Reset metrics. Should be called after :meth:`compute_metrics()`.
        """
        raise NotImplementedError
