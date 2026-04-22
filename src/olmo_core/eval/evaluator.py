from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, Iterable, Iterator, Optional

import torch

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
    :param batches_factory: A callable that returns an iterable over batches. This is an
        alternative to providing the ``batches`` argument directly.
    :param device: The device to compute/reduce metrics on.
    :param deterministic: When ``True`` and :data:`batches` is a
        :class:`~olmo_core.data.data_loader.DataLoaderBase`, each evaluation pass resets the data
        loader and reshuffles with ``epoch=1`` so repeated evals read the same batches in the same
        order. This is useful when eval loops are truncated via
        :class:`~olmo_core.train.common.Duration`. When ``False``, the data loader still resets to
        batch 0 before each pass, but reshuffles without pinning the epoch so the batch order may
        change between eval runs. This does not implement a moving window across evals; if an eval
        is truncated, different reshuffles may result in different instances being evaluated each
        time.
    """

    def __init__(
        self,
        *,
        name: str,
        batches: Optional[Iterable[Dict[str, Any]]] = None,
        batches_factory: Optional[Callable[[], Iterable[Dict[str, Any]]]] = None,
        device: Optional[torch.device] = None,
        deterministic: bool = True,
    ):
        if batches is None:
            assert (
                batches_factory is not None
            ), "Either 'batches' or 'batches_factory' must be provided."
        else:
            assert (
                batches_factory is None
            ), "'batches' and 'batches_factory' cannot both be provided."
        self.name = name
        self.batches = batches
        self.batches_factory = batches_factory
        self.device = device
        self.deterministic = deterministic

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterator over the evaluator's batches.
        """
        if self.batches is None:
            assert self.batches_factory is not None
            self.batches = self.batches_factory()
        if isinstance(self.batches, DataLoaderBase):
            # Reset bookkeeping before reshuffling so eval_duration-limited evals always restart
            # from batch 0 instead of resuming from the previous partial pass.
            self.batches.reset()
            if self.deterministic:
                self.batches.reshuffle(epoch=1, in_memory=True)
            else:
                self.batches.reshuffle(in_memory=True)
        for batch in self.batches:
            yield batch
        if isinstance(self.batches, DataLoaderBase):
            self.batches.reset()

    @property
    def display_name(self) -> str:
        return self.name

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
        self, batch: Dict[str, Any], ce_loss: Optional[torch.Tensor], logits: Optional[torch.Tensor]
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
