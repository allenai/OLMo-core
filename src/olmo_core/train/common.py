from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from ..config import StrEnum
from ..data.utils import get_labels


class DurationUnit(StrEnum):
    """
    Units that can be used to define a :class:`Duration`.
    """

    steps = "steps"
    """
    Steps (batches).
    """
    epochs = "epochs"
    """
    Epochs.
    """
    tokens = "tokens"
    """
    Tokens.
    """


@dataclass
class Duration:
    value: int
    """
    The value of the duration.
    """
    unit: DurationUnit
    """
    The unit associated with the :data:`value`.
    """

    @classmethod
    def steps(cls, steps: int) -> "Duration":
        """
        Define a duration from a number of steps.
        """
        return cls(value=steps, unit=DurationUnit.steps)

    @classmethod
    def epochs(cls, epochs: int) -> "Duration":
        """
        Define a duration from a number of epochs.
        """
        return cls(value=epochs, unit=DurationUnit.epochs)

    @classmethod
    def tokens(cls, tokens: int) -> "Duration":
        """
        Define a duration from a number of tokens.
        """
        return cls(value=tokens, unit=DurationUnit.tokens)

    def due(self, *, step: int, tokens: int, epoch: int) -> bool:
        """
        Check if the duration is due.
        """
        if self.unit == DurationUnit.steps:
            return step >= self.value
        elif self.unit == DurationUnit.tokens:
            return tokens >= self.value
        elif self.unit == DurationUnit.epochs:
            return epoch > self.value
        else:
            raise NotImplementedError


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


class ReduceType(StrEnum):
    """
    An enumeration of the allowed ways to reduce a metric across ranks.
    """

    mean = "mean"
    """
    Average across the process group.
    """

    sum = "sum"
    """
    Add across the process group.
    """

    max = "max"
    """
    Take the max across the process group.
    """

    l2_norm = "l2_norm"
    """
    For metrics that are computed as L2 norms on each rank, this will correctly reduce the norm
    across the process group to produce the global L2 norm.
    """


def get_inputs_for_loss(
    batch: Dict[str, Any], logits: torch.Tensor, label_ignore_index: int = -100
) -> Tuple[torch.Tensor, torch.Tensor]:
    # shape: (batch_size, seq_len - 1, vocab_size)
    logits_for_loss = logits[..., :-1, :].contiguous()
    # shape: (batch_size * (seq_len - 1), vocab_size)
    logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))

    # shape: (batch_size, seq_len - 1)
    labels = batch.get("labels", get_labels(batch, label_ignore_index=label_ignore_index))
    # shape: (batch_size * (seq_len - 1),)
    labels = labels.view(-1)

    return logits, labels
