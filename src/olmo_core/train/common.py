from dataclasses import dataclass

from ..config import StrEnum


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
    The trainer will attempt to load a checkpoint from the save folder or load path (in that order)
    but will train from scratch if no checkoint is found.
    """

    always = "always"
    """
    The trainer will attempt to load a checkpoint from the save folder or load path (in that order)
    and raise an error if no checkpoint is found.
    """

    never = "never"
    """
    The trainer will never load a checkpoint even if one exists in the save folder or load path.
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
