from dataclasses import dataclass

from olmo_core.config import Config


@dataclass
class ExpertParallelConfig(Config):
    """
    Configuration class for expert parallelism (EP).
    """

    degree: int
    """
    The EP degree.
    """
