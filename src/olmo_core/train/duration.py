from dataclasses import dataclass

from olmo_core.config import StrEnum


class DurationUnit(StrEnum):
    steps = "steps"
    epochs = "epochs"
    tokens = "tokens"


@dataclass
class Duration:
    value: int
    unit: DurationUnit
