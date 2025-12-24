from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from olmo_core.config import Config

if TYPE_CHECKING:
    from .train_module import TrainModule


@dataclass
class TrainModuleConfig(Config):
    @abstractmethod
    def build(self, *args, **kargs) -> "TrainModule":
        raise NotImplementedError
