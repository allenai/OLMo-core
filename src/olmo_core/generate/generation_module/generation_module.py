import logging
from abc import ABCMeta, abstractmethod
from typing import Any

import torch.distributed as dist
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.stateful import Stateful

log = logging.getLogger(__name__)


class GenerationModule(Stateful, metaclass=ABCMeta):
    @property
    def dp_process_group(self) -> dist.ProcessGroup | None:
        """
        Should return the data parallel process group if it's anything other than the default
        process group.
        """
        return None

    def state_dict_to_load(self, metadata: Metadata) -> dict[str, Any]:
        del metadata
        return self.state_dict()

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load a state dict.
        """
        raise NotImplementedError
