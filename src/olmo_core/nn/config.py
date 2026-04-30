from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import torch.nn as nn

from ..config import Config


@dataclass
class ModuleConfig(Config, metaclass=ABCMeta):
    """Base configuration class for neural network modules."""

    @abstractmethod
    def build(self, *args, **kwargs) -> nn.Module:
        """Build the corresponding module."""
        raise NotImplementedError


@dataclass
class ModelConfig(ModuleConfig, metaclass=ABCMeta):
    """Base configuration class for neural network models."""

    @property
    @abstractmethod
    def num_params(self) -> int:
        """The total number of parameters in the model once built."""
        raise NotImplementedError

    @property
    def num_active_params(self) -> int:
        """The number of active parameters in the model once built."""
        return self.num_params

    @property
    @abstractmethod
    def num_non_embedding_params(self) -> int:
        """The total number of non-embedding parameters in the model once built."""
        raise NotImplementedError

    @property
    def num_active_non_embedding_params(self) -> int:
        """The number of active non-embedding parameters in the model once built."""
        return self.num_non_embedding_params
