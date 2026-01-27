from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement

from olmo_core.config import Registrable

from ..buffer_cache import BufferCache
from ..config import ModuleConfig
from .ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)


class SequenceMixerBase(nn.Module):
    """
    Base class for sequence mixing modules (e.g. attention, recurrent, convolution, etc.).
    """

    @abstractmethod
    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        raise NotImplementedError

    @abstractmethod
    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def num_flops_per_token(self, seq_len: int) -> int:
        raise NotImplementedError


SeqMixer = TypeVar("SeqMixer", bound=SequenceMixerBase)


@dataclass
class SequenceMixerConfig(ModuleConfig, Registrable, Generic[SeqMixer]):
    def num_params(self, d_model: int) -> int:
        raise NotImplementedError

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> SeqMixer:
        raise NotImplementedError
