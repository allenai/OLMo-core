from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement

from olmo_core.config import Registrable

from ..buffer_cache import BufferCache
from ..config import ModuleConfig
from .ring import RingContextParallelStyle, UlyssesContextParallelStyle

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod


class SequenceMixer(nn.Module):
    """
    Base class for sequence mixing modules (e.g. attention, recurrent, convolution, etc.).
    """

    @abstractmethod
    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Placement | None = None,
        output_layout: Placement | None = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        raise NotImplementedError

    @abstractmethod
    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: RingContextParallelStyle | None = None,
        uly: UlyssesContextParallelStyle | None = None,
    ):
        raise NotImplementedError

    @abstractmethod
    def num_flops_per_token(self, seq_len: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def init_weights(
        self,
        *,
        init_method: "InitMethod",
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: torch.Generator | None = None,
    ) -> None:
        raise NotImplementedError


SeqMixer = TypeVar("SeqMixer", bound=SequenceMixer)


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
        cache: BufferCache | None = None,
    ) -> SeqMixer:
        raise NotImplementedError
