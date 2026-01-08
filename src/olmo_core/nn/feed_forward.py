import functools
import math
from dataclasses import dataclass
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.tensor.placement_types import Placement, Replicate

from ..config import DType, StrEnum
from ..doc_utils import beta_feature
from ..exceptions import OLMoConfigurationError
from .config import ModuleConfig
from .functional import l2_normalize
from .utils import get_tp_wrappers

__all__ = [
    "ActivationFunction",
    "FeedForwardType",
    "FeedForwardConfig",
    "FeedForward",
    "NormalizedFeedForward",
]


class ActivationFunction(StrEnum):
    """
    An enumeration of the supported activation functions for feed-forward modules.
    """

    silu = "silu"
    """
    SiLU/Swish activation function, used for SwiGLU.
    """

    gelu_tanh = "gelu_tanh"
    """
    GELU with tanh approximation, used for GeGLU.
    """

    def get_fn(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self == ActivationFunction.silu:
            return F.silu
        elif self == ActivationFunction.gelu_tanh:
            return functools.partial(F.gelu, approximate="tanh")
        else:
            raise NotImplementedError(self)


class FeedForwardType(StrEnum):
    """
    An enumeration of the different feed-forward / MLP implementations.
    """

    default = "default"
    """
    ➡️ :class:`FeedForward`
    """

    normalized = "normalized"
    """
    ➡️ :class:`NormalizedFeedForward`
    """


@dataclass
class FeedForwardConfig(ModuleConfig):
    """
    A config for building :class:`FeedForward` modules.
    """

    hidden_size: int
    name: FeedForwardType = FeedForwardType.default
    """
    The name of the implementation.
    """
    bias: Optional[bool] = None
    dtype: Optional[DType] = None
    activation: ActivationFunction = ActivationFunction.silu
    """
    The activation function to use. See :class:`ActivationFunction` for options.
    """

    def num_params(self, d_model: int) -> int:
        """
        The number of params that the module will have once built.

        :param d_model: The model dimensionality.
        """
        bias = self.bias if self.bias is not None else self.name != FeedForwardType.normalized

        params = 0

        params += 3 * d_model * self.hidden_size
        if bias:
            params += 2 * self.hidden_size + d_model

        # w1 + w3 scaling factors
        if self.name == FeedForwardType.normalized:
            params += 2 * self.hidden_size

        return params

    def build(
        self, d_model: int, *, dtype: Optional[torch.dtype] = None, init_device: str = "cpu"
    ) -> "FeedForward":
        """
        Build the corresponding feed-forward module.

        :param d_model: The model dimensionality.
        :param init_device: The device initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True)
        kwargs.pop("name")
        kwargs.update(d_model=d_model, init_device=init_device)
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype.as_pt()
        elif dtype is not None:
            kwargs["dtype"] = dtype

        try:
            if self.name == FeedForwardType.default:
                return FeedForward(**kwargs)
            elif self.name == FeedForwardType.normalized:
                activation = kwargs.pop("activation", ActivationFunction.silu)
                if activation != ActivationFunction.silu:
                    raise OLMoConfigurationError(
                        f"NormalizedFeedForward only supports 'silu' activation, got '{activation}'"
                    )
                return NormalizedFeedForward(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


class FeedForward(nn.Module):
    """
    Basic feed-forward module with gated activation (SwiGLU or GeGLU).
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        activation: Union[str, ActivationFunction] = ActivationFunction.silu,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        if isinstance(activation, str):
            activation = ActivationFunction(activation)
        self.activation = activation
        self.activation_fn = activation.get_fn()
        self.w1 = nn.Linear(d_model, hidden_size, bias=bias, dtype=dtype, device=init_device)
        self.w2 = nn.Linear(hidden_size, d_model, bias=bias, dtype=dtype, device=init_device)
        self.w3 = nn.Linear(d_model, hidden_size, bias=bias, dtype=dtype, device=init_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the feed-forward on the input ``x``.

        :param x: The input of shape ``(*, d_model)``.
        """
        return self.w2(self.activation_fn(self.w1(x)) * self.w3(x))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        rowwise_parallel, colwise_parallel, prepare_module_input = get_tp_wrappers(
            float8_enabled=float8_enabled
        )

        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan=prepare_module_input(
                input_layouts=None if input_layout is None else (input_layout,),
                desired_input_layouts=(Replicate(),),
            ),
        )

        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan={
                "w1": colwise_parallel(),
                "w2": rowwise_parallel(
                    output_layouts=output_layout, use_local_output=use_local_output
                ),
                "w3": colwise_parallel(),
            },
        )

    def num_flops_per_token(self, seq_len: int) -> int:
        del seq_len
        # 6 FLOPs per parameter (2 ops * 3 for forward+backward)
        return 6 * sum(p.numel() for p in self.parameters())


@beta_feature
class NormalizedFeedForward(FeedForward):
    """
    An nGPT feed-forward implementation.
    """

    def __init__(
        self,
        *,
        d_model: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__(
            d_model=d_model,
            hidden_size=hidden_size,
            dtype=dtype,
            init_device=init_device,
            bias=False,
        )
        self.sw_init_value = 1.0
        self.sw_init_scaling = 1.0
        self.sw1 = torch.nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=init_device))
        self.sw3 = torch.nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=init_device))
        self.sqrt_d_model = math.sqrt(d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.sw1)
        nn.init.ones_(self.sw3)
        with torch.no_grad():
            self.sw1.mul_(self.sw_init_scaling)
            self.sw3.mul_(self.sw_init_scaling)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sw1 = self.sw1 * ((self.sw_init_value / self.sw_init_scaling) * self.sqrt_d_model)
        sw3 = self.sw3 * (self.sw_init_value / self.sw_init_scaling)
        return self.w2(F.silu(sw1 * self.w1(x)) * (sw3 * self.w3(x)))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled

        raise NotImplementedError(
            "TP is not implemented yet for the normalized feed-forward variant"
        )

    @torch.no_grad()
    def normalize_matrices(self):
        """
        Normalize the weights in all matrices. This should be called after each optimizer step, which
        the :class:`~olmo_core.train.train_module.TransformerTrainModule` will handle for you.
        """
        self._normalize_matrix(self.w1.weight)
        self._normalize_matrix(self.w2.weight, dim=0)
        self._normalize_matrix(self.w3.weight)

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(l2_normalize(w, dim=dim))
