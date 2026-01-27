from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DType, StrEnum
from ..exceptions import OLMoConfigurationError
from .config import ModuleConfig
from .functional import l2_normalize

__all__ = [
    "LayerNormType",
    "LayerNormConfig",
    "LayerNorm",
    "RMSNorm",
    "CuTeRMSNorm",
    "FusedRMSNorm",
    "L2Norm",
]


class LayerNormType(StrEnum):
    """
    An enumeration of the different layer norm implementations.
    """

    default = "default"
    """
    ➡️ :class:`LayerNorm`
    """
    rms = "rms"
    """
    ➡️ :class:`RMSNorm`
    """
    cute_rms = "cute_rms"
    """
    ➡️ :class:`CuTeRMSNorm`
    """
    fused_rms = "fused_rms"
    """
    ➡️ :class:`FusedRMSNorm`
    """
    l2_norm = "l2_norm"
    """
    ➡️ :class:`L2Norm`
    """


@dataclass
class LayerNormConfig(ModuleConfig):
    """
    A config for conveniently building any one of the different layer norm classes.

    See the :class:`LayerNorm` subclasses to learn which fields are valid for each implementation.
    """

    name: LayerNormType = LayerNormType.default
    """
    The name of the implementation.
    """
    eps: Optional[float] = None
    elementwise_affine: Optional[bool] = None
    bias: Optional[bool] = None
    full_precision: Optional[bool] = None
    dtype: Optional[DType] = None

    def num_params(self, size: int) -> int:
        """
        The number of parameters in the module once built.

        :param size: The size of the input along the dimension to be normalized.
        """
        elementwise_affine = (
            self.elementwise_affine
            if self.elementwise_affine is not None
            else self.name != LayerNormType.l2_norm
        )
        bias = self.bias if self.bias is not None else self.name != LayerNormType.l2_norm
        ln_params = 0
        if elementwise_affine:
            ln_params += size
            if bias:
                ln_params += size
        return ln_params

    def build(self, size: int, init_device: str = "cpu") -> "LayerNorm":
        """
        Construct the corresponding LayerNorm class.

        :param size: The size of the input along the dimension to be normalized.
        :param init_device: The device initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True)
        kwargs.pop("name")
        if (dtype := kwargs.pop("dtype", None)) is not None:
            kwargs.update(dtype=dtype.as_pt())

        try:
            if self.name == LayerNormType.default:
                return LayerNorm(size=size, init_device=init_device, **kwargs)
            elif self.name == LayerNormType.rms:
                return RMSNorm(size=size, init_device=init_device, **kwargs)
            elif self.name == LayerNormType.cute_rms:
                return CuTeRMSNorm(size=size, init_device=init_device, **kwargs)
            elif self.name == LayerNormType.fused_rms:
                return FusedRMSNorm(size=size, init_device=init_device, **kwargs)
            elif self.name == LayerNormType.l2_norm:
                return L2Norm(size=size, **kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


class LayerNorm(nn.Module):
    """
    Layer normalization.

    :param size: The size of the input along the dimension to be normalized.
    :param eps: The epsilon used for numerical stability.
    :param elementwise_affine: Whether to include an element-wise affine transform.
    :param bias: Whether the element-wise affine should include an element-wise bias.
        Ignored if ``elementwise_affine=False``.
    :param full_precision: Force the operation to run in full precision regardless of the input
        data type.
    :param dtype: The default data type to use for the weight and bias in the element-wise affine.
        If ``full_precision=False`` it can be useful to set this to the expected input data type.
        Ignored if ``elementwise_affine=False``.
    :param init_device: The device used when initializing the element-wise weight/bias.
    """

    def __init__(
        self,
        *,
        size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        full_precision: bool = True,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.normalized_shape = (size,)
        self.eps = eps
        self.full_precision = full_precision
        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.ones(self.normalized_shape, dtype=dtype, device=init_device)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.zeros(self.normalized_shape, dtype=dtype, device=init_device)
                )
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def extra_repr(self):
        if self.weight is not None and self.bias is not None:
            return f"{tuple(self.weight.shape)}, bias=True, eps={self.eps}"
        elif self.weight is not None:
            return f"{tuple(self.weight.shape)}, eps={self.eps}"
        else:
            return f"eps={self.eps}"

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply layer norm.

        :param x: The input.
        :param residual: Optional residual tensor to add to the normalized output.
            If provided, the result is norm(x) + residual.
        """
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype

            if self.full_precision:
                x = x.float()

            x = F.layer_norm(
                x,
                self.normalized_shape,
                weight=None if self.weight is None else self.weight.type_as(x),
                bias=None if self.bias is None else self.bias.type_as(x),
                eps=self.eps,
            ).to(og_dtype)

            if residual is not None:
                x = x + residual
            return x


class RMSNorm(LayerNorm):
    """
    RMSNorm, a simplified layer norm implementation.
    """

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply RMSNorm.

        :param x: The input.
        :param residual: Optional residual tensor to add to the normalized output.
            If provided, the result is rms_norm(x) + residual.
        """
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype

            if self.full_precision:
                x = x.float()

            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)

            if self.weight is not None:
                if self.bias is not None:
                    x = self.weight.type_as(x) * x + self.bias.type_as(x)
                else:
                    x = self.weight.type_as(x) * x

            if residual is not None:
                x = x + residual

            return x.to(og_dtype)


class CuTeRMSNorm(RMSNorm):
    """
    A CuTe-based implementation from the QuACK library.

    .. warning::
        This requires `quack <https://github.com/Dao-AILab/quack>`_ to be installed.

    """

    def __init__(
        self,
        *,
        size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        full_precision: bool = True,
        init_device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        from quack import rmsnorm as rms_norm_fn  # type: ignore

        if not full_precision:
            # the CUTE kernel always casts to full precision internally
            raise NotImplementedError(
                f"Currently only 'full_precision=True' is supported with '{self.__class__.__name__}'"
            )

        super().__init__(
            size=size,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            full_precision=full_precision,
            dtype=dtype,
            init_device=init_device,
        )
        self._rms_norm_fn = rms_norm_fn

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply CuTe-based RMSNorm.

        :param x: The input.
        :param residual: Optional residual tensor to add to the normalized output.
            If provided, the result is rms_norm(x) + residual.
        """
        result = self._rms_norm_fn(
            x,
            weight=None if self.weight is None else self.weight.type_as(x),
            bias=None if self.bias is None else self.bias.type_as(x),
            residual=residual,
            eps=self.eps,
        )
        return result.to(x.dtype)

class FusedRMSNorm(RMSNorm):
    """
    A "fused" triton-based implementation of :class:`RMSNorm`.

    .. warning::
        This requires `flash-attn <https://github.com/Dao-AILab/flash-attention>`_ to be installed.

    .. warning::
        Currently only ``elementwise_affine=True`` is supported.
    """

    def __init__(
        self,
        *,
        size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        full_precision: bool = True,
        init_device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        from flash_attn.ops.triton.layer_norm import rms_norm_fn  # type: ignore

        if not elementwise_affine:
            raise NotImplementedError(
                f"Currently only 'elementwise_affine=True' is supported with '{self.__class__.__name__}'"
            )
        if not full_precision:
            # the triton kernel always casts to full precision internally
            raise NotImplementedError(
                f"Currently only 'full_precision=True' is supported with '{self.__class__.__name__}'"
            )

        super().__init__(
            size=size,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            full_precision=full_precision,
            dtype=dtype,
            init_device=init_device,
        )
        self._rms_norm_fn = rms_norm_fn

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply fused Triton-based RMSNorm.

        :param x: The input.
        :param residual: Optional residual tensor to add to the normalized output.
            If provided, the result is rms_norm(x) + residual.
        """
        og_dtype = x.dtype
        if self.full_precision:
            x = x.float()
        result = self._rms_norm_fn(
            x,
            self.weight.type_as(x),
            None if self.bias is None else self.bias.type_as(x),
            eps=self.eps,
        ).to(og_dtype)
        if residual is not None:
            result = result + residual
        return result


class L2Norm(LayerNorm):
    """
    A variant of layer norm that just normalizes the last dimension of the input by its L2 norm,
    as done in nGPT.

    :param size: The size of the input along the dimension to be normalized.
    """

    def __init__(
        self,
        *,
        size: int,
    ):
        super().__init__(size=size, elementwise_affine=False, bias=False)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        """
        Apply L2 normalization.

        :param x: The input.
        :param residual: Optional residual tensor to add to the normalized output.
            If provided, the result is l2_norm(x) + residual.
        """
        result = l2_normalize(x)
        if residual is not None:
            result = result + residual
        return result
