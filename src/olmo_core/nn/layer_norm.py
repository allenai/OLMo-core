from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config, DType, StrEnum

__all__ = ["LayerNormType", "LayerNormConfig", "LayerNorm", "RMSNorm", "FusedRMSNorm"]


class LayerNormType(StrEnum):
    """
    An enumeration of the different layer norm implementations.

    - "default" ➡️ :class:`LayerNorm`
    - "rms" ➡️ :class:`RMSNorm`
    - "fused_rms" ➡️ :class:`FusedRMSNorm`
    """

    default = "default"
    rms = "rms"
    fused_rms = "fused_rms"


@dataclass
class LayerNormConfig(Config):
    """
    A config for conveniently building any one of the different layer norm classes.

    See :class:`LayerNorm` for a description of the parameters.
    """

    name: LayerNormType = LayerNormType.default
    """
    - "default" ➡️ :class:`LayerNorm`
    - "rms" ➡️ :class:`RMSNorm`
    - "fused_rms" ➡️ :class:`FusedRMSNorm`
    """
    eps: float = 1e-5
    elementwise_affine: bool = True
    bias: bool = True
    full_precision: bool = True
    dtype: DType = DType.float32

    def build(self, size: int, init_device: str = "cpu") -> "LayerNorm":
        """
        Construct the corresponding LayerNorm class.

        See :class:`LayerNorm` for a description of the parameters.
        """
        kwargs = self.as_dict(exclude_none=True)
        kwargs.pop("name")
        dtype = kwargs["dtype"].as_pt()
        kwargs.update(
            dict(
                size=size,
                init_device=init_device,
                dtype=dtype,
            )
        )

        if self.name == LayerNormType.default:
            return LayerNorm(**kwargs)
        elif self.name == LayerNormType.rms:
            return RMSNorm(**kwargs)
        elif self.name == LayerNormType.fused_rms:
            return FusedRMSNorm(**kwargs)
        else:
            raise NotImplementedError(self.name)


class LayerNorm(nn.Module):
    """
    Layer normalization.

    .. seealso::
        - :class:`RMSNorm`
        - :class:`FusedRMSNorm`

    :param size: The hidden size / dimensionality of the input.
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

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer norm.

        :param x: The input.
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
            )

            return x.to(og_dtype)


class RMSNorm(LayerNorm):
    """
    RMS norm, a simplified layer norm implementation.

    .. seealso::
        - :class:`LayerNorm`
        - :class:`FusedRMSNorm`
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS norm.

        :param x: The input.
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

            return x.to(og_dtype)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        og_dtype = x.dtype
        if self.full_precision:
            x = x.float()
        return self._rms_norm_fn(
            x,
            self.weight.type_as(x),
            None if self.bias is None else self.bias.type_as(x),
            eps=self.eps,
        ).to(og_dtype)
