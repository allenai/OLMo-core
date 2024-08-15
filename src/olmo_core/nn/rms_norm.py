import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    RMS norm, a simplified layer norm implementation
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
    ):
        super().__init__()
        self.normalized_shape = (size,)
        self.eps = eps
        self.full_precision = full_precision
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=init_device))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=init_device))
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


class FusedRMSNorm(nn.Module):
    """
    A fused version of :class:`RMSNorm`.

    .. warning::
        This requires `flash-attn <https://github.com/Dao-AILab/flash-attention>`_ to be installed.
    """

    def __init__(
        self,
        *,
        size: int,
        eps: float = 1e-5,
        bias: bool = True,
        full_precision: bool = True,
        init_device: str = "cpu",
    ):
        from flash_attn.ops.triton.layer_norm import rms_norm_fn  # type: ignore

        super().__init__()
        self.eps = eps
        self.full_precision = full_precision
        self.weight = nn.Parameter(torch.ones(size, device=init_device))
        if bias:
            self.bias = nn.Parameter(torch.zeros(size, device=init_device))
        else:
            self.register_parameter("bias", None)
        self._rms_norm_fn = rms_norm_fn

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS norm.

        :param x: The input.
        """
        og_dtype = x.dtype
        if self.full_precision:
            x = x.float()
        return self._rms_norm_fn(
            x,
            self.weight.type_as(x),
            None if self.bias is None else self.bias.type_as(x),
            eps=self.eps,
        ).to(og_dtype)
