from typing import Tuple

import torch


def cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cast a tensor to FP8 with scaling factors calculated from contiguous blocks of 128 elements
    in last dimension of the tensor. The size of the last dimension must be divisible by 128.

    :returns: The FP8 tensor and its scaling factors.
    """
    assert x.dim() >= 2
    assert x.size(-1) % 128 == 0
    in_shape = x.shape
    x = x.view(*in_shape[:-1], -1, 128)
    x_amax = x.abs().float().amax(dim=-1).view(*in_shape[:-1], -1).clamp(1e-4)
    x = (x * (448.0 / x_amax.unsqueeze(-1))).to(torch.float8_e4m3fn)
    return x.view(in_shape), x_amax / 448.0


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cast a tensor with shape ``(*, m, n)`` to FP8 with scaling factors calculated from interior
    128 x 128 blocks.

    :returns: The FP8 tensor and its scaling factors.
    """
    assert x.dim() >= 2
    assert x.size(-1) % 128 == 0 and x.size(-2) % 128 == 0
    in_shape = x.shape
    m, n = in_shape[-2:]
    x = x.view(*in_shape[:-2], m // 128, 128, n // 128, 128)
    x_amax = x.abs().float().amax(dim=(-3, -1), keepdim=True).clamp(1e-4)
    x = (x * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x.view(in_shape).contiguous(), (x_amax / 448.0).view(*in_shape[:-2], m // 128, n // 128)
