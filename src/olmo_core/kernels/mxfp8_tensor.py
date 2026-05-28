from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ..doc_utils import beta_feature
from .mxfp8_utils import (
    _quantize_to_mxfp8_torch,
    dequantize_rows_from_mxfp8,
    quantize_rows_to_mxfp8,
)


def _dequantize_rows_from_mxfp8_torch(
    qdata: torch.Tensor,
    scales: torch.Tensor,
    *,
    block_size: int = 32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    m, k = qdata.shape
    q_blocks = qdata.reshape(m, k // block_size, block_size)
    x = q_blocks.to(torch.float32) * scales.to(torch.float32).unsqueeze(-1)
    return x.reshape(m, k).to(out_dtype)


@beta_feature
class OlmoMXFP8Tensor(torch.Tensor):
    """
    Prototype tensor subclass for rowwise MXFP8 payloads.

    This is intentionally narrow:
    - rank-2 tensors only
    - e4m3 qdata plus e8m0 row/block scales
    - wrapper dtype is the original high-precision dtype

    The goal is to test whether qdata + scales can travel as one Tensor-like
    value through custom autograd boundaries. Generic torch ops conservatively
    dequantize to a normal high-precision tensor.
    """

    qdata: torch.Tensor
    scales: torch.Tensor
    block_size: int
    orig_dtype: torch.dtype
    prefer_triton: bool

    @staticmethod
    def __new__(
        cls,
        qdata: torch.Tensor,
        scales: torch.Tensor,
        *,
        block_size: int = 32,
        orig_dtype: torch.dtype = torch.bfloat16,
        prefer_triton: bool = True,
    ) -> "OlmoMXFP8Tensor":
        if qdata.ndim != 2 or scales.ndim != 2:
            raise ValueError(
                "OlmoMXFP8Tensor expects rank-2 qdata/scales, "
                f"got qdata={tuple(qdata.shape)} scales={tuple(scales.shape)}"
            )
        if block_size != 32:
            raise ValueError(f"Only block_size=32 is supported (got {block_size})")
        if qdata.dtype != torch.float8_e4m3fn:
            raise ValueError(f"qdata must be float8_e4m3fn, got {qdata.dtype}")
        if scales.dtype != torch.float8_e8m0fnu:
            raise ValueError(f"scales must be float8_e8m0fnu, got {scales.dtype}")
        expected_scales_shape = (qdata.shape[0], qdata.shape[1] // block_size)
        if tuple(scales.shape) != expected_scales_shape:
            raise ValueError(
                "scales shape mismatch: "
                f"expected {expected_scales_shape}, got {tuple(scales.shape)}"
            )
        self = torch.Tensor._make_wrapper_subclass(
            cls,
            qdata.shape,
            strides=qdata.stride(),
            storage_offset=qdata.storage_offset(),
            dtype=orig_dtype,
            layout=qdata.layout,
            device=qdata.device,
            requires_grad=False,
        )
        self.qdata = qdata
        self.scales = scales
        self.block_size = int(block_size)
        self.orig_dtype = orig_dtype
        self.prefer_triton = bool(prefer_triton)
        return self

    @staticmethod
    def from_hp(
        x: torch.Tensor,
        *,
        block_size: int = 32,
        prefer_triton: bool = True,
    ) -> "OlmoMXFP8Tensor":
        if prefer_triton:
            qdata, scales = quantize_rows_to_mxfp8(x, block_size=block_size)
        else:
            qdata, scales = _quantize_to_mxfp8_torch(x, block_size=block_size)
        return OlmoMXFP8Tensor(
            qdata,
            scales,
            block_size=block_size,
            orig_dtype=x.dtype,
            prefer_triton=prefer_triton,
        )

    @staticmethod
    def from_qdata_scales(
        qdata: torch.Tensor,
        scales: torch.Tensor,
        *,
        block_size: int = 32,
        orig_dtype: torch.dtype = torch.bfloat16,
        prefer_triton: bool = True,
    ) -> "OlmoMXFP8Tensor":
        return OlmoMXFP8Tensor(
            qdata,
            scales,
            block_size=block_size,
            orig_dtype=orig_dtype,
            prefer_triton=prefer_triton,
        )

    def dequantize(self, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        resolved_out_dtype = self.orig_dtype if out_dtype is None else out_dtype
        if not self.prefer_triton:
            return _dequantize_rows_from_mxfp8_torch(
                self.qdata,
                self.scales,
                block_size=self.block_size,
                out_dtype=resolved_out_dtype,
            )
        return dequantize_rows_from_mxfp8(
            self.qdata,
            self.scales,
            block_size=self.block_size,
            out_dtype=resolved_out_dtype,
        )

    def as_scaled_grouped_mm_prequantized_lhs(self):
        from .scaled_grouped_mm import ScaledGroupedMMPrequantizedLHS

        return ScaledGroupedMMPrequantizedLHS(
            mat_a_q=self.qdata,
            scale_a=self.scales,
            mat_a_shape=tuple(self.shape),
            scales_are_blocked=False,
        )

    def __repr__(self) -> str:
        return (
            "OlmoMXFP8Tensor("
            f"shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"qdata_dtype={self.qdata.dtype}, scales_dtype={self.scales.dtype}, "
            f"block_size={self.block_size}, prefer_triton={self.prefer_triton})"
        )

    def __tensor_flatten__(self) -> Tuple[list[str], Dict[str, Any]]:
        tensor_names = ["qdata", "scales"]
        return tensor_names, {
            "block_size": self.block_size,
            "orig_dtype": self.orig_dtype,
            "prefer_triton": self.prefer_triton,
        }

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: Dict[str, torch.Tensor],
        meta: Dict[str, Any],
        outer_size: torch.Size,
        outer_stride: Tuple[int, ...],
    ) -> "OlmoMXFP8Tensor":
        del outer_size, outer_stride
        return OlmoMXFP8Tensor(
            inner_tensors["qdata"],
            inner_tensors["scales"],
            block_size=int(meta["block_size"]),
            orig_dtype=meta["orig_dtype"],
            prefer_triton=bool(meta["prefer_triton"]),
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        del types
        kwargs = {} if kwargs is None else kwargs

        def unwrap(value):
            if isinstance(value, OlmoMXFP8Tensor):
                return value.dequantize()
            if isinstance(value, tuple):
                return tuple(unwrap(item) for item in value)
            if isinstance(value, list):
                return [unwrap(item) for item in value]
            if isinstance(value, dict):
                return {key: unwrap(item) for key, item in value.items()}
            return value

        return func(*unwrap(args), **unwrap(kwargs))


__all__ = ["OlmoMXFP8Tensor"]
