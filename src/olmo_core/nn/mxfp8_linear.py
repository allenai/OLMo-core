from __future__ import annotations

from typing import Iterator, Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor

from olmo_core.kernels.mxfp8_linear import (
    prequantize_scaled_mm_rhs,
    scaled_mm_mxfp8_fp8_weight,
)

from .fp8_weight import FP8WeightCacheSpec, FP8WeightStore


def _linear_rhs(weight: Tensor) -> Tensor:
    return weight.transpose(0, 1)


def _linear_rhs_for_dgrad(weight: Tensor) -> Tensor:
    return weight


_MXFP8_LINEAR_CACHE_SPECS = (
    FP8WeightCacheSpec("rhs", _linear_rhs),
    FP8WeightCacheSpec("rhs_for_dgrad", _linear_rhs_for_dgrad),
)


class MXFP8Linear(nn.Linear):
    """
    Linear layer with MXFP8 GEMMs and optimizer-owned logical weights.

    The module keeps `weight` only as an initialization/checkpoint/cache anchor.
    `named_fp8_weight_stores()` exposes the logical weight to the fused
    optimizer so it can own the fp32 main parameter and refresh this layer's
    MXFP8 RHS caches.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        *,
        enabled: bool = True,
        save_wgrad_input: Literal["mxfp8", "bf16"] = "mxfp8",
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.enabled = enabled
        if save_wgrad_input not in ("mxfp8", "bf16"):
            raise ValueError(
                "MXFP8Linear save_wgrad_input must be 'mxfp8' or 'bf16', "
                f"got {save_wgrad_input!r}"
            )
        self.save_wgrad_input = save_wgrad_input
        self._mxfp8_weight = FP8WeightStore(
            logical_name="weight",
            logical_shape=tuple(self.weight.shape),
            cache_specs=_MXFP8_LINEAR_CACHE_SPECS,
            anchor_param=self.weight,
            optimizer_enabled=True,
            prequantizer=prequantize_scaled_mm_rhs,
        )
        self.weight.requires_grad_(False)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        enabled: bool = True,
        save_wgrad_input: Literal["mxfp8", "bf16"] = "mxfp8",
    ) -> "MXFP8Linear":
        new_linear = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            enabled=enabled,
            save_wgrad_input=save_wgrad_input,
        )
        new_linear.weight = linear.weight
        new_linear.bias = linear.bias
        new_linear._mxfp8_weight.anchor_param = new_linear.weight
        new_linear.weight.requires_grad_(False)
        return new_linear

    @property
    def fp8_weight_store(self) -> FP8WeightStore:
        self._sync_fp8_weight_store()
        return self._mxfp8_weight

    def _sync_fp8_weight_store(self) -> None:
        self._mxfp8_weight.anchor_param = self.weight
        self._mxfp8_weight.logical_shape = tuple(self.weight.shape)
        self._mxfp8_weight.optimizer_enabled = True
        self.weight.requires_grad_(False)

    def named_fp8_weight_stores(self) -> Iterator[tuple[str, FP8WeightStore]]:
        self._sync_fp8_weight_store()
        yield "weight", self._mxfp8_weight

    def zero_mxfp8_weight_grads(self, set_to_none: bool = True) -> None:
        self._mxfp8_weight.zero_grad(set_to_none=set_to_none)

    def set_mxfp8_weight_main_grads_to_none(self) -> None:
        self._mxfp8_weight.set_main_grad_to_none()

    def zero_grad(self, set_to_none: bool = True) -> None:
        super().zero_grad(set_to_none=set_to_none)
        self.zero_mxfp8_weight_grads(set_to_none=set_to_none)

    def disable_mxfp8_anchor_grads(self) -> None:
        self._sync_fp8_weight_store()
        self.weight.grad = None
        if hasattr(self.weight, "_main_grad_fp32"):
            self.weight._main_grad_fp32 = None  # type: ignore[attr-defined]

    def release_mxfp8_anchor_storage(self) -> None:
        self._sync_fp8_weight_store()
        self._mxfp8_weight.release_anchor_storage()

    def invalidate_mxfp8_cache(self) -> None:
        self._mxfp8_weight.invalidate()

    @torch.no_grad()
    def refresh_mxfp8_cache(self) -> None:
        self._sync_fp8_weight_store()
        if not self.enabled:
            self.invalidate_mxfp8_cache()
            return
        if self.weight.device.type != "cuda":
            self.invalidate_mxfp8_cache()
            return
        if self._mxfp8_weight.anchor_storage_released:
            if (
                self._mxfp8_weight.prequantized_rhs is None
                or self._mxfp8_weight.prequantized_rhs_for_dgrad is None
            ):
                raise RuntimeError(
                    "Cannot refresh MXFP8Linear caches from released anchor storage. "
                    "The optimizer must refresh MXFP8 stores directly from fp32 main params."
                )
            return
        self._mxfp8_weight.refresh_from_logical_weight(
            self.weight,
            version_tensors=(self.weight,),
        )

    def _use_mxfp8(self, x: Tensor) -> bool:
        if not self.enabled:
            return False
        if x.device.type != "cuda":
            return False
        if x.dtype != torch.bfloat16:
            return False
        return True

    def _assert_supported_mxfp8_shape(self, x_2d: Tensor) -> None:
        assert x_2d.dtype == torch.bfloat16, (
            "MXFP8Linear currently requires bf16 activations, "
            f"got {x_2d.dtype}"
        )
        assert self.in_features % 32 == 0, (
            "MXFP8Linear requires in_features divisible by 32, "
            f"got {self.in_features}"
        )
        assert self.out_features % 32 == 0, (
            "MXFP8Linear training requires out_features divisible by 32, "
            f"got {self.out_features}"
        )
        if torch.is_grad_enabled():
            assert x_2d.shape[0] % 32 == 0, (
                "MXFP8Linear wgrad reduction dim must be divisible by 32, "
                f"got flattened input shape {tuple(x_2d.shape)}"
            )

    def _cache_is_current(self) -> bool:
        if (
            self._mxfp8_weight.prequantized_rhs is None
            or self._mxfp8_weight.prequantized_rhs_for_dgrad is None
        ):
            return False
        if self._mxfp8_weight.anchor_storage_released:
            return True
        return self._mxfp8_weight.weight_versions == (int(self.weight._version),)

    def forward(self, x: Tensor) -> Tensor:
        if not self._use_mxfp8(x):
            raise RuntimeError("MXFP8Linear requires CUDA bf16 activations for forward")

        x_shape = x.shape
        x_2d = x.reshape(-1, x_shape[-1])
        self._assert_supported_mxfp8_shape(x_2d)

        self._sync_fp8_weight_store()
        if not self._cache_is_current():
            self.refresh_mxfp8_cache()

        out = scaled_mm_mxfp8_fp8_weight(
            x_2d,
            prequantized_rhs=self._mxfp8_weight.require_prequantized_rhs(),
            prequantized_rhs_for_dgrad=self._mxfp8_weight.require_prequantized_rhs_for_dgrad(),
            wgrad_sink=self._mxfp8_weight if torch.is_grad_enabled() else None,
            save_wgrad_input_as_mxfp8=self.save_wgrad_input == "mxfp8",
        )
        out = out.reshape(*x_shape[:-1], self.out_features)
        if self.bias is not None:
            out = out + self.bias.to(dtype=out.dtype)
        return out

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return f"{base}, enabled={self.enabled}, save_wgrad_input={self.save_wgrad_input}"


__all__ = ["MXFP8Linear"]
