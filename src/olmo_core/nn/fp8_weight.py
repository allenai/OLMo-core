from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import torch

from olmo_core.kernels import (
    ScaledGroupedMMPrequantizedRHS,
    prequantize_scaled_grouped_mm_rhs,
)


@dataclass(frozen=True)
class FP8WeightCacheSpec:
    """
    Describes one cached FP8 RHS layout derived from a logical weight.

    The transform is intentionally explicit. MoE routed experts, shared
    experts, and future attention projections can each provide the layouts
    their kernels need without the store guessing from parameter names.
    """

    name: str
    transform: Callable[[torch.Tensor], torch.Tensor]


class FP8WeightStore:
    """
    Generic logical weight store for fp32-main -> FP8-model training.

    This object presents just enough tensor-like surface for the fused optimizer
    while physically storing named FP8 RHS caches plus BF16 logical gradients.
    A normal parameter can remain attached as a shape/device anchor until the
    optimizer has initialized fp32 main params, after which its backing storage
    may be released.
    """

    def __init__(
        self,
        *,
        logical_name: str,
        logical_shape: tuple[int, ...],
        cache_specs: Sequence[FP8WeightCacheSpec] = (),
        anchor_param: Optional[torch.nn.Parameter] = None,
        optimizer_enabled: bool = False,
        prequantized_rhs: Optional[ScaledGroupedMMPrequantizedRHS] = None,
        prequantized_rhs_for_dgrad: Optional[ScaledGroupedMMPrequantizedRHS] = None,
    ) -> None:
        self.logical_name = logical_name
        self.logical_shape = logical_shape
        self.cache_specs = tuple(cache_specs)
        self.anchor_param = anchor_param
        self.optimizer_enabled = optimizer_enabled
        self.cache_values: dict[str, ScaledGroupedMMPrequantizedRHS] = {}
        if prequantized_rhs is not None:
            self.cache_values["rhs"] = prequantized_rhs
        if prequantized_rhs_for_dgrad is not None:
            self.cache_values["rhs_for_dgrad"] = prequantized_rhs_for_dgrad
        self.weight_versions: Optional[tuple[int, ...]] = None
        self.grad_bf16: Optional[torch.Tensor] = None
        self.main_grad_fp32: Optional[torch.Tensor] = None
        self.accumulate_wgrad_in_fp32 = False
        self.anchor_storage_released = False

    @property
    def requires_grad(self) -> bool:
        return True

    @property
    def shape(self) -> tuple[int, ...]:
        return self.logical_shape

    @property
    def ndim(self) -> int:
        return len(self.logical_shape)

    @property
    def dtype(self) -> torch.dtype:
        if self.anchor_param is not None:
            return self.anchor_param.dtype
        if self.grad_bf16 is not None:
            return self.grad_bf16.dtype
        return torch.bfloat16

    @property
    def device(self) -> torch.device:
        if self.anchor_param is not None:
            return self.anchor_param.device
        if self.grad_bf16 is not None:
            return self.grad_bf16.device
        for cache in self.cache_values.values():
            return cache.mat_b_q.device
        return torch.device("cpu")

    @property
    def data(self) -> torch.Tensor:
        if self.anchor_param is None:
            raise RuntimeError(
                f"FP8 weight store '{self.logical_name}' has no anchor tensor to expose as data"
            )
        return self.anchor_param.data

    @property
    def grad(self) -> Optional[torch.Tensor]:
        return self.grad_bf16

    @property
    def _main_grad_fp32(self) -> Optional[torch.Tensor]:
        return self.main_grad_fp32

    @_main_grad_fp32.setter
    def _main_grad_fp32(self, value: Optional[torch.Tensor]) -> None:
        self.main_grad_fp32 = value

    @property
    def prequantized_rhs(self) -> Optional[ScaledGroupedMMPrequantizedRHS]:
        return self.cache_values.get("rhs")

    @prequantized_rhs.setter
    def prequantized_rhs(self, value: Optional[ScaledGroupedMMPrequantizedRHS]) -> None:
        self._set_cache("rhs", value)

    @property
    def prequantized_rhs_for_dgrad(self) -> Optional[ScaledGroupedMMPrequantizedRHS]:
        return self.cache_values.get("rhs_for_dgrad")

    @prequantized_rhs_for_dgrad.setter
    def prequantized_rhs_for_dgrad(self, value: Optional[ScaledGroupedMMPrequantizedRHS]) -> None:
        self._set_cache("rhs_for_dgrad", value)

    def _set_cache(self, name: str, value: Optional[ScaledGroupedMMPrequantizedRHS]) -> None:
        if value is None:
            self.cache_values.pop(name, None)
            return
        self.cache_values[name] = value

    def numel(self) -> int:
        out = 1
        for dim in self.logical_shape:
            out *= dim
        return out

    def element_size(self) -> int:
        return torch.tensor([], dtype=self.dtype).element_size()

    def type(self) -> str:
        device_prefix = "torch.cuda" if self.device.type == "cuda" else "torch"
        if self.dtype == torch.bfloat16:
            return f"{device_prefix}.BFloat16Tensor"
        if self.dtype == torch.float16:
            return f"{device_prefix}.HalfTensor"
        if self.dtype == torch.float32:
            return f"{device_prefix}.FloatTensor"
        return str(self.dtype)

    def invalidate(self) -> None:
        self.cache_values.clear()
        self.weight_versions = None

    def zero_grad(self, set_to_none: bool = True) -> None:
        if set_to_none:
            self.grad_bf16 = None
            self.main_grad_fp32 = None
            return
        if self.grad_bf16 is not None:
            self.grad_bf16.zero_()
        if self.main_grad_fp32 is not None:
            self.main_grad_fp32.zero_()

    def set_main_grad_to_none(self) -> None:
        self.main_grad_fp32 = None

    def replace_wgrad(self, grad: torch.Tensor) -> None:
        if self.accumulate_wgrad_in_fp32:
            self.main_grad_fp32 = grad.detach().to(torch.float32).contiguous()
            self.grad_bf16 = None
        else:
            self.grad_bf16 = grad.detach().to(torch.bfloat16).contiguous()
            self.main_grad_fp32 = None

    @torch.no_grad()
    def release_anchor_storage(self) -> None:
        if self.anchor_param is None or self.anchor_storage_released:
            return
        missing = [spec.name for spec in self.cache_specs if spec.name not in self.cache_values]
        if missing:
            raise RuntimeError(
                f"Cannot release bf16 anchor storage for FP8 weight store '{self.logical_name}' "
                f"before its FP8 caches are initialized: missing {missing}"
            )
        base = torch.empty(
            (1,),
            dtype=self.anchor_param.dtype,
            device=self.anchor_param.device,
        )
        placeholder = torch.as_strided(
            base,
            size=self.logical_shape,
            stride=(0,) * len(self.logical_shape),
        )
        self.anchor_param.requires_grad_(False)
        self.anchor_param.grad = None
        if hasattr(self.anchor_param, "_main_grad_fp32"):
            self.anchor_param._main_grad_fp32 = None  # type: ignore[attr-defined]
        self.anchor_param.data = placeholder
        self.anchor_storage_released = True

    @torch.no_grad()
    def refresh_from_logical_weight(
        self,
        logical_weight: torch.Tensor,
        *,
        version_tensors: Sequence[torch.Tensor] = (),
        update_anchor: bool = False,
    ) -> None:
        if tuple(logical_weight.shape) != self.logical_shape:
            raise ValueError(
                f"FP8 weight store '{self.logical_name}' expected logical shape "
                f"{self.logical_shape}, got {tuple(logical_weight.shape)}"
            )
        if update_anchor and self.anchor_param is not None and not self.anchor_storage_released:
            self.anchor_param.data.copy_(logical_weight.to(dtype=self.anchor_param.dtype))
        if not self.cache_specs:
            raise RuntimeError(f"FP8 weight store '{self.logical_name}' has no cache specs")
        self.refresh_from_tensors(
            cache_tensors={spec.name: spec.transform(logical_weight) for spec in self.cache_specs},
            version_tensors=version_tensors,
        )

    @torch.no_grad()
    def refresh_from_tensors(
        self,
        *,
        cache_tensors: Optional[Mapping[str, torch.Tensor]] = None,
        rhs: Optional[torch.Tensor] = None,
        rhs_for_dgrad: Optional[torch.Tensor] = None,
        version_tensors: Sequence[torch.Tensor] = (),
    ) -> None:
        tensors: dict[str, torch.Tensor] = {}
        if cache_tensors is not None:
            tensors.update(cache_tensors)
        if rhs is not None:
            tensors["rhs"] = rhs
        if rhs_for_dgrad is not None:
            tensors["rhs_for_dgrad"] = rhs_for_dgrad
        if not tensors:
            raise ValueError("refresh_from_tensors requires at least one cache tensor")
        for name, tensor in tensors.items():
            self.cache_values[name] = prequantize_scaled_grouped_mm_rhs(
                tensor,
                check_mat_b_version=False,
            )
        self.weight_versions = tuple(int(t._version) for t in version_tensors)

    def require_cache(self, name: str) -> ScaledGroupedMMPrequantizedRHS:
        cache = self.cache_values.get(name)
        if cache is None:
            raise RuntimeError(
                f"FP8 weight store '{self.logical_name}' cache '{name}' is not initialized"
            )
        return cache

    def require_prequantized_rhs(self) -> ScaledGroupedMMPrequantizedRHS:
        return self.require_cache("rhs")

    def require_prequantized_rhs_for_dgrad(self) -> ScaledGroupedMMPrequantizedRHS:
        return self.require_cache("rhs_for_dgrad")

    def iter_prequantized_caches(self):
        return self.cache_values.values()

    def accumulate_wgrad(self, grad: torch.Tensor) -> None:
        if self.accumulate_wgrad_in_fp32:
            grad_fp32 = grad.detach().to(torch.float32)
            if self.main_grad_fp32 is None:
                self.main_grad_fp32 = grad_fp32.contiguous()
            else:
                self.main_grad_fp32.add_(grad_fp32)
            self.grad_bf16 = None
        else:
            grad_bf16 = grad.detach().to(torch.bfloat16)
            if self.grad_bf16 is None:
                self.grad_bf16 = grad_bf16.contiguous()
            else:
                self.grad_bf16.add_(grad_bf16)
            self.main_grad_fp32 = None


__all__ = ["FP8WeightCacheSpec", "FP8WeightStore"]
