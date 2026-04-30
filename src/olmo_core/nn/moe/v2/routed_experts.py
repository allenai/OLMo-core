"""
Notes on torch grouped_mm backend behavior (PyTorch 2.10):

- torch.nn.functional.grouped_mm(...) calls aten::_grouped_mm.
- CUDA path in aten/native/cuda/GroupedBlas.cpp chooses:
  1) Fast path: at::cuda::detail::bf16bf16_grouped_mm(...)
  2) Fallback: _grouped_mm_fallback(...)

Fast path conditions (CUDA):
- mat_a dtype == bf16
- mat_b dtype == bf16
- out_dtype (or default) == bf16
- _scaled_mm_allowed_device(sm90_only=true, sm100_only=true) is true
  (in 2.10 this means device major == 9 or 10 only)
- grouped-mm CUTLASS kernels are built
  (!USE_ROCM, !Windows, CUDA_VERSION >= 12.0)

Fallback behavior:
- Uses offs.cpu() and loops over groups with mm_out/bmm_out.
- This can introduce D2H sync for device offs.

Implication:
- On CUDA devices outside the fastpath gate (e.g. major 12 in PyTorch 2.10),
  grouped_mm falls back even if offs is on GPU.
"""

import weakref
from dataclasses import dataclass
from typing import Optional, Tuple, cast

import grouped_gemm  # type: ignore
import grouped_gemm.ops
import nvtx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.config import Config, DType
from olmo_core.kernels import (
    ScaledGroupedMMPrequantizedLHS,
    ScaledGroupedMMPrequantizedRHS,
)
from olmo_core.kernels import grouped_mm as grouped_mm_with_buffers
from olmo_core.kernels import prequantize_scaled_grouped_mm_rhs, scaled_grouped_mm_q
from olmo_core.kernels.mxfp8_utils import (
    dequantize_rows_from_mxfp8,
    swiglu_quantize_rows_from_mxfp8,
    swiglu_quantize_rows_to_mxfp8,
)

from .fp8 import MoERowwiseFP8Config, normalize_rowwise_fp8_config


def _debug_is_inf_or_nan(x):
    return torch.logical_or(~torch.isfinite(x), torch.isnan(x))


def _debug_get_row_indices_for_nan_or_inf(x):
    return torch.where(_debug_is_inf_or_nan(x).any(dim=-1))[0]


def _debug_get_row_indices_for_nan_or_inf_before_end(x, end):
    naninf_row_indices = _debug_get_row_indices_for_nan_or_inf(x)
    return naninf_row_indices[naninf_row_indices < end]


class _SwiGLUQuantizeRowsMXFP8Autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, up_gate: torch.Tensor, block_size: int):  # type: ignore[override]
        h, h_q, h_scales = swiglu_quantize_rows_to_mxfp8(
            up_gate,
            block_size=int(block_size),
        )
        if up_gate.requires_grad:
            ctx.save_for_backward(up_gate)
        else:
            ctx.save_for_backward()
        return h, h_q, h_scales

    @staticmethod
    def backward(ctx, grad_h: torch.Tensor, grad_h_q: torch.Tensor, grad_h_scales: torch.Tensor):  # type: ignore[override]
        del grad_h_q, grad_h_scales
        if not ctx.needs_input_grad[0]:
            return None, None
        if grad_h is None:
            up_gate = ctx.saved_tensors[0]
            return torch.zeros_like(up_gate), None

        (up_gate,) = ctx.saved_tensors
        hidden = up_gate.shape[-1] // 2
        up = up_gate[:, :hidden]
        gate = up_gate[:, hidden:]

        gate_f32 = gate.to(torch.float32)
        grad_h_f32 = grad_h.to(torch.float32)
        up_f32 = up.to(torch.float32)

        sig = torch.sigmoid(gate_f32)
        silu_gate = gate_f32 * sig
        dsilu = sig * (1.0 + gate_f32 * (1.0 - sig))

        grad_up = grad_h_f32 * silu_gate
        grad_gate = grad_h_f32 * up_f32 * dsilu
        grad_up_gate = torch.cat((grad_up, grad_gate), dim=-1).to(dtype=up_gate.dtype)
        return grad_up_gate, None


class _SwiGLUQuantizeRowsFromMXFP8Autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, up_gate_q: torch.Tensor, up_gate_scales: torch.Tensor, block_size: int):  # type: ignore[override]
        h, h_q, h_scales = swiglu_quantize_rows_from_mxfp8(
            up_gate_q,
            up_gate_scales,
            block_size=int(block_size),
        )
        if up_gate_q.requires_grad:
            ctx.save_for_backward(up_gate_q, up_gate_scales)
            ctx.block_size = int(block_size)
        else:
            ctx.save_for_backward()
            ctx.block_size = int(block_size)
        return h, h_q, h_scales

    @staticmethod
    def backward(ctx, grad_h: torch.Tensor, grad_h_q: torch.Tensor, grad_h_scales: torch.Tensor):  # type: ignore[override]
        del grad_h_q, grad_h_scales
        if not ctx.needs_input_grad[0]:
            return None, None, None

        up_gate_q, up_gate_scales = ctx.saved_tensors
        if grad_h is None:
            return (
                torch.zeros(
                    up_gate_q.shape,
                    dtype=torch.bfloat16,
                    device=up_gate_q.device,
                ),
                None,
                None,
            )

        up_gate = dequantize_rows_from_mxfp8(
            up_gate_q,
            up_gate_scales,
            block_size=int(ctx.block_size),
            out_dtype=torch.bfloat16,
        )
        hidden = up_gate.shape[-1] // 2
        up = up_gate[:, :hidden]
        gate = up_gate[:, hidden:]

        gate_f32 = gate.to(torch.float32)
        grad_h_f32 = grad_h.to(torch.float32)
        up_f32 = up.to(torch.float32)

        sig = torch.sigmoid(gate_f32)
        silu_gate = gate_f32 * sig
        dsilu = sig * (1.0 + gate_f32 * (1.0 - sig))

        grad_up = grad_h_f32 * silu_gate
        grad_gate = grad_h_f32 * up_f32 * dsilu
        grad_up_gate = torch.cat((grad_up, grad_gate), dim=-1).to(dtype=torch.bfloat16)
        return grad_up_gate, None, None


@torch.compiler.disable
def gmm_no_compile(a, b, batch_sizes, trans_b=False):
    return grouped_gemm.ops.gmm(a, b, batch_sizes, trans_b)


def gmm(
    a: torch.Tensor,
    b: torch.Tensor,
    batch_sizes: torch.Tensor,
    trans_b: bool = False,
    out: Optional[torch.Tensor] = None,
    input_grad_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if use_torch_grouped_mm():
        # torch.nn.functional.grouped_mm has no trans_b argument.
        # It expects mat_b to be (num_groups, K, N), so we transpose when
        # emulating grouped_gemm(..., trans_b=True).
        b_grouped_mm = b.transpose(1, 2) if trans_b else b
        offs = torch.cumsum(batch_sizes.to(dtype=torch.int32), dim=0, dtype=torch.int32)
        if out is not None or input_grad_out is not None:
            out_tensor = grouped_mm_with_buffers(
                a,
                b_grouped_mm,
                offs=offs,
                out=out,
                input_grad_out=input_grad_out,
            )
        else:
            out_tensor = F.grouped_mm(a, b_grouped_mm, offs=offs)
        # WARNING: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grouped_mm.html
        # "offs[i] marks the end of group i and offs[-1] must be strictly less than the total length of that operand’s sliced dimension"
        # so padded positions are always necessary? I think "strictly less than" is a documentation mistake

        # BIG NOTE: grouped_mm's returned value containes uninitialized values for the padded positions (pos > offsets[-1]), which can be NaN and may cause later ops to produce NaN in valid positions.
        # if _debug_get_row_indices_for_nan_or_inf_before_end(out, batch_sizes.sum()).numel() > 0:
        #     raise RuntimeError(f"NaN or Inf detected in grouped_mm output in valid tokens. batch_sizes={batch_sizes} offs={offs} out={out}")
        return out_tensor

    if out is not None or input_grad_out is not None:
        raise RuntimeError("gmm(out=..., input_grad_out=...) requires torch grouped_mm backend")
    return gmm_no_compile(a, b, batch_sizes, trans_b)


# if env variable OLMO_USE_TORCH_GROUPED_MM is set, use its value to determine whether to use torch grouped_mm;
import os

env_val = os.getenv("OLMO_USE_TORCH_GROUPED_MM")
USE_TORCH_GROUPED_MM: Optional[bool]
if env_val is not None:
    if env_val.lower() in ("1", "true", "yes"):
        USE_TORCH_GROUPED_MM = True
    elif env_val.lower() in ("0", "false", "no"):
        USE_TORCH_GROUPED_MM = False
    else:
        raise ValueError(
            f"Invalid value for OLMO_USE_TORCH_GROUPED_MM: {env_val}. Expected one of (1, 0, true, false, yes, no)."
        )
else:
    # otherwise, use feature detection and version gate.
    USE_TORCH_GROUPED_MM = None


def use_torch_grouped_mm():
    global USE_TORCH_GROUPED_MM
    if USE_TORCH_GROUPED_MM is not None:
        return USE_TORCH_GROUPED_MM

    torch_version = torch.__version__.split("+")[0]  # strip local build suffix, e.g. +cu128
    try:
        major_str, minor_str, *_ = torch_version.split(".")
        major, minor = int(major_str), int(minor_str)
        meets_version_gate = major > 2 or (major == 2 and minor >= 10)
    except (ValueError, TypeError):
        # Fall back to feature detection on unusual version strings.
        meets_version_gate = hasattr(F, "grouped_mm")

    # grouped_mm was added in torch 2.10; hasattr keeps this robust to local builds.
    USE_TORCH_GROUPED_MM = meets_version_gate and hasattr(F, "grouped_mm")
    return USE_TORCH_GROUPED_MM


REQUIRES_HOST_SIDE_SPLIT_SIZES = None  # cache the result of whether host-side split sizes are required, since it does not change during runtime and checking it requires parsing torch version every time.


def requires_host_side_split_sizes():
    # read from cache if available
    global REQUIRES_HOST_SIDE_SPLIT_SIZES
    if REQUIRES_HOST_SIDE_SPLIT_SIZES is not None:
        return REQUIRES_HOST_SIDE_SPLIT_SIZES

    # grouped_gemm cublas mode requires host-side split sizes, grouped_mm does not.
    REQUIRES_HOST_SIDE_SPLIT_SIZES = not use_torch_grouped_mm()

    return REQUIRES_HOST_SIDE_SPLIT_SIZES


@dataclass
class RoutedExpertsConfig(Config):
    """Configuration for routed experts in a MoE block."""

    # Input (and output) dimension of the experts
    d_model: int

    # Hidden (intermediate) dimension of the experts
    hidden_size: int

    # Number of routed experts
    num_experts: int

    # Whether to use bias in the experts
    bias: bool

    # default dtype for the experts
    dtype: DType

    # Optional FP8 config used by rowwise EP no-sync path.
    rowwise_fp8: Optional[MoERowwiseFP8Config] = None

    def build(
        self,
        init_device: str = "cpu",
    ) -> "RoutedExperts":
        kwargs = self.as_dict()
        return RoutedExperts(init_device=init_device, **kwargs)

    def num_params(self) -> int:
        """
        The number of params that the module will have once built.

        :param d_model: The model dimensionality.
        """

        params = 3 * self.d_model * self.hidden_size  # up, gate, down
        if self.bias:
            params += 2 * self.hidden_size  # up and gate bias
            params += self.d_model  # down bias

        params *= self.num_experts  # for each expert

        return params

    def num_active_params(self, top_k: int) -> int:
        """
        The number of params that the module will have once built, given the top_k experts.

        :param top_k: The number of experts to use.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if top_k > self.num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot be greater than num_experts ({self.num_experts})"
            )

        params = 3 * self.d_model * self.hidden_size  # up, gate, down
        if self.bias:
            params += 2 * self.hidden_size  # up and gate bias
            params += self.d_model  # down bias

        params *= top_k  # for each expert

        return params


class RoutedExperts(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_size: int,
        num_experts: int,
        bias: bool,
        dtype: DType,
        rowwise_fp8: Optional[MoERowwiseFP8Config] = None,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        assert bias == False, "Routed experts do not support bias for now."
        self.w_up_gate = nn.Parameter(
            torch.empty(
                num_experts, 2 * hidden_size, d_model, dtype=dtype.as_pt(), device=init_device
            ),
        )

        self.w_down = nn.Parameter(
            torch.empty(num_experts, hidden_size, d_model, dtype=dtype.as_pt(), device=init_device),
        )
        owner_ref = weakref.ref(self)
        self.w_up_gate._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
        self.w_down._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]

        # assume no ep in init
        self.num_local_experts: int = num_experts
        self.ep_dim: int = 1
        self.ep_rank: int = 0
        self.rowwise_fp8 = normalize_rowwise_fp8_config(rowwise_fp8)
        self._rowwise_fp8_checked = False
        self._rowwise_fp8_up_gate_prequant: Optional[ScaledGroupedMMPrequantizedRHS] = None
        self._rowwise_fp8_down_prequant: Optional[ScaledGroupedMMPrequantizedRHS] = None
        self._rowwise_fp8_up_gate_prequant_t: Optional[ScaledGroupedMMPrequantizedRHS] = None
        self._rowwise_fp8_down_prequant_t: Optional[ScaledGroupedMMPrequantizedRHS] = None
        self._rowwise_fp8_weight_versions: Optional[tuple[int, int]] = None

    def invalidate_rowwise_fp8_cache(self) -> None:
        self._rowwise_fp8_up_gate_prequant = None
        self._rowwise_fp8_down_prequant = None
        self._rowwise_fp8_up_gate_prequant_t = None
        self._rowwise_fp8_down_prequant_t = None
        self._rowwise_fp8_weight_versions = None

    @torch.no_grad()
    def refresh_rowwise_fp8_cache(self) -> None:
        cfg = self.rowwise_fp8
        if cfg is None or not cfg.enabled:
            self.invalidate_rowwise_fp8_cache()
            return
        if self.w_up_gate.device.type != "cuda" or self.w_down.device.type != "cuda":
            self.invalidate_rowwise_fp8_cache()
            return
        up_gate_rhs = self.w_up_gate.transpose(1, 2)
        self._rowwise_fp8_up_gate_prequant = prequantize_scaled_grouped_mm_rhs(up_gate_rhs)
        self._rowwise_fp8_up_gate_prequant_t = prequantize_scaled_grouped_mm_rhs(self.w_up_gate)
        self._rowwise_fp8_down_prequant = prequantize_scaled_grouped_mm_rhs(self.w_down)
        self._rowwise_fp8_down_prequant_t = prequantize_scaled_grouped_mm_rhs(
            self.w_down.transpose(1, 2)
        )
        self._rowwise_fp8_weight_versions = (
            int(self.w_up_gate._version),
            int(self.w_down._version),
        )

    def _maybe_refresh_rowwise_fp8_cache(self) -> None:
        cfg = self.rowwise_fp8
        if cfg is None or not cfg.enabled:
            return
        versions = (int(self.w_up_gate._version), int(self.w_down._version))
        if (
            self._rowwise_fp8_up_gate_prequant is None
            or self._rowwise_fp8_down_prequant is None
            or self._rowwise_fp8_weight_versions != versions
        ):
            self.refresh_rowwise_fp8_cache()

    def _use_rowwise_fp8(self, x: torch.Tensor, *, enabled: bool) -> bool:
        if not enabled:
            return False
        cfg = self.rowwise_fp8
        if cfg is None or not cfg.enabled:
            return False
        if x.device.type != "cuda":
            return False
        if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            return False
        if not self._rowwise_fp8_checked:
            cfg.assert_runtime_supported()
            self._rowwise_fp8_checked = True
        return True

    def _forward_rowwise_fp8(
        self,
        x: torch.Tensor,
        batch_size_per_expert_tensor: torch.Tensor,
        *,
        prequantized_input_q: Optional[torch.Tensor] = None,
        prequantized_input_scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        cfg = self.rowwise_fp8
        assert cfg is not None

        self._maybe_refresh_rowwise_fp8_cache()
        offs = torch.cumsum(
            batch_size_per_expert_tensor.to(dtype=torch.int32), dim=0, dtype=torch.int32
        )
        prequantized_lhs = None
        if prequantized_input_q is not None and prequantized_input_scales is not None:
            prequantized_lhs = ScaledGroupedMMPrequantizedLHS(
                mat_a_q=prequantized_input_q,
                scale_a=prequantized_input_scales,
                mat_a_shape=cast(Tuple[int, int], tuple(x.shape)),
                scales_are_blocked=False,
            )

        up_kwargs = dict(
            offs=offs,
            use_fast_accum=cfg.use_fast_accum,
            prequantized_lhs=prequantized_lhs,
            prequantized_rhs=self._rowwise_fp8_up_gate_prequant,
        )
        if self._rowwise_fp8_up_gate_prequant_t is not None:
            up_kwargs["prequantized_rhs_for_dgrad"] = self._rowwise_fp8_up_gate_prequant_t
        up_gate = scaled_grouped_mm_q(
            x,
            self.w_up_gate.transpose(1, 2),
            **up_kwargs,
        )
        up_gate = cast(torch.Tensor, up_gate)
        h, h_q, h_scales = _SwiGLUQuantizeRowsMXFP8Autograd.apply(
            up_gate,
            int(cfg.block_size),
        )
        h_prequantized_lhs = ScaledGroupedMMPrequantizedLHS(
            mat_a_q=h_q,
            scale_a=h_scales,
            mat_a_shape=cast(Tuple[int, int], tuple(h.shape)),
            scales_are_blocked=False,
        )

        down_kwargs = dict(
            offs=offs,
            use_fast_accum=cfg.use_fast_accum,
            prequantized_lhs=h_prequantized_lhs,
            prequantized_rhs=self._rowwise_fp8_down_prequant,
        )
        if self._rowwise_fp8_down_prequant_t is not None:
            down_kwargs["prequantized_rhs_for_dgrad"] = self._rowwise_fp8_down_prequant_t
        down = scaled_grouped_mm_q(
            h,
            self.w_down,
            **down_kwargs,
        )

        return cast(torch.Tensor, down)

    # @torch.compiler.disable(recursive=False)
    @nvtx.annotate("RoutedExperts.forward", color="blue")
    def forward(
        self,
        x: torch.Tensor,
        batch_size_per_expert: torch.Tensor,
        *,
        down_proj_out: Optional[torch.Tensor] = None,
        up_proj_input_grad_out: Optional[torch.Tensor] = None,
        use_rowwise_fp8: bool = False,
        rowwise_fp8_input_q: Optional[torch.Tensor] = None,
        rowwise_fp8_input_scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        `batch_size_per_expert` specifies the number of tokens in x for each expert.
        """

        assert isinstance(
            batch_size_per_expert, torch.Tensor
        ), "only accept Tensor for batch_size_per_expert"

        if requires_host_side_split_sizes():
            # CPU-side split sizes are required by grouped_gemm cublas mode.
            # grouped_gemm CUTLASS mode can accept device-side split sizes, but it is slow.
            # Always assume grouped_gemm runs in cublas mode.
            assert (
                batch_size_per_expert.device.type == "cpu"
            ), "batch_size_per_expert must be on cpu"
            batch_size_per_expert_tensor = batch_size_per_expert.to(
                dtype=torch.int64
            )  # int64 required for grouped_gemm
        else:
            assert (
                batch_size_per_expert.device.type == "cuda"
            ), "batch_size_per_expert expected to be on GPU"
            # grouped_mm expects int32 offsets derived from split sizes.
            batch_size_per_expert_tensor = batch_size_per_expert.to(dtype=torch.int32)

        if x.numel() == 0:
            if down_proj_out is not None:
                return down_proj_out
            return x

        if self._use_rowwise_fp8(x, enabled=use_rowwise_fp8):
            return self._forward_rowwise_fp8(
                x,
                batch_size_per_expert_tensor,
                prequantized_input_q=rowwise_fp8_input_q,
                prequantized_input_scales=rowwise_fp8_input_scales,
            )

        w_up_gate = self.w_up_gate  # (E, H, 2D)
        w_down = self.w_down  # (E, H, D)

        # up + gate projection
        up_gate = gmm(
            x,
            w_up_gate,
            batch_size_per_expert_tensor,
            trans_b=True,
            input_grad_out=up_proj_input_grad_out,
        )  # -> (BS, 2H)

        up_gate = cast(torch.Tensor, up_gate)  # ensure type is Tensor

        h = self.chunk_and_activate(up_gate)  # -> (BS, H)

        # down projection
        down = gmm(
            h,
            w_down,
            batch_size_per_expert_tensor,
            trans_b=False,
            out=down_proj_out,
        )  # -> (BS, D)

        return cast(torch.Tensor, down)  # ensure type is Tensor

    def act_and_down(
        self, up_gate: torch.Tensor, batch_size_per_expert_tensor: torch.Tensor
    ) -> torch.Tensor:
        # swiglu + down projection
        # so that it apply activation checkpointing if needed
        h = self.chunk_and_activate(up_gate)  # -> (BS, H)

        down = gmm(h, self.w_down, batch_size_per_expert_tensor, trans_b=False)  # -> (BS, H)
        return down

    def chunk_and_activate(self, up_gate: torch.Tensor) -> torch.Tensor:
        # NOTE: this might include pad tokens, but I decide not to exlude pads:
        # 1 chunk_and_activate is cheap relative to MoE GEMMs, even at 2x capacity.
        # 2 Excluding tail pads without sync is hard with stock PyTorch ops;
        #   true compute-skipping usually needs dynamic slicing (.item() sync) or a custom kernel.
        # 3 Extra pad-handling ops can cost more than just doing SiLU on full buffer.
        up, gate = up_gate.chunk(2, dim=-1)
        h = up * F.silu(gate)  # -> (BS, H)
        return h

    def apply_ep(self, ep_mesh: DeviceMesh, **kwargs):
        # shard dim 0 to ep_mp, replicate on ep_dp mesh
        self.ep_mesh = ep_mesh["ep_dp", "ep_mp"]
        # with torch.no_grad():  # just to avoid tracking the rebind below
        self.ep_dim = ep_mesh["ep_mp"].size()
        self.ep_rank = ep_mesh["ep_mp"].get_local_rank()

        assert (
            self.num_experts % self.ep_dim == 0
        ), "num_experts must be divisible by the number of expert partitions"
        self.num_local_experts = self.num_experts // self.ep_dim

        self.w_up_gate = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                2 * self.hidden_size,
                self.d_model,
                dtype=self.w_up_gate.dtype,
                device=self.w_up_gate.device,
            ),
        )

        self.w_down = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                self.d_model,
                dtype=self.w_down.dtype,
                device=self.w_down.device,
            ),
        )
        owner_ref = weakref.ref(self)
        self.w_up_gate._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
        self.w_down._moe_rowwise_fp8_cache_owner = owner_ref  # type: ignore[attr-defined]
        self.invalidate_rowwise_fp8_cache()

        self._ep_sharded = True

    def extra_repr(self):
        return f"num_experts={self.num_experts}, hidden_size={self.hidden_size}, d_model={self.d_model}"
