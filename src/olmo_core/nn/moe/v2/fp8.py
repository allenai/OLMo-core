from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple

import nvtx
import torch

from olmo_core.config import Config, StrEnum
from olmo_core.kernels import (
    prequantize_scaled_grouped_mm_rhs,
    scaled_grouped_mm_q,
    scaled_grouped_mm_q_fp8_weight,
)

if TYPE_CHECKING:
    from .block import MoEFusedV2TransformerBlock


class MoERowwiseFP8ScaleMode(StrEnum):
    rceil = "rceil"


@dataclass
class MoERowwiseFP8Config(Config):
    enabled: bool = True
    block_size: int = 32
    scale_mode: MoERowwiseFP8ScaleMode = MoERowwiseFP8ScaleMode.rceil
    use_fast_accum: bool = True
    fp8_only_params: bool = True

    def validate(self) -> None:
        if self.block_size != 32:
            raise ValueError(f"Only block_size=32 is supported for MoE rowwise FP8 (got {self.block_size})")

    def assert_runtime_supported(self) -> None:
        self.validate()
        if not torch.cuda.is_available():
            raise RuntimeError("MoE rowwise FP8 requires CUDA")

        major, _minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        if major < 10:
            raise RuntimeError(
                "MoE rowwise FP8 is fail-closed on pre-SM100 GPUs in this implementation. "
                f"Detected compute capability major={major}."
            )

        if not hasattr(torch.nn.functional, "scaled_grouped_mm"):
            raise RuntimeError("torch.nn.functional.scaled_grouped_mm is required for MoE rowwise FP8")


def normalize_rowwise_fp8_config(
    value: Optional[MoERowwiseFP8Config | Mapping[str, Any]],
) -> Optional[MoERowwiseFP8Config]:
    if value is None:
        return None
    if isinstance(value, MoERowwiseFP8Config):
        return value
    if isinstance(value, Mapping):
        return MoERowwiseFP8Config.from_dict(dict(value))
    raise TypeError(
        "rowwise_fp8 must be MoERowwiseFP8Config, mapping/dict, or None "
        f"(got {type(value)!r})"
    )


def reset_shared_rowwise_fp8_cache(block: MoEFusedV2TransformerBlock) -> None:
    block._shared_rowwise_fp8_up_prequant = None
    block._shared_rowwise_fp8_down_prequant = None
    block._shared_rowwise_fp8_up_prequant_t = None
    block._shared_rowwise_fp8_down_prequant_t = None
    block._shared_rowwise_fp8_weight_versions = None
    up_weight = getattr(block, "_shared_rowwise_fp8_up_gate_weight", None)
    down_weight = getattr(block, "_shared_rowwise_fp8_down_weight", None)
    if up_weight is not None:
        up_weight.invalidate()
    if down_weight is not None:
        down_weight.invalidate()


def _rowwise_fp8_enabled(cfg: Optional[MoERowwiseFP8Config]) -> bool:
    return cfg is not None and cfg.enabled


def invalidate_rowwise_fp8_cache(block: MoEFusedV2TransformerBlock) -> None:
    if block.routed_experts is not None:
        block.routed_experts.invalidate_rowwise_fp8_cache()
    reset_shared_rowwise_fp8_cache(block)


@torch.no_grad()
def refresh_shared_rowwise_fp8_cache(block: MoEFusedV2TransformerBlock) -> None:
    cfg = block.rowwise_fp8
    if not _rowwise_fp8_enabled(cfg):
        reset_shared_rowwise_fp8_cache(block)
        return
    if block.shared_experts is None:
        reset_shared_rowwise_fp8_cache(block)
        return
    sync_shared_weights = getattr(block, "_sync_shared_rowwise_fp8_weight_anchors", None)
    if sync_shared_weights is not None:
        sync_shared_weights()
    up_weight = getattr(block, "_shared_rowwise_fp8_up_gate_weight", None)
    down_weight = getattr(block, "_shared_rowwise_fp8_down_weight", None)
    if (
        cfg.fp8_only_params
        and up_weight is not None
        and down_weight is not None
        and (up_weight.anchor_storage_released or down_weight.anchor_storage_released)
    ):
        if (
            up_weight.prequantized_rhs is None
            or up_weight.prequantized_rhs_for_dgrad is None
            or down_weight.prequantized_rhs is None
            or down_weight.prequantized_rhs_for_dgrad is None
        ):
            raise RuntimeError(
                "Cannot refresh rowwise FP8 shared expert caches from released bf16 anchors. "
                "The optimizer must refresh FP8 stores directly from fp32 main params."
            )
        block._shared_rowwise_fp8_up_prequant = up_weight.prequantized_rhs
        block._shared_rowwise_fp8_up_prequant_t = up_weight.prequantized_rhs_for_dgrad
        block._shared_rowwise_fp8_down_prequant = down_weight.prequantized_rhs
        block._shared_rowwise_fp8_down_prequant_t = down_weight.prequantized_rhs_for_dgrad
        block._shared_rowwise_fp8_weight_versions = None
        return
    if block.shared_experts.w_up_gate.device.type != "cuda":
        reset_shared_rowwise_fp8_cache(block)
        return

    with nvtx.annotate("moe_rowwise_fp8_param_refresh_shared_weight_prequant"):
        if up_weight is not None and down_weight is not None:
            up_weight.refresh_from_logical_weight(
                block.shared_experts.w_up_gate,
                version_tensors=(block.shared_experts.w_up_gate,),
            )
            down_weight.refresh_from_logical_weight(
                block.shared_experts.w_down,
                version_tensors=(block.shared_experts.w_down,),
            )
            block._shared_rowwise_fp8_up_prequant = up_weight.prequantized_rhs
            block._shared_rowwise_fp8_up_prequant_t = up_weight.prequantized_rhs_for_dgrad
            block._shared_rowwise_fp8_down_prequant = down_weight.prequantized_rhs
            block._shared_rowwise_fp8_down_prequant_t = down_weight.prequantized_rhs_for_dgrad
        else:
            up_rhs = block.shared_experts.w_up_gate.unsqueeze(0)
            down_rhs = block.shared_experts.w_down
            up_rhs_t = block.shared_experts.w_up_gate.transpose(0, 1).unsqueeze(0)
            down_rhs_t = block.shared_experts.w_down.transpose(1, 2)
            block._shared_rowwise_fp8_up_prequant = prequantize_scaled_grouped_mm_rhs(
                up_rhs,
                check_mat_b_version=False,
            )
            block._shared_rowwise_fp8_down_prequant = prequantize_scaled_grouped_mm_rhs(
                down_rhs,
                check_mat_b_version=False,
            )
            block._shared_rowwise_fp8_up_prequant_t = prequantize_scaled_grouped_mm_rhs(
                up_rhs_t,
                check_mat_b_version=False,
            )
            block._shared_rowwise_fp8_down_prequant_t = prequantize_scaled_grouped_mm_rhs(
                down_rhs_t,
                check_mat_b_version=False,
            )
        block._shared_rowwise_fp8_weight_versions = (
            int(block.shared_experts.w_up_gate._version),
            int(block.shared_experts.w_down._version),
        )


@torch.no_grad()
def refresh_rowwise_fp8_cache(block: MoEFusedV2TransformerBlock) -> None:
    block_cfg = block.rowwise_fp8
    routed_cfg = (
        block.routed_experts.rowwise_fp8 if block.routed_experts is not None else None
    )
    shared_enabled = _rowwise_fp8_enabled(block_cfg)
    routed_enabled = _rowwise_fp8_enabled(routed_cfg)
    if not shared_enabled and not routed_enabled:
        invalidate_rowwise_fp8_cache(block)
        return

    if block.routed_experts is not None:
        if routed_enabled:
            with nvtx.annotate("moe_rowwise_fp8_param_refresh_routed_weight_prequant"):
                block.routed_experts.refresh_rowwise_fp8_cache()
        else:
            block.routed_experts.invalidate_rowwise_fp8_cache()
    refresh_shared_rowwise_fp8_cache(block)


def shared_experts_forward1_rowwise_fp8(
    block: MoEFusedV2TransformerBlock,
    x: torch.Tensor,
    *,
    use_fast_accum: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert block.shared_experts is not None
    if x.ndim == 3:
        B, S, D = x.shape
        x2 = x.reshape(B * S, D)
    elif x.ndim == 2:
        x2 = x
        D = x.shape[1]
    else:
        raise RuntimeError(
            "shared_experts_forward1_rowwise_fp8 expects x to be [B, S, D] or [N, D], "
            f"got shape={tuple(x.shape)}"
        )
    E, H = block.shared_experts.num_experts, block.shared_experts.hidden_size
    BS = x2.shape[0]

    up_prequant = block._shared_rowwise_fp8_up_prequant
    up_prequant_t = block._shared_rowwise_fp8_up_prequant_t
    if up_prequant is None or up_prequant_t is None:
        raise RuntimeError("shared rowwise FP8 up/gate prequant buffers were not initialized")
    offs = torch.tensor([BS], device=x.device, dtype=torch.int32)
    up_kwargs = dict(
        offs=offs,
        use_fast_accum=use_fast_accum,
        prequantized_rhs=up_prequant,
        prequantized_rhs_for_dgrad=up_prequant_t,
    )
    cfg = block.rowwise_fp8
    fp8_only_params = cfg is not None and cfg.fp8_only_params
    up_anchor = block.shared_experts.w_up_gate.unsqueeze(0)
    if fp8_only_params:
        up_weight = getattr(block, "_shared_rowwise_fp8_up_gate_weight", None)
        if up_weight is None:
            raise RuntimeError("shared rowwise FP8 up/gate weight store is not initialized")
        up_kwargs["wgrad_sink"] = up_weight
        up_kwargs["wgrad_sink_squeeze_first_dim"] = True
        up_anchor = up_anchor.detach()
        mm_impl = scaled_grouped_mm_q_fp8_weight
    else:
        mm_impl = scaled_grouped_mm_q
    up_gate = mm_impl(
        x2,
        up_anchor,
        **up_kwargs,
    )
    up_gate = up_gate.view(BS, E, 2, H).permute(1, 0, 2, 3)
    up, gate = up_gate.unbind(dim=2)
    return up, gate


def shared_experts_forward_rowwise_fp8(
    block: MoEFusedV2TransformerBlock,
    x: torch.Tensor,
    *,
    use_fast_accum: bool,
) -> torch.Tensor:
    assert block.shared_experts is not None
    if x.ndim != 3:
        raise RuntimeError(
            "shared_experts_forward_rowwise_fp8 expects x to be [B, S, D], "
            f"got shape={tuple(x.shape)}"
        )
    B, S, D = x.shape
    E, H = block.shared_experts.num_experts, block.shared_experts.hidden_size
    BS = B * S

    x2 = x.reshape(BS, D)
    up_prequant = block._shared_rowwise_fp8_up_prequant
    up_prequant_t = block._shared_rowwise_fp8_up_prequant_t
    if up_prequant is None or up_prequant_t is None:
        raise RuntimeError("shared rowwise FP8 up/gate prequant buffers were not initialized")
    up_offs = torch.tensor([BS], device=x.device, dtype=torch.int32)
    up_kwargs = dict(
        offs=up_offs,
        use_fast_accum=use_fast_accum,
        prequantized_rhs=up_prequant,
        prequantized_rhs_for_dgrad=up_prequant_t,
    )
    cfg = block.rowwise_fp8
    fp8_only_params = cfg is not None and cfg.fp8_only_params
    up_anchor = block.shared_experts.w_up_gate.unsqueeze(0)
    if fp8_only_params:
        up_weight = getattr(block, "_shared_rowwise_fp8_up_gate_weight", None)
        if up_weight is None:
            raise RuntimeError("shared rowwise FP8 up/gate weight store is not initialized")
        up_kwargs["wgrad_sink"] = up_weight
        up_kwargs["wgrad_sink_squeeze_first_dim"] = True
        up_anchor = up_anchor.detach()
        up_mm_impl = scaled_grouped_mm_q_fp8_weight
    else:
        up_mm_impl = scaled_grouped_mm_q
    up_gate = up_mm_impl(
        x2,
        up_anchor,
        **up_kwargs,
    )

    up_gate = up_gate.view(BS, E, 2, H).permute(1, 0, 2, 3)
    up, gate = up_gate.unbind(dim=2)
    gate = torch.nn.functional.silu(gate)
    hidden = up * gate

    down_prequant = block._shared_rowwise_fp8_down_prequant
    down_prequant_t = block._shared_rowwise_fp8_down_prequant_t
    if down_prequant is None or down_prequant_t is None:
        raise RuntimeError("shared rowwise FP8 down prequant buffers were not initialized")
    hidden_2d = hidden.reshape(E * BS, -1)
    down_offs = torch.arange(BS, E * BS + 1, BS, device=hidden.device, dtype=torch.int32)
    down_kwargs = dict(
        offs=down_offs,
        use_fast_accum=use_fast_accum,
        prequantized_rhs=down_prequant,
        prequantized_rhs_for_dgrad=down_prequant_t,
    )
    down_anchor = block.shared_experts.w_down
    if fp8_only_params:
        down_weight = getattr(block, "_shared_rowwise_fp8_down_weight", None)
        if down_weight is None:
            raise RuntimeError("shared rowwise FP8 down weight store is not initialized")
        down_kwargs["wgrad_sink"] = down_weight
        down_anchor = down_anchor.detach()
        down_mm_impl = scaled_grouped_mm_q_fp8_weight
    else:
        down_mm_impl = scaled_grouped_mm_q
    out_2d = down_mm_impl(
        hidden_2d,
        down_anchor,
        **down_kwargs,
    )
    return out_2d.view(E, BS, D).view(E, B, S, D)


def shared_experts_forward2_rowwise_fp8(
    block: MoEFusedV2TransformerBlock,
    up: torch.Tensor,
    gate: torch.Tensor,
    xshape: torch.Size,
    *,
    use_fast_accum: bool,
) -> torch.Tensor:
    assert block.shared_experts is not None
    E, _H = block.shared_experts.num_experts, block.shared_experts.hidden_size
    B, S, D = xshape
    BS = B * S

    gate = torch.nn.functional.silu(gate)
    hidden = up * gate

    down_prequant = block._shared_rowwise_fp8_down_prequant
    down_prequant_t = block._shared_rowwise_fp8_down_prequant_t
    if down_prequant is None or down_prequant_t is None:
        raise RuntimeError("shared rowwise FP8 down prequant buffers were not initialized")
    hidden_2d = hidden.reshape(E * BS, -1)
    offs = torch.arange(BS, E * BS + 1, BS, device=hidden.device, dtype=torch.int32)
    down_kwargs = dict(
        offs=offs,
        use_fast_accum=use_fast_accum,
        prequantized_rhs=down_prequant,
        prequantized_rhs_for_dgrad=down_prequant_t,
    )
    cfg = block.rowwise_fp8
    fp8_only_params = cfg is not None and cfg.fp8_only_params
    down_anchor = block.shared_experts.w_down
    if fp8_only_params:
        down_weight = getattr(block, "_shared_rowwise_fp8_down_weight", None)
        if down_weight is None:
            raise RuntimeError("shared rowwise FP8 down weight store is not initialized")
        down_kwargs["wgrad_sink"] = down_weight
        down_anchor = down_anchor.detach()
        mm_impl = scaled_grouped_mm_q_fp8_weight
    else:
        mm_impl = scaled_grouped_mm_q
    out_2d = mm_impl(
        hidden_2d,
        down_anchor,
        **down_kwargs,
    )
    return out_2d.view(E, BS, D).view(E, B, S, D)
