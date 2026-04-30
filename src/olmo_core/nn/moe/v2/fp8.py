from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional, Tuple

import torch

from olmo_core.config import Config, StrEnum
from olmo_core.kernels import prequantize_scaled_grouped_mm_rhs, scaled_grouped_mm_q

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

    def validate(self) -> None:
        if self.block_size != 32:
            raise ValueError(
                f"Only block_size=32 is supported for MoE rowwise FP8 (got {self.block_size})"
            )

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
            raise RuntimeError(
                "torch.nn.functional.scaled_grouped_mm is required for MoE rowwise FP8"
            )


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
        "rowwise_fp8 must be MoERowwiseFP8Config, mapping/dict, or None " f"(got {type(value)!r})"
    )


def reset_shared_rowwise_fp8_cache(block: MoEFusedV2TransformerBlock) -> None:
    block._shared_rowwise_fp8_up_prequant = None
    block._shared_rowwise_fp8_down_prequant = None
    block._shared_rowwise_fp8_up_prequant_t = None
    block._shared_rowwise_fp8_down_prequant_t = None
    block._shared_rowwise_fp8_weight_versions = None


def invalidate_rowwise_fp8_cache(block: MoEFusedV2TransformerBlock) -> None:
    if block.routed_experts is not None:
        block.routed_experts.invalidate_rowwise_fp8_cache()
    reset_shared_rowwise_fp8_cache(block)


@torch.no_grad()
def refresh_rowwise_fp8_cache(block: MoEFusedV2TransformerBlock) -> None:
    cfg = block.rowwise_fp8
    if cfg is None or not cfg.enabled:
        invalidate_rowwise_fp8_cache(block)
        return
    if block.routed_experts is not None:
        block.routed_experts.refresh_rowwise_fp8_cache()
    if block.shared_experts is None:
        reset_shared_rowwise_fp8_cache(block)
        return
    if block.shared_experts.w_up_gate.device.type != "cuda":
        reset_shared_rowwise_fp8_cache(block)
        return

    up_rhs = block.shared_experts.w_up_gate.unsqueeze(0)
    down_rhs = block.shared_experts.w_down
    up_rhs_t = block.shared_experts.w_up_gate.transpose(0, 1).unsqueeze(0)
    down_rhs_t = block.shared_experts.w_down.transpose(1, 2)
    block._shared_rowwise_fp8_up_prequant = prequantize_scaled_grouped_mm_rhs(up_rhs)
    block._shared_rowwise_fp8_down_prequant = prequantize_scaled_grouped_mm_rhs(down_rhs)
    block._shared_rowwise_fp8_up_prequant_t = prequantize_scaled_grouped_mm_rhs(up_rhs_t)
    block._shared_rowwise_fp8_down_prequant_t = prequantize_scaled_grouped_mm_rhs(down_rhs_t)
    block._shared_rowwise_fp8_weight_versions = (
        int(block.shared_experts.w_up_gate._version),
        int(block.shared_experts.w_down._version),
    )


def maybe_refresh_shared_rowwise_fp8_cache(block: MoEFusedV2TransformerBlock) -> None:
    cfg = block.rowwise_fp8
    if cfg is None or not cfg.enabled:
        return
    if block.shared_experts is None:
        return
    versions = (
        int(block.shared_experts.w_up_gate._version),
        int(block.shared_experts.w_down._version),
    )
    if (
        block._shared_rowwise_fp8_up_prequant is None
        or block._shared_rowwise_fp8_down_prequant is None
        or block._shared_rowwise_fp8_weight_versions != versions
    ):
        refresh_rowwise_fp8_cache(block)


def shared_experts_forward1_rowwise_fp8(
    block: MoEFusedV2TransformerBlock,
    x: torch.Tensor,
    *,
    use_fast_accum: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert block.shared_experts is not None
    B, S, D = x.shape
    E, H = block.shared_experts.num_experts, block.shared_experts.hidden_size
    BS = B * S

    maybe_refresh_shared_rowwise_fp8_cache(block)
    x2 = x.reshape(BS, D)
    offs = torch.tensor([BS], device=x.device, dtype=torch.int32)
    up_kwargs = dict(
        offs=offs,
        use_fast_accum=use_fast_accum,
        prequantized_rhs=block._shared_rowwise_fp8_up_prequant,
    )
    up_prequant_t = getattr(block, "_shared_rowwise_fp8_up_prequant_t", None)
    if up_prequant_t is not None:
        up_kwargs["prequantized_rhs_for_dgrad"] = up_prequant_t
    up_gate = scaled_grouped_mm_q(
        x2,
        block.shared_experts.w_up_gate.unsqueeze(0),
        **up_kwargs,
    )
    up_gate = up_gate.view(BS, E, 2, H).permute(1, 0, 2, 3)
    up, gate = up_gate.unbind(dim=2)
    return up, gate


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

    maybe_refresh_shared_rowwise_fp8_cache(block)
    hidden_2d = hidden.reshape(E * BS, -1)
    offs = torch.arange(BS, E * BS + 1, BS, device=hidden.device, dtype=torch.int32)
    down_kwargs = dict(
        offs=offs,
        use_fast_accum=use_fast_accum,
        prequantized_rhs=block._shared_rowwise_fp8_down_prequant,
    )
    down_prequant_t = getattr(block, "_shared_rowwise_fp8_down_prequant_t", None)
    if down_prequant_t is not None:
        down_kwargs["prequantized_rhs_for_dgrad"] = down_prequant_t
    out_2d = scaled_grouped_mm_q(
        hidden_2d,
        block.shared_experts.w_down,
        **down_kwargs,
    )
    return out_2d.view(E, BS, D).view(E, B, S, D)
