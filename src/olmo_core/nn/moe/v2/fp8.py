from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from olmo_core.config import Config, StrEnum


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
