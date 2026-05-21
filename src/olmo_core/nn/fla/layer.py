"""
Compatibility module for old ``olmo_core.nn.fla.layer`` config paths.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention.flash_linear_attn_api import (
    dispatch_causal_conv1d,
    dispatch_chunk_gated_delta_rule,
    has_fla,
)
from olmo_core.nn.attention.recurrent import GatedDeltaNet, GatedDeltaNetConfig
from olmo_core.nn.buffer_cache import BufferCache


FLA = GatedDeltaNet


@dataclass
class FLAConfig(GatedDeltaNetConfig):
    """
    Legacy FLA config adapter.

    Older checkpoints used ``FLAConfig(name="GatedDeltaNet", fla_layer_kwargs=...)``.
    The current equivalent is :class:`GatedDeltaNetConfig`, so this class translates
    the old fields during config decoding.
    """

    name: Optional[str] = field(default=None, repr=False, compare=False)
    fla_layer_kwargs: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self, type: Optional[str] = None):
        super().__post_init__(type=type)

        if self.name is not None and self.name != "GatedDeltaNet":
            raise OLMoConfigurationError(
                f"Legacy FLA layer '{self.name}' cannot be converted to GatedDeltaNetConfig"
            )
        if self.fla_layer_kwargs.get("use_gate", True) is not True:
            raise OLMoConfigurationError("Legacy FLAConfig with use_gate=False is not supported")
        if self.fla_layer_kwargs.get("use_short_conv", True) is not True:
            raise OLMoConfigurationError(
                "Legacy FLAConfig with use_short_conv=False is not supported"
            )

        for old_key, new_key in {
            "num_heads": "n_heads",
            "n_heads": "n_heads",
            "num_v_heads": "n_v_heads",
            "n_v_heads": "n_v_heads",
            "n_kv_heads": "n_v_heads",
            "head_dim": "head_dim",
            "expand_v": "expand_v",
            "allow_neg_eigval": "allow_neg_eigval",
            "conv_size": "conv_size",
            "conv_bias": "conv_bias",
            "norm_eps": "norm_eps",
        }.items():
            if old_key in self.fla_layer_kwargs:
                setattr(self, new_key, self.fla_layer_kwargs[old_key])

        # Match the historical GatedDeltaNet default when decoding an old FLAConfig.
        if self.name == "GatedDeltaNet" and "allow_neg_eigval" not in self.fla_layer_kwargs:
            self.allow_neg_eigval = False

        self.name = None
        self.fla_layer_kwargs = {}

    def build(
        self,
        d_model: int,
        n_heads: int | None = None,
        *,
        layer_idx: int = 0,
        n_layers: int = 1,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> GatedDeltaNet:
        if n_heads is not None:
            self.n_heads = n_heads
        return super().build(
            d_model,
            layer_idx=layer_idx,
            n_layers=n_layers,
            init_device=init_device,
            cache=cache,
        )


__all__ = [
    "FLA",
    "FLAConfig",
    "GatedDeltaNet",
    "GatedDeltaNetConfig",
    "dispatch_causal_conv1d",
    "dispatch_chunk_gated_delta_rule",
    "has_fla",
]
